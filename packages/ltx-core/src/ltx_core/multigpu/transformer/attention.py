import os

import torch
import torch.distributed as dist

from ltx_core.model.transformer.attention import AttentionCallable, MaskedAttentionCallable

# Mirrors the kernel's DEFAULT_BARRIER_TIMEOUT_SECONDS (configs.cuh), which All2All converts to
# cycles via the device peak SM clock. Stored so the timeout can be read back to reset after a raise.
_DEFAULT_ALL2ALL_TIMEOUT_SECONDS = 10.0
_GRAPH_BREAK_PROBE = os.environ.get("SPARK_CP_GRAPH_BREAK_PROBE", "0") == "1"


class AttentionManager:
    def __init__(
        self,
        max_tokens: int,
        num_heads: int,
        head_dim: int,
        tensor_dtype: torch.dtype,
        group: torch.distributed.ProcessGroup,
        copy_out_: bool = False,
    ) -> None:
        # Lazy: ltx_kernels is an optional GPU-only dep, and this constructor already
        # requires CUDA -- so importing it here (not at module scope) keeps the multigpu
        # modules importable without the kernels installed (e.g. CPU CI test collection).
        from ltx_kernels import All2All  # noqa: PLC0415

        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        self.max_tokens = max_tokens
        hidden_dim = num_heads * head_dim
        num_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        self.copy_out = copy_out_
        self.async_qkv = os.environ.get("SPARK_CP_ASYNC_QKV", "0") == "1"
        buffer_seqlen = (max_tokens + self.world_size - 1) // self.world_size
        self.all2all_heads, self.all2all_q = (
            All2All(
                rank=self.rank,
                world_size=self.world_size,
                seqlen=buffer_seqlen,
                hidden_dim=hidden_dim,
                num_sms=num_sms,
                tensor_dtype=tensor_dtype,
                group=group,
            )
            for _ in range(2)
        )
        self.all2all_k, self.all2all_v = (
            All2All(
                rank=self.rank,
                world_size=self.world_size,
                seqlen=buffer_seqlen,
                hidden_dim=hidden_dim,
                num_sms=num_sms,
                tensor_dtype=tensor_dtype,
                group=group,
            )
            if not self.copy_out
            else self.all2all_q
            for _ in range(2)
        )
        self.group = group
        self._all2all_timeout_seconds = _DEFAULT_ALL2ALL_TIMEOUT_SECONDS
        # v4 head-group pipeline (PipelinedAll2AllAttention): its exchanges use their own NCCL communicator with a
        # high-priority internal stream, so that a group's NCCL kernel gets SMs while the attention grid runs.
        # v4 SPARK_CP_PACK_IN_GRAPH: the head pack before each exchange is graph code (NCCLAll2All
        # send_recv_heads_packed), so Inductor writes it from the producing kernel instead of a separate copy.
        self.pack_in_graph = (os.environ.get("SPARK_CP_PACK_IN_GRAPH", "0") == "1" and self.world_size > 1
                              and hasattr(self.all2all_q, "send_recv_heads_packed"))
        self.pipeline_groups = int(os.environ.get("SPARK_CP_PIPELINE_GROUPS", "1"))
        self.pipeline_group = None
        self.pipeline_stream = None
        self.pipeline_buffers: dict[tuple, list[torch.Tensor]] = {}  # per shape: send q/k/v, recv q/k/v, gathered
        if self.pipeline_groups > 1 and self.world_size > 1:
            options = dist.ProcessGroupNCCL.Options()
            options.is_high_priority_stream = os.environ.get("SPARK_CP_PIPELINE_HIGH_PRIORITY", "1") == "1"
            ranks = dist.get_process_group_ranks(group) if group is not None else list(range(self.world_size))
            self.pipeline_group = dist.new_group(ranks=ranks, backend="nccl", pg_options=options)
            self.pipeline_stream = torch.cuda.Stream(priority=-1)
        # Optional transport for the pipeline's exchanges instead of NCCL (2 ranks), set by the caller:
        # pipeline_transport_factory(group, nbytes, slots) -> transport, slots = [(send_off, recv_off, bytes)] of one
        # send and one receive buffer, the transport offering view(which, offset, shape, dtype) -> tensor,
        # post(slot, stream), wait(slot, stream), error() and close() (e.g. Sol-H3-Spark's spark_rdma: RDMA from
        # host-pinned buffers on DGX Spark, which has no GPUDirect RDMA). Created at the first pipelined call for
        # that call's shape (other shapes keep NCCL); if creating it fails on either rank, both keep NCCL.
        self.pipeline_transport_factory = None
        self.pipeline_transport = None
        self.pipeline_transport_key = None

    def set_seqlen_all2all(self, seqlens: list[int]) -> None:
        # Route through the wrappers so the registered custom ops' fake-impl
        # shape info gets updated alongside the C++ runtime's rank_tokens.
        self.all2all_q.set_rank_tokens(seqlens)
        self.all2all_k.set_rank_tokens(seqlens)
        self.all2all_v.set_rank_tokens(seqlens)
        self.all2all_heads.set_rank_tokens(seqlens)

    @property
    def all2all_timeout_seconds(self) -> float:
        """The all2all barrier (deadlock-detection) timeout, in seconds, applied to every instance."""
        return self._all2all_timeout_seconds

    @all2all_timeout_seconds.setter
    def all2all_timeout_seconds(self, seconds: float) -> None:
        # Raise it for the first ``torch.compile`` forward -- where one rank's recompile can delay its
        # all2all kernel launch past the steady-state timeout, tripping the barrier -- then reset to
        # the prior value. ``all2all_k``/``all2all_v`` may alias ``all2all_q`` (copy-out path);
        # setting twice is idempotent. Fan out first (it validates) so a rejected value leaves the
        # stored steady-state value untouched.
        for a2a in (self.all2all_q, self.all2all_k, self.all2all_v, self.all2all_heads):
            a2a.set_timeout_seconds(seconds)
        self._all2all_timeout_seconds = seconds

    def send_recv_qkv(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.async_qkv:
            t_q, t_k, t_v = self.all2all_q.send_recv_async([q, k, v], copy_out=self.copy_out)
            return t_q, t_k, t_v
        if self.pack_in_graph:
            return (self.all2all_q.send_recv_heads_packed(q), self.all2all_k.send_recv_heads_packed(k),
                    self.all2all_v.send_recv_heads_packed(v))
        t_q = self.all2all_q.send_recv_heads(q, copy_out=self.copy_out)
        t_k = self.all2all_k.send_recv_heads(k, copy_out=self.copy_out)
        t_v = self.all2all_v.send_recv_heads(v, copy_out=self.copy_out)
        return t_q, t_k, t_v

    def gather_heads(self, heads_local: torch.Tensor) -> torch.Tensor:
        out = self.all2all_heads.gather_heads(heads_local, copy_out=self.copy_out)
        return out


class _All2AllRedistribute:
    """Shared redistribute/gather pipeline for self-attention SP wrappers.
    Folds the head dim view-and-shuffle so the masked and unmasked variants only
    have to choose how to invoke the inner attention (with or without the mask
    kwarg) -- the rest of the SP plumbing is identical.
    """

    def __init__(self, manager: AttentionManager) -> None:
        self.manager = manager

    def redistribute(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        if heads % self.manager.world_size != 0:
            raise ValueError(f"heads ({heads}) must be divisible by world_size ({self.manager.world_size})")

        head_dim = q.shape[-1] // heads
        q = q.view(q.shape[0], q.shape[1], heads, head_dim)
        k = k.view(k.shape[0], k.shape[1], heads, head_dim)
        v = v.view(v.shape[0], v.shape[1], heads, head_dim)

        t_q, t_k, t_v = self.manager.send_recv_qkv(q, k, v)
        local_heads = heads // self.manager.world_size

        # `flatten` / `unflatten` collapse only the head dims, avoiding a `-1` in the
        # seq position -- that would otherwise be ambiguous if the seq is 0 for a
        # zero-token modality.
        t_q = t_q.flatten(-2)
        t_k = t_k.flatten(-2)
        t_v = t_v.flatten(-2)
        return t_q, t_k, t_v, local_heads, head_dim

    def gather(self, hidden_states: torch.Tensor, local_heads: int, head_dim: int) -> torch.Tensor:
        hidden_states = hidden_states.unflatten(-1, (local_heads, head_dim))
        hidden_states = self.manager.gather_heads(hidden_states)
        return hidden_states.flatten(-2)


class All2AllAttention(AttentionCallable):
    def __init__(self, manager: AttentionManager, original_attention: AttentionCallable):
        self._sp = _All2AllRedistribute(manager)
        self.original_attention = original_attention

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
    ) -> torch.Tensor:
        t_q, t_k, t_v, local_heads, head_dim = self._sp.redistribute(q, k, v, heads)
        hidden_states = self.original_attention(q=t_q, k=t_k, v=t_v, heads=local_heads)
        return self._sp.gather(hidden_states, local_heads, head_dim)


class PipelinedAll2AllAttention(AttentionCallable):
    """Self-attention SP wrapper with the Ulysses exchanges pipelined over head groups (v4, SPARK_CP_PIPELINE_GROUPS).

    Each rank's local heads (heads / world) split into G groups. After one pack of q, k, v into (G, world, tokens,
    heads/(world*G), head_dim), every group's q, k, v exchange is queued at once as one NCCL group call
    (batch_isend_irecv on the manager's high-priority communicator), so group g + 1 travels while group g's
    attention runs; each group's output goes back to its token owners as soon as its attention is done. Separate
    collectives do not overlap on GB10: the attention grid takes every SM before the next NCCL kernel can start
    (h3-unified-20261027 bench/bench_overlap.py). The inner attention is called once per group; Sol-Attn routes per
    (query block, head) and the dense path is per head too, so the result equals the unpipelined one
    (bench/bench_ltx_pipeline.py; SPARK_CP_PIPELINE_CHECK=N compares the first N calls in the run). Falls back to
    the unpipelined exchange for B != 1, cross-attention shapes or head counts G does not divide.
    """

    def __init__(self, manager: AttentionManager, original_attention: AttentionCallable):
        self._sp = _All2AllRedistribute(manager)
        self.manager = manager
        self.original_attention = original_attention
        # Set by the Stage 2 Sol-Attn adapter when its callable counts one layer per group sequence (part, parts).
        self.original_attention_accepts_parts = False
        self._checks = int(os.environ.get("SPARK_CP_PIPELINE_CHECK", "0"))

    def _reference(self, q, k, v, heads, **kwargs):
        t_q, t_k, t_v, local_heads, head_dim = self._sp.redistribute(q, k, v, heads)
        hidden_states = self.original_attention(q=t_q, k=t_k, v=t_v, heads=local_heads, **kwargs)
        return self._sp.gather(hidden_states, local_heads, head_dim)

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
    ) -> torch.Tensor:
        groups, world = self.manager.pipeline_groups, self.manager.world_size
        if (self.manager.pipeline_group is None or heads % (world * groups) or q.shape[0] != 1
                or k.shape != q.shape or v.shape != q.shape):
            if _GRAPH_BREAK_PROBE and q.shape[0] == 1 and k.shape == q.shape and v.shape == q.shape:
                return self._reference_outside_graph(q, k, v, heads)
            return self._reference(q, k, v, heads)
        return self._pipelined(q, k, v, heads, groups)

    # SPARK_CP_GRAPH_BREAK_PROBE=1 (diagnosis): the unpipelined exchange, same values, run outside the compiled graph
    # exactly where _pipelined is, to tell the graph break's cost from the pipeline's.
    @torch.compiler.disable
    def _reference_outside_graph(self, q, k, v, heads):
        return self._reference(q, k, v, heads)

    def _transport(self, key: tuple, t: int, hg: int, head_dim: int, groups: int, dtype: torch.dtype):
        """The manager's pipeline transport for this shape, created on first use (collective on both ranks)."""
        manager = self.manager
        if manager.pipeline_transport_factory is None or manager.world_size != 2:
            return None
        if manager.pipeline_transport_key is None:
            manager.pipeline_transport_key = key
            esize = torch.empty((), dtype=dtype).element_size()
            bq, bo = 3 * t * hg * head_dim * esize, t * hg * head_dim * esize
            slots = ([(g * bq, g * bq, bq) for g in range(groups)]
                     + [(groups * bq + g * bo, groups * bq + g * bo, bo) for g in range(groups)])
            transport, error = None, None
            try:
                transport = manager.pipeline_transport_factory(manager.group, groups * (bq + bo), slots)
            except Exception as exc:  # noqa: BLE001 -- any failure means NCCL on both ranks
                error = exc
            ok = torch.tensor([0 if transport is None else 1], dtype=torch.int32, device="cuda")
            dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=manager.group)
            if int(ok.item()) == 1:
                manager.pipeline_transport = transport
                print(f"[v4 transport] rank {manager.rank}: {type(transport).__name__}, {len(slots)} slots, "
                      f"{groups * (bq + bo) / 2**20:.1f} MiB each way, shape {key}", flush=True)
            else:
                if transport is not None:
                    transport.close()
                print(f"[v4 transport] rank {manager.rank}: unavailable ({error or 'failed on the peer'}), "
                      f"the pipeline keeps NCCL", flush=True)
        return manager.pipeline_transport if manager.pipeline_transport_key == key else None

    @torch.compiler.disable
    def _pipelined(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, groups: int) -> torch.Tensor:
        manager = self.manager
        if manager.pipeline_transport_factory is not None:
            _, t, hidden = q.shape
            hg, head_dim = heads // manager.world_size // groups, hidden // heads
            transport = self._transport((groups, manager.world_size, t, hg, head_dim, q.dtype), t, hg, head_dim,
                                        groups, q.dtype)
            if transport is not None:
                return self._pipelined_transport(transport, q, k, v, heads, groups)
        world, rank, group, comm = manager.world_size, manager.rank, manager.pipeline_group, manager.pipeline_stream
        _, t, hidden = q.shape
        head_dim = hidden // heads
        local_heads = heads // world
        hg = local_heads // groups
        peers = [(r, dist.get_global_rank(group, r)) for r in range(world) if r != rank]
        main = torch.cuda.current_stream()
        # x (1, t, world * G * hg * d) -> (G, world, t, hg, d): send[g, r] are my tokens' heads r*lh + g*hg .. for
        # rank r; my own share goes straight into its slot of the receive buffer.
        # Buffers live on the manager, one set per shape, reused by every layer: each call ends with the main stream
        # waiting for the exchange stream, so the next layer's pack cannot overwrite a buffer still in flight.
        key = (groups, world, t, hg, head_dim, q.dtype)
        buffers = manager.pipeline_buffers.get(key)
        if buffers is None:
            buffers = [torch.empty(key[:5], dtype=q.dtype, device=q.device) for _ in range(7)]
            manager.pipeline_buffers[key] = buffers
        send, recv, gathered = buffers[0:3], buffers[3:6], buffers[6]
        for x, s_, r_ in zip((q, k, v), send, recv):
            grouped = x.view(t, world, groups, hg, head_dim).permute(2, 1, 0, 3, 4)
            for peer, _ in peers:
                s_[:, peer].copy_(grouped[:, peer])
            r_[:, rank].copy_(grouped[:, rank])
        packed = torch.cuda.Event()
        packed.record(main)
        ready = [torch.cuda.Event() for _ in range(groups)]
        comm.wait_event(packed)
        with torch.cuda.stream(comm):
            for g in range(groups):
                ops = []
                for i in range(3):
                    for peer, global_peer in peers:
                        ops.append(dist.P2POp(dist.isend, send[i][g, peer], global_peer, group))
                        ops.append(dist.P2POp(dist.irecv, recv[i][g, peer], global_peer, group))
                for work in dist.batch_isend_irecv(ops):
                    work.wait()
                ready[g].record(comm)
        outputs = []
        for g in range(groups):
            main.wait_event(ready[g])
            q_g, k_g, v_g = (recv[i][g].view(1, world * t, hg * head_dim) for i in range(3))
            parts = {"part": g, "parts": groups} if self.original_attention_accepts_parts else {}
            out = self.original_attention(q=q_g, k=k_g, v=v_g, heads=hg, **parts).reshape(1, world * t, hg, head_dim)
            outputs.append(out)
            done = torch.cuda.Event()
            done.record(main)
            with torch.cuda.stream(comm):
                comm.wait_event(done)
                gathered[g, rank].copy_(out[0, rank * t:(rank + 1) * t])
                ops = []
                for peer, global_peer in peers:
                    ops.append(dist.P2POp(dist.isend, out[0, peer * t:(peer + 1) * t], global_peer, group))
                    ops.append(dist.P2POp(dist.irecv, gathered[g, peer], global_peer, group))
                for work in dist.batch_isend_irecv(ops):
                    work.wait()
        main.wait_stream(comm)
        # (G, world, t, hg, d) -> (1, t, heads * d) with rank r's heads r*lh + g*hg + j, as gather_heads.
        result = gathered.permute(2, 1, 0, 3, 4).reshape(1, t, heads * head_dim)
        if self._checks > 0:
            self._checks -= 1
            extra = {"count": False} if self.original_attention_accepts_parts else {}
            same = bool(torch.equal(result, self._reference(q, k, v, heads, **extra)))
            print(f"[v4 pipeline] rank {rank} groups {groups} heads {heads} tokens {t}: equal to the unpipelined "
                  f"exchange {same}", flush=True)
            if not same:
                raise RuntimeError("pipelined attention differs from the unpipelined exchange")
        return result


    def _pipelined_transport(self, rx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int,
                             groups: int) -> torch.Tensor:
        """_pipelined's schedule over the manager's transport (2 ranks). The pack writes the transport's send buffer;
        every group's q, k, v slot is posted at once on the main stream; per group the exchange stream waits for its
        slot and copies it into device memory (a transport such as spark_rdma keeps its buffers in host-pinned memory,
        which the GPU does not hold in L2, so attention must read a device copy) while the main stream runs the previous
        group's attention; each group's output goes back through its own slot. Slots are reused by every layer: a slot
        is posted only after the previous layer's consumers of it (the call ends with the main stream waiting for the
        exchange stream)."""
        manager = self.manager
        world, rank, comm = manager.world_size, manager.rank, manager.pipeline_stream
        peer = 1 - rank
        _, t, hidden = q.shape
        head_dim = hidden // heads
        hg = heads // world // groups
        main = torch.cuda.current_stream()
        key = (groups, world, t, hg, head_dim, q.dtype)
        buffers = manager.pipeline_buffers.get(key)
        if buffers is None:
            buffers = [torch.empty(key[:5], dtype=q.dtype, device=q.device) for _ in range(7)]
            manager.pipeline_buffers[key] = buffers
        recv, gathered = buffers[3:6], buffers[6]
        bq = 3 * t * hg * head_dim * q.element_size()
        send_qkv = rx.view("send", 0, (groups, 3, t, hg, head_dim), q.dtype)
        recv_qkv = rx.view("recv", 0, (groups, 3, t, hg, head_dim), q.dtype)
        send_out = rx.view("send", groups * bq, (groups, t, hg, head_dim), q.dtype)
        recv_out = rx.view("recv", groups * bq, (groups, t, hg, head_dim), q.dtype)
        for i, x in enumerate((q, k, v)):
            grouped = x.view(t, world, groups, hg, head_dim).permute(2, 1, 0, 3, 4)
            send_qkv[:, i].copy_(grouped[:, peer])
            recv[i][:, rank].copy_(grouped[:, rank])
        for g in range(groups):
            rx.post(g, main)
        ready = [torch.cuda.Event() for _ in range(groups)]
        comm.wait_stream(main)
        with torch.cuda.stream(comm):
            for g in range(groups):
                rx.wait(g, comm)
                for i in range(3):
                    recv[i][g, peer].copy_(recv_qkv[g, i])
                ready[g].record(comm)
        outputs = []
        for g in range(groups):
            main.wait_event(ready[g])
            q_g, k_g, v_g = (recv[i][g].view(1, world * t, hg * head_dim) for i in range(3))
            parts = {"part": g, "parts": groups} if self.original_attention_accepts_parts else {}
            out = self.original_attention(q=q_g, k=k_g, v=v_g, heads=hg, **parts).reshape(1, world * t, hg, head_dim)
            outputs.append(out)
            gathered[g, rank].copy_(out[0, rank * t:(rank + 1) * t])
            done = torch.cuda.Event()
            done.record(main)
            with torch.cuda.stream(comm):
                comm.wait_event(done)
                send_out[g].copy_(out[0, peer * t:(peer + 1) * t])
                rx.post(groups + g, comm)
                rx.wait(groups + g, comm)
                gathered[g, peer].copy_(recv_out[g])
        main.wait_stream(comm)
        for out in outputs:
            out.record_stream(comm)
        if rx.error():
            raise RuntimeError("v4 pipeline transport error")
        result = gathered.permute(2, 1, 0, 3, 4).reshape(1, t, heads * head_dim)
        if self._checks > 0:
            self._checks -= 1
            extra = {"count": False} if self.original_attention_accepts_parts else {}
            same = bool(torch.equal(result, self._reference(q, k, v, heads, **extra)))
            print(f"[v4 transport] rank {rank} groups {groups} heads {heads} tokens {t}: equal to the unpipelined "
                  f"exchange {same}", flush=True)
            if not same:
                raise RuntimeError("transport-pipelined attention differs from the unpipelined exchange")
        return result


class MaskedAll2AllAttention(MaskedAttentionCallable):
    def __init__(self, manager: AttentionManager, original_attention: MaskedAttentionCallable):
        self._sp = _All2AllRedistribute(manager)
        self.original_attention = original_attention

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        t_q, t_k, t_v, local_heads, head_dim = self._sp.redistribute(q, k, v, heads)
        hidden_states = self.original_attention(q=t_q, k=t_k, v=t_v, heads=local_heads, mask=mask)
        return self._sp.gather(hidden_states, local_heads, head_dim)


class _AudioAll2AllRedistribute:
    """Shared redistribute/gather pipeline for audio cross-attention SP wrappers.
    Q is sliced locally per rank (no cross-rank shuffle on Q because the audio
    sequence length is small enough to replicate); K/V are redistributed across
    ranks via ``send_recv_heads``; outputs are gathered via
    ``all_gather_into_tensor`` along the head dimension. The masked and unmasked
    variants share this plumbing and only differ in how they invoke the inner
    attention.
    """

    def __init__(self, manager: AttentionManager) -> None:
        self.manager = manager

    def redistribute(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        if heads % self.manager.world_size != 0:
            raise ValueError(f"heads ({heads}) must be divisible by world_size ({self.manager.world_size})")

        head_dim = q.shape[-1] // heads
        heads_per_rank = heads // self.manager.world_size
        rank = self.manager.rank

        q = q.view(q.shape[0], q.shape[1], heads, head_dim)
        k = k.view(k.shape[0], k.shape[1], heads, head_dim)
        v = v.view(v.shape[0], v.shape[1], heads, head_dim)

        t_q = q[:, :, heads_per_rank * rank : heads_per_rank * (rank + 1), :].clone()
        if self.manager.async_qkv:
            t_k, t_v = self.manager.all2all_k.send_recv_async([k, v], copy_out=self.manager.copy_out)
        elif self.manager.pack_in_graph:
            t_k = self.manager.all2all_k.send_recv_heads_packed(k)
            t_v = self.manager.all2all_v.send_recv_heads_packed(v)
        else:
            t_k = self.manager.all2all_k.send_recv_heads(k, copy_out=self.manager.copy_out)
            t_v = self.manager.all2all_v.send_recv_heads(v, copy_out=self.manager.copy_out)

        # `flatten` / `unflatten` collapse only the head dims, avoiding a `-1` in the
        # seq position -- that would otherwise be ambiguous if the seq is 0 for a
        # zero-token modality.
        t_q = t_q.flatten(-2)
        t_k = t_k.flatten(-2)
        t_v = t_v.flatten(-2)
        return t_q, t_k, t_v, heads_per_rank, head_dim

    def gather(self, hidden_states: torch.Tensor, heads_per_rank: int, head_dim: int) -> torch.Tensor:
        # (B, S, heads_per_rank, head_dim). Move head dim to dim 0 so all_gather_into_tensor
        # gathers along it; permute back after the collective.
        hidden_states = hidden_states.unflatten(-1, (heads_per_rank, head_dim)).permute(2, 0, 1, 3).contiguous()
        gathered = torch.empty(
            (heads_per_rank * self.manager.world_size, *hidden_states.shape[1:]),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        dist.all_gather_into_tensor(gathered, hidden_states, group=self.manager.group)
        # (heads, B, S, head_dim) -> (B, S, heads, head_dim) -> (B, S, heads * head_dim)
        return gathered.permute(1, 2, 0, 3).flatten(-2)


class AudioAll2AllAttention(AttentionCallable):
    """All2All attention for audio cross-attention (video_to_audio).
    Q is sliced locally per rank, K/V are redistributed via send_recv_heads,
    then outputs are gathered via all_gather across the head dimension.
    """

    def __init__(self, manager: AttentionManager, original_attention: AttentionCallable):
        self._sp = _AudioAll2AllRedistribute(manager)
        self.original_attention = original_attention

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
    ) -> torch.Tensor:
        t_q, t_k, t_v, heads_per_rank, head_dim = self._sp.redistribute(q, k, v, heads)
        hidden_states = self.original_attention(q=t_q, k=t_k, v=t_v, heads=heads_per_rank)
        return self._sp.gather(hidden_states, heads_per_rank, head_dim)


class MaskedAudioAll2AllAttention(MaskedAttentionCallable):
    """Masked counterpart to :class:`AudioAll2AllAttention`.
    No current caller invokes A2V / V2A cross-attention with a mask, so the SP
    mutator pre-installs an unmasked-only :class:`AudioAll2AllAttention` and the
    masked slot stays at the model default. Defined now so adding masked audio
    cross-attention later is just an SP-mutator change, not a missing-piece
    discovery.
    """

    def __init__(self, manager: AttentionManager, original_attention: MaskedAttentionCallable):
        self._sp = _AudioAll2AllRedistribute(manager)
        self.original_attention = original_attention

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        t_q, t_k, t_v, heads_per_rank, head_dim = self._sp.redistribute(q, k, v, heads)
        hidden_states = self.original_attention(q=t_q, k=t_k, v=t_v, heads=heads_per_rank, mask=mask)
        return self._sp.gather(hidden_states, heads_per_rank, head_dim)
