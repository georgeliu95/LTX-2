"""CPU layout contracts for actual source classes; CUDA/Inductor needs the GPU probe."""
import ast
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

SOURCE = Path(__file__).parents[1] / "src/ltx_core/multigpu/transformer/attention.py"


class Tensor:
    def __init__(self, data):
        self.data = np.asarray(data)

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def device(self):
        return "cpu"

    def view(self, *shape):
        return Tensor(self.data.reshape(shape))

    def __getitem__(self, index):
        return Tensor(self.data[index])

    def clone(self):
        return Tensor(self.data.copy())

    def flatten(self, start_dim, end_dim=-1):
        start_dim %= self.data.ndim
        end_dim %= self.data.ndim
        return self.view(*(self.shape[:start_dim] + (-1,) + self.shape[end_dim + 1:]))

    def unflatten(self, dim, shape):
        return self.view(*(self.shape[:dim] + tuple(shape) + self.shape[dim + 1:]))

    def unbind(self, dim):
        return tuple(self[(slice(None),) * dim + (index,)] for index in range(self.shape[dim]))


@pytest.fixture
def classes():
    tree = ast.parse(SOURCE.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.ClassDef)
                and node.name in ("AttentionManager", "_AudioAll2AllRedistribute")]
    fake_torch = SimpleNamespace(Tensor=Tensor, dtype=np.dtype,
        distributed=SimpleNamespace(ProcessGroup=type("ProcessGroup", (), {})),
        cuda=SimpleNamespace(current_device=lambda: 0,
            get_device_properties=lambda device: SimpleNamespace(multi_processor_count=48)),
        stack=lambda tensors, dim: Tensor(np.stack([x.data for x in tensors], axis=dim)))
    namespace = {"torch": fake_torch, "dist": SimpleNamespace(get_rank=lambda group: 0,
        get_world_size=lambda group: 4), "os": __import__("os"), "_DEFAULT_ALL2ALL_TIMEOUT_SECONDS": 10.}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return SimpleNamespace(**namespace)


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_coalescing_preserves_baseline_layout_and_independent_outputs(classes, batch, rank, dtype):
    # Given offset-backed source tensors that distinguish rank/batch/token/head/K/V.
    world, tokens, heads, dim = 4, 5, 8, 3
    values = np.arange(world * batch * tokens * heads * dim).reshape(world, batch, tokens, heads, dim)
    sources = (values.astype(dtype), (values + 4096).astype(dtype))
    local = [Tensor(np.concatenate([np.zeros((batch, 1, heads, dim), dtype=dtype), x[rank]], axis=1)[:, 1:])
             for x in sources]
    q = Tensor(np.ones((batch, 7, heads * dim), dtype=dtype))
    calls = []
    class Exchange:
        def __init__(self, kind):
            self.kind = kind

        def send_recv_heads_packed(self, value):
            calls.append((self.kind, value.shape[0]))
            chosen = range(2) if value.shape[0] == 2 * batch else (self.kind,)
            np.testing.assert_array_equal(value.data, np.concatenate([sources[kind][rank] for kind in chosen], axis=0))
            received = [np.concatenate([sources[kind][peer, :, :, rank * 2:(rank + 1) * 2]
                                       for peer in range(world)], axis=1) for kind in chosen]
            return Tensor(np.concatenate(received, axis=0))
    manager = SimpleNamespace(world_size=world, rank=rank, async_qkv=False, pack_in_graph=True,
        copy_out=False, coalesce_audio_kv=False, all2all_k=Exchange(0), all2all_v=Exchange(1))
    redistribute = classes._AudioAll2AllRedistribute(manager)
    baseline = redistribute.redistribute(q, *(x.view(batch, tokens, heads * dim) for x in local), heads)
    calls.clear()
    manager.coalesce_audio_kv = True
    # When the actual source branch coalesces both payloads into a single exchange.
    actual = redistribute.redistribute(q, *(x.view(batch, tokens, heads * dim) for x in local), heads)
    # Then values/order/dtype match baseline, one exchange occurs, and outputs cannot overwrite each other.
    assert calls == [(0, 2 * batch)]
    for expected, result in zip(baseline[:3], actual[:3]):
        np.testing.assert_array_equal(expected.data, result.data)
        assert expected.dtype == result.dtype
        assert expected.shape == result.shape
    assert actual[3:] == (2, dim)
    saved_v = actual[2].data.copy()
    actual[1].data.fill(-1)
    np.testing.assert_array_equal(actual[2].data, saved_v)
    np.testing.assert_array_equal(local[0].data, sources[0][rank])
    retained = [value.data.copy() for value in actual[:3]]
    for source, own in zip(sources, local):
        source *= -1
        own.data *= -1
    changed = redistribute.redistribute(q, *(x.view(batch, tokens, heads * dim) for x in local), heads)
    for old, saved in zip(actual[:3], retained):
        np.testing.assert_array_equal(old.data, saved)
    np.testing.assert_array_equal(changed[1].data, -baseline[1].data)
    np.testing.assert_array_equal(changed[2].data, -baseline[2].data)


@pytest.mark.parametrize("packed,async_qkv", [(False, False), (True, True)])
def test_incompatible_configuration_fails_at_manager_construction(classes, monkeypatch, packed, async_qkv):
    # Given an explicit opt-in with an incompatible manager configuration.
    kernels = ModuleType("ltx_kernels")
    kernels.All2All = lambda **kwargs: SimpleNamespace(send_recv_heads_packed=lambda x: x)
    monkeypatch.setitem(__import__("sys").modules, "ltx_kernels", kernels)
    monkeypatch.setenv("SPARK_CP_COALESCE_AUDIO_KV", "1")
    monkeypatch.setenv("SPARK_CP_PACK_IN_GRAPH", str(int(packed)))
    monkeypatch.setenv("SPARK_CP_ASYNC_QKV", str(int(async_qkv)))
    monkeypatch.setenv("SPARK_CP_PIPELINE_GROUPS", "1")
    # When / Then: no fallback route is silently selected.
    with pytest.raises(ValueError):
        classes.AttentionManager(128, 8, 3, np.dtype("float16"), None)


def test_opt_in_is_captured_and_defaults_off(classes, monkeypatch):
    # Given a valid packed manager and no opt-in; when environment later changes; then its setting stays off.
    kernels = ModuleType("ltx_kernels")
    kernels.All2All = lambda **kwargs: SimpleNamespace(send_recv_heads_packed=lambda x: x)
    monkeypatch.setitem(__import__("sys").modules, "ltx_kernels", kernels)
    monkeypatch.delenv("SPARK_CP_COALESCE_AUDIO_KV", raising=False)
    monkeypatch.setenv("SPARK_CP_PACK_IN_GRAPH", "1")
    monkeypatch.setenv("SPARK_CP_ASYNC_QKV", "0")
    monkeypatch.setenv("SPARK_CP_PIPELINE_GROUPS", "1")
    manager = classes.AttentionManager(128, 8, 3, np.dtype("float16"), None)
    monkeypatch.setenv("SPARK_CP_COALESCE_AUDIO_KV", "1")
    assert not manager.coalesce_audio_kv


def test_mismatched_kv_fails_before_any_exchange(classes):
    # Given coalescing with incompatible K/V token counts; when redistributing; then fail explicitly before transport.
    manager = SimpleNamespace(world_size=4, rank=0, async_qkv=False, pack_in_graph=True,
                              coalesce_audio_kv=True)
    q, k, v = (Tensor(np.zeros((1, length, 24))) for length in (7, 5, 6))
    with pytest.raises(ValueError):
        classes._AudioAll2AllRedistribute(manager).redistribute(q, k, v, 8)
