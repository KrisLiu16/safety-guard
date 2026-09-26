"""Fixed Triton autotuning for sharded evaluation (T036).

Triton 3.3 autotunes a kernel the first time it meets a key and keeps the choice inside the process. Several eval
processes sharing one L20 time each other's benchmark runs, so one of them can pick another config than a lone
process would, and every record it scores then differs a little (v4 with 4 shards: 69% of Run A lines differed,
max |d logprob| 0.1; with the choices fixed a 4-shard run matched the single process line for line). record writes
the config each autotuned kernel chose, at exit; pin loads that file into every kernel's cache before its first
launch, so only keys missing from the file are benchmarked (listed in state()["unpinned"]).
"""
from __future__ import annotations

import atexit
import json
from pathlib import Path

_state = {"mode": None, "pinned": 0, "unpinned": []}
_tuners = []


def kernel_name(tuner):
    fn = getattr(tuner, "base_fn", None) or tuner.fn
    return f"{fn.__module__}.{fn.__qualname__}"


def config_fields(config):
    return {"kwargs": dict(config.kwargs), "num_warps": config.num_warps, "num_stages": config.num_stages,
            "num_ctas": config.num_ctas, "maxnreg": getattr(config, "maxnreg", None)}


def dump():
    """{kernel: {json key: config fields}} for every autotuned kernel seen so far."""
    out = {}
    for tuner in _tuners:
        for key, config in tuner.cache.items():
            out.setdefault(kernel_name(tuner), {})[json.dumps(list(key))] = config_fields(config)
    return out


def install(mode, path, autotuner=None, write_at_exit=True):
    """mode "record" or "pin"; path: the JSON file (record writes it at exit). autotuner: the module holding
    Autotuner (tests pass a fake)."""
    if mode not in ("record", "pin"):
        raise ValueError(mode)
    if autotuner is None:
        from triton.runtime import autotuner
    table = json.loads(Path(path).read_text()) if mode == "pin" else {}
    original = autotuner.Autotuner.run
    _state["mode"] = mode

    def run(self, *args, **kwargs):
        if self not in _tuners:
            _tuners.append(self)
            for key_text, fields in table.get(kernel_name(self), {}).items():
                match = next((c for c in self.configs if config_fields(c) == fields), None)
                if match is None:
                    raise ValueError(f"{kernel_name(self)}: a pinned config is not among the kernel's configs")
                self.cache[tuple(json.loads(key_text))] = match
                _state["pinned"] += 1
        before = set(self.cache)
        out = original(self, *args, **kwargs)
        if mode == "pin":
            _state["unpinned"] += [[kernel_name(self), json.dumps(list(k))] for k in set(self.cache) - before]
        return out

    autotuner.Autotuner.run = run
    if mode == "record" and write_at_exit:
        atexit.register(lambda: Path(path).write_text(json.dumps(dump(), indent=1, sort_keys=True) + "\n"))


def state():
    return {"mode": _state["mode"], "pinned": _state["pinned"], "unpinned": list(_state["unpinned"])}
