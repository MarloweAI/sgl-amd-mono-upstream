"""How many DeepSeek-V4.1 verify steps the plugin served at each row count, in a served process:
with this directory on ``PYTHONPATH``, each routed step (``spec._Built.forward``) adds one to its
row count's device counter, captured into the graphs, so every replay counts. Every 60 s each
process prints

    WIDTH_HIST rank=<r> rows=<m> path=<mono|wide> steps=<n>

A test tool only (no timing)."""

import importlib.abc
import importlib.util
import os
import sys
import threading
import time

TARGET = "sgl_amd_mono.deepseek_v41.spec"
PERIOD_S = 60
MAX = 1024


def _install(mod):
    import torch

    state = {}
    lock = threading.Lock()
    begin, end = torch.cuda.CUDAGraph.capture_begin, torch.cuda.CUDAGraph.capture_end

    def capture_begin(self, *a, **k):
        lock.acquire()
        try:
            return begin(self, *a, **k)
        except BaseException:
            lock.release()
            raise

    def capture_end(self, *a, **k):
        try:
            return end(self, *a, **k)
        finally:
            lock.release()

    torch.cuda.CUDAGraph.capture_begin = capture_begin
    torch.cuda.CUDAGraph.capture_end = capture_end

    def report(device):
        torch.cuda.set_device(device)
        side = torch.cuda.Stream(device)
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        while True:
            time.sleep(PERIOD_S)
            with lock, torch.cuda.stream(side):
                hist = state["hist"].to("cpu").tolist()
            for m, n in enumerate(hist):
                if n:
                    path = "mono" if m <= mod.MAX_ROWS else "wide"
                    print(f"WIDTH_HIST rank={rank} rows={m} path={path} steps={n}", file=sys.stderr, flush=True)

    forward = mod._Built.forward

    def counted(self, plan, input_ids, *args, **kwargs):
        dev = input_ids.device
        if "hist" not in state:
            assert not torch.cuda.is_current_stream_capturing(), "no eager step first"
            state["hist"] = torch.zeros(MAX, dtype=torch.int64, device=dev)
            state["one"] = torch.ones((), dtype=torch.int64, device=dev)
            threading.Thread(target=report, args=(dev,), daemon=True).start()
        state["hist"][plan.key.rows].add_(state["one"])
        return forward(self, plan, input_ids, *args, **kwargs)

    mod._Built.forward = counted
    print(f"WIDTH_HIST installed pid={os.getpid()}", file=sys.stderr, flush=True)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != TARGET:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        if spec is None:
            return None
        run = spec.loader.exec_module

        def exec_module(module):
            run(module)
            _install(module)

        spec.loader.exec_module = exec_module
        return spec


sys.meta_path.insert(0, _Finder())


def _next_sitecustomize():
    here = os.path.dirname(os.path.abspath(__file__))
    for d in sys.path:
        f = os.path.join(os.path.abspath(d or "."), "sitecustomize.py")
        if os.path.dirname(f) != here and os.path.isfile(f):
            exec(compile(open(f).read(), f, "exec"), {"__name__": "sitecustomize", "__file__": f})
            return


_next_sitecustomize()
