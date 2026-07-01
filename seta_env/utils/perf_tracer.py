"""
Hierarchical Performance Tracer — Chrome Perfetto compatible.

Produces trace files loadable in chrome://tracing or ui.perfetto.dev.
One PerfTracer per terminal_env trajectory; saved to the trial folder at end.
Multiple per-trajectory files can later be merged via PerfTracer.merge_files().

Tracks (tid constants):
    TID_ENV   (1) — TerminalEnvironment lifecycle stages
    TID_AGENT (2) — Agent iteration loop
    TID_MODEL (3) — LLM request / response round-trips
    TID_TOOL  (4) — Tool call execution

Basic usage:
    tracer = PerfTracer(session_id="my_task_t0_abc123")

    async with tracer.span("1_reset_env", cat="env"):
        ...

    async with tracer.span("model_request", cat="model",
                           tid=PerfTracer.TID_MODEL,
                           args={"iteration": 3}):
        ...

    tracer.instant("task_finished", cat="agent")
    tracer.save("/path/to/trial/perf_trace.json")

Merging traces across runs:
    PerfTracer.merge_files(
        ["/trials/t0/perf_trace.json", "/trials/t1/perf_trace.json"],
        output_path="/logs/merged_trace.json",
    )
"""

import json
import time
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Dict, List, Optional, Union


class PerfTracer:
    """Chrome JSON tracing (Perfetto-compatible) performance recorder.

    Thread/async safety: all mutations append to a list; CPython's GIL makes
    individual list.append() atomic, so concurrent async tasks sharing one
    PerfTracer instance are safe without extra locking.
    """

    # ------------------------------------------------------------------ #
    # Track-ID constants — map to distinct swimlanes in Perfetto           #
    # ------------------------------------------------------------------ #
    TID_ENV   = 1   # TerminalEnvironment lifecycle
    TID_AGENT = 2   # Agent iteration loop
    TID_MODEL = 3   # LLM request / response
    TID_TOOL  = 4   # Tool call execution

    _TRACK_NAMES: Dict[int, str] = {
        TID_ENV:   "Environment",
        TID_AGENT: "Agent",
        TID_MODEL: "Model",
        TID_TOOL:  "Tools",
    }

    # ------------------------------------------------------------------ #
    # Construction                                                          #
    # ------------------------------------------------------------------ #

    def __init__(self, session_id: str, pid: int = 1) -> None:
        """
        Args:
            session_id: Human-readable label for this trajectory, e.g.
                        "my_task_t0_abc123".  Shown as the process name in
                        Perfetto.
            pid:        Logical process ID.  Use distinct values when merging
                        traces from multiple trajectories into one file so
                        their tracks don't collide.
        """
        self.session_id = session_id
        self._pid = pid
        self._events: List[Dict[str, Any]] = []

        # Reference times: wall-clock + monotonic pair so that
        # timestamps are expressed as real wall-clock microseconds but
        # avoid the drift/jumps of repeated time.time() calls.
        self._t0_wall  = time.time()
        self._t0_mono  = time.monotonic()

        self._emit_metadata()

    # ------------------------------------------------------------------ #
    # Internal helpers                                                      #
    # ------------------------------------------------------------------ #

    def _now_us(self) -> int:
        """Microseconds since Unix epoch, using monotonic clock for accuracy."""
        elapsed = time.monotonic() - self._t0_mono
        return int((self._t0_wall + elapsed) * 1_000_000)

    def _emit_metadata(self) -> None:
        """Emit Perfetto metadata events (process name, track names/order)."""
        self._events.append({
            "ph": "M", "name": "process_name",
            "pid": self._pid, "tid": 0,
            "args": {"name": self.session_id},
        })
        for tid, track_name in self._TRACK_NAMES.items():
            self._events.append({
                "ph": "M", "name": "thread_name",
                "pid": self._pid, "tid": tid,
                "args": {"name": track_name},
            })
            # thread_sort_index keeps the swimlanes in a predictable order
            self._events.append({
                "ph": "M", "name": "thread_sort_index",
                "pid": self._pid, "tid": tid,
                "args": {"sort_index": tid},
            })

    # ------------------------------------------------------------------ #
    # Low-level event API                                                   #
    # ------------------------------------------------------------------ #

    def begin(
        self,
        name: str,
        *,
        cat: str = "",
        tid: int = TID_ENV,
        args: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit a begin (ph='B') event."""
        ev: Dict[str, Any] = {
            "ph": "B", "name": name, "cat": cat,
            "pid": self._pid, "tid": tid, "ts": self._now_us(),
        }
        if args:
            ev["args"] = args
        self._events.append(ev)

    def end(
        self,
        name: str,
        *,
        cat: str = "",
        tid: int = TID_ENV,
        args: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit an end (ph='E') event."""
        ev: Dict[str, Any] = {
            "ph": "E", "name": name, "cat": cat,
            "pid": self._pid, "tid": tid, "ts": self._now_us(),
        }
        if args:
            ev["args"] = args
        self._events.append(ev)

    def instant(
        self,
        name: str,
        *,
        cat: str = "",
        tid: int = TID_ENV,
        scope: str = "t",
        args: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit an instant (ph='i') event — no duration, just a marker."""
        ev: Dict[str, Any] = {
            "ph": "i", "name": name, "cat": cat,
            "pid": self._pid, "tid": tid, "ts": self._now_us(), "s": scope,
        }
        if args:
            ev["args"] = args
        self._events.append(ev)

    # ------------------------------------------------------------------ #
    # High-level context-manager API                                        #
    # ------------------------------------------------------------------ #

    @asynccontextmanager
    async def span(
        self,
        name: str,
        *,
        cat: str = "",
        tid: int = TID_ENV,
        args: Optional[Dict[str, Any]] = None,
    ):
        """Async context manager: emits begin on enter, end on exit.

        The end event is emitted even if an exception is raised so the trace
        always has balanced begin/end pairs.

        Usage::

            async with tracer.span("model_request", cat="model",
                                   tid=PerfTracer.TID_MODEL,
                                   args={"iteration": 3}):
                response = await model.generate(...)
        """
        self.begin(name, cat=cat, tid=tid, args=args)
        try:
            yield self
        finally:
            self.end(name, cat=cat, tid=tid)

    @contextmanager
    def span_sync(
        self,
        name: str,
        *,
        cat: str = "",
        tid: int = TID_ENV,
        args: Optional[Dict[str, Any]] = None,
    ):
        """Synchronous variant of :meth:`span` for non-async code."""
        self.begin(name, cat=cat, tid=tid, args=args)
        try:
            yield self
        finally:
            self.end(name, cat=cat, tid=tid)

    # ------------------------------------------------------------------ #
    # Persistence                                                           #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        """Return the trace as a Chrome JSON tracing dict."""
        return {
            "traceEvents": list(self._events),
            "displayTimeUnit": "ms",
            "metadata": {
                "session_id": self.session_id,
                "recorded_at_wall": self._t0_wall,
            },
        }

    def save(self, path: str) -> None:
        """Write the trace to *path* in Chrome JSON format (compact)."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, separators=(",", ":"))

    # ------------------------------------------------------------------ #
    # Merging utilities                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def merge(
        tracers: List[Union["PerfTracer", Dict[str, Any]]],
        output_path: str,
    ) -> None:
        """Merge in-memory tracers into one Perfetto file.

        Each tracer gets a distinct pid so their tracks don't overlap.
        """
        all_events: List[Dict[str, Any]] = []
        for i, t in enumerate(tracers):
            d = t.to_dict() if isinstance(t, PerfTracer) else t
            for ev in d.get("traceEvents", []):
                ev = dict(ev)
                ev["pid"] = i
                all_events.append(ev)
        with open(output_path, "w") as f:
            json.dump(
                {"traceEvents": all_events, "displayTimeUnit": "ms"},
                f, separators=(",", ":"),
            )

    @staticmethod
    def merge_files(paths: List[str], output_path: str) -> None:
        """Merge saved per-trajectory trace files into one Perfetto file.

        Example — collect all traces from a logging directory::

            import glob
            trace_files = glob.glob("/logs/trials/**/perf_trace.json", recursive=True)
            PerfTracer.merge_files(trace_files, "/logs/merged_trace.json")
        """
        all_events: List[Dict[str, Any]] = []
        for i, path in enumerate(paths):
            with open(path) as f:
                d = json.load(f)
            for ev in d.get("traceEvents", []):
                ev = dict(ev)
                ev["pid"] = i
                all_events.append(ev)
        with open(output_path, "w") as f:
            json.dump(
                {"traceEvents": all_events, "displayTimeUnit": "ms"},
                f, separators=(",", ":"),
            )


# --------------------------------------------------------------------------- #
# CLI entry point                                                               #
# --------------------------------------------------------------------------- #

def _main() -> None:
    import argparse
    import glob as _glob
    import os

    parser = argparse.ArgumentParser(
        description=(
            "Merge all perf_trace.json files found recursively under INPUT_DIR "
            "into a single Chrome/Perfetto trace file."
        )
    )
    parser.add_argument(
        "-i", "--input_dir",
        help="Root directory to search for perf_trace.json files.",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help=(
            "Output path for the merged trace file. "
            "Defaults to <input_dir>/merged_trace.json."
        ),
    )
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_path = args.output or os.path.join(input_dir, "merged_trace.json")

    pattern = os.path.join(input_dir, "**", "perf_trace.json")
    paths = sorted(_glob.glob(pattern, recursive=True))

    # Exclude the output file itself in case it was named perf_trace.json
    paths = [p for p in paths if os.path.abspath(p) != os.path.abspath(output_path)]

    if not paths:
        print(f"No perf_trace.json files found under {input_dir}")
        return

    print(f"Merging {len(paths)} trace file(s) -> {output_path}")
    for p in paths:
        print(f"  {p}")

    PerfTracer.merge_files(paths, output_path)
    print("Done.")


if __name__ == "__main__":
    _main()
