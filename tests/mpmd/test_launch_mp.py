import os
import sys
import socket
import subprocess
import time
import json
from pathlib import Path

import pytest


HERE = Path(__file__).parent
MP_TEST = HERE / "run_test_potrs.py"


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _gpu_count_from_env() -> int:
    """Best-effort GPU count from CUDA_VISIBLE_DEVICES; returns -1 if unset."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cvd:
        return -1
    # Accept formats like "0,1" or "0" or "" (empty -> 0)
    parts = [p for p in cvd.split(",") if p.strip() != ""]
    return len(parts)


@pytest.mark.multi_gpu
@pytest.mark.parametrize("requested_procs", (1, 2, 3, 4))
def test_launch_mpmd_collect_results(requested_procs):
    """
    Launch mp_test.py with two processes on localhost and assert both complete.

    This test is skipped when we can't confidently run multi-GPU locally
    (no GPUs visible and no CUDA_VISIBLE_DEVICES preset).
    """

    gpu_count = _gpu_count_from_env()
    if gpu_count == -1:
        # Fall back: detect via /dev/nvidia*
        has_gpu_nodes = any(Path("/dev").glob("nvidia[0-9]*"))
        if not has_gpu_nodes:
            pytest.skip("No GPUs detected and CUDA_VISIBLE_DEVICES not set; skipping MPMD GPU test")
    elif gpu_count != requested_procs:
        pytest.skip(
            f"Need {requested_procs} GPUs in CUDA_VISIBLE_DEVICES to run this test (have {gpu_count})"
        )

    port = _find_free_port()
    coord = f"127.0.0.1:{port}"

    procs = []
    logs = []
    env = os.environ.copy()
    # Make JAX friendlier for tests
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    # Print launch context to aid debugging under -s
    print(
        "\n".join(
            [
                "[launcher] starting mp run",
                f"[launcher] requested_procs={requested_procs}",
                f"[launcher] coord={coord}",
                f"[launcher] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')}",
                f"[launcher] XLA_PYTHON_CLIENT_PREALLOCATE={env.get('XLA_PYTHON_CLIENT_PREALLOCATE','')}",
            ]
        )
    )

    for i in range(requested_procs):
        p = subprocess.Popen(
            [sys.executable, "-u", str(MP_TEST), coord, str(i), str(requested_procs)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(HERE),
            text=True,
            bufsize=1,
        )
        procs.append(p)

    # Collect output with a timeout to avoid hanging the test suite
    deadline = time.time() + 150
    for idx, p in enumerate(procs):
        out_chunks = []
        while p.poll() is None and time.time() < deadline:
            line = p.stdout.readline()
            if line:
                # Stream output to console with proc prefix for readability under -s
                out_chunks.append(line)
        # Drain any remaining output
        remaining = p.stdout.read() or ""
        if remaining:
            out_chunks.append(remaining)
        logs.append("".join(out_chunks))

    # Ensure all processes exited
    exits = [p.wait(timeout=5) for p in procs]
    for idx, code in enumerate(exits):
        if code != 0:
            print(f"===== mp_test proc {idx} combined output =====")
            print(logs[idx])
        assert code == 0, f"mp_test process {idx} failed with exit code {code}"

    # Parse MPTEST JSON lines and aggregate results
    parsed = []
    per_proc_seen = set()
    discover = {}
    summary = {}
    tests_by_proc = {}
    for idx, log in enumerate(logs):
        for line in log.splitlines():
            try:
                if line.startswith("MPTEST_DISCOVER "):
                    payload = json.loads(line.split(" ", 1)[1])
                    discover[payload.get("proc")] = payload
                elif line.startswith("MPTEST_RESULT "):
                    payload = json.loads(line.split(" ", 1)[1])
                    parsed.append(payload)
                    per_proc_seen.add(payload.get("proc"))
                    tests_by_proc.setdefault(payload.get("proc"), []).append(payload)
                elif line.startswith("MPTEST_SUMMARY "):
                    payload = json.loads(line.split(" ", 1)[1])
                    summary[payload.get("proc")] = payload
                    per_proc_seen.add(payload.get("proc"))
            except json.JSONDecodeError:
                # Ignore malformed lines; they will still be visible in raw logs
                pass

    # Ensure we observed all expected procs
    expected_procs = set(range(requested_procs))
    assert expected_procs.issubset(per_proc_seen), (
        f"Missing results from some processes. expected={sorted(expected_procs)} seen={sorted(per_proc_seen)}\n"
        f"Raw logs:\n" + "\n\n".join(f"===== proc {i} =====\n{l}" for i, l in enumerate(logs))
    )

    # Human-friendly summary printed for visibility under -s
    print("\n[launcher] per-process summary:")
    for pidx in sorted(expected_procs):
        disc = discover.get(pidx, {})
        summ = summary.get(pidx, {"ok": 0, "fail": 0, "total": 0})
        sel = disc.get("selected", [])
        print(
            "\n".join(
                [
                    f"[launcher] proc={pidx}",
                    f"[launcher]  selected={sel if sel else '<unknown>'}",
                    f"[launcher]  ok={summ.get('ok',0)} fail={summ.get('fail',0)} total={summ.get('total',0)}",
                ]
            )
        )

    # Summarize and assert on failures
    failures = [r for r in parsed if r.get("status") == "fail"]
    if failures:
        def _short_msg(tb: str) -> str:
            if not tb:
                return ""
            lines = [ln for ln in tb.splitlines() if ln.strip()]
            # Use the final exception line for brevity
            return lines[-1] if lines else tb.strip()

        summary_lines = [
            "MPMD test failures:",
        ]
        # Group failures per proc and test name with concise message
        for r in failures:
            summary_lines.append(
                f"- proc {r.get('proc')} :: {r.get('name')}: {_short_msg(r.get('traceback',''))}"
            )

        # Add compact per-proc stats and selections
        summary_lines.append("")
        summary_lines.append("Per-process selection and summary:")
        for pidx in sorted(expected_procs):
            sel = discover.get(pidx, {}).get("selected", [])
            summ = summary.get(pidx, {"ok": 0, "fail": 0, "total": 0})
            summary_lines.append(
                f"  proc {pidx}: selected={sel if sel else '<unknown>'} | ok={summ.get('ok',0)} fail={summ.get('fail',0)} total={summ.get('total',0)}"
            )

        # Optionally include logs; filtered by default to hide MPTEST_* noise
        show_raw = os.environ.get("JAXMG_MP_SHOW_RAW", "0")
        if show_raw != "0":
            summary_lines.append("")
            summary_lines.append(
                "Raw logs (set JAXMG_MP_SHOW_RAW=2 to include MPTEST_* lines):"
            )
            for i, l in enumerate(logs):
                if show_raw == "1":
                    filtered = "\n".join(
                        ln for ln in l.splitlines() if not ln.startswith("MPTEST_")
                    )
                    summary_lines.append(f"===== proc {i} (filtered) =====\n{filtered}")
                else:
                    summary_lines.append(f"===== proc {i} =====\n{l}")

        # Final fail
        pytest.fail("\n".join(summary_lines))

    # As a smoke check, ensure we saw at least one ok result per proc
    ok_by_proc = {p: 0 for p in expected_procs}
    for r in parsed:
        if r.get("status") == "ok":
            ok_by_proc[r.get("proc")] = ok_by_proc.get(r.get("proc"), 0) + 1
    assert all(v > 0 for v in ok_by_proc.values()), (
        f"Expected at least one passing test per process, got: {ok_by_proc}\n"
        f"Raw logs:\n" + "\n\n".join(f"===== proc {i} =====\n{l}" for i, l in enumerate(logs))
    )