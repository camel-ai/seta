# ========= Copyright 2023-2026 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2026 @ CAMEL-AI.org. All Rights Reserved. =========
"""A near-zero-cost agent for stress/placement testing of the env + Daytona path.

Purpose: isolate the *environment lifecycle* (build-gate -> create sandbox from
snapshot -> [tiny agent] -> evaluate -> teardown) from model latency, so a driver
can fire the exact `task x sample` shape of training and measure, per snapshot,
whether Daytona's "No available runners" (and runner-proxy exec failures) hit
**none / some / all** of a group's sandboxes.

It does NOT call the model. After the runtime is up (sandbox created), `astep()`
optionally runs a few cheap `shell_exec` liveness probes to confirm the sandbox is
actually functional (runner proxy reachable), optionally holds the sandbox to
emulate a real trajectory's lifetime, then returns. Capped to <=2 effective turns.

Tunables (env vars, re-read each `astep` so they can change between rounds without
restarting env_service):
  NOOP_EXEC_PROBES  int    number of `echo` liveness probes (default 1; 0 = pure
                           create test, no exec). Clamped to <=2.
  NOOP_HOLD_SEC     float  seconds to hold the sandbox after probes (default 0).
                           Use to emulate a realistic trajectory lifetime so the
                           create/free churn matches training.
  NOOP_PROBE_CMD    str    command template; `{i}` is the probe index.
"""
import asyncio
import logging
import os
import time

from seta_env.agent.train_agent import AgentTrain

logger = logging.getLogger(__name__)

# Marker the analyzer / driver greps for to confirm a probe actually executed
# inside the sandbox (proves the runner proxy + tmux session work end-to-end).
_OK = "__NOOP_OK_{i}__"


class AgentNoop(AgentTrain):
    """Model-free agent: exercises the sandbox, never calls the LLM.

    Inherits AgentTrain's constructor/reset/meta_info_record contract unchanged;
    only `astep` is overridden, so it plugs into TerminalEnvironment.run_agent
    (which calls `await agent.astep(instruction)` then reads `agent.meta_info_record`).
    """

    async def astep(self, input_message=None, *args, **kwargs):  # noqa: D401
        t0 = time.monotonic()
        try:
            n_probe = min(2, max(0, int(os.environ.get("NOOP_EXEC_PROBES", "1"))))
        except ValueError:
            n_probe = 1
        try:
            hold = float(os.environ.get("NOOP_HOLD_SEC", "0"))
        except ValueError:
            hold = 0.0
        cmd_tmpl = os.environ.get("NOOP_PROBE_CMD", "echo " + _OK + " $(hostname)")

        tools = getattr(self, "_internal_tools", {}) or {}
        tool = tools.get("shell_exec")

        exec_ok = 0
        outs: list[str] = []
        for i in range(n_probe):
            if tool is None:
                outs.append("NO_SHELL_EXEC_TOOL")
                break
            try:
                res = await tool.async_call(
                    id="noop", command=cmd_tmpl.format(i=i), block=True
                )
                s = str(res)
                outs.append(s[:160])
                if _OK.format(i=i) in s:
                    exec_ok += 1
            except Exception as e:  # noqa: BLE001 — record, never raise (this is a probe)
                outs.append(f"EXEC_FAIL:{type(e).__name__}:{str(e)[:120]}")

        if hold > 0:
            await asyncio.sleep(hold)

        # meta_info_record drives run_info.agent_summary; "task_finished" is in
        # TerminalEnvironment._IMPORTANT_TERMINATION_REASONS so the run is marked
        # as a clean finish. exec fields let the analyzer separate "sandbox created"
        # from "sandbox actually usable".
        self.meta_info_record["iteration_count"] = n_probe
        self.meta_info_record["termination_reason"] = "task_finished"
        self.meta_info_record["noop_exec_ok"] = exec_ok
        self.meta_info_record["noop_exec_attempts"] = n_probe
        self.meta_info_record["noop_outs"] = outs
        self.meta_info_record["noop_wall_sec"] = round(time.monotonic() - t0, 2)

        logger.info(
            "[noop] task=%s exec_ok=%d/%d wall=%.1fs",
            getattr(self, "task_name", "?"), exec_ok, n_probe, time.monotonic() - t0,
        )
        return f"[noop] exec_ok={exec_ok}/{n_probe}"
