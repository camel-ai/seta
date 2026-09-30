"""scripts/miles/common/harbor_agent.py: request building, timeouts and the /flush abort hook.

Needs Miles importable (``miles.utils.http_utils``); runs inside the Miles container.
"""

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "miles"))
from common import harbor_agent  # noqa: E402


class RunTest(unittest.IsolatedAsyncioTestCase):
    async def test_agent_call_timeout_is_configurable(self):
        mocked_post = AsyncMock(return_value={"reward": 1.0})
        observed_timeout = None

        async def capture_timeout(awaitable, *, timeout):
            nonlocal observed_timeout
            observed_timeout = timeout
            return await awaitable

        env = {
            "AGENT_SERVER_URL": "http://agent:11000",
            "HARBOR_AGENT_CALL_TIMEOUT_SEC": "86400",
        }

        with patch.dict(os.environ, env, clear=True), patch.object(
            harbor_agent, "post", mocked_post
        ), patch.object(
            harbor_agent.asyncio, "wait_for", side_effect=capture_timeout
        ):
            await harbor_agent.run("http://session:30000", "prompt")

        self.assertEqual(observed_timeout, 86400.0)

    async def test_agent_call_timeout_must_be_positive(self):
        with patch.dict(
            os.environ,
            {"HARBOR_AGENT_CALL_TIMEOUT_SEC": "0"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "must be positive"):
                await harbor_agent.run("http://session:30000", "prompt")

    async def test_passes_explicit_terminus_xml_and_interleaved_thinking(self):
        mocked_post = AsyncMock(return_value={"reward": 1.0})
        env = {
            "AGENT_SERVER_URL": "http://agent:11000",
            "AGENT_MODEL_NAME": "model",
            "HARBOR_AGENT_NAME": "terminus-2",
            "HARBOR_AGENT_MAX_ITERATIONS": "50",
            "HARBOR_MAX_SEQ_LEN": "1048576",
            "HARBOR_TERMINUS_PARSER": "xml",
            "HARBOR_INTERLEAVED_THINKING": "true",
        }

        with patch.dict(os.environ, env), patch.object(
            harbor_agent, "post", mocked_post
        ):
            await harbor_agent.run(
                "http://session:30000",
                "prompt",
                request_kwargs={"temperature": 0.7},
                metadata={"instance_id": "task", "max_seq_len": 32768},
            )

        request = mocked_post.await_args.args[1]
        self.assertEqual(request["sampling_params"]["parser_name"], "xml")
        self.assertIs(request["sampling_params"]["interleaved_thinking"], True)
        self.assertEqual(request["sampling_params"]["max_iterations"], 50)
        self.assertNotIn("max_seq_len", request["sampling_params"])
        # Backward compatibility: an unset HARBOR_TERMINUS_ENABLE_SUMMARIZE
        # sends nothing, so Harbor's default decides and a run recorded before
        # this variable existed behaves exactly as it did.
        self.assertNotIn("enable_summarize", request["sampling_params"])

    async def test_forwards_camel_compaction_budget_when_requested(self):
        """Compaction is agent policy: Harbor pops it into agent kwargs.

        Left in sampling_params it would reach the model config instead.
        """
        mocked_post = AsyncMock(return_value={"reward": 1.0})
        env = {
            "AGENT_SERVER_URL": "http://agent:11000",
            "AGENT_MODEL_NAME": "model",
            "HARBOR_AGENT_NAME": "camel",
            "HARBOR_AGENT_MAX_ITERATIONS": "75",
            "HARBOR_MAX_SEQ_LEN": "49152",
            "HARBOR_CAMEL_MAX_COMPACTIONS": "12",
        }

        with patch.dict(os.environ, env, clear=True), patch.object(
            harbor_agent, "post", mocked_post
        ):
            await harbor_agent.run(
                "http://session:30000",
                "prompt",
                request_kwargs={"temperature": 1.0},
                metadata={"instance_id": "task"},
            )

        params = mocked_post.await_args.args[1]["sampling_params"]
        self.assertEqual(params["max_compactions"], 12)
        # CAMEL keeps length-cut output as terminal policy rather than
        # regenerating it; the compaction budget must not disturb that.
        self.assertIs(params["response_feedback"], False)
        self.assertEqual(params["max_response_feedback"], 0)

    async def test_unset_camel_compaction_sends_nothing(self):
        """A run recorded before this variable existed must be unchanged."""
        mocked_post = AsyncMock(return_value={"reward": 1.0})
        env = {
            "AGENT_SERVER_URL": "http://agent:11000",
            "HARBOR_AGENT_NAME": "camel",
        }

        with patch.dict(os.environ, env, clear=True), patch.object(
            harbor_agent, "post", mocked_post
        ):
            await harbor_agent.run(
                "http://session:30000", "prompt", metadata={"instance_id": "t"}
            )

        params = mocked_post.await_args.args[1]["sampling_params"]
        self.assertNotIn("max_compactions", params)

    async def test_camel_compaction_rejects_a_non_integer(self):
        mocked_post = AsyncMock(return_value={"reward": 1.0})
        env = {
            "AGENT_SERVER_URL": "http://agent:11000",
            "HARBOR_AGENT_NAME": "camel",
            "HARBOR_CAMEL_MAX_COMPACTIONS": "twelve",
        }

        with patch.dict(os.environ, env, clear=True), patch.object(
            harbor_agent, "post", mocked_post
        ):
            with self.assertRaises(ValueError):
                await harbor_agent.run(
                    "http://session:30000", "prompt", metadata={"instance_id": "t"}
                )

    async def test_compaction_budget_does_not_leak_into_terminus_2(self):
        """The variable is camel-only."""
        mocked_post = AsyncMock(return_value={"reward": 1.0})
        env = {
            "AGENT_SERVER_URL": "http://agent:11000",
            "HARBOR_AGENT_NAME": "terminus-2",
            "HARBOR_CAMEL_MAX_COMPACTIONS": "12",
        }

        with patch.dict(os.environ, env, clear=True), patch.object(
            harbor_agent, "post", mocked_post
        ):
            await harbor_agent.run(
                "http://session:30000", "prompt", metadata={"instance_id": "t"}
            )

        params = mocked_post.await_args.args[1]["sampling_params"]
        self.assertNotIn("max_compactions", params)

    async def test_forwards_terminus_summarization_when_requested(self):
        mocked_post = AsyncMock(return_value={"reward": 1.0})
        env = {
            "AGENT_SERVER_URL": "http://agent:11000",
            "AGENT_MODEL_NAME": "model",
            "HARBOR_AGENT_NAME": "terminus-2",
            "HARBOR_AGENT_MAX_ITERATIONS": "200",
            "HARBOR_MAX_SEQ_LEN": "65536",
            "HARBOR_TERMINUS_PARSER": "xml",
            "HARBOR_INTERLEAVED_THINKING": "true",
            "HARBOR_TERMINUS_ENABLE_SUMMARIZE": "true",
        }

        with patch.dict(os.environ, env, clear=True), patch.object(
            harbor_agent, "post", mocked_post
        ):
            await harbor_agent.run(
                "http://session:30000",
                "prompt",
                request_kwargs={"temperature": 0.8},
                metadata={"instance_id": "task"},
            )

        request = mocked_post.await_args.args[1]
        self.assertIs(request["sampling_params"]["enable_summarize"], True)

    async def test_terminus_summarization_can_be_explicitly_disabled(self):
        mocked_post = AsyncMock(return_value={"reward": 1.0})
        env = {
            "AGENT_SERVER_URL": "http://agent:11000",
            "HARBOR_AGENT_NAME": "terminus-2",
            "HARBOR_TERMINUS_ENABLE_SUMMARIZE": "false",
        }

        with patch.dict(os.environ, env, clear=True), patch.object(
            harbor_agent, "post", mocked_post
        ):
            await harbor_agent.run("http://session:30000", "prompt")

        request = mocked_post.await_args.args[1]
        self.assertIs(request["sampling_params"]["enable_summarize"], False)

    async def test_camel_uses_native_tools_without_terminus_only_kwargs(self):
        mocked_post = AsyncMock(return_value={"reward": 1.0})
        env = {
            "AGENT_SERVER_URL": "http://agent:11000",
            "AGENT_MODEL_NAME": "model",
            "HARBOR_AGENT_NAME": "camel",
            "HARBOR_AGENT_MAX_ITERATIONS": "50",
            "HARBOR_MAX_SEQ_LEN": "1048576",
            # These may remain in a caller's environment after a Terminus2 run.
            # CAMEL owns none of them and must not forward any to SGLang.
            "HARBOR_TERMINUS_PARSER": "xml",
            "HARBOR_INTERLEAVED_THINKING": "not-a-bool",
            "HARBOR_TERMINUS_ENABLE_SUMMARIZE": "true",
        }

        with patch.dict(os.environ, env, clear=True), patch.object(
            harbor_agent, "post", mocked_post
        ):
            await harbor_agent.run(
                "http://session:30000",
                "prompt",
                request_kwargs={"temperature": 0.7},
                metadata={"instance_id": "task", "max_seq_len": 32768},
            )

        request = mocked_post.await_args.args[1]
        self.assertEqual(request["agent_name"], "camel")
        self.assertEqual(
            request["sampling_params"],
            {
                "temperature": 0.7,
                "max_iterations": 50,
                "response_feedback": False,
                "max_response_feedback": 0,
            },
        )


class AbortTest(unittest.IsolatedAsyncioTestCase):
    async def test_flushes_every_instance_in_plural_map(self):
        args = SimpleNamespace(
            session_server_instance_ids={30001: "instance-b", 30000: "instance-a"}
        )
        responses = [
            {"session_server_instance_id": "instance-a", "cancelled": 2},
            {"session_server_instance_id": "instance-b", "cancelled": 3},
        ]
        mocked_post = AsyncMock(side_effect=responses)

        with patch.dict(os.environ, {"AGENT_SERVER_URL": "http://agent:11000"}), patch.object(
            harbor_agent, "post", mocked_post
        ):
            await harbor_agent.abort(args)

        self.assertEqual(
            [call.args for call in mocked_post.await_args_list],
            [
                (
                    "http://agent:11000/flush",
                    {"session_server_instance_id": "instance-a"},
                ),
                (
                    "http://agent:11000/flush",
                    {"session_server_instance_id": "instance-b"},
                ),
            ],
        )
        self.assertTrue(
            all(call.kwargs == {"max_retries": 3} for call in mocked_post.await_args_list)
        )

    async def test_retains_singular_instance_compatibility(self):
        args = SimpleNamespace(session_server_instance_id="instance-a")
        mocked_post = AsyncMock(return_value={"cancelled": 1})

        with patch.dict(os.environ, {"AGENT_SERVER_URL": "http://agent:11000"}), patch.object(
            harbor_agent, "post", mocked_post
        ):
            await harbor_agent.abort(args)

        mocked_post.assert_awaited_once_with(
            "http://agent:11000/flush",
            {"session_server_instance_id": "instance-a"},
            max_retries=3,
        )

    async def test_missing_instance_identity_is_loud(self):
        with patch.dict(os.environ, {"AGENT_SERVER_URL": "http://agent:11000"}):
            with self.assertLogs(harbor_agent.logger, level="ERROR") as captured:
                with self.assertRaisesRegex(RuntimeError, "neither session_server"):
                    await harbor_agent.abort(SimpleNamespace())

        self.assertIn("[HARBOR-ABORT] FAILED", "\n".join(captured.output))

    async def test_flush_failure_reports_instance_and_reason(self):
        args = SimpleNamespace(session_server_instance_ids={30000: "instance-a"})
        mocked_post = AsyncMock(side_effect=ConnectionError("server unreachable"))

        with patch.dict(os.environ, {"AGENT_SERVER_URL": "http://agent:11000"}), patch.object(
            harbor_agent, "post", mocked_post
        ):
            with self.assertLogs(harbor_agent.logger, level="ERROR") as captured:
                with self.assertRaisesRegex(RuntimeError, "server unreachable"):
                    await harbor_agent.abort(args)

        output = "\n".join(captured.output)
        self.assertIn("[HARBOR-ABORT] FAILED", output)
        self.assertIn("instance-a", output)


if __name__ == "__main__":
    unittest.main()
