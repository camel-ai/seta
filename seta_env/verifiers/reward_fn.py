

async def reward_fn_pass_ratio(evaluation_results: dict, **kwargs) -> float:
    r"""
    A reward function that calculates the pass ratio of unit tests.

    Args:
        evaluation_results (dict): A dictionary containing the results of unit tests, where the key is the test name and the value is 0 (fail) or 1 (pass).

    Returns:
        float: The pass ratio, calculated as the number of passed tests divided by the total number of tests. If there are no tests, returns 0.0.
    """
    if not evaluation_results:
        return None

    total_tests = len(evaluation_results)
    passed_tests = sum(evaluation_results.values())

    pass_ratio = passed_tests / total_tests
    return pass_ratio


async def reward_fn_pass_ratio_with_bonus(
    evaluation_results: dict,
    full_pass_bonus: float = 0.2,
    **kwargs,
) -> float:
    r"""Pass ratio plus a bonus when every unit test passes.

    Args:
        evaluation_results: ``{test_name: 0|1}`` dict from the verifier.
        full_pass_bonus: Bonus added when ``passed == total``. Default 0.2
            (empirically validated), which doubles the reward of a clean
            pass relative to "almost all passed".
    """
    if not evaluation_results:
        return None

    total_tests = len(evaluation_results)
    passed_tests = sum(evaluation_results.values())
    reward = passed_tests / total_tests
    if passed_tests == total_tests:
        reward += full_pass_bonus
    return reward


async def reward_fn_pass_ratio_with_parallel_penalty(
    evaluation_results: dict,
    agent_info: dict,
    # NOTE: parallel_threshold is intentionally NOT wired to
    # AgentConfig.max_parallel_tool_calls. The agent's hard cap and the
    # reward soft penalty serve different purposes and may legitimately
    # diverge. If you tune one, consider whether the other should follow,
    # but do not enforce equality.
    parallel_threshold: int = 5,
    penalty_per_excess: float = 0.05,
    max_penalty: float = 0.2,
    full_pass_bonus: float = 1.0,
    **kwargs,
) -> float:
    r"""Pass ratio + full-pass bonus minus a soft penalty for excessive parallel tool calls.

    The penalty is intentionally small: a successful trajectory
    (1.0 + ``full_pass_bonus`` = 2.0) must always score higher than a failed
    one (<= 1.0) even after the worst-case penalty. The penalty shapes
    behavior, it does not override task completion.

    Args:
        evaluation_results: ``{test_name: 0|1}`` from the verifier.
        agent_info: The agent's ``meta_info_record``. Must contain
            ``max_parallel_tool_call`` (int).
        parallel_threshold: No penalty when the largest parallel batch in
            any single turn is at or below this value.
        penalty_per_excess: Linear penalty per parallel call above
            ``parallel_threshold``.
        max_penalty: Hard cap on the penalty.
        full_pass_bonus: Bonus added when every unit test passes.
    """
    if not evaluation_results:
        return None

    total_tests = len(evaluation_results)
    passed_tests = sum(evaluation_results.values())
    reward = passed_tests / total_tests
    if passed_tests == total_tests:
        reward += full_pass_bonus

    max_parallel = (agent_info or {}).get("max_parallel_tool_call", 0) or 0
    if max_parallel > parallel_threshold:
        excess = max_parallel - parallel_threshold
        penalty = min(excess * penalty_per_excess, max_penalty)
        reward -= penalty
    return reward

async def reward_fn_binary(
    evaluation_results: dict,
    # full_pass_bonus: float = 0.2,
    **kwargs,
) -> float:
    r"""Pass ratio plus a bonus when every unit test passes.

    Args:
        evaluation_results: ``{test_name: 0|1}`` dict from the verifier.
        full_pass_bonus: Bonus added when ``passed == total``. Default 0.2
            (empirically validated), which doubles the reward of a clean
            pass relative to "almost all passed".
    """
    if not evaluation_results:
        return None

    total_tests = len(evaluation_results)
    passed_tests = sum(evaluation_results.values())
    reward = passed_tests / total_tests
    reward = 1.0 if reward == 1 else 0
    return reward


def _require_kwarg(reward_fn_name: str, kwargs: dict, key: str) -> None:
    if key not in kwargs or kwargs[key] is None:
        raise ValueError(
            f"reward_fn '{reward_fn_name}' requires kwarg '{key}'"
        )


async def reward_factory(reward_fn_name: str, **kwargs) -> float:
    r"""Dispatch to a reward function by name.

    Each branch validates its required kwargs explicitly and raises
    ``ValueError`` if anything is missing — silent defaults masked plumbing
    bugs in the past.
    """
    if reward_fn_name == "pass_ratio":
        _require_kwarg(reward_fn_name, kwargs, "evaluation_results")
        return await reward_fn_pass_ratio(kwargs["evaluation_results"])

    if reward_fn_name == "pass_ratio_with_bonus":
        _require_kwarg(reward_fn_name, kwargs, "evaluation_results")
        return await reward_fn_pass_ratio_with_bonus(
            kwargs["evaluation_results"]
        )

    if reward_fn_name == "pass_ratio_parallel_penalty":
        _require_kwarg(reward_fn_name, kwargs, "evaluation_results")
        _require_kwarg(reward_fn_name, kwargs, "agent_info")
        return await reward_fn_pass_ratio_with_parallel_penalty(
            evaluation_results=kwargs["evaluation_results"],
            agent_info=kwargs["agent_info"],
        )

    if reward_fn_name == "binary":
        _require_kwarg(reward_fn_name, kwargs, "evaluation_results")
        return await reward_fn_binary(kwargs["evaluation_results"])

    raise ValueError(
        f"Unsupported reward function: {reward_fn_name}. Supported functions are: "
        "'pass_ratio', 'pass_ratio_with_bonus', 'pass_ratio_parallel_penalty'."
    )
