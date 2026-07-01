import pytest
from seta_env.verifiers.reward_fn import reward_fn_pass_ratio, reward_factory


# ---------------------------------------------------------------------------
# reward_fn_pass_ratio
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_all_pass():
    result = await reward_fn_pass_ratio({"t1": 1, "t2": 1, "t3": 1})
    assert result == 1.0

@pytest.mark.asyncio
async def test_all_fail():
    result = await reward_fn_pass_ratio({"t1": 0, "t2": 0})
    assert result == 0.0

@pytest.mark.asyncio
async def test_mixed():
    result = await reward_fn_pass_ratio({"t1": 1, "t2": 1, "t3": 1, "t4": 0, "t5": 0})
    assert result == pytest.approx(0.6)

@pytest.mark.asyncio
async def test_single_pass():
    result = await reward_fn_pass_ratio({"t1": 1})
    assert result == 1.0

@pytest.mark.asyncio
async def test_single_fail():
    result = await reward_fn_pass_ratio({"t1": 0})
    assert result == 0.0

@pytest.mark.asyncio
async def test_empty():
    result = await reward_fn_pass_ratio({})
    assert result == 0.0

@pytest.mark.asyncio
async def test_float_values():
    # sum(values) / len works for floats too
    result = await reward_fn_pass_ratio({"t1": 0.5, "t2": 0.5})
    assert result == pytest.approx(0.5)

@pytest.mark.asyncio
async def test_extra_kwargs_ignored():
    # kwargs like trajectory should not affect the result
    result = await reward_fn_pass_ratio({"t1": 1}, trajectory="anything", extra="ignored")
    assert result == 1.0


# ---------------------------------------------------------------------------
# reward_factory
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_factory_pass_ratio_all_pass():
    result = await reward_factory("pass_ratio", evaluation_results={"t1": 1})
    assert result == 1.0

@pytest.mark.asyncio
async def test_factory_pass_ratio_mixed():
    result = await reward_factory("pass_ratio", evaluation_results={"t1": 1, "t2": 0})
    assert result == pytest.approx(0.5)

@pytest.mark.asyncio
async def test_factory_pass_ratio_empty():
    result = await reward_factory("pass_ratio", evaluation_results={})
    assert result == 0.0

@pytest.mark.asyncio
async def test_factory_trajectory_kwarg_ignored():
    result = await reward_factory(
        "pass_ratio",
        evaluation_results={"t1": 1},
        trajectory="[User]: do something\n[Assistant]: done",
    )
    assert result == 1.0

@pytest.mark.asyncio
async def test_factory_unknown_name():
    with pytest.raises(ValueError, match="Unsupported reward function"):
        await reward_factory("nonexistent_fn", evaluation_results={"t1": 1})

@pytest.mark.asyncio
async def test_factory_missing_evaluation_results_defaults_empty():
    # reward_factory passes kwargs["evaluation_results"] defaulting to {}
    result = await reward_factory("pass_ratio")
    assert result == 0.0
