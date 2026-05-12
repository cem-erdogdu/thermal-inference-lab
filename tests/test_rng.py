import numpy as np
import pytest

from thermal_inference_lab.core.rng import RNG, as_rng, split_rng


def test_seed_produces_deterministic_draws():
    a = RNG(42).normal(size=10)
    b = RNG(42).normal(size=10)
    np.testing.assert_array_equal(a, b)


def test_different_seeds_differ():
    a = RNG(1).uniform(size=10)
    b = RNG(2).uniform(size=10)
    assert not np.allclose(a, b)


def test_as_rng_passthrough_and_int():
    r = RNG(7)
    assert as_rng(r) is r
    r2 = as_rng(11)
    assert isinstance(r2, RNG) and r2.seed == 11
    r3 = as_rng(None, default_seed=3)
    assert r3.seed == 3


def test_spawn_returns_independent_streams():
    parent = RNG(2024)
    children = parent.spawn(4)
    assert len(children) == 4
    draws = [c.uniform(size=20) for c in children]
    # No two child streams should produce identical sequences.
    for i in range(4):
        for j in range(i + 1, 4):
            assert not np.allclose(draws[i], draws[j])


def test_split_rng_names_streams():
    streams = split_rng(RNG(0), ["a", "b"])
    assert set(streams) == {"a", "b"}
    assert not np.allclose(streams["a"].uniform(size=5), streams["b"].uniform(size=5))


def test_spawn_count_validation():
    with pytest.raises(ValueError):
        RNG(0).spawn(0)
