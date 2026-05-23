import numpy as np

from thermal_inference_lab.core.config import IsingConfig, SamplerConfig
from thermal_inference_lab.energy.ising import IsingModel
from thermal_inference_lab.samplers.gibbs import GibbsSampler
from thermal_inference_lab.samplers.metropolis import MetropolisSampler


def _run(sampler_cls, seed):
    model = IsingModel(IsingConfig(shape=(6, 6)))
    sampler = sampler_cls(model, temperature=2.5, rng=seed)
    return sampler.run(SamplerConfig(n_samples=25, burn_in=20, interval=1, seed=seed))


def test_metropolis_same_seed_identical_trace():
    a = _run(MetropolisSampler, seed=7)
    b = _run(MetropolisSampler, seed=7)
    np.testing.assert_array_equal(a.samples, b.samples)
    np.testing.assert_array_equal(a.energies, b.energies)
    assert a.acceptance_rate == b.acceptance_rate


def test_gibbs_same_seed_identical_trace():
    a = _run(GibbsSampler, seed=11)
    b = _run(GibbsSampler, seed=11)
    np.testing.assert_array_equal(a.samples, b.samples)


def test_metropolis_different_seeds_differ():
    a = _run(MetropolisSampler, seed=1)
    b = _run(MetropolisSampler, seed=2)
    assert not np.array_equal(a.samples, b.samples)
