import numpy as np
import pytest

from thermal_inference_lab.core.config import IsingConfig, SamplerConfig
from thermal_inference_lab.distributions.observables import mean_absolute_magnetization
from thermal_inference_lab.energy.ising import IsingModel
from thermal_inference_lab.samplers.gibbs import GibbsSampler


def test_gibbs_low_T_magnetises():
    model = IsingModel(IsingConfig(shape=(8, 8)))
    sampler = GibbsSampler(model, temperature=1.0, rng=0)
    r = sampler.run(SamplerConfig(n_samples=120, burn_in=300, interval=2, seed=0))
    assert mean_absolute_magnetization(r.samples) > 0.85


def test_gibbs_acceptance_is_one():
    model = IsingModel(IsingConfig(shape=(6, 6)))
    sampler = GibbsSampler(model, temperature=2.5, rng=0)
    r = sampler.run(SamplerConfig(n_samples=20, burn_in=10, seed=0))
    assert r.acceptance_rate == pytest.approx(1.0)


def test_gibbs_rejects_non_ising_model():
    from thermal_inference_lab.energy.lattice_gas import LatticeGasModel
    with pytest.raises(TypeError):
        GibbsSampler(LatticeGasModel(), temperature=1.0, rng=0)
