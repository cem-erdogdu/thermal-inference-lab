import numpy as np
import pytest

from thermal_inference_lab.core.config import IsingConfig, SamplerConfig
from thermal_inference_lab.distributions.observables import mean_absolute_magnetization
from thermal_inference_lab.energy.ising import IsingModel
from thermal_inference_lab.samplers.metropolis import MetropolisSampler


def test_metropolis_acceptance_in_unit_interval():
    model = IsingModel(IsingConfig(shape=(6, 6)))
    sampler = MetropolisSampler(model, temperature=2.5, rng=0)
    r = sampler.run(SamplerConfig(n_samples=50, burn_in=50, interval=1, seed=0))
    assert 0.0 <= r.acceptance_rate <= 1.0


def test_metropolis_at_low_T_magnetises():
    """Well below T_c the system should magnetise strongly."""
    model = IsingModel(IsingConfig(shape=(8, 8)))
    sampler = MetropolisSampler(model, temperature=1.0, rng=0)
    r = sampler.run(SamplerConfig(n_samples=120, burn_in=400, interval=2, seed=0))
    assert mean_absolute_magnetization(r.samples) > 0.85


def test_metropolis_at_high_T_disorders():
    """Well above T_c the magnetisation should be small."""
    model = IsingModel(IsingConfig(shape=(8, 8)))
    sampler = MetropolisSampler(model, temperature=10.0, rng=0)
    r = sampler.run(SamplerConfig(n_samples=200, burn_in=400, interval=2, seed=0))
    assert mean_absolute_magnetization(r.samples) < 0.3


def test_record_energy_flag():
    model = IsingModel(IsingConfig(shape=(4, 4)))
    sampler = MetropolisSampler(model, temperature=2.5, rng=0)
    r1 = sampler.run(SamplerConfig(n_samples=20, burn_in=10, record_energy=True, seed=0))
    r2 = sampler.run(SamplerConfig(n_samples=20, burn_in=10, record_energy=False, seed=0))
    assert r1.energies.size == 20
    assert r2.energies.size == 0
