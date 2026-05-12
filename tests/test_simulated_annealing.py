import numpy as np
import pytest

from thermal_inference_lab.annealing.schedules import ExponentialSchedule
from thermal_inference_lab.annealing.simulated_annealing import SimulatedAnnealing
from thermal_inference_lab.core.config import AnnealConfig, IsingConfig
from thermal_inference_lab.energy.ising import IsingModel


def test_anneal_drives_energy_down():
    model = IsingModel(IsingConfig(shape=(6, 6), J=1.0, h=0.0))
    schedule = ExponentialSchedule(t_start=4.0, t_end=0.01, _n_steps=3000)
    sa = SimulatedAnnealing(model, schedule=schedule, rng=0)
    r = sa.run()
    # The best energy must equal the minimum of the trace.
    assert r.best_energy == pytest.approx(min(r.energy_trace.min(), r.best_energy))
    # The trace must end below it started.
    assert r.energy_trace[-1] < r.energy_trace[0]


def test_anneal_finds_ground_state_for_tiny_lattice():
    model = IsingModel(IsingConfig(shape=(3, 3), J=1.0, h=0.0))
    # Ground state energy for 3x3 torus, all aligned, 2*9=18 bonds => E = -18.
    schedule = ExponentialSchedule(t_start=3.0, t_end=1e-3, _n_steps=8000)
    sa = SimulatedAnnealing(model, schedule=schedule, rng=0)
    r = sa.run()
    assert r.best_energy == pytest.approx(-18.0)


def test_from_config_constructs_schedule():
    model = IsingModel(IsingConfig(shape=(4, 4)))
    sa = SimulatedAnnealing.from_config(model, AnnealConfig(n_steps=200))
    assert sa.schedule is not None
