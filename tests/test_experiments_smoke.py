"""Smoke tests for the experiment scripts.

These keep parameters small so the suite runs fast but exercise every
import path and CLI entry. They guard against regressions in
top-level scripts that would otherwise only fail when run by hand.
"""

import os

import pytest


def test_estimate_temperature_runs(capsys):
    from thermal_inference_lab.experiments import estimate_temperature as exp
    exp.main(true_temperature=2.5, size=5, n_samples=60, seed=0)
    out = capsys.readouterr().out
    assert "true T" in out


def test_recover_parameters_runs(capsys):
    from thermal_inference_lab.experiments import recover_ising_parameters as exp
    exp.main(true_J=1.0, true_h=0.0, size=5, temperature=3.0, n_samples=60, seed=0)
    out = capsys.readouterr().out
    assert "recovered" in out


def test_compare_samplers_runs(capsys):
    from thermal_inference_lab.experiments import compare_samplers as exp
    exp.main(size=5, temperature=2.5, n_samples=50, seed=0)
    out = capsys.readouterr().out
    assert "metropolis" in out
    assert "gibbs" in out


def test_train_rbm_runs(capsys):
    from thermal_inference_lab.experiments import train_rbm as exp
    exp.main(size=3, n_hidden=4, n_epochs=3, seed=0)
    out = capsys.readouterr().out
    assert "MSE" in out


def test_anneal_runs(capsys):
    from thermal_inference_lab.experiments import anneal_binary_problem as exp
    exp.main(n=12, n_steps=2000, seed=0)
    out = capsys.readouterr().out
    assert "best energy" in out


def test_run_ising_runs(tmp_path, capsys):
    from thermal_inference_lab.experiments import run_ising as exp
    exp.main(size=4, n_samples=40, burn_in=50, interval=1, out_dir=str(tmp_path), seed=0)
    out = capsys.readouterr().out
    assert "Wrote" in out
    assert os.path.exists(tmp_path / "ising_magnetization.png")
