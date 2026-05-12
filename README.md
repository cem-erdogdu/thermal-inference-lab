# thermal-inference-lab

A NumPy-only scientific computing lab for **Boltzmann-style physical systems**:
simulate them, estimate their equilibrium observables, and infer the hidden
physical parameters from generated samples.

The package is deliberately framework-free. There is **no PyTorch, JAX,
TensorFlow, scipy, sklearn, numba, networkx, or probabilistic-programming
library** in the dependency tree — only the Python standard library, NumPy,
matplotlib, and pytest. Everything that looks like a deep-learning idiom
(parameter dictionaries, Adam, contrastive divergence) is rebuilt on top of
plain ``np.ndarray``.

## Why this exists

Many machine-learning textbooks introduce energy-based models (the Boltzmann
machine, the RBM, score matching) without ever connecting them back to the
classical statistical mechanics they were lifted from. This repo treats the
two as a single subject:

- **Energy models** are Hamiltonians: 2D Ising, lattice gas, crystal defects,
  RBM (Section *energy/*).
- **Sampling** is Monte Carlo Markov chain: Metropolis and heat-bath Gibbs
  (Section *samplers/*).
- **Annealing** drives a system toward its ground state by lowering the
  temperature schedule (Section *annealing/*).
- **Inference** recovers the parameters of the Hamiltonian from samples,
  either by maximum pseudolikelihood (Ising) or contrastive divergence (RBM)
  (Section *inference/*).
- **Observables** turn raw samples into thermodynamic quantities — energy,
  magnetisation, specific heat, susceptibility, Binder cumulant
  (Section *distributions/observables.py*).

## The physics in one screen

For a classical system in thermal equilibrium with a bath at temperature *T*,
the probability of a microstate *s* is

```
p(s; T) = exp(-H(s) / T) / Z(T)
```

with the **partition function**

```
Z(T) = sum_{s} exp(-H(s) / T)
```

and *k_B = 1* in our reduced units. The 2D Ising Hamiltonian on a square
lattice with periodic boundary conditions is

```
H(s) = -J * sum_{<i,j>} s_i * s_j  -  h * sum_i s_i,   s_i in {-1, +1}.
```

The local energy delta for a single-site flip — central to single-spin
Metropolis — is

```
ΔE(i) = 2 * s_i * (J * sum_{j in nn(i)} s_j + h).
```

This is implemented in ``IsingModel.local_energy_delta`` and exercised
extensively by the tests.

## Sampling

The package implements two samplers:

- **Metropolis-Hastings** with single-spin proposals (``samplers/metropolis.py``).
  Accept with probability ``min(1, exp(-ΔE/T))``.
- **Heat-bath Gibbs** (``samplers/gibbs.py``). Sample each spin from its
  conditional ``p(s_i = +1 | rest) = sigmoid(2 β (J Σ_nn + h))``. We do a
  checkerboard sweep, which is provably ergodic and avoids the bias of
  parallel updates from stale neighbour state.

Both samplers reuse a common driver (``samplers/base.py``) that does burn-in,
thinning, energy-trace recording, and acceptance bookkeeping.

## Inference

Two complementary routines turn samples back into model parameters:

- **Maximum pseudolikelihood for the Ising model** (``inference/parameter_estimation.py``)
  recovers *(J, h)* from a batch of equilibrium configurations. The full
  likelihood is intractable because of the partition function, so we use
  Besag's pseudolikelihood: the product of one-site conditional probabilities.
  Closed-form gradients are passed to the package's own NumPy optimizers
  (SGD, Momentum, Nesterov, Adam, AdamW).
- **Temperature estimation** (``inference/temperature_estimation.py``) scans
  *β* on a log-uniform grid when *(J, h)* are known.
- **Contrastive Divergence** (``inference/contrastive_divergence.py``) trains
  an RBM by approximating ``<∂F/∂θ>_model`` with *k* Gibbs steps starting from
  the data.

## Installation

```bash
git clone <this repo>
cd thermal-inference-lab
pip install -e ".[dev]"
```

## Running the tests

```bash
pytest -q
```

Expected output: **89 passed**. The full suite takes about 3 seconds on a
laptop.

## Running experiments

Every script in ``thermal_inference_lab/experiments/`` is runnable from the
command line:

```bash
python -m thermal_inference_lab.experiments.run_ising --size 16
python -m thermal_inference_lab.experiments.estimate_temperature --true-temperature 2.6
python -m thermal_inference_lab.experiments.recover_ising_parameters --true-J 1.0 --true-h 0.3
python -m thermal_inference_lab.experiments.compare_samplers --size 12
python -m thermal_inference_lab.experiments.train_rbm
python -m thermal_inference_lab.experiments.anneal_binary_problem --n 24
```

``run_ising`` writes ``artifacts/ising_magnetization.png`` showing the
magnetisation and specific heat curves across the phase transition; the rest
print summary statistics to stdout.

## Package layout

```
thermal_inference_lab/
  core/            # config, RNG, validation, exceptions, math helpers
  energy/          # Ising / RBM / lattice-gas / crystal-defect Hamiltonians
  distributions/   # Boltzmann weights, partition functions, observables
  samplers/        # Metropolis, Gibbs, chain orchestration, diagnostics
  annealing/       # cooling schedules + simulated-annealing driver
  optimizers/      # SGD, Momentum, Nesterov, Adam, AdamW (dict-of-arrays API)
  inference/       # pseudolikelihood, CD-k, parameter estimation, fit results
  experiments/     # runnable scripts that produce plots and summaries
  visualization/   # lattice / energy / phase / diagnostics matplotlib helpers
  datasets/        # synthetic spin and pattern data generators
tests/             # 20 test modules, 89 cases
```

## License

MIT.
