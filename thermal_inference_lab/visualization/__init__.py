"""Matplotlib visualisation helpers (importable without a display)."""

import matplotlib

# Default to a non-interactive backend so experiments run headless.
matplotlib.use("Agg", force=False)

from thermal_inference_lab.visualization.lattice_plot import plot_lattice, plot_lattice_grid
from thermal_inference_lab.visualization.energy_plot import plot_energy_trace, plot_energy_histogram
from thermal_inference_lab.visualization.phase_plot import plot_magnetization_curve
from thermal_inference_lab.visualization.diagnostics_plot import plot_autocorrelation

__all__ = [
    "plot_lattice",
    "plot_lattice_grid",
    "plot_energy_trace",
    "plot_energy_histogram",
    "plot_magnetization_curve",
    "plot_autocorrelation",
]
