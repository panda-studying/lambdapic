"""Saving / printing of Baier--Katkov spectrum results."""

from __future__ import annotations

import numpy as np

from .types import Spectrum

__all__ = ["spectrum_table", "format_spectrum", "save_spectrum_text"]


def format_spectrum(spectrum: Spectrum) -> str:
    """Form a human-readable summary table of a :class:`Spectrum`."""
    om = spectrum.omega
    dW = spectrum.dW_domega
    dE = spectrum.dE_domega if spectrum.dE_domega is not None else om * dW

    lines = []
    lines.append("  omega          dW/domega           dE/domega")
    lines.append("  " + "-" * 46)
    for o, w, e in zip(om, dW, dE):
        lines.append(f"  {o:12.6e}  {w:18.8e}  {e:18.8e}")
    lines.append("  " + "-" * 46)
    lines.append(f"  total probability W = {spectrum.total_probability():.8e}")
    lines.append(f"  total radiated energy = {spectrum.total_energy():.8e}")
    meta = spectrum.metadata
    if meta:
        lines.append("  metadata:")
        for k, v in meta.items():
            lines.append(f"    {k}: {v}")
    return "\n".join(lines)


def spectrum_table(spectrum: Spectrum) -> np.ndarray:
    """Return an (Nw, 3) array of ``[omega, dW/domega, dE/domega]``."""
    om = spectrum.omega
    dW = spectrum.dW_domega
    dE = spectrum.dE_domega if spectrum.dE_domega is not None else om * dW
    return np.column_stack([om, dW, dE])


def save_spectrum_text(spectrum: Spectrum, path: str) -> None:
    """Write the spectrum as a three-column text file plus a header comment."""
    table = spectrum_table(spectrum)
    header = (
        "# Baier-Katkov single-particle spectrum (natural units c=hbar=1, m_e=1)\n"
        "# columns: omega, dW/domega, dE/domega\n"
    )
    for k, v in spectrum.metadata.items():
        header += f"# {k}: {v}\n"
    np.savetxt(path, table, header=header.rstrip("\n"), comments="")