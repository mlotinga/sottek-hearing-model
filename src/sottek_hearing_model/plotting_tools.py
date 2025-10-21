# -*- coding: utf-8 -*-
# %% Preamble
"""
plotting_tools.py
-----------------

Provides utilities for creating and displaying plots in various environments
(scripts, interactive shells, Jupyter notebooks, headless).

Automatically chooses the best backend for the user environment:
- QtAgg (PyQt6) for interactive GUI plots
- module-based IPython/Jupyter integration
- Agg fallback for headless runs

Requirements
------------
matplotlib

Ownership and Quality Assurance
-------------------------------
Author: Mike JB Lotinga (m.j.lotinga@edu.salford.ac.uk)
Institution: University of Salford
Date created: 20/10/2025
Date last modified: 21/10/2025
Python version: 3.11

Copyright statement: This code has been developed during work undertaken within
the RefMap project (www.refmap.eu), based on the RefMap code repository

"""

import os
import sys
import matplotlib as mpl
from matplotlib import pyplot as plt

# %% Internal Functions

# _is_jupyter
def _is_jupyter():
    """Detect if running inside Jupyter/IPython notebook."""
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        return shell in ("ZMQInteractiveShell",)  # Jupyter
    except Exception:
        return False


# _is_interactive_console
def _is_interactive_console():
    """Detect if running in an interactive shell (IPython, Spyder, etc.)."""
    return hasattr(sys, "ps1") or "SPYDER" in os.environ or "VSCODE_PID" in os.environ


# _ensure_backend
def _ensure_backend(preferred="QtAgg"):
    """
    Select a safe Matplotlib backend:
    - Jupyter → inline
    - Interactive shell (Spyder, VS Code) → QtAgg if available
    - Headless → Agg
    """
    current = mpl.get_backend().lower()

    # 1. Jupyter Notebook
    if _is_jupyter():
        try:
            from IPython import get_ipython
            get_ipython().run_line_magic("matplotlib", "inline")
            return
        except Exception:
            pass  # fallback below

    # 2. Interactive shell or desktop session
    if _is_interactive_console() or os.environ.get("DISPLAY") or sys.platform == "win32":
        try:
            import PyQt6  # noqa: F401
            if "qt" not in current.lower():
                mpl.use(preferred, force=True)
            return
        except Exception:
            pass  # fallback below

    # 3. Headless
    mpl.use("Agg", force=True)


# %% Public Functions

# create_figure
def create_figure(**kwargs):
    """Create a Matplotlib figure using an environment-safe backend."""
    _ensure_backend()
    
    fig, ax = plt.subplots(**kwargs)
    return fig, ax


# show_plot
def show_plot(fig=None, block=True):
    """Show a Matplotlib figure safely in GUI, notebook, or headless mode."""
    _ensure_backend()

    try:
        # jupyter notebook
        if _is_jupyter():
            from IPython.display import display
            if fig is not None:
                display(fig)
            else:
                display(plt.gcf())
            return

        # interactive IDE console (spyder/VS Code)
        if _is_interactive_console() and ("JPY_PARENT_PID" in os.environ or "IPYKERNEL" in os.environ):
            # Avoid plt.show() duplication when IDE manages rendering
            if fig is not None:
                return fig
            return plt.gcf()

        # Python or cmd/shell
        if fig is not None:
            fig.show()
        else:
            plt.show(block=block)

    except Exception as e:
        print(f"[plotting] Could not open GUI plot: {e}")
        try:
            outfile = os.path.abspath("plot_output.png")
            plt.savefig(outfile, dpi=300, bbox_inches='tight')
            print(f"[plotting] Saved fallback figure to: {outfile}")
        except Exception:
            pass
