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

Functions
---------

Ownership and Quality Assurance
-------------------------------
Author: Mike JB Lotinga (m.j.lotinga@edu.salford.ac.uk)
Institution: University of Salford
Date created: 20/10/2025
Date last modified: 22/10/2025
Python version: 3.11

Copyright statement: This code has been developed during work undertaken within
the RefMap project (www.refmap.eu), based on the RefMap code repository

"""

# %% Import block

import os
import matplotlib as mpl
from matplotlib import pyplot as plt

# %% _is_jupyter
def _is_jupyter():
    """_is_jupyter()
    
    Detect if running inside a Jupyter-like environment
    (incl. VS Code notebooks).
    
    """

    try:
        from IPython import get_ipython
        shell = get_ipython()

        if not shell:
            return False

        return shell.__class__.__name__ in (
            "ZMQInteractiveShell",
            "Shell",
            "TerminalInteractiveShell",
        )

    except Exception:
        return False
# end of _is_jupyter internal function


# %% _is_interactive_console 
def _is_interactive_console():
    """_is_interactive_console()
    Detect VS Code or Spyder interactive consoles (non-Jupyter).
    """

    return "IPYKERNEL" in os.environ or "JPY_PARENT_PID" in os.environ
# end of _is_interactive_console internal function


# %% _ensure_backend
def _ensure_backend():
    """_ensure_backend()
    Ensure Matplotlib uses a GUI backend when available.
    """

    try:
        current = mpl.get_backend().lower()
        if "agg" in current:
            mpl.use("QtAgg", force=True)

    except Exception:
        pass
# end of _ensure_backend internal function


# %% create_figure
def create_figure(*args, **kwargs):
    """create_figure(*args, **kwargs)

    Create a Matplotlib figure safely.

    - Temporarily disables interactive mode to prevent auto-display
      in IDEs or notebooks.
    - Restores prior state afterward.
    """
    interactive = mpl.is_interactive()
    plt.ioff()
    try:
        fig, axs = plt.subplots(*args, **kwargs)

    finally:
        if interactive:
            plt.ion()

    return fig, axs
# end of create_figure function


# %% show_plot
def show_plot(fig=None, block=True):
    """show_plot(fig=None, block=True)

    Display a Matplotlib figure safely across environments.

    - Jupyter/VS Code: relies on auto-display (no duplicate)
    - Shell/script: opens GUI window
    - Headless: saves fallback PNG

    Closes figure after display to prevent IDEs from duplicating it.
    """
    _ensure_backend()

    if fig is None:
        fig = plt.gcf()

    try:
        # jupyter or VS Code notebook ---
        if _is_jupyter():
            from IPython.display import display
            display(fig)
            plt.close(fig)
            return None

        # IDE interactive console (Spyder / VS Code terminal) ---
        if _is_interactive_console() and mpl.is_interactive():
            plt.draw()
            plt.close(fig)
            return None

        # regular Python script / shell ---
        fig.show()
        plt.close(fig)
        return None

    except Exception as e:
        print(f"[plotting] Could not open GUI plot ({e}). Saving to file.")
        try:
            outfile = os.path.abspath("plot_output.png")
            plt.savefig(outfile, dpi=300, bbox_inches="tight")
            print(f"[plotting] Fallback figure saved to: {outfile}")

        except Exception:
            pass

        return None
# end of show_plot function