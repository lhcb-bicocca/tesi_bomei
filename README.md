# Limit Setting for $\tau\to\mu\mu\mu$

This package acts as a codebase for Alessandro Bomei's bachelor thesis (2026).

The project aims at developing a software to calculate upper limits on the branching fraction of $\tau\to\mu\mu\mu$ decays.


## `uv` based

The template is based on [uv](https://docs.astral.sh/uv/) an "extremely fast Python package and project manager, written in Rust".

Once the repository is prepared following the instructions above, run `uv sync` to install the dependencies and prepare the package.
If new dependencies are needed, install them with `uv add <package>`. Remember to add `pyproject.toml` and `uv.lock` to the commit for keeping track of the new dependencies.
In case the environment needs to be rebuilt, simply run `uv sync` again.

For running macros or code within the environment, use

```bash
uv run python macros/run_myfunc.py
```

or 

```bash
source .venv/bin/activate
python macros/run_myfunc.py
```

## Dependencies

A dependency on the [analysis_helpers](https://cpviolation.github.io/analysis_helpers/) package is included.

For loading functions within the package act as usual:

```py
from analysis_helpers.plotting import plot_hist
```

## Package Structure

The package is structured as follows:

```
my-project/
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions CI pipeline (add it here if needed)
│       └── template-sync.yml    # GitHub Actions CI pipeline for syncing the template
├── macros
│   └── run_myfunc.py            # example macro 
├── notebooks
│   └── mynotebook.ipynb         # example notebook
├── src
│   └── mypkg
│       └── library.py           # example library of functions
├── LICENSE
├── main.py
├── pyproject.toml               # Project dependencies and metadata (uv)
├── README.md
└── uv.lock                      # Lockfile for reproducible builds
```

It is suggested to write all the functions to the `mypkg` directory, eventually in multiple files.
With this approach they will not always be accessible in the `macros`, but also in the `notebooks`.
Use `macros` for running programs that do not require testing, and `notebooks` to test your functions of showing examples.
Before committing any notebook remember to re-run them from scratch to test for inconsistencies.
