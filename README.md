# LHCb Bicocca Template Package
This is a template package for developing the codebase of analyses and projects developed within the LHCb Bicocca group.

## Use the template

### Dual-Remote Approach
To use the template we follow the Dual-Remote approach
```
                     ┌────────────────────────┐
                     │ Base Template Repo     │
                     └───────────┬────────────┘
                                 │
                   (Fetch updates via `template`)
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Downstream Project Repository                                           │
│                                                                         │
│  Remote 'origin'   ──► git@github.com:lhcb-bicocca/my-new-project.git   │
│  Remote 'template' ──► git@github.com:lhcb-bicocca/pkg-template.git     │
└─────────────────────────────────────────────────────────────────────────┘
```

In this workflow, every repository created from the template keeps two Git remotes:

- origin: Points to the individual project repository (where normal work happens).
- template: Points back to the original base template repository.

#### Step 1: Initialize the Downstream Repo

Instead of using GitHub's "Use this template" button, clone the base template, create the new repository on GitHub, and re-link the remotes:

```bash
# 1. Clone the base template into your new project directory
git clone git@github.com:lhcb-bicocca/pkg-template my-new-project
cd my-new-project

# 2. Rename 'origin' to 'template'
git remote rename origin template

# 3. Create your new empty repository in the org on GitHub, then link it as 'origin'
git remote add origin git@github.com:lhcb-bicocca/my-new-project.git

# 4. Push the initial code to your new repo
git push -u origin main
```

#### Step 2: Pull Updates from the Template Later
Whenever you add features or bug fixes to base-template, developers in child repositories can pull those updates with standard Git merge or rebase operations:

```bash
# Fetch latest changes from the base template
git fetch template

# Merge updates into your active branch (or use rebase)
git merge template/main --allow-unrelated-histories
```

### Automated Sync via GitHub Actions

A `template-sync` workflow is added to `.github/workflows`.
Child repositories inherit this workflow.
Every night (or on manual trigger), the action checks `lhcb-bicocca/pkg-template` for new commits.
If new features exist, it automatically creates a Pull Request in the child repository with the new code, letting maintainers review and resolve conflicts via GitHub's standard PR UI


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
