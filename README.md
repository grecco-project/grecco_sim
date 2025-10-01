# Simulation Environment for GreCCo EC control

This python framework offers a testbed for energy community control within in the GreCCo project.

Its core is a simulator.

Starting a simulation is done using the script
grecco_sim/scripts/start_simulation.py

# Documentation

Generate the Documentation using `sphinx-build doc _build`.
If in doubt about style:

https://google.github.io/styleguide/pyguide.html


## Installation (Attention: has changed in Feb 2025)

The project is managed with `uv`.
Get `uv` from astral:
https://docs.astral.sh/uv/getting-started/installation/#installation-methods

E.g. on linux via:
```
wget -qO- https://astral.sh/uv/install.sh | sh
```

Then set up a virtual environment using 
```
uv sync --extra sim --extra dev
```
`uv` will set up a virtual environment in the main folder in `.venv`.
Use this environment in the following.

A bare installation will install the dependencies necessary for using the types defined in grecco_sim for coordination.
The extra dependency `sim` makes sure that packages necessary for simulation are installed. Install also `dev` to executre linting and pytests.

You can editably install the project using `uv pip install -e ./` when in the main project directory.

In general, it should be sufficient to clone the project and execute the start script.
There is sample simulation data in the `data` folder 


### Adding dependencies
Depencies are added using `uv add <packet name>`.


### Gurobi (Academic licence) - may be needed for on/off flexibilities

DO NOT INSTALL CASADI FROM CONDA CHANNELS!
Instead, use pip to install casadi. This way, solver binaries are included.

Install Gurobi on your system.
Some environment variables have to be set. See them here:

https://github.com/casadi/casadi/wiki/FAQ%3A-how-to-get-third-party-solvers-to-work%3F

Hint: the LD_LIBRARY_PATH must be set *before* starting the python script.
In pycharm this can be achieved using the run configurations.
