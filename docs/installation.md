# Setup environments

How to setup environments for running the code.

## Clone the repository to your local machine

Clone from the upstream repository:

```sh
git clone https://github.com/yhma-lab/DeepLearningFlappyBird.git
```

or from your forked repository:

```sh
git clone https://<your-github-username>/DeepLearningFlappyBird.git
```

## Install dependencies

### If you have `uv` installed

```sh
uv sync --dev
```

That's it!

```sh
# activate the environment and run the game
source ./.venv/bin/activate
python3 run_game.py

# or run script using `uv`
uv run run_game.py
```

Enjoy!

### If you have `conda` installed

Create a conda virtual environment in the project directory:

```sh
conda create -p ./.conda-venv -c conda-forge python=3.12 numpy pygame ruff mypy jupyterlab
# optional if not to train the model
conda install -p ./.conda-venv -c conda-forge jax flax opencv
```

Activate your conda virtual environment:

```sh
conda activate ./.conda-venv
# DO NOT FORGET to install the local flappybird package
pip install -e ./modules/flappybird
```

Try to run the game:

```sh
python3 run_game.py
```

Enjoy!

### If you prefer original `pip` and `venv`

```sh
# Check your python version, which should be >=3.9,<3.13
python3 --version

# Create a virtual environment in the project directory
python3 -m venv ./.venv

# Be sure to use the correct `pip` path
#
# How to check the `pip` path:
#
#     pip --version
#
# If the path is not `./.venv/bin/pip`, you should use the correct path.
# Or you can use the following command to install the dependencies either.
#
#     python3 -m pip install -r requirements.txt
#
pip install -r requirements.txt
```
