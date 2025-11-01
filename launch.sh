curl -LsSf https://astral.sh/uv/install.sh
pip install uv
uv venv --clear
source .venv/bin/activate
uv pip install -e .
uv cache prune
uv run tests/test_env.py
source .venv/bin/activate