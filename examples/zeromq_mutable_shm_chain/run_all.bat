start "mstore" cmd /k "uv run python -m mstore.cli"

start "sink"   cmd /k "uv run python sink.py --after-stage 3"

start "stage3" cmd /k "uv run python stage.py --stage 3"
start "stage2" cmd /k "uv run python stage.py --stage 2"
start "stage1" cmd /k "uv run python stage.py --stage 1"

start "producer" cmd /k "uv run python producer.py --gib 2"