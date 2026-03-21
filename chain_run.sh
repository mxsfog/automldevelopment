#!/bin/bash
# Цепочка UAF сессий: каждая продолжает предыдущую
# Использование: ./chain_run.sh <task_yaml> <n_sessions> [hours_per_session]
# Пример: ./chain_run.sh data/sports_betting/task.yaml 5 1.5

TASK=$1
N=${2:-5}
HOURS=${3:-1.5}
ITERATIONS=100
MODEL=claude-sonnet-4-6

export PATH="$HOME/.local/bin:$PATH"

prev_session=""
for i in $(seq 1 $N); do
    ts=$(date +%b%d_%H%M | tr '[:upper:]' '[:lower:]')
    session_id="chain_${i}_${ts}"

    echo "=== Chain session $i/$N: $session_id ==="

    if [ -z "$prev_session" ]; then
        uv run uaf run \
            --task "$TASK" \
            --session-id "$session_id" \
            --time "$HOURS" \
            --budget-iterations "$ITERATIONS" \
            --autonomous \
            --model "$MODEL" \
            2>&1 | tee ".uaf/sessions/${session_id}.log"
    else
        uv run uaf run \
            --task "$TASK" \
            --session-id "$session_id" \
            --prev-session "$prev_session" \
            --time "$HOURS" \
            --budget-iterations "$ITERATIONS" \
            --autonomous \
            --model "$MODEL" \
            2>&1 | tee ".uaf/sessions/${session_id}.log"
    fi

    prev_session="$session_id"
    echo "=== Session $i done. Best metric logged to MLflow ==="
done

echo "=== Chain complete: $N sessions ==="
