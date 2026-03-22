#!/bin/bash
# Цепочка UAF сессий: каждая продолжает предыдущую
# Использование: ./chain_run.sh <task_yaml> <n_sessions> [hours_per_session]
# Пример: ./chain_run.sh data/<task>/task.yaml 5 1.5

TASK=$1
N=${2:-5}
HOURS=${3:-1.5}
ITERATIONS=100
MODEL=claude-sonnet-4-6
LEAKAGE_THRESHOLD=60.0

export PATH="$HOME/.local/bin:$PATH"

# Читаем metric_name из task.yaml
METRIC_NAME=$(python3 -c "
import yaml, sys
try:
    d = yaml.safe_load(open('$TASK'))
    print(d.get('metric', {}).get('name', 'metric'))
except:
    print('metric')
" 2>/dev/null || echo "metric")

# Возвращает значение целевой метрики из metadata.json сессии, или -999 если нет/leakage
get_metric() {
    local sid=$1
    local meta=".uaf/sessions/${sid}/models/best/metadata.json"
    [ -f "$meta" ] || { echo "-999"; return; }
    python3 -c "
import json
try:
    d = json.load(open('$meta'))
    v = d.get('$METRIC_NAME') or d.get('roi') or d.get('auc') or -999
    print(v)
except:
    print(-999)
"
}

prev_session=""
last_clean_session=""  # последняя сессия без leakage

for i in $(seq 1 $N); do
    ts=$(date +%b%d_%H%M | tr '[:upper:]' '[:lower:]')
    session_id="chain_${i}_${ts}"

    echo "=== Chain session $i/$N: $session_id ==="

    # Определяем от кого продолжать: берём last_clean_session, не просто prev
    chain_from="$last_clean_session"

    if [ -z "$chain_from" ]; then
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
            --prev-session "$chain_from" \
            --time "$HOURS" \
            --budget-iterations "$ITERATIONS" \
            --autonomous \
            --model "$MODEL" \
            2>&1 | tee ".uaf/sessions/${session_id}.log"
    fi

    # Проверяем метрику сессии — обновляем last_clean только если нет leakage
    metric_val=$(get_metric "$session_id")
    echo "=== Session $i done. ${METRIC_NAME}=${metric_val}% ==="

    if python3 -c "
v = float('$metric_val')
thr = float('$LEAKAGE_THRESHOLD')
import sys
sys.exit(0 if v > -900 and v <= thr else 1)
" 2>/dev/null; then
        last_clean_session="$session_id"
        echo "=== Clean session, chain continues from $session_id ==="
    else
        echo "=== Session $session_id skipped for chain (${METRIC_NAME}=$metric_val > threshold or no model) ==="
        echo "=== Next session will continue from: ${last_clean_session:-none} ==="
    fi
done

echo "=== Chain complete: $N sessions ==="
