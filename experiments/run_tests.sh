#!/usr/bin/env bash
# =============================================================================
# run_all_tests.sh (compatible with CSV-only run_attacks.py)
# =============================================================================

mkdir -p results
OUTPUT="results/attack_results.txt"
> "$OUTPUT"   # clear previous results

CMD="python experiments/run_attacks.py"

run() {
    local label="$1"
    shift
    echo ""
    echo ">>> $label"
    echo "" >> "$OUTPUT"
    echo "##############################################################" >> "$OUTPUT"
    echo "# TEST: $label" >> "$OUTPUT"
    echo "##############################################################" >> "$OUTPUT"
    $CMD "$@" >> "$OUTPUT" 2>&1
    echo "    Done. Output saved."
}

echo "============================================================="
echo "  ADVERSARIAL ATTACK TEST SUITE"
echo "  Output: $OUTPUT"
echo "============================================================="

# -----------------------
# GROUP 1 — SEVERITY SWEEP
# -----------------------
echo ""
echo "===== GROUP 1: SEVERITY SWEEP ================================"

for sev in 0.05 0.1 0.2 0.3 0.5 0.75 1.0 1.5 2.0 3.0
do
    run "SEVERITY $sev" --severity $sev
done

# -----------------------
# GROUP 2 — FRACTION SWEEP
# -----------------------
echo ""
echo "===== GROUP 2: FRACTION SWEEP ================================"

for frac in 0.05 0.1 0.2 0.3 0.5 0.7 0.9 1.0
do
    run "FRACTION $frac" --severity 0.5 --fraction $frac
done

# -----------------------
# GROUP 3 — LOOKBACK SWEEP
# -----------------------
echo ""
echo "===== GROUP 3: LOOKBACK SWEEP ================================"

for lb in 5 10 25 50 100 200 500
do
    run "LOOKBACK $lb" --severity 0.5 --fraction 0.4 --lookback $lb
done

# -----------------------
# GROUP 4 — SENSOR TARGETING
# -----------------------
echo ""
echo "===== GROUP 4: SENSOR TARGETING =============================="

run "ALL SENSORS" --severity 0.5
run "CURRENT SENSORS ONLY" --severity 0.5 --target-cols 0 1 2
run "VOLTAGE SENSORS ONLY" --severity 0.5 --target-cols 3 4 5
run "SINGLE SENSOR Ia" --severity 0.5 --target-cols 0
run "SINGLE SENSOR Va" --severity 0.5 --target-cols 3
run "ALL SENSORS HIGH SEV" --severity 1.5
run "CURRENT SENSORS HIGH SEV" --severity 1.5 --target-cols 0 1 2
run "VOLTAGE SENSORS HIGH SEV" --severity 1.5 --target-cols 3 4 5

# -----------------------
# GROUP 5 — SEED VERIFICATION
# -----------------------
echo ""
echo "===== GROUP 5: SEED VERIFICATION ============================="

for seed in 42 123 999 7 2024
do
    run "SEED $seed" --severity 0.5 --fraction 0.3 --seed $seed
done

# -----------------------
# GROUP 6 — COMPOUND SCENARIOS
# -----------------------
echo ""
echo "===== GROUP 6: COMPOUND SCENARIOS ============================"

run "Subtle bias current sensors" --severity 0.15 --target-cols 0 1 2
run "Widespread low-noise all sensors" --severity 0.2 --fraction 0.7 --seed 42
run "Single sensor high-noise" --severity 1.5 --fraction 0.6 --target-cols 0 --seed 42
run "Replay attack current sensors" --severity 0.5 --fraction 0.6 --lookback 300 --target-cols 0 1 2 --seed 42
run "Late-stage voltage drift" --severity 2.5 --target-cols 3 4 5
run "Moderate attack all sensors" --severity 0.8 --fraction 0.5 --lookback 150 --seed 42
run "Worst-case max pressure" --severity 2.0 --fraction 0.9 --lookback 500 --seed 42
run "Near-imperceptible attack" --severity 0.05 --fraction 0.05 --lookback 5 --seed 42

echo ""
echo "============================================================="
echo "  ALL TESTS COMPLETE"
echo "  Full logs saved to: $OUTPUT"
echo "============================================================="