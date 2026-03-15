#!/usr/bin/env bash
# =============================================================================
# run_all_tests.sh
# =============================================================================
# Run from the PROJECT ROOT directory:
#
#   chmod +x experiments/run_all_tests.sh
#   ./experiments/run_all_tests.sh
#
# All output is saved to results/attack_results.txt so you can review,
# compare, and present a full picture of model robustness.
#
# Test groups:
#   1. Severity sweep          — all attacks, low → high severity
#   2. Fraction sweep          — noise + replay, sparse → widespread
#   3. Lookback sweep          — replay only, recent → distant past
#   4. Sensor targeting        — current sensors vs voltage sensors vs all
#   5. Seed verification       — confirm stochastic results are consistent
#   6. Compound configurations — realistic combined parameter scenarios
# =============================================================================

# --- Setup -------------------------------------------------------------------
mkdir -p results
OUTPUT="results/attack_results.txt"
> "$OUTPUT"   # Clear any previous results

CMD="python experiments/run_attacks.py"

run() {
    # Prints a label to screen AND appends full output to the results file
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


# =============================================================================
# GROUP 1 — SEVERITY SWEEP
# =============================================================================
# Purpose: Establish how model performance degrades as attacks get stronger.
# All other parameters are held at defaults (fraction=0.3, lookback=50,
# seed=42, all columns targeted).
#
# What to look for in results:
#   - At what severity does recall first drop below 90%?  80%?
#   - Is the degradation gradual (linear) or sudden (cliff)?
#   - Which attack type causes the steepest drop at each severity?
#   - Does the detection agent or diagnostic agent degrade faster?
# =============================================================================
echo ""
echo "===== GROUP 1: SEVERITY SWEEP ================================"

run "SEVERITY 0.05 — Barely perceptible (5% of 1 std dev)" \
    --severity 0.05

run "SEVERITY 0.1 — Very low (10% of 1 std dev)" \
    --severity 0.1

run "SEVERITY 0.2 — Low" \
    --severity 0.2

run "SEVERITY 0.3 — Low-moderate" \
    --severity 0.3

run "SEVERITY 0.5 — Moderate (default)" \
    --severity 0.5

run "SEVERITY 0.75 — Moderate-high" \
    --severity 0.75

run "SEVERITY 1.0 — High (1 full std dev offset)" \
    --severity 1.0

run "SEVERITY 1.5 — Very high" \
    --severity 1.5

run "SEVERITY 2.0 — Extreme (2 std dev offset)" \
    --severity 2.0

run "SEVERITY 3.0 — Catastrophic" \
    --severity 3.0


# =============================================================================
# GROUP 2 — FRACTION SWEEP (noise + replay)
# =============================================================================
# Purpose: Understand how the proportion of corrupted samples affects results.
# Severity is held constant at 0.5 so only fraction varies.
#
# What to look for in results:
#   - Does sparse noise (10%) affect the model noticeably?
#   - Is there a fraction threshold where results collapse?
#   - Compare noise_injection vs replay_attack at identical fraction values —
#     which is more dangerous at the same coverage?
# =============================================================================
echo ""
echo "===== GROUP 2: FRACTION SWEEP ================================"

run "FRACTION 0.05 — Very sparse (5% of samples corrupted)" \
    --severity 0.5 --fraction 0.05

run "FRACTION 0.1 — Sparse (10%)" \
    --severity 0.5 --fraction 0.1

run "FRACTION 0.2 — Low (20%)" \
    --severity 0.5 --fraction 0.2

run "FRACTION 0.3 — Moderate default (30%)" \
    --severity 0.5 --fraction 0.3

run "FRACTION 0.5 — Half of all samples" \
    --severity 0.5 --fraction 0.5

run "FRACTION 0.7 — High coverage (70%)" \
    --severity 0.5 --fraction 0.7

run "FRACTION 0.9 — Near-total coverage (90%)" \
    --severity 0.5 --fraction 0.9

run "FRACTION 1.0 — All samples corrupted" \
    --severity 0.5 --fraction 1.0


# =============================================================================
# GROUP 3 — LOOKBACK SWEEP (replay attack only)
# =============================================================================
# Purpose: Test whether replaying recent data vs distant past data matters.
# All parameters held constant except lookback.
#
# What to look for in results:
#   - Does lookback=5 (nearly identical readings) affect the model at all?
#   - At what lookback does replay become genuinely dangerous?
#   - The replay attack insight: danger comes from HOW DIFFERENT the
#     replayed data is, not just that data was replaced.
#
# Note: Bias and noise results will be identical across all these runs
# (lookback doesn't affect them). Focus on replay_attack rows.
# =============================================================================
echo ""
echo "===== GROUP 3: LOOKBACK SWEEP (replay attack) ================"

run "LOOKBACK 5 — Replay from 5 samples ago (nearly identical)" \
    --severity 0.5 --fraction 0.4 --lookback 5

run "LOOKBACK 10 — Replay from up to 10 samples ago" \
    --severity 0.5 --fraction 0.4 --lookback 10

run "LOOKBACK 25 — Replay from up to 25 samples ago" \
    --severity 0.5 --fraction 0.4 --lookback 25

run "LOOKBACK 50 — Default lookback" \
    --severity 0.5 --fraction 0.4 --lookback 50

run "LOOKBACK 100 — Replay from up to 100 samples ago" \
    --severity 0.5 --fraction 0.4 --lookback 100

run "LOOKBACK 200 — Replay from a significantly different time window" \
    --severity 0.5 --fraction 0.4 --lookback 200

run "LOOKBACK 500 — Replay from distant past (maximum threat)" \
    --severity 0.5 --fraction 0.4 --lookback 500


# =============================================================================
# GROUP 4 — SENSOR TARGETING
# =============================================================================
# Purpose: Compare attack impact when targeting current sensors only,
# voltage sensors only, or all sensors. Helps identify which sensors
# are the critical attack surface.
#
# Column indices:
#   0=Ia  1=Ib  2=Ic  (current sensors)
#   3=Va  4=Vb  5=Vc  (voltage sensors)
#
# What to look for in results:
#   - Does corrupting current sensors cause more damage than voltage?
#   - Are there single sensors that are disproportionately important?
#   - Is attacking all columns significantly worse than one group?
#
# Run at moderate and high severity to see if the pattern changes.
# =============================================================================
echo ""
echo "===== GROUP 4: SENSOR TARGETING =============================="

run "ALL SENSORS — Baseline for comparison (severity 0.5)" \
    --severity 0.5

run "CURRENT SENSORS ONLY — Ia Ib Ic (severity 0.5)" \
    --severity 0.5 --target-cols 0 1 2

run "VOLTAGE SENSORS ONLY — Va Vb Vc (severity 0.5)" \
    --severity 0.5 --target-cols 3 4 5

run "SINGLE SENSOR — Ia only (severity 0.5)" \
    --severity 0.5 --target-cols 0

run "SINGLE SENSOR — Va only (severity 0.5)" \
    --severity 0.5 --target-cols 3

# Repeat at higher severity to see if pattern changes
run "ALL SENSORS — High severity baseline (severity 1.5)" \
    --severity 1.5

run "CURRENT SENSORS ONLY — High severity (severity 1.5)" \
    --severity 1.5 --target-cols 0 1 2

run "VOLTAGE SENSORS ONLY — High severity (severity 1.5)" \
    --severity 1.5 --target-cols 3 4 5


# =============================================================================
# GROUP 5 — SEED VERIFICATION (stochastic attacks)
# =============================================================================
# Purpose: Confirm that noise and replay results are consistent across
# different random seeds. If results vary wildly by seed, the findings
# are not reliable and need averaging across multiple seeds.
#
# What to look for in results:
#   - Are noise_injection and replay_attack results stable across seeds?
#   - Bias and drift will be IDENTICAL across all seeds (deterministic).
#   - If stochastic results vary by more than ~2-3%, report an average
#     rather than a single run.
# =============================================================================
echo ""
echo "===== GROUP 5: SEED VERIFICATION ============================="

run "SEED 42 — Default seed" \
    --severity 0.5 --fraction 0.3 --seed 42

run "SEED 123 — Alternative seed" \
    --severity 0.5 --fraction 0.3 --seed 123

run "SEED 999 — Alternative seed" \
    --severity 0.5 --fraction 0.3 --seed 999

run "SEED 7 — Alternative seed" \
    --severity 0.5 --fraction 0.3 --seed 7

run "SEED 2024 — Alternative seed" \
    --severity 0.5 --fraction 0.3 --seed 2024


# =============================================================================
# GROUP 6 — COMPOUND CONFIGURATIONS
# =============================================================================
# Purpose: Test realistic combined scenarios rather than isolated parameters.
# These represent specific threat models you might actually encounter.
#
# Scenario descriptions are included so output is self-explanatory
# when presenting results.
# =============================================================================
echo ""
echo "===== GROUP 6: COMPOUND CONFIGURATIONS ======================="

# --- Subtle persistent bias on current sensors
# Models a miscalibrated current sensor after maintenance.
# Low severity, all three current channels, no stochastic elements.
run "SCENARIO: Miscalibrated current sensors after maintenance" \
    --severity 0.15 --target-cols 0 1 2

# --- Widespread low-noise across all sensors
# Models general EM interference across the measurement system.
# Low per-sample corruption but affects 70% of readings.
run "SCENARIO: Widespread EM interference, low intensity" \
    --severity 0.2 --fraction 0.7 --seed 42

# --- Targeted high-noise on single phase
# Models a single faulty sensor transmitting corrupted readings.
# High severity but only one sensor column.
run "SCENARIO: Single sensor failure, high corruption" \
    --severity 1.5 --fraction 0.6 --target-cols 0 --seed 42

# --- Replay attack masking an active fault
# The most dangerous realistic replay scenario:
# high coverage, long lookback, targeting current sensors.
run "SCENARIO: Active SCADA replay attack on current sensors" \
    --severity 0.5 --fraction 0.6 --lookback 300 --target-cols 0 1 2 --seed 42

# --- Late-stage sensor drift
# Models a sensor nearing complete failure.
# Very high severity drift on voltage sensors.
run "SCENARIO: Late-stage voltage sensor degradation" \
    --severity 2.5 --target-cols 3 4 5

# --- Combined moderate attack on all sensors
# A coordinated multi-vector attack at moderate severity.
run "SCENARIO: Coordinated moderate attack, all sensors" \
    --severity 0.8 --fraction 0.5 --lookback 150 --seed 42

# --- High severity, all sensors, high fraction
# Worst-case scenario: everything under maximum pressure.
run "SCENARIO: Worst-case maximum pressure test" \
    --severity 2.0 --fraction 0.9 --lookback 500 --seed 42

# --- Near-imperceptible attack — testing detection limits
# How subtle can an attack be while still degrading recall?
run "SCENARIO: Near-imperceptible attack — detection lower bound" \
    --severity 0.05 --fraction 0.05 --lookback 5 --seed 42


# =============================================================================
# DONE
# =============================================================================
echo ""
echo "============================================================="
echo "  ALL TESTS COMPLETE"
echo "  Full results saved to: $OUTPUT"
echo ""
echo "  Summary of test groups:"
echo "    Group 1 — Severity sweep        (10 runs)"
echo "    Group 2 — Fraction sweep         (8 runs)"
echo "    Group 3 — Lookback sweep         (7 runs)"
echo "    Group 4 — Sensor targeting       (8 runs)"
echo "    Group 5 — Seed verification      (5 runs)"
echo "    Group 6 — Compound scenarios     (8 runs)"
echo "                                    --------"
echo "    Total                           46 runs"
echo "============================================================="