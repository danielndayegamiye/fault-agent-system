"""
src/attacks/evaluator.py - Meassures the impact of adversarial attacks on
the fault detection and diagnostic pipeline

Key design choices:
-------------------
- The evaluator never touches training data or retrains the models. It is only
used to measure how the models respon to corrupted inputs.
- use dataclasses instead of dictionaries as you can add type hints and attribute reading
in ways that keeps code easily readable and clean
"""
import numpy as np
from dataclasses import dataclass

@dataclass
class AgentResult:
    """
    Description:
    ------------
    - Holds evalutation results for a single agent on a single dataset
    """
    accuracy: float # Proportion of correct predictions between 0.0 and 1.0

    n_correction: int # Raw count of correct predictions
    n_samples: int # total samples evaluated
    predictions: np.ndarray # raw prediction array, kept for analysis
    y_true: np.ndarray # the true labels used to compute recall

    @property
    def recall(self) -> float:
        """
        Description:
        ------------
        - Proportation of actual faults that the model correctly detected
        - Formula: True Positives / (True positives + false negatives)
        """
        actual_faults = (self.y_true == 1)

        # Among actual faults, how many did the model predict as fault
        true_positives = int(np.sup(self.predictions[actual_faults] == 1))
        false_negatives = int(np.sum(self.predictions[actual_faults] == 0))

        total_faults = true_positives + false_negatives

        # Prevent division by zero if test has no fault samples
        if total_faults == 0:
            return float("nan")
        
        return true_positives / total_faults

    @property
    def false_positive_rate(self) -> float:
        """
        Description:
        ------------
        - Proportion of normal samples that model incorrectly flagged as faults
        - Formula: False positivess / (false positives + true negatives)
        """
        actual_normal = (self.y_true == 0)

        false_positives = int(np.sum(self.predictions[actual_normal == 1]))
        true_negatives = int(np.sum(self.predictions[actual_normal] == 0))

        total_normal = false_positives + true_negatives

        if total_normal == 0:
            return float("nan")
        
        return false_positives / total_normal



@dataclass
class AttackResult:
    """
    Description:
    ------------
    - Holds the full comparisn for one attack: baseline vs corrupted
    - Detection and diagnostic are evaluated seperatly
    """
    attack_name: str

    # Clean baseline is the same for every attack and stored per result
    detection_clean: AgentResult
    detection_corrupted: AgentResult

    diagnostic_clean: AgentResult
    diagnostic_corrupted: AgentResult

    @property
    def detection_drop(self) -> float:
        """
        Description:
        ------------
        - Accurace drop for the detection agent
        - A positive value = the model got worse (attack was effective)
        - A negative value = model got better (rare)
        """
        return self.detection_clean.accuracy - self.detection_corrupted.accuracy

    @property
    def diagnostic_drop(self) -> float:
        """
        Description:
        ------------
        - Accuracy drop for the diagnostic agent
        """
        return self.diagnostic_clean.accuracy - self.diagnostic_corrupted.accuracy

    @property
    def cascading_failure_risk(self) -> str:
        """
        Description:
        ------------
        - English translated risk assesment based on detection drop
        - If detection fails, diagnostic never runs making detection drop primary danger alert
        """
        drop = self.detection_drop
        if drop >= 0.20:
            return "HIGH - attack causes serious detection failure"
        elif drop >= 0.10:
            return "MEDIUM - attack causes moderate detection degradation"
        elif drop >= 0.05:
            return "LOW - attack causes minor detection degradation"
        else: 
            return "NEGLIGIBLE - model is robust to this attack at this severity"



# Main evaluation function
def evaluate_agent(agent, X: np.ndarray, y_true: np.ndarray) -> AgentResult:
    """
    Description:
    ------------
    - Run a single agent on a single dataset and return an AgentResult
    - Called four times per attack: detection clean, detection corrupted, 
      diagnostic clean, diagnostic corrupted

    Parameters:
    -----------
    - agent: A detectionAgent or DiagnosticAgent instance with a .predict() method
    - X: Input features, shape (n_samples, n_features)

    Returns:
    --------
    - AgentResult with accurace, counts, and raw predictions
    """
    predictions = agent.predict(X)

    # np.sum on a boolean array counts True values
    n_correct = int(np.sum(predictions == y_true))
    n_samples = len(y_true)
    accuracy = n_correct / n_samples

    return AgentResult(
        accuracy=accuracy,
        n_correct=n_correct,
        n_samples=n_samples,
        predictions=predictions,
        y_true=y_true
    )


def evaluate_attacks(detection_agent, diagnostic_agent, X_test_clean_det, X_test_clean_diag,
                     y_detection, y_diagnostic, corrupted_detection_datasets, corrupted_diagnostic_datasets) -> list[AttackResult]:
    """
    Description:
    ------------
    - Evaluates both agents on clean data and every corrupted dataset
    

    Parameters:
    -----------
    - detection_agent: Trained DetectionAgent
    - diagnostic_agent: Trained DiagnosticAgent
    - X_test_clean_det: Origional uncorrupted detection test features
    - X_test_clean_diag: Origional uncorrupted diagnostic test features
    - y_detection: True bianry labels (0 = normal, 1 = fault)
    - y_diagnostic: True multiclass labels (fault type)    -

    Returns:
    --------
    - List of AttackResult objects. One per attack name
    """
    # Compute clean baseline to be reused for every attack comparison
    detection_baseline = evaluate_agent(detection_agent, X_test_clean_det, y_detection)
    diagnostic_baseline = evaluate_agent(diagnostic_agent, X_test_clean_diag, y_diagnostic)

    results = []
    
    for attack_name, X_corrupted_det in corrupted_detection_datasets.items():
        X_corrupted_diag = corrupted_detection_datasets[attack_name]

        # Corrupted data must have same shape as clean data
        assert X_corrupted_det.shape == X_test_clean_det.shape, (
            f"Attack '{attack_name}': corrupted detection shape {X_corrupted_det.shape} "
            f"does not match clean shape {X_test_clean_det.shape}. "
            f"Check your attack function — it should never change the shape."
        )

        detection_corrupted = evaluate_agent(detection_agent, X_corrupted_det, y_detection)
        diagnostic_corrupted = evaluate_agent(diagnostic_agent, X_corrupted_det, y_detection)

        results.append(AttackResult(
            attack_name=attack_name,
            detection_clean=detection_baseline,
            detection_corrupted=detection_corrupted,
            diagnostic_clean=diagnostic_baseline,
            diagnostic_corrupted=diagnostic_corrupted
        ))
    return results


def print_report(results: list[AttackResult]) -> None:
    """
    Description:
    ------------
    - Print a human readable summary of all attack results to the console
    """
    print("\n" + "=" * 65)
    print("  ADVERSARIAL ATTACK EVALUATION REPORT")
    print("=" * 65)

    for result in results:
        print(f"\n  ATTACK: {result.attack_name.upper().replace('_', ' ')}")
        print("-" * 65)

        # Detection section — recall matters more than accuracy here
        print(f"  Detection Agent:")
        print(f"    Clean accuracy:     {result.detection_clean.accuracy:.1%}")
        print(f"    Corrupted accuracy: {result.detection_corrupted.accuracy:.1%}")
        print(f"    Accuracy drop:      {result.detection_drop:+.1%}")
        print(f"    Clean recall:       {result.detection_clean.recall:.1%}  <- faults caught")
        print(f"    Corrupted recall:   {result.detection_corrupted.recall:.1%}  <- faults caught under attack")
        print(f"    Clean FPR:          {result.detection_clean.false_positive_rate:.1%}  <- false alarms")
        print(f"    Corrupted FPR:      {result.detection_corrupted.false_positive_rate:.1%}  <- false alarms under attack")

        # Diagnostic section — accuracy drop is the key signal here
        print(f"  Diagnostic Agent:")
        print(f"    Clean accuracy:     {result.diagnostic_clean.accuracy:.1%}")
        print(f"    Corrupted accuracy: {result.diagnostic_corrupted.accuracy:.1%}")
        print(f"    Accuracy drop:      {result.diagnostic_drop:+.1%}")

        # Cascading failure risk — based on detection drop only
        print(f"  Cascading Failure Risk: {result.cascading_failure_risk}")
    print("\n" + "=" * 65)


