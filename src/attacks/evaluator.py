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
        
        """


    @property
    def false_positive_rate(self) -> float:
        """
        
        """




@dataclass
class AttackResult:
    """
    
    """


    @property
    def detection_drop(self) -> float:
        """
        
        
        """


    @property
    def diagnostic_drop(self) -> float:
        """
        
        """


    @property
    def cascading_failure_risk(self) -> str:
        """
        
        """



# Main evaluation function
def evaluate_agent() -> AgentResult:
    """
    
    """



# functions for reporting results
def print_report(results: list[AttackResult]) -> None:
    """
    
    """


