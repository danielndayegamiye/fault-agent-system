import argparse
import logging
from pathlib import Path

from src.data.loader import load_data
from src.agents.detection_agent import DetectionAgent
from src.agents.diagnostic_agent import DiagnosticAgent

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def save_importances(agent, task_name, save_dir):
    importances = agent.feature_importance()
    filepath = save_dir / f"{task_name}_feature_importances.txt"
    
    with open(filepath, "w") as f:
        f.write(f"Feature Importances for {task_name.capitalize()} Agent\n")
        f.write("=" * 40 + "\n")
        
        sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        for feat, score in sorted_imp:
            f.write(f"{feat:10s} : {score:.6f}\n")
            
    logger.info("Saved %s feature importances to %s", task_name, filepath)

def main():
    parser = argparse.ArgumentParser(description="Train detection and diagnostic fault agents.")
    
    parser.add_argument("--detection", type=str, help="Path to the binary detection dataset CSV")
    parser.add_argument("--diagnostic", type=str, help="Path to the multiclass diagnostic dataset CSV")
    parser.add_argument("--save-dir", type=str, required=True, help="Directory to save the trained models")
    parser.add_argument("--n-estimators", type=int, default=300, help="Number of XGBoost trees")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="XGBoost learning rate")
    
    args = parser.parse_args()
    
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    agent_config = {
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate
    }
    
    if args.detection:
        logger.info("Running Detection Pipeline...")
        det_data = load_data(args.detection, task="detection")
        det_agent = DetectionAgent(config=agent_config)
        
        det_agent.train(
            det_data["X_train"], 
            det_data["y_train"], 
            feature_names=det_data["feature_names"]
        )
        det_agent.evaluate(det_data["X_test"], det_data["y_test"])
        det_agent.save(save_dir / "detection_agent.pkl")
        save_importances(det_agent, "detection", save_dir)
        
    if args.diagnostic:
        logger.info("Running Diagnostic Pipeline...")
        diag_data = load_data(args.diagnostic, task="multiclass")
        diag_agent = DiagnosticAgent(config=agent_config)
        
        diag_agent.train(
            diag_data["X_train"], 
            diag_data["y_train"], 
            feature_names=diag_data["feature_names"],
            label_encoder=diag_data.get("label_encoder"),
            class_names=diag_data.get("class_names")
        )
        diag_agent.evaluate(diag_data["X_test"], diag_data["y_test"])
        diag_agent.save(save_dir / "diagnostic_agent.pkl")
        save_importances(diag_agent, "diagnostic", save_dir)

if __name__ == "__main__":
    main()