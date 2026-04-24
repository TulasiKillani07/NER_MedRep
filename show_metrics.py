"""
Quick script to display model metrics.
Run after training and evaluation.
"""
import json
from pathlib import Path

def show_metrics():
    """Display saved metrics in a readable format."""
    
    metrics_file = Path("model/metrics.json")
    summary_file = Path("model/training_summary.json")
    
    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE METRICS")
    print("=" * 80)
    
    # Show training summary if available
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        print("\n📚 TRAINING SUMMARY:")
        print("-" * 80)
        print(f"Total Examples:        {summary.get('total_examples', 'N/A')}")
        print(f"Training Examples:     {summary.get('train_examples', 'N/A')}")
        print(f"Validation Examples:   {summary.get('val_examples', 'N/A')}")
        print(f"Epochs Trained:        {summary.get('epochs', 'N/A')}")
        print(f"Best Epoch:            {summary.get('best_epoch', 'N/A')}")
        print(f"Best F1 Score:         {summary.get('best_f1', 0):.2%}")
        print("-" * 80)
    
    # Show evaluation metrics if available
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        print("\n📊 OVERALL PERFORMANCE:")
        print("-" * 80)
        overall = metrics.get('overall', {})
        print(f"Precision:             {overall.get('precision', 0):.2%}")
        print(f"Recall:                {overall.get('recall', 0):.2%}")
        print(f"F1-Score:              {overall.get('f1_score', 0):.2%}")
        print("-" * 80)
        
        print("\n📋 PER-ENTITY PERFORMANCE:")
        print("-" * 80)
        print(f"{'Entity':<15} {'Precision':>12} {'Recall':>12} {'F1-Score':>12}")
        print("-" * 80)
        
        per_entity = metrics.get('per_entity', {})
        for entity_type in ["SYMPTOM", "INDICATION", "SEVERITY", "DURATION"]:
            if entity_type in per_entity:
                ent_metrics = per_entity[entity_type]
                print(f"{entity_type:<15} "
                      f"{ent_metrics.get('precision', 0):>11.2%} "
                      f"{ent_metrics.get('recall', 0):>11.2%} "
                      f"{ent_metrics.get('f1_score', 0):>11.2%}")
            else:
                print(f"{entity_type:<15} {'N/A':>12} {'N/A':>12} {'N/A':>12}")
        
        print("-" * 80)
    else:
        print("\n⚠️  No evaluation metrics found.")
        print("Run evaluation first: python -m training.evaluate")
    
    print("\n" + "=" * 80)
    
    # Show recommendations
    if metrics_file.exists():
        overall = metrics.get('overall', {})
        f1 = overall.get('f1_score', 0)
        
        print("\n💡 RECOMMENDATIONS:")
        print("-" * 80)
        
        if f1 >= 0.90:
            print("✓ Excellent performance! Model is production-ready.")
        elif f1 >= 0.80:
            print("✓ Good performance. Consider fine-tuning for better results.")
        elif f1 >= 0.70:
            print("⚠ Moderate performance. Add more training data or adjust hyperparameters.")
        else:
            print("✗ Low performance. Review training data quality and increase dataset size.")
        
        print("-" * 80)
    
    print()

if __name__ == "__main__":
    show_metrics()
