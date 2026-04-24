"""
Evaluate the trained NER model and display comprehensive metrics.
Shows precision, recall, F1-score per entity type and overall.
"""
import json
import random
from pathlib import Path
from collections import defaultdict

import spacy
from spacy.training import Example
from spacy.scorer import Scorer
from loguru import logger

# Paths
DATA_DIR = Path(__file__).parent.parent / "data" / "annotated"
MODEL_PATH = Path(__file__).parent.parent / "model" / "output"


def load_test_data(test_split=0.2):
    """
    Load and split data into train/test sets.
    Returns test set for evaluation.
    """
    all_data = []
    
    json_files = list(DATA_DIR.glob("*.json"))
    logger.info(f"Loading data from {len(json_files)} files...")
    
    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        for item in data:
            text = item["text"]
            entities = []
            for ent in item["entities"]:
                entities.append((ent["start"], ent["end"], ent["label"]))
            
            all_data.append((text, {"entities": entities}))
    
    # Shuffle and split
    random.seed(42)  # For reproducibility
    random.shuffle(all_data)
    
    split_idx = int(len(all_data) * (1 - test_split))
    train_data = all_data[:split_idx]
    test_data = all_data[split_idx:]
    
    logger.info(f"Total examples: {len(all_data)}")
    logger.info(f"Train: {len(train_data)} | Test: {len(test_data)}")
    
    return test_data


def calculate_metrics(true_positives, false_positives, false_negatives):
    """Calculate precision, recall, and F1 score."""
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1


def evaluate_model():
    """Evaluate the trained model on test data."""
    
    # Load model
    if not MODEL_PATH.exists():
        logger.error(f"Model not found at {MODEL_PATH}")
        logger.error("Please train the model first using: python -m training.train")
        return
    
    logger.info(f"Loading model from {MODEL_PATH}...")
    nlp = spacy.load(MODEL_PATH)
    
    # Load test data
    test_data = load_test_data(test_split=0.2)
    
    # Convert to Examples
    examples = []
    for text, annotations in test_data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        examples.append(example)
    
    logger.info(f"Evaluating on {len(examples)} test examples...")
    
    # Use spaCy's built-in scorer
    scorer = Scorer()
    scores = scorer.score(examples)
    
    # Display results
    print("\n" + "=" * 80)
    print("MODEL EVALUATION METRICS")
    print("=" * 80)
    
    # Overall metrics
    ner_scores = scores.get("ents_per_type", {})
    
    print("\n📊 OVERALL NER PERFORMANCE:")
    print("-" * 80)
    print(f"{'Metric':<20} {'Value':>10}")
    print("-" * 80)
    print(f"{'Precision':<20} {scores.get('ents_p', 0):>9.2%}")
    print(f"{'Recall':<20} {scores.get('ents_r', 0):>9.2%}")
    print(f"{'F1-Score':<20} {scores.get('ents_f', 0):>9.2%}")
    print("-" * 80)
    
    # Per-entity metrics
    print("\n📋 PER-ENTITY TYPE PERFORMANCE:")
    print("-" * 80)
    print(f"{'Entity Type':<20} {'Precision':>12} {'Recall':>12} {'F1-Score':>12}")
    print("-" * 80)
    
    entity_types = ["SYMPTOM", "INDICATION", "SEVERITY", "DURATION"]
    
    for entity_type in entity_types:
        if entity_type in ner_scores:
            metrics = ner_scores[entity_type]
            precision = metrics.get('p', 0)
            recall = metrics.get('r', 0)
            f1 = metrics.get('f', 0)
            
            print(f"{entity_type:<20} {precision:>11.2%} {recall:>11.2%} {f1:>11.2%}")
        else:
            print(f"{entity_type:<20} {'N/A':>12} {'N/A':>12} {'N/A':>12}")
    
    print("-" * 80)
    
    # Additional statistics
    print("\n📈 ADDITIONAL STATISTICS:")
    print("-" * 80)
    
    # Count predictions
    total_predicted = 0
    total_gold = 0
    correct_predictions = 0
    
    entity_counts = defaultdict(lambda: {"predicted": 0, "gold": 0, "correct": 0})
    
    for example in examples:
        predicted_ents = example.predicted.ents
        gold_ents = example.reference.ents
        
        total_predicted += len(predicted_ents)
        total_gold += len(gold_ents)
        
        # Count by entity type
        for ent in predicted_ents:
            entity_counts[ent.label_]["predicted"] += 1
        
        for ent in gold_ents:
            entity_counts[ent.label_]["gold"] += 1
        
        # Count correct predictions (exact match)
        gold_set = {(ent.start_char, ent.end_char, ent.label_) for ent in gold_ents}
        pred_set = {(ent.start_char, ent.end_char, ent.label_) for ent in predicted_ents}
        
        correct = len(gold_set & pred_set)
        correct_predictions += correct
        
        for ent in gold_set & pred_set:
            entity_counts[ent[2]]["correct"] += 1
    
    print(f"{'Total Gold Entities':<30} {total_gold:>10}")
    print(f"{'Total Predicted Entities':<30} {total_predicted:>10}")
    print(f"{'Correct Predictions':<30} {correct_predictions:>10}")
    print(f"{'Accuracy (Exact Match)':<30} {correct_predictions/total_gold if total_gold > 0 else 0:>9.2%}")
    print("-" * 80)
    
    # Entity type distribution
    print("\n📊 ENTITY TYPE DISTRIBUTION:")
    print("-" * 80)
    print(f"{'Entity Type':<20} {'Gold':>10} {'Predicted':>12} {'Correct':>10}")
    print("-" * 80)
    
    for entity_type in entity_types:
        gold = entity_counts[entity_type]["gold"]
        predicted = entity_counts[entity_type]["predicted"]
        correct = entity_counts[entity_type]["correct"]
        
        print(f"{entity_type:<20} {gold:>10} {predicted:>12} {correct:>10}")
    
    print("-" * 80)
    
    # Sample predictions
    print("\n🔍 SAMPLE PREDICTIONS (First 5 test examples):")
    print("=" * 80)
    
    for i, example in enumerate(examples[:5], 1):
        text = example.reference.text
        gold_ents = [(ent.text, ent.label_) for ent in example.reference.ents]
        pred_ents = [(ent.text, ent.label_) for ent in example.predicted.ents]
        
        print(f"\nExample {i}:")
        print(f"  Text: {text}")
        print(f"  Gold: {gold_ents}")
        print(f"  Pred: {pred_ents}")
        
        # Check if correct
        gold_set = {(ent.start_char, ent.end_char, ent.label_) for ent in example.reference.ents}
        pred_set = {(ent.start_char, ent.end_char, ent.label_) for ent in example.predicted.ents}
        
        if gold_set == pred_set:
            print(f"  ✓ Perfect match!")
        else:
            missing = gold_set - pred_set
            extra = pred_set - gold_set
            if missing:
                print(f"  ✗ Missing: {missing}")
            if extra:
                print(f"  ✗ Extra: {extra}")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    
    # Save metrics to file
    metrics_file = Path(__file__).parent.parent / "model" / "metrics.json"
    metrics_data = {
        "overall": {
            "precision": scores.get('ents_p', 0),
            "recall": scores.get('ents_r', 0),
            "f1_score": scores.get('ents_f', 0)
        },
        "per_entity": {}
    }
    
    for entity_type in entity_types:
        if entity_type in ner_scores:
            metrics_data["per_entity"][entity_type] = {
                "precision": ner_scores[entity_type].get('p', 0),
                "recall": ner_scores[entity_type].get('r', 0),
                "f1_score": ner_scores[entity_type].get('f', 0)
            }
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    logger.success(f"Metrics saved to: {metrics_file}")


if __name__ == "__main__":
    evaluate_model()
