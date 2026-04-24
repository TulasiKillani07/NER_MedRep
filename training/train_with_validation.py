"""
Enhanced training script with validation metrics during training.
Shows loss and validation scores after each epoch.
"""
import json
import random
import warnings
from pathlib import Path

import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
from spacy.scorer import Scorer
from loguru import logger

# Paths
DATA_DIR = Path(__file__).parent.parent / "data" / "annotated"
MODEL_OUTPUT_PATH = Path(__file__).parent.parent / "model" / "output"

# Base model
BASE_MODEL = "en_core_web_md"

# Training hyperparameters
N_ITER = 30
DROP_RATE = 0.3
BATCH_SIZE_START = 4
BATCH_SIZE_END = 32
VALIDATION_SPLIT = 0.15  # 15% for validation


def load_training_data():
    """Load all JSON training files."""
    training_data = []
    
    json_files = list(DATA_DIR.glob("*.json"))
    logger.info(f"Found {len(json_files)} training files")
    
    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        for item in data:
            text = item["text"]
            entities = []
            for ent in item["entities"]:
                entities.append((ent["start"], ent["end"], ent["label"]))
            
            training_data.append((text, {"entities": entities}))
    
    logger.info(f"Loaded {len(training_data)} total examples")
    return training_data


def split_data(data, validation_split=0.15):
    """Split data into train and validation sets."""
    random.seed(42)
    random.shuffle(data)
    
    split_idx = int(len(data) * (1 - validation_split))
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    logger.info(f"Train: {len(train_data)} | Validation: {len(val_data)}")
    return train_data, val_data


def evaluate_on_validation(nlp, val_data):
    """Evaluate model on validation set."""
    # Create fresh examples with current model predictions
    examples = []
    for text, annotations in val_data:
        # Get prediction from current model
        doc = nlp(text)
        # Create reference doc
        ref_doc = nlp.make_doc(text)
        example = Example.from_dict(ref_doc, annotations)
        # Update with predictions
        example.predicted = doc
        examples.append(example)
    
    scorer = Scorer()
    scores = scorer.score(examples)
    
    return {
        "precision": scores.get('ents_p', 0),
        "recall": scores.get('ents_r', 0),
        "f1_score": scores.get('ents_f', 0)
    }


def train():
    logger.info(f"Loading base model: {BASE_MODEL}")
    nlp = spacy.load(BASE_MODEL)

    # Load and split data
    all_data = load_training_data()
    train_data, val_data = split_data(all_data, VALIDATION_SPLIT)

    # Add NER to pipeline
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
        logger.info("Added new NER component")
    else:
        ner = nlp.get_pipe("ner")
        logger.info("Using existing NER component")

    # Add labels
    for _, annotations in train_data:
        for _, _, label in annotations["entities"]:
            ner.add_label(label)

    logger.info(f"Entity labels: {ner.labels}")

    # Convert to Examples
    train_examples = []
    for text, annotations in train_data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        train_examples.append(example)

    # Training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    
    logger.info(f"Starting training for {N_ITER} iterations...")
    print("\n" + "=" * 80)
    print("TRAINING PROGRESS")
    print("=" * 80)
    print(f"{'Epoch':<8} {'Loss':>12} {'Val Precision':>15} {'Val Recall':>12} {'Val F1':>10}")
    print("-" * 80)

    best_f1 = 0
    best_epoch = 0

    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.resume_training()

        for iteration in range(N_ITER):
            random.shuffle(train_examples)
            losses = {}

            # Training
            batches = minibatch(train_examples, size=compounding(BATCH_SIZE_START, BATCH_SIZE_END, 1.001))
            for batch in batches:
                nlp.update(batch, drop=DROP_RATE, losses=losses)

            # Validation every epoch - pass val_data not val_examples
            val_metrics = evaluate_on_validation(nlp, val_data)
            
            # Print metrics
            print(f"{iteration + 1:<8} {losses.get('ner', 0):>12.4f} "
                  f"{val_metrics['precision']:>14.2%} "
                  f"{val_metrics['recall']:>12.2%} "
                  f"{val_metrics['f1_score']:>10.2%}")
            
            # Track best model
            if val_metrics['f1_score'] > best_f1:
                best_f1 = val_metrics['f1_score']
                best_epoch = iteration + 1

    print("-" * 80)
    print(f"Best F1: {best_f1:.2%} at epoch {best_epoch}")
    print("=" * 80)

    # Save model
    MODEL_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(MODEL_OUTPUT_PATH)
    logger.success(f"Model saved to: {MODEL_OUTPUT_PATH}")
    
    # Save training summary
    summary = {
        "total_examples": len(all_data),
        "train_examples": len(train_data),
        "val_examples": len(val_data),
        "epochs": N_ITER,
        "best_epoch": best_epoch,
        "best_f1": best_f1,
        "final_val_metrics": val_metrics
    }
    
    summary_file = MODEL_OUTPUT_PATH.parent / "training_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.success(f"Training summary saved to: {summary_file}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    train()
