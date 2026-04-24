# train.py
# Trains the spaCy NER model on all annotated data from JSON files
# Uses en_core_web_md as the base model (CPU friendly, has word vectors)
# Saves the trained model to model/output/

import json
import random
import warnings
from pathlib import Path

import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
from loguru import logger

# Where to find training data
DATA_DIR = Path(__file__).parent.parent / "data" / "annotated"

# Where to save the trained model
MODEL_OUTPUT_PATH = Path(__file__).parent.parent / "model" / "output"

# Base model — en_core_web_md gives us word vectors which help generalization
BASE_MODEL = "en_core_web_md"

# Training hyperparameters
N_ITER = 30          # number of training iterations (epochs)
DROP_RATE = 0.3      # dropout rate — prevents overfitting
BATCH_SIZE_START = 4
BATCH_SIZE_END = 32


def load_training_data():
    """Load all JSON training files from data/annotated/"""
    training_data = []
    
    json_files = list(DATA_DIR.glob("*.json"))
    logger.info(f"Found {len(json_files)} training files")
    
    for json_file in json_files:
        logger.info(f"Loading {json_file.name}...")
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # Convert from our JSON format to spaCy format
        for item in data:
            text = item["text"]
            entities = []
            for ent in item["entities"]:
                entities.append((ent["start"], ent["end"], ent["label"]))
            
            training_data.append((text, {"entities": entities}))
    
    logger.info(f"Loaded {len(training_data)} total training examples")
    return training_data


def train():
    logger.info(f"Loading base model: {BASE_MODEL}")
    nlp = spacy.load(BASE_MODEL)

    # Load all training data from JSON files
    TRAIN_DATA = load_training_data()

    # Add NER to the pipeline if not already present
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
        logger.info("Added new NER component to pipeline")
    else:
        ner = nlp.get_pipe("ner")
        logger.info("Using existing NER component")

    # Add entity labels from training data
    for _, annotations in TRAIN_DATA:
        for _, _, label in annotations["entities"]:
            ner.add_label(label)

    logger.info(f"Entity labels: {ner.labels}")

    # Convert training data to spaCy Example objects
    examples = []
    for text, annotations in TRAIN_DATA:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        examples.append(example)

    # Disable other pipeline components during training
    # We only want to update the NER component
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]

    logger.info(f"Starting training for {N_ITER} iterations on {len(examples)} examples...")

    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.resume_training()

        for iteration in range(N_ITER):
            random.shuffle(examples)
            losses = {}

            # Create mini-batches with compounding batch size
            batches = minibatch(examples, size=compounding(BATCH_SIZE_START, BATCH_SIZE_END, 1.001))

            for batch in batches:
                nlp.update(batch, drop=DROP_RATE, losses=losses)

            logger.info(f"Iteration {iteration + 1}/{N_ITER} — Loss: {losses.get('ner', 0):.4f}")

    # Save the trained model
    MODEL_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(MODEL_OUTPUT_PATH)
    logger.success(f"Model saved to: {MODEL_OUTPUT_PATH}")
    logger.success(f"Training complete! Trained on {len(examples)} examples for {N_ITER} iterations")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    train()
