# Multi-Symptom Detector

Domain-specific NER model to detect symptoms, duration, severity from medical queries.
Built with spaCy, designed for CPU-only environments.

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_md
```

## Train the model

```bash
cd multisymptomp_detector
python -m training.train
```

## Run the API

```bash
cd multisymptomp_detector
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

## Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I have fever, headache and knee pain from 3 days"}'
```

## Project Structure

```
multisymptomp_detector/
├── data/
│   ├── raw/                    # raw user queries (collected over time)
│   ├── annotated/              # annotated .spacy training files
│   └── patterns/               # EntityRuler patterns + synonym/whitelist files
├── preprocessing/
│   ├── normalizer.py           # lowercase, remove special chars, expand abbreviations
│   ├── spell_checker.py        # symspellpy-based spell correction with medical whitelist
│   ├── synonym_mapper.py       # maps informal terms to canonical medical terms
│   └── negation_detector.py    # detects negated entities (no fever, denies pain)
├── training/
│   ├── train_data.py           # annotated training examples
│   └── train.py                # training script
├── model/
│   ├── ner_pipeline.py         # full inference pipeline
│   └── output/                 # saved trained model (generated after training)
├── api/
│   └── app.py                  # FastAPI REST API
└── logs/                       # application logs
```
