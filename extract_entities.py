"""
Extract all symptoms and indications from training data and patterns
"""
import json
from pathlib import Path

symptoms = set()
indications = set()

# Get from training data
print("Extracting from training data...")
for file in Path('data/annotated').glob('*.json'):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            text = item['text'].lower()
            for entity in item.get('entities', []):
                if isinstance(entity, dict):
                    start = entity['start']
                    end = entity['end']
                    label = entity['label']
                    entity_text = text[start:end].strip()
                    if label == 'SYMPTOM':
                        symptoms.add(entity_text)
                    elif label == 'INDICATION':
                        indications.add(entity_text)
                elif isinstance(entity, list) and len(entity) == 3:
                    start, end, label = entity
                    entity_text = text[start:end].strip()
                    if label == 'SYMPTOM':
                        symptoms.add(entity_text)
                    elif label == 'INDICATION':
                        indications.add(entity_text)

# Get from patterns
print("Extracting from patterns...")
for file in Path('data/patterns').glob('*.json'):
    with open(file, 'r', encoding='utf-8') as f:
        patterns = json.load(f)
        for pattern in patterns:
            if 'label' in pattern:
                if pattern['label'] == 'SYMPTOM':
                    if 'pattern' in pattern:
                        if isinstance(pattern['pattern'], str):
                            symptoms.add(pattern['pattern'].lower())
                        elif isinstance(pattern['pattern'], list):
                            text_parts = []
                            for p in pattern['pattern']:
                                if isinstance(p, dict) and 'LOWER' in p:
                                    text_parts.append(p['LOWER'])
                            if text_parts:
                                symptoms.add(' '.join(text_parts))
                elif pattern['label'] == 'INDICATION':
                    if 'pattern' in pattern:
                        if isinstance(pattern['pattern'], str):
                            indications.add(pattern['pattern'].lower())
                        elif isinstance(pattern['pattern'], list):
                            text_parts = []
                            for p in pattern['pattern']:
                                if isinstance(p, dict) and 'LOWER' in p:
                                    text_parts.append(p['LOWER'])
                            if text_parts:
                                indications.add(' '.join(text_parts))

print(f"\n=== SYMPTOMS ({len(symptoms)}) ===")
for s in sorted(symptoms):
    print(f"  - {s}")

print(f"\n=== INDICATIONS ({len(indications)}) ===")
for i in sorted(indications):
    print(f"  - {i}")

# Save to files
with open('symptoms_list.txt', 'w', encoding='utf-8') as f:
    for s in sorted(symptoms):
        f.write(f"{s}\n")

with open('indications_list.txt', 'w', encoding='utf-8') as f:
    for i in sorted(indications):
        f.write(f"{i}\n")

print(f"\n✓ Saved to symptoms_list.txt and indications_list.txt")
