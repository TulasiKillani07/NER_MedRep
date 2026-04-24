# Model Metrics Guide

This guide explains how to train, evaluate, and view metrics for your NER model.

## 📊 Available Scripts

### 1. **Basic Training** (Original)
```bash
python -m training.train
```
- Trains the model on all data
- Shows loss per epoch
- No validation metrics during training
- Fastest option

### 2. **Training with Validation** (Recommended)
```bash
python -m training.train_with_validation
```
- Trains with 85% train / 15% validation split
- Shows validation metrics (Precision, Recall, F1) after each epoch
- Tracks best performing epoch
- Saves training summary to `model/training_summary.json`

**Output Example:**
```
================================================================================
TRAINING PROGRESS
================================================================================
Epoch         Loss    Val Precision   Val Recall     Val F1
--------------------------------------------------------------------------------
1           45.2341          87.23%       84.56%     85.87%
2           32.1234          89.45%       87.12%     88.27%
...
30           5.4321          92.34%       91.23%     91.78%
--------------------------------------------------------------------------------
Best F1: 91.78% at epoch 30
================================================================================
```

### 3. **Model Evaluation**
```bash
python -m training.evaluate
```
- Evaluates trained model on 20% test set
- Shows comprehensive metrics:
  - Overall Precision, Recall, F1
  - Per-entity type metrics (SYMPTOM, INDICATION, SEVERITY, DURATION)
  - Entity distribution statistics
  - Sample predictions with comparison
- Saves metrics to `model/metrics.json`

**Output Example:**
```
================================================================================
MODEL EVALUATION METRICS
================================================================================

📊 OVERALL NER PERFORMANCE:
--------------------------------------------------------------------------------
Metric               Value
--------------------------------------------------------------------------------
Precision            91.23%
Recall               89.45%
F1-Score             90.33%
--------------------------------------------------------------------------------

📋 PER-ENTITY TYPE PERFORMANCE:
--------------------------------------------------------------------------------
Entity Type          Precision       Recall    F1-Score
--------------------------------------------------------------------------------
SYMPTOM                  92.45%       91.23%      91.83%
INDICATION               89.12%       87.34%      88.22%
SEVERITY                 93.67%       92.11%      92.88%
DURATION                 88.90%       86.45%      87.66%
--------------------------------------------------------------------------------
```

### 4. **Quick Metrics View**
```bash
python show_metrics.py
```
- Displays saved metrics in readable format
- Shows training summary + evaluation results
- Provides performance recommendations
- No computation, just displays saved data

## 🎯 Recommended Workflow

### First Time Training:
```bash
# 1. Train with validation metrics
python -m training.train_with_validation

# 2. Evaluate on test set
python -m training.evaluate

# 3. View summary
python show_metrics.py
```

### Quick Check:
```bash
# Just view existing metrics
python show_metrics.py
```

### Re-training:
```bash
# Train again (e.g., after adding more data)
python -m training.train_with_validation

# Re-evaluate
python -m training.evaluate

# Check improvements
python show_metrics.py
```

## 📈 Understanding Metrics

### **Precision**
- What percentage of predicted entities are correct?
- High precision = Few false positives
- Example: If model predicts 100 entities and 92 are correct → 92% precision

### **Recall**
- What percentage of actual entities did we find?
- High recall = Few false negatives
- Example: If there are 100 entities and model finds 89 → 89% recall

### **F1-Score**
- Harmonic mean of precision and recall
- Balanced measure of overall performance
- **Target: > 85% for production use**

### **Per-Entity Metrics**
- Shows performance for each entity type
- Helps identify which entities need more training data
- Example: Low DURATION recall → Add more duration examples

## 🎓 Performance Benchmarks

| F1-Score | Quality | Action |
|----------|---------|--------|
| **> 90%** | Excellent | Production ready ✓ |
| **80-90%** | Good | Fine-tune or deploy |
| **70-80%** | Moderate | Add more data |
| **< 70%** | Poor | Review data quality |

## 📁 Output Files

After training and evaluation, you'll have:

```
model/
├── output/              # Trained model files
├── metrics.json         # Evaluation metrics
└── training_summary.json # Training statistics
```

### `metrics.json` Structure:
```json
{
  "overall": {
    "precision": 0.9123,
    "recall": 0.8945,
    "f1_score": 0.9033
  },
  "per_entity": {
    "SYMPTOM": {
      "precision": 0.9245,
      "recall": 0.9123,
      "f1_score": 0.9183
    },
    ...
  }
}
```

### `training_summary.json` Structure:
```json
{
  "total_examples": 3093,
  "train_examples": 2629,
  "val_examples": 464,
  "epochs": 30,
  "best_epoch": 28,
  "best_f1": 0.9178,
  "final_val_metrics": {
    "precision": 0.9234,
    "recall": 0.9123,
    "f1_score": 0.9178
  }
}
```

## 🔧 Troubleshooting

### Low Precision
- Model predicting too many false entities
- **Fix**: Review pattern files, add negative examples

### Low Recall
- Model missing many entities
- **Fix**: Add more diverse training examples

### Specific Entity Type Low
- One entity type performing poorly
- **Fix**: Add 100+ more examples for that entity type

### Overfitting (Train good, Test poor)
- Model memorizing training data
- **Fix**: Increase dropout rate, add more diverse data

## 💡 Tips for Better Metrics

1. **Balanced Data**: Ensure all entity types have similar counts
2. **Diverse Examples**: Include various phrasings and contexts
3. **Quality over Quantity**: 1000 good examples > 5000 poor ones
4. **Regular Evaluation**: Check metrics after every major data addition
5. **Iterative Improvement**: Focus on weakest entity type each iteration

## 🚀 Next Steps

After achieving good metrics:
1. Test the API: `python -m test_api`
2. Deploy the model
3. Monitor real-world performance
4. Collect edge cases for retraining
