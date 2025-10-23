# DNA-Molecule Interaction Model with Foundation Models

## Overview

This implementation uses state-of-the-art foundation models with cross-attention to improve AUC performance:

- **DNA-BERT2** (117M parameters): Pretrained on DNA sequences for gene encoding
- **ChemBERTa** (77M parameters): Pretrained on chemical structures for molecule encoding
- **Cross-Attention**: Bidirectional attention mechanism between DNA and molecule embeddings

## Architecture

```
DNA Sequence → DNA-BERT2 → DNA Embeddings (512D)
                                            ↓
                                     Cross-Attention
                                            ↓
SMILES String → ChemBERTa → Mol Embeddings (512D)
                                            ↓
                                    Fusion Layers
                                            ↓
                                   Interaction Prediction
```

### Key Components

1. **DNABERT2Encoder** (`DNAMoleculeModelFoundation.py:8-46`)
   - Uses `zhihan1996/DNABERT-2-117M` pretrained model
   - Projects embeddings to 512 dimensions
   - Supports backbone freezing for faster initial training

2. **MoleculeFoundationEncoder** (`DNAMoleculeModelFoundation.py:49-82`)
   - Uses `DeepChem/ChemBERTa-77M-MLM` pretrained model
   - Projects embeddings to 512 dimensions
   - Handles SMILES tokenization

3. **CrossAttention** (`DNAMoleculeModelFoundation.py:85-148`)
   - 8-head multi-head attention (configurable)
   - Bidirectional: DNA→Molecule and Molecule→DNA
   - Layer normalization and feed-forward networks
   - Concatenates attended representations

4. **DNAMoleculeInteractionModelFoundation** (`DNAMoleculeModelFoundation.py:151-211`)
   - Main model combining all components
   - Fusion layers reduce dimensionality
   - Binary classification head with dropout

## Installation

```bash
pip install -r requirements.txt
```

New dependencies:
- `transformers>=4.30.0` - HuggingFace models
- `sentencepiece` - Tokenization support

## Usage

### Quick Test

Verify the model loads correctly:

```bash
python test_foundation_model.py
```

### Training

```bash
python train_foundation.py
```

**Key Training Parameters** (`train_foundation.py`):
- `batch_size=16` - Smaller than original due to larger models (line 148)
- `lr=1e-4` - Lower learning rate for foundation models (line 169)
- `weight_decay=1e-4` - Regularization (line 23)
- `freeze_backbones=False` - Fine-tune foundation models (line 162)

### Training Strategy

**Option 1: End-to-End Fine-tuning** (Current setup)
- Fine-tunes all foundation model weights
- Better performance but slower and more memory-intensive
- Set `freeze_backbones=False` in `train_foundation.py:162`

**Option 2: Frozen Backbone** (Faster initial training)
- Freezes DNA-BERT2 and ChemBERTa weights
- Only trains cross-attention and fusion layers
- Set `freeze_backbones=True` in `train_foundation.py:162`
- Can unfreeze later for fine-tuning

### Memory Considerations

Foundation models require more GPU memory:

- **Batch size**: Start with 16, reduce if OOM errors occur
- **Gradient checkpointing**: Can be added for very large models
- **Mixed precision**: Consider using `torch.cuda.amp` for faster training

Adjust batch size in `train_foundation.py:148`:
```python
batch_size = 16  # Reduce to 8 or 4 if memory issues
```

## Model Comparison

| Feature | Original Model | Foundation Model |
|---------|---------------|------------------|
| DNA Encoder | CNN (256D) | DNA-BERT2 (512D) |
| Molecule Encoder | LSTM + Attention (256D) | ChemBERTa (512D) |
| Interaction | Concatenation | Cross-Attention |
| Parameters | ~5M | ~200M |
| Training Speed | Fast | Slower |
| Expected AUC | ~0.64 | **>0.70** (target) |

## Expected Improvements

1. **Better Representations**: Pretrained on massive datasets
2. **Transfer Learning**: General chemical/biological knowledge
3. **Cross-Attention**: Captures interaction patterns between modalities
4. **Regularization**: Foundation models are well-regularized

## Files

- `DNAMoleculeModelFoundation.py` - Model architecture
- `train_foundation.py` - Training script
- `test_foundation_model.py` - Quick verification test
- `requirements.txt` - Updated dependencies

## Troubleshooting

### Out of Memory Errors
- Reduce batch size: `batch_size = 8` or `batch_size = 4`
- Freeze backbones: `freeze_backbones=True`
- Use gradient checkpointing (requires code modification)

### Model Download Issues
- Models are downloaded from HuggingFace on first run
- Requires internet connection
- Models cached in `~/.cache/huggingface/`

### Slow Training
- Normal for foundation models (10-100x slower than CNN/LSTM)
- Use GPU for significant speedup
- Consider freezing backbones initially

## Next Steps

1. Run `python test_foundation_model.py` to verify setup
2. Start training with `python train_foundation.py`
3. Monitor validation AUC - should improve beyond 0.64
4. Adjust hyperparameters based on results
5. Consider ensemble with original model

## Performance Tips

- Use mixed precision training for 2x speedup
- Gradient accumulation if batch size is too small
- Learning rate warmup for stable training
- Cosine annealing schedule for better convergence
