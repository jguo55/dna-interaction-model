"""
Quick test script to verify the foundation model can be loaded and run
"""
import torch
from DNAMoleculeModelFoundation import DNAMoleculeInteractionModelFoundation

print("Testing Foundation Model Setup")
print("=" * 60)

# Test 1: Model instantiation
print("\n[Test 1] Instantiating model...")
try:
    model = DNAMoleculeInteractionModelFoundation(
        dna_model_name="zhihan1996/DNABERT-2-117M",
        mol_model_name="DeepChem/ChemBERTa-77M-MLM",
        freeze_backbones=False,
        num_attention_heads=8
    )
    print("[PASS] Model instantiated successfully")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {trainable:,}")
except Exception as e:
    print(f"[FAIL] Failed to instantiate model: {e}")
    exit(1)

# Test 2: Forward pass with dummy data
print("\n[Test 2] Testing forward pass with dummy data...")
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")
    model = model.to(device)
    model.eval()

    # Create dummy batch
    dummy_dna = ["ATCGATCGATCGATCG", "GCTAGCTAGCTAGCTA"]
    dummy_smiles = ["CCO", "C1=CC=CC=C1"]

    with torch.no_grad():
        predictions = model(dummy_dna, dummy_smiles)

    print(f"[PASS] Forward pass successful")
    print(f"  Input batch size: 2")
    print(f"  Output shape: {predictions.shape}")
    print(f"  Sample predictions: {predictions}")

except Exception as e:
    print(f"[FAIL] Forward pass failed: {e}")
    exit(1)

# Test 3: Check gradient computation
print("\n[Test 3] Testing gradient computation...")
try:
    model.train()
    predictions = model(dummy_dna, dummy_smiles)
    loss = predictions.mean()
    loss.backward()
    print("[PASS] Gradient computation successful")

except Exception as e:
    print(f"[FAIL] Gradient computation failed: {e}")
    exit(1)

print("\n" + "=" * 60)
print("All tests passed! The model is ready for training.")
print("\nNext steps:")
print("1. Install dependencies: pip install -r requirements.txt")
print("2. Run training: python train_foundation.py")
print("=" * 60)
