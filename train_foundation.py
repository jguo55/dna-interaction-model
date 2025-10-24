import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import time
from DNAMoleculeModelFoundation import (
    DNAMoleculeInteractionModelFoundation,
    DNAMoleculeDatasetFoundation,
    collate_fn_foundation
)
import h5py
import csv
import os

import sys


def train_model(model, train_loader, val_loader, num_epochs: int = 50, lr: float = 0.0001, patience=10, use_patience=True):
    """Train the foundation model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Foundation models are already on device via device_map
    # Move custom layers (projection, cross-attention, fusion, classifier) to device
    if hasattr(model, 'dna_encoder'):
        model.dna_encoder.projection = model.dna_encoder.projection.to(device)
        model.dna_encoder.dropout = model.dna_encoder.dropout.to(device)
    if hasattr(model, 'molecule_encoder'):
        model.molecule_encoder.projection = model.molecule_encoder.projection.to(device)
        model.molecule_encoder.dropout = model.molecule_encoder.dropout.to(device)
    if hasattr(model, 'cross_attention'):
        model.cross_attention = model.cross_attention.to(device)
    if hasattr(model, 'fusion_layer'):
        model.fusion_layer = model.fusion_layer.to(device)
    if hasattr(model, 'classifier'):
        model.classifier = model.classifier.to(device)

    # Use lower learning rate for foundation models
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5, min_lr=1e-7)

    best_val_auc = 0
    epochs_no_improvement = 0

    for epoch in range(num_epochs):
        epoch_time = time.time()

        # Training
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []

        batch_time = time.time()

        for batch_idx, (dna_seqs, smiles_list, labels) in enumerate(train_loader):
            labels = labels.to(device)

            optimizer.zero_grad()
            predictions = model(dna_seqs, smiles_list)
            loss = criterion(predictions, labels)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            train_preds.extend(predictions.cpu().detach().numpy())
            train_labels.extend(labels.cpu().detach().numpy())

            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}, batch_time = {time.time()-batch_time:.2f}s")
                batch_time = time.time()

        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for dna_seqs, smiles_list, labels in val_loader:
                labels = labels.to(device)

                predictions = model(dna_seqs, smiles_list)
                loss = criterion(predictions, labels)

                val_loss += loss.item()
                val_preds.extend(predictions.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        train_auc = roc_auc_score(train_labels, train_preds)
        val_auc = roc_auc_score(val_labels, val_preds)

        scheduler.step(val_loss)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            epochs_no_improvement = 0
            torch.save(model.state_dict(), 'best_dna_molecule_model_foundation.pt')
            torch.save(model.state_dict(), 'data_general/best_dna_molecule_model_foundation.pt')
            print(f"✓ New best model saved! Val AUC: {val_auc:.4f}")
        else:
            epochs_no_improvement += 1
            print(f"Epochs with no improvement: {epochs_no_improvement}")

        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, '
              f'Train AUC={train_auc:.4f}, Val AUC={val_auc:.4f}, '
              f'LR={current_lr:.2e}, Time = {time.time()-epoch_time:.2f}s')
        print("-" * 80)

        if epochs_no_improvement >= patience and use_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    return model


def get_data(filepath):
    """Load training data from CSV files"""
    train_dna_ids = []
    train_smiles_list = []
    train_labels = []

    for dirpath, _, filenames in os.walk(os.path.join(filepath, "train")):
        for i, f in enumerate(filenames):
            with open(os.path.join(dirpath, f), 'r') as c:
                reader = csv.DictReader(c)
                for row in reader:
                    train_dna_ids.append(row['GeneID'])
                    train_smiles_list.append(row['SMILES'])
                    train_labels.append(int(row['Label']))
            print(f"Processed {i+1} of {len(filenames)} files")

    return train_dna_ids, train_smiles_list, train_labels


def load_gene_sequences(filepath):
    """Load gene sequences from HDF5 file"""
    print("Loading gene sequences into RAM...")
    gene_seq = {}
    with h5py.File(filepath, 'r') as f:
        print(f"Loading {len(f.keys())} sequences...")
        for key in f.keys():
            gene_seq[key] = f[key][()]
    print(f"Successfully loaded {len(gene_seq.keys())} sequences!")
    return gene_seq


if __name__ == "__main__":
    sys.modules['triton'] = None

    print("=" * 80)
    print("DNA-Small Molecule Interaction Prediction Model (Foundation Model Version)")
    print("Using DNA-BERT2 + ChemBERTa with Cross-Attention")
    print("=" * 80)

    torch.manual_seed(67)
    np.random.seed(67)

    basepath = ""

    print("\n[1/6] Getting Data...")
    dna_seqs, smiles, labels = get_data(basepath + "data_general/")

    print(f"\n[2/6] Splitting data...")
    # Split data to val set
    X_dna_train, X_dna_test, X_mol_train, X_mol_test, y_train, y_test = train_test_split(
        dna_seqs, smiles, labels, test_size=0.2, random_state=67)

    print(f"Training samples: {len(X_dna_train)}")
    print(f"Validation samples: {len(X_dna_test)}")

    print(f"\n[3/6] Loading gene sequences...")
    # Note: This path (../tmp/genes.hdf5) is used in production (Kubernetes mounts ../tmp)
    # The dataset class has a fallback to data_general/genes.hdf5 if gene_seq is not provided
    gene_seq = load_gene_sequences("../tmp/genes.hdf5")

    print(f"\n[4/6] Creating datasets...")
    train_dataset = DNAMoleculeDatasetFoundation(X_dna_train, X_mol_train, y_train, gene_seq)
    val_dataset = DNAMoleculeDatasetFoundation(X_dna_test, X_mol_test, y_test, gene_seq)

    # Create data loaders with custom collate function
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             collate_fn=collate_fn_foundation, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           collate_fn=collate_fn_foundation, num_workers=4, pin_memory=True)


    print(f"\n[5/6] Initializing model...")
    print("Loading DNA-BERT2 and ChemBERTa models...")

    # Initialize model
    # Freeze foundation model backbones for faster training (train only task-specific layers)
    model = DNAMoleculeInteractionModelFoundation(
        dna_model_name="zhihan1996/DNABERT-2-117M",
        mol_model_name="DeepChem/ChemBERTa-77M-MLM",
        freeze_backbones=True,  # Freeze foundation models - only train projection/attention/classifier
        num_attention_heads=8
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")

    # Train model
    print("\n[6/6] Starting training...")
    print("-" * 80)
    trained_model = train_model(model, train_loader, val_loader,
                                num_epochs=50, lr=1e-4, patience=10, use_patience=True)

    # Test model
    print("\n" + "=" * 80)
    print("Evaluating on test set...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trained_model.eval()
    test_preds = []
    test_labels = []

    with torch.no_grad():
        for dna_seqs, smiles_list, labels in val_loader:
            predictions = trained_model(dna_seqs, smiles_list)
            test_preds.extend(predictions.cpu().numpy())
            test_labels.extend(labels.numpy())

    test_auc = roc_auc_score(test_labels, test_preds)
    # Model outputs logits (not probabilities), so threshold is 0.0, not 0.5
    test_acc = accuracy_score(test_labels, [1 if p > 0.0 else 0 for p in test_preds])

    print(f"\nFinal Test Results:")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print("=" * 80)
