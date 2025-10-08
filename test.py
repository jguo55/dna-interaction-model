import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import os
import pickle
from DNAMoleculeModel import DNAEncoder, DNAMoleculeDataset, DNAMoleculeInteractionModel, MoleculeEncoder
import csv
    
def get_data(filepath):
    train_dna_ids = []
    train_smiles_list = []
    train_labels = []

    for dirpath, _, filenames in os.walk(filepath+"test"):
        for i, f in enumerate(filenames):
            with open(dirpath+"/"+f, 'r') as c:
                reader = csv.DictReader(c)
                for row in reader:
                    train_dna_ids.append(row['GeneID'])
                    train_smiles_list.append(row['SMILES'])
                    train_labels.append(int(row['Label']))
            print(f"Processed {i+1} of {len(filenames)} files")

    return train_dna_ids, train_smiles_list, train_labels    
        
if __name__ == "__main__":
    print("DNA-Small Molecule Interaction Prediction Model")
    print("=" * 50)
    torch.manual_seed(67)

    basepath = ""

    print("Getting Data...")

    # Create character vocabulary.
    mol_encoder = MoleculeEncoder()
    if not os.path.exists(basepath + "smiles_vocab.pkl"):
        print("smiles vocab not found. creating vocab...")
        smiles_set = set()
        for i in range(1, 15):
            print(f"Processing file {i}/14")
            with open(basepath + f'data_general/train/train_{i}.csv', 'r') as c:
                reader = csv.DictReader(c)
                for row in reader:
                    smiles_set.add(row['SMILES'])
        print(f"found {len(smiles_set)} unique molecules")
        with open(basepath + "smiles_vocab.pkl", 'wb') as f:
            pickle.dump(smiles_set, f)
    else:
        with open(basepath + "smiles_vocab.pkl", 'rb') as f:
            smiles_set = pickle.load(f)
        print(f"loaded {len(smiles_set)} unique molecules")
        
    char_to_idx = mol_encoder.create_char_vocab(list(smiles_set))

    dna_seqs, smiles, labels = get_data(basepath + "data_general/")
    # Split data to val set 
        
    dna_encoder = DNAEncoder()
    test_dataset = DNAMoleculeDataset(dna_seqs, smiles, labels, char_to_idx, dna_encoder)

    # Create data loaders
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize model
    trained_model = DNAMoleculeInteractionModel(mol_vocab_size=len(char_to_idx))
    trained_model.load_state_dict(torch.load('best_dna_molecule_model.pt', weights_only=True))

    print(f"Model parameters: {sum(p.numel() for p in trained_model.parameters()):,}")
    print(f"Testing samples: {len(test_dataset)}")

    # Test model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trained_model.eval()
    test_preds = []
    test_labels = []

    trained_model.to(device)

    with torch.no_grad():
        for i, (dna_batch, mol_batch, batch_labels) in enumerate(test_loader):
            dna_batch, mol_batch = dna_batch.to(device), mol_batch.to(device)
            predictions = trained_model(dna_batch, mol_batch)
            test_preds.extend(predictions.cpu().numpy())
            test_labels.extend(batch_labels.numpy())

            if i % 1000 == 0:
                print(f"batch {i}/{len(test_loader)}")

    test_auc = roc_auc_score(test_labels, test_preds)
    test_acc = accuracy_score(test_labels, [1 if p > 0.5 else 0 for p in test_preds])

    print(f"\nFinal Test Results:")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")