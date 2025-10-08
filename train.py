import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import os
import pickle
import time
from DNAMoleculeModel import DNAEncoder, DNAMoleculeDataset, DNAMoleculeInteractionModel, MoleculeEncoder
import h5py
import csv
    
def train_model(model, train_loader, val_loader, num_epochs: int = 50, lr: float = 0.001, patience=10, use_patience=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_auc = 0

    epochs_no_improvement = 0

    start_time = time.time()

    for epoch in range(num_epochs):

        epoch_time = time.time()

        # Training
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []

        batch_time = time.time()
        
        for batch_idx, (dna_batch, mol_batch, labels) in enumerate(train_loader):
            dna_batch, mol_batch, labels = dna_batch.to(device), mol_batch.to(device), labels.to(device)

            optimizer.zero_grad()
            predictions = model(dna_batch, mol_batch)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend(predictions.cpu().detach().numpy())
            train_labels.extend(labels.cpu().detach().numpy())

            if batch_idx % 1000 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], batch_time = {time.time()-batch_time}")
                batch_time = time.time() #reset time

        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for dna_batch, mol_batch, labels in val_loader:
                dna_batch, mol_batch, labels = dna_batch.to(device), mol_batch.to(device), labels.to(device)

                predictions = model(dna_batch, mol_batch)
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
            torch.save(model.state_dict(), 'best_dna_molecule_model.pt')
        else:
            epochs_no_improvement += 1

        print(f'Epoch {epoch}: Train Loss={train_loss/len(train_loader):.4f}, '
            f'Train AUC={train_auc:.4f}, Val AUC={val_auc:.4f}, Time = {time.time()-epoch_time}')
        
        if epochs_no_improvement >= patience and use_patience:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"Total training time: {time.time()-start_time}")
    return model

def get_data(filepath):
    train_dna_ids = []
    train_smiles_list = []
    train_labels = []

    for dirpath, _, filenames in os.walk(filepath+"train"):
        for i, f in enumerate(filenames):
            with open(dirpath+"/"+f, 'r') as c:
                reader = csv.DictReader(c)
                for row in reader:
                    train_dna_ids.append(row['GeneID'])
                    train_smiles_list.append(row['SMILES'])
                    train_labels.append(int(row['Label']))
            print(f"Processed {i+1} of {len(filenames)} files")

    return train_dna_ids, train_smiles_list, train_labels    

def load_gene_sequences(filepath):
    print("Loading gene sequences into ram...")
    gene_seq = {}
    '''
    with open(filepath, 'r') as f:
        print("opened file...")
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            gene_seq[row["GeneID"]] = row["GeneSequence"]
    print("loaded sequences")
    return gene_seq
    '''
    with h5py.File(filepath, 'r') as f:
        print(f"Loading {len(f.keys())} sequences...")
        for key in f.keys():
            gene_seq[key] = f[key][()]
    print(f"Succesfully loaded {len(gene_seq.keys())} sequences!")
    return gene_seq
        
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
    X_dna_train, X_dna_test, X_mol_train, X_mol_test, y_train, y_test = train_test_split(
        dna_seqs, smiles, labels, test_size=0.2, random_state=67)
    
    gene_seq = load_gene_sequences("../tmp/genes.hdf5")
    
    dna_encoder = DNAEncoder()
    train_dataset = DNAMoleculeDataset(X_dna_train, X_mol_train, y_train, char_to_idx, dna_encoder, gene_seq)
    val_dataset = DNAMoleculeDataset(X_dna_test, X_mol_test, y_test, char_to_idx, dna_encoder, gene_seq)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

    # Initialize model
    model = DNAMoleculeInteractionModel(mol_vocab_size=len(char_to_idx))

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Train model
    print("\nStarting training...")
    trained_model = train_model(model, train_loader, val_loader, num_epochs=1000, use_patience=True)

    # Test model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trained_model.eval()
    test_preds = []
    test_labels = []

    with torch.no_grad():
        for dna_batch, mol_batch, labels in val_loader:
            dna_batch, mol_batch = dna_batch.to(device), mol_batch.to(device)
            predictions = trained_model(dna_batch, mol_batch)
            test_preds.extend(predictions.cpu().numpy())
            test_labels.extend(labels.numpy())

    test_auc = roc_auc_score(test_labels, test_preds)
    test_acc = accuracy_score(test_labels, [1 if p > 0.5 else 0 for p in test_preds])

    print(f"\nFinal Test Results:")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")