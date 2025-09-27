import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from typing import List
import csv
import h5py
import os
import pickle
import time
import random

class DNAEncoder(nn.Module):
    def __init__(self, max_seq_length: int = 1000, embed_dim: int = 128):
        super(DNAEncoder, self).__init__()
        self.max_seq_length = max_seq_length
        self.embed_dim = embed_dim

        # DNA nucleotide vocabulary: A, T, G, C, N (unknown)
        self.vocab_size = 5
        self.embedding = nn.Embedding(self.vocab_size, embed_dim)

        # Convolutional layers for motif detection
        self.conv1 = nn.Conv1d(embed_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=7, padding=3)

        # Pooling and normalization
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.batch_norm3 = nn.BatchNorm1d(256)

        # Global pooling
        self.global_pool = nn.AdaptiveMaxPool1d(1)

    def encode_sequence(self, sequence: str) -> torch.Tensor:
        """Convert DNA sequence string to tensor of indices"""
        nucleotide_to_idx = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4}
        sequence = sequence.upper()[:self.max_seq_length]

        # Convert to indices
        indices = [nucleotide_to_idx.get(nuc, 4) for nuc in sequence]

        # Pad sequence to max length
        while len(indices) < self.max_seq_length:
            indices.append(4)  # Pad with 'N'

        return torch.tensor(indices, dtype=torch.long)

    def forward(self, x):
        # x shape: (batch_size, seq_length)
        x = self.embedding(x)  # (batch_size, seq_length, embed_dim)
        x = x.transpose(1, 2)  # (batch_size, embed_dim, seq_length)

        # Convolutional layers
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout(x)

        x = F.relu(self.batch_norm2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)

        x = F.relu(self.batch_norm3(self.conv3(x)))
        x = self.global_pool(x)  # (batch_size, 256, 1)

        return x.squeeze(-1)  # (batch_size, 256)

class MoleculeEncoder(nn.Module):
    def __init__(self, vocab_size: int = 100, embed_dim: int = 128, max_length: int = 200):
        super(MoleculeEncoder, self).__init__()
        self.max_length = max_length
        self.embed_dim = embed_dim

        # SMILES character embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # LSTM for sequential processing of SMILES
        self.lstm = nn.LSTM(embed_dim, 128, batch_first=True, bidirectional=True, num_layers=2)

        # Attention mechanism
        self.attention = nn.Linear(256, 1)
        self.dropout = nn.Dropout(0.3)

        # Final projection
        self.projection = nn.Linear(256, 256)

    def create_char_vocab(self, smiles_list: List[str]) -> dict:
        """Create character vocabulary from SMILES strings"""
        chars = set()
        for smiles in smiles_list:
            chars.update(smiles)

        char_to_idx = {char: idx for idx, char in enumerate(sorted(chars))}
        char_to_idx['<PAD>'] = len(char_to_idx)
        char_to_idx['<UNK>'] = len(char_to_idx)

        return char_to_idx

    def encode_smiles(self, smiles: str, char_to_idx: dict) -> torch.Tensor:
        """Convert SMILES string to tensor of indices"""
        indices = [char_to_idx.get(char, char_to_idx['<UNK>']) for char in smiles[:self.max_length]]

        # Pad sequence
        while len(indices) < self.max_length:
            indices.append(char_to_idx['<PAD>'])

        return torch.tensor(indices, dtype=torch.long)

    def forward(self, x):
        # x shape: (batch_size, seq_length)
        embedded = self.embedding(x)  # (batch_size, seq_length, embed_dim)
        embedded = self.dropout(embedded)

        # LSTM processing
        lstm_out, _ = self.lstm(embedded)  # (batch_size, seq_length, 256)

        # Attention mechanism
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)  # (batch_size, seq_length, 1)
        attended = torch.sum(attention_weights * lstm_out, dim=1)  # (batch_size, 256)

        # Final projection
        output = self.projection(attended)
        return output

class DNAMoleculeInteractionModel(nn.Module):
    def __init__(self, dna_max_length: int = 1000, mol_max_length: int = 200,
                 mol_vocab_size: int = 100):
        super(DNAMoleculeInteractionModel, self).__init__()

        # Encoders
        self.dna_encoder = DNAEncoder(dna_max_length)
        self.molecule_encoder = MoleculeEncoder(mol_vocab_size, max_length=mol_max_length)

        # Interaction layers
        self.fusion_layer = nn.Sequential(
            nn.Linear(256 + 256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Prediction head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, dna_seq, mol_seq):
        # Encode inputs
        dna_features = self.dna_encoder(dna_seq)  # (batch_size, 256)
        mol_features = self.molecule_encoder(mol_seq)  # (batch_size, 256)

        # Concatenate features
        combined = torch.cat([dna_features, mol_features], dim=1)  # (batch_size, 512)

        # Fusion and prediction
        fused = self.fusion_layer(combined)  # (batch_size, 256)
        prediction = self.classifier(fused)  # (batch_size, 1)

        return prediction.squeeze()

class DNAMoleculeDataset(Dataset):
    def __init__(self, dna_ids: List[str], smiles: List[str],
                 labels: List[int], char_to_idx: dict, dna_encoder: DNAEncoder, balanced=True):
        self.dna_ids = dna_ids
        self.smiles = smiles
        self.labels = labels
        self.char_to_idx = char_to_idx
        self.dna_encoder = dna_encoder
        
    def __len__(self):
        return len(self.dna_ids)

    def __getitem__(self, idx):
        with h5py.File('claude-test/data/genes.hdf5', 'r') as f:
            dna_sequence = f[self.dna_ids[idx]][()]
        dna_encoded = self.dna_encoder.encode_sequence(dna_sequence)
        mol_encoded = torch.tensor([self.char_to_idx.get(char, self.char_to_idx['<UNK>'])
                                   for char in self.smiles[idx][:200]], dtype=torch.long)

        # Pad molecule sequence
        if len(mol_encoded) < 200:
            padding = torch.full((200 - len(mol_encoded),), self.char_to_idx['<PAD>'], dtype=torch.long)
            mol_encoded = torch.cat([mol_encoded, padding])

        return dna_encoded, mol_encoded, torch.tensor(self.labels[idx], dtype=torch.float)
    
def train_model(model, train_loader, val_loader, num_epochs: int = 50, lr: float = 0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_auc = 0

    for epoch in range(num_epochs):
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
            torch.save(model.state_dict(), 'best_dna_molecule_model.pt')

        print(f'Epoch {epoch}: Train Loss={train_loss/len(train_loader):.4f}, '
            f'Train AUC={train_auc:.4f}, Val AUC={val_auc:.4f}')

    return model

def get_data(filepath="claude-test/data/"):
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
            print(f"Processed {i+1} of {len(filenames)+1} files")

    return train_dna_ids, train_smiles_list, train_labels
        
if __name__ == "__main__":
    print("DNA-Small Molecule Interaction Prediction Model")
    print("=" * 50)
    torch.manual_seed(67)

    print("Getting Data...")

    # Create character vocabulary. TODO FIX THIS TO WORK WITH THE GENERATED TRAINING SET
    mol_encoder = MoleculeEncoder()
    if not os.path.exists("claude-test/smiles_vocab.pkl"):
        print("smiles vocab not found. creating vocab...")
        smiles_set = set()
        for i in range(1, 11):
            print(f"Processing file {i}/10")
            with open(f'claude-test/data/ixns_file_{i}_of_10_with_SMILES.csv', 'r') as c:
                reader = csv.DictReader(c)
                for row in reader:
                    smiles_set.add(row['SMILES'])
        print(f"found {len(smiles_set)} unique molecules")
        with open("claude-test/smiles_vocab.pkl", 'wb') as f:
            pickle.dump(smiles_set, f)
    else:
        with open("claude-test/smiles_vocab.pkl", 'rb') as f:
            smiles_set = pickle.load(f)
        print(f"loaded {len(smiles_set)} unique molecules")
        
    char_to_idx = mol_encoder.create_char_vocab(list(smiles_set))

    dna_seqs, smiles, labels = get_data()
    # Split data to val set 
    X_dna_train, X_dna_test, X_mol_train, X_mol_test, y_train, y_test = train_test_split(
        dna_seqs, smiles, labels, test_size=0.2, random_state=67)
    
    dna_encoder = DNAEncoder()
    train_dataset = DNAMoleculeDataset(X_dna_train, X_mol_train, y_train, char_to_idx, dna_encoder)
    val_dataset = DNAMoleculeDataset(X_dna_test, X_mol_test, y_test, char_to_idx, dna_encoder)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize model
    model = DNAMoleculeInteractionModel(mol_vocab_size=len(char_to_idx))

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Train model
    print("\nStarting training...")
    trained_model = train_model(model, train_loader, val_loader, num_epochs=20)

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