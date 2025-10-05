from torch.utils.data import Dataset
from typing import List
import torch
import torch.nn as nn
import h5py
import torch.nn.functional as F

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
                 labels: List[int], char_to_idx: dict, dna_encoder: DNAEncoder, gene_seq:dict=None):
        self.dna_ids = dna_ids
        self.smiles = smiles
        self.labels = labels
        self.char_to_idx = char_to_idx
        self.dna_encoder = dna_encoder
        self.gene_seq = gene_seq #intended for training only. load all gene sequences into ram
        
    def __len__(self):
        return len(self.dna_ids)

    def __getitem__(self, idx):
        '''
        if self.preloaded:
            dna_sequence = self.gene_seq[self.dna_ids[idx]]
        else:
            if self.h5 is None:
                self.h5 = h5py.File('data/genes.hdf5', 'r')
            dna_sequence = self.h5[self.dna_ids[idx]][()]
        '''
        if self.gene_seq: 
            dna_sequence = self.gene_seq[self.dna_ids[idx]]
        else:
            with h5py.File("data/genes.hdf5", 'r') as f:
                dna_sequence = f[self.dna_ids[idx]][()]
        dna_encoded = self.dna_encoder.encode_sequence(dna_sequence)
        mol_encoded = torch.tensor([self.char_to_idx.get(char, self.char_to_idx['<UNK>'])
                                   for char in self.smiles[idx][:200]], dtype=torch.long)

        # Pad molecule sequence
        if len(mol_encoded) < 200:
            padding = torch.full((200 - len(mol_encoded),), self.char_to_idx['<PAD>'], dtype=torch.long)
            mol_encoded = torch.cat([mol_encoded, padding])

        return dna_encoded, mol_encoded, torch.tensor(self.labels[idx], dtype=torch.float)