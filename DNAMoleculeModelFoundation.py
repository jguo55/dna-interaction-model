from torch.utils.data import Dataset
from typing import List
import torch
import torch.nn as nn
import h5py
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig

class DNABERT2Encoder(nn.Module):
    """DNA encoder using DNABERT-2 foundation model"""
    def __init__(self, model_name: str = "zhihan1996/DNABERT-2-117M", freeze_backbone: bool = False):
        super(DNABERT2Encoder, self).__init__()

        # Use local patched BERT version
        local_model_path = "./bert_patch/"

        # Load patched BERT tokenizer from local directory
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

        # Load config from local directory
        config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # Load model from local patched directory
        # Determine device for loading
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            self.model = AutoModel.from_pretrained(
                model_name,
                config=config,
                trust_remote_code=True
            ).to(device)
        except ValueError as e:
            # If we get the config mismatch error, try loading as BertModel directly
            if "config_class" in str(e):
                from transformers import BertModel
                self.model = BertModel.from_pretrained(
                    model_name,
                    config=config,
                    trust_remote_code=True
                ).to(device)
            else:
                raise e

        # Freeze backbone if specified (for faster training initially)
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        # Get hidden size from model config
        self.hidden_size = self.model.config.hidden_size

        # Projection layer to reduce dimensionality if needed
        self.projection = nn.Linear(self.hidden_size, 512)
        self.dropout = nn.Dropout(0.1)

    def forward(self, sequences: List[str]):
        """
        Args:
            sequences: List of DNA sequences as strings
        Returns:
            Tensor of shape (batch_size, 512)
        """
        # Tokenize sequences
        inputs = self.tokenizer(sequences, return_tensors='pt', padding=True,
                               truncation=True, max_length=512).to(self.model.device)

        # Get embeddings from DNABERT-2
        outputs = self.model(**inputs, return_dict=True)

        # Use [CLS] token representation (first token)
        # Handle case where model returns tuple instead of object
        last_hidden_state = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
        cls_embeddings = last_hidden_state[:, 0, :]  # (batch_size, hidden_size)

        # Project to common dimension
        projected = self.projection(cls_embeddings)  # (batch_size, 512)
        projected = self.dropout(projected)

        return projected, last_hidden_state  # Return both for cross-attention


class MoleculeFoundationEncoder(nn.Module):
    """Small molecule encoder using ChemBERTa foundation model"""
    def __init__(self, model_name: str = "DeepChem/ChemBERTa-77M-MLM", freeze_backbone: bool = False):
        super(MoleculeFoundationEncoder, self).__init__()

        # Load ChemBERTa model and tokenizer
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)

        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        # Get hidden size from model config
        self.hidden_size = self.model.config.hidden_size

        # Projection layer
        self.projection = nn.Linear(self.hidden_size, 512)
        self.dropout = nn.Dropout(0.1)

    def forward(self, smiles_list: List[str]):
        """
        Args:
            smiles_list: List of SMILES strings
        Returns:
            Tensor of shape (batch_size, 512)
        """
        # Tokenize SMILES
        inputs = self.tokenizer(smiles_list, return_tensors='pt', padding=True,
                               truncation=True, max_length=512).to(self.model.device)

        # Get embeddings from ChemBERTa
        outputs = self.model(**inputs, return_dict=True)

        # Use [CLS] token representation
        # Handle case where model returns tuple instead of object
        last_hidden_state = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
        cls_embeddings = last_hidden_state[:, 0, :]  # (batch_size, hidden_size)

        # Project to common dimension
        projected = self.projection(cls_embeddings)  # (batch_size, 512)
        projected = self.dropout(projected)

        return projected, last_hidden_state  # Return both for cross-attention


class CrossAttention(nn.Module):
    """Cross-attention mechanism between DNA and molecule embeddings"""
    def __init__(self, embed_dim: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super(CrossAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Multi-head attention layers
        # DNA attends to molecule
        self.dna_to_mol_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        # Molecule attends to DNA
        self.mol_to_dna_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Feed-forward networks
        self.ffn1 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout)
        )

        self.ffn2 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout)
        )

        self.norm3 = nn.LayerNorm(embed_dim)
        self.norm4 = nn.LayerNorm(embed_dim)

    def forward(self, dna_embeds, mol_embeds):
        """
        Args:
            dna_embeds: DNA embeddings (batch_size, embed_dim)
            mol_embeds: Molecule embeddings (batch_size, embed_dim)
        Returns:
            Fused representation (batch_size, embed_dim * 2)
        """
        # Add sequence dimension for attention
        dna_embeds = dna_embeds.unsqueeze(1)  # (batch_size, 1, embed_dim)
        mol_embeds = mol_embeds.unsqueeze(1)  # (batch_size, 1, embed_dim)

        # DNA attends to molecule
        dna_attended, _ = self.dna_to_mol_attn(dna_embeds, mol_embeds, mol_embeds)
        dna_attended = self.norm1(dna_embeds + dna_attended)
        dna_attended = self.norm3(dna_attended + self.ffn1(dna_attended))

        # Molecule attends to DNA
        mol_attended, _ = self.mol_to_dna_attn(mol_embeds, dna_embeds, dna_embeds)
        mol_attended = self.norm2(mol_embeds + mol_attended)
        mol_attended = self.norm4(mol_attended + self.ffn2(mol_attended))

        # Remove sequence dimension and concatenate
        dna_attended = dna_attended.squeeze(1)  # (batch_size, embed_dim)
        mol_attended = mol_attended.squeeze(1)  # (batch_size, embed_dim)

        # Concatenate attended representations
        fused = torch.cat([dna_attended, mol_attended], dim=-1)  # (batch_size, embed_dim * 2)

        return fused


class DNAMoleculeInteractionModelFoundation(nn.Module):
    """
    DNA-Molecule interaction model using foundation models with cross-attention
    """
    def __init__(self,
                 dna_model_name: str = "zhihan1996/DNABERT-2-117M",
                 mol_model_name: str = "DeepChem/ChemBERTa-77M-MLM",
                 freeze_backbones: bool = False,
                 num_attention_heads: int = 8):
        super(DNAMoleculeInteractionModelFoundation, self).__init__()

        # Foundation model encoders
        self.dna_encoder = DNABERT2Encoder(dna_model_name, freeze_backbone=freeze_backbones)
        self.molecule_encoder = MoleculeFoundationEncoder(mol_model_name, freeze_backbone=freeze_backbones)

        # Cross-attention mechanism
        self.cross_attention = CrossAttention(embed_dim=512, num_heads=num_attention_heads)

        # Fusion and prediction layers
        self.fusion_layer = nn.Sequential(
            nn.Linear(1024, 512),  # 512 * 2 from cross-attention
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Prediction head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, dna_sequences: List[str], smiles_list: List[str]):
        """
        Args:
            dna_sequences: List of DNA sequences as strings
            smiles_list: List of SMILES strings
        Returns:
            Predictions (batch_size,)
        """
        # Encode DNA and molecules
        dna_embeds, dna_hidden = self.dna_encoder(dna_sequences)  # (batch_size, 512)
        mol_embeds, mol_hidden = self.molecule_encoder(smiles_list)  # (batch_size, 512)

        # Apply cross-attention
        fused_embeds = self.cross_attention(dna_embeds, mol_embeds)  # (batch_size, 1024)

        # Fusion and prediction
        fused = self.fusion_layer(fused_embeds)  # (batch_size, 256)
        predictions = self.classifier(fused)  # (batch_size, 1)

        return predictions.squeeze()


class DNAMoleculeDatasetFoundation(Dataset):
    """Dataset for DNA-Molecule interactions using foundation models"""
    def __init__(self, dna_ids: List[str], smiles: List[str],
                 labels: List[int], gene_seq: dict = None):
        self.dna_ids = dna_ids
        self.smiles = smiles
        self.labels = labels
        self.gene_seq = gene_seq  # Preloaded gene sequences

    def __len__(self):
        return len(self.dna_ids)

    def __getitem__(self, idx):
        # Load gene sequence
        if self.gene_seq:
            dna_sequence = self.gene_seq[self.dna_ids[idx]]
        else:
            with h5py.File("data_general/genes.hdf5", 'r') as f:
                dna_sequence = f[self.dna_ids[idx]][()]

        # Decode if bytes
        if isinstance(dna_sequence, bytes):
            dna_sequence = dna_sequence.decode('utf-8')

        # Return raw strings - tokenization happens in the model
        return dna_sequence, self.smiles[idx], torch.tensor(self.labels[idx], dtype=torch.float)


def collate_fn_foundation(batch):
    """Custom collate function for foundation model dataset"""
    dna_seqs, smiles_list, labels = zip(*batch)
    labels = torch.stack(labels)
    return list(dna_seqs), list(smiles_list), labels
