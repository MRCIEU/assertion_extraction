"""
Evidence Sentence Selector (ESS) Model

Binary classifier to predict whether a sentence is evidence for a relation.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Optional


class EvidenceSentenceSelector(nn.Module):
    """
    Evidence Sentence Selector model.
    
    Task: Given (sentence, head_entity, tail_entity), predict if sentence is evidence.
    
    Input format:
        [HEAD:DRUG] Ensartinib [/HEAD] [TAIL:GENE] ALK [/TAIL] <sent>
    
    Output:
        p_evidence ∈ [0, 1]
    """
    
    def __init__(
        self,
        model_name: str = "michiyasunaga/BioLinkBERT-base",
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Load encoder (let transformers choose the format)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        
        # Binary classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, 1)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            return_features: if True, return CLS features
        
        Returns:
            logits: [batch, 1] - raw logits
            probs: [batch, 1] - sigmoid probabilities
            features: [batch, hidden] - optional CLS features
        """
        # Encode
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # [CLS] representation
        cls_repr = outputs.last_hidden_state[:, 0, :]  # [batch, hidden]
        
        # Classification
        cls_repr = self.dropout(cls_repr)
        logits = self.classifier(cls_repr)  # [batch, 1]
        probs = torch.sigmoid(logits)
        
        result = {
            'logits': logits,
            'probs': probs
        }
        
        if return_features:
            result['features'] = cls_repr
        
        return result
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        threshold: float = 0.5
    ) -> torch.Tensor:
        """
        Predict binary labels.
        
        Returns:
            predictions: [batch] - binary labels (0 or 1)
        """
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            probs = outputs['probs'].squeeze(-1)  # [batch]
            preds = (probs > threshold).long()
        return preds


def format_ess_input(
    sentence: str,
    head_text: str,
    head_type: str,
    tail_text: str,
    tail_type: str,
    max_length: int = 256
) -> str:
    """
    Format input for ESS model.
    
    Format: [HEAD:TYPE] text [/HEAD] [TAIL:TYPE] text [/TAIL] <sentence>
    
    Example:
        [HEAD:DRUG] Ensartinib [/HEAD] [TAIL:GENE] ALK [/TAIL] 
        Ensartinib is a novel ALK inhibitor.
    """
    input_text = (
        f"[HEAD:{head_type}] {head_text} [/HEAD] "
        f"[TAIL:{tail_type}] {tail_text} [/TAIL] "
        f"{sentence}"
    )
    
    # Truncate if too long (should rarely happen for single sentences)
    if len(input_text) > max_length * 5:  # Rough char limit
        # Keep markers, truncate sentence
        markers = f"[HEAD:{head_type}] {head_text} [/HEAD] [TAIL:{tail_type}] {tail_text} [/TAIL] "
        remaining = max_length * 5 - len(markers)
        input_text = markers + sentence[:remaining]
    
    return input_text


# For compatibility with training script
__all__ = ['EvidenceSentenceSelector', 'format_ess_input']

