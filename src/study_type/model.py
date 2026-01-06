"""
Study Type Multi-label Classifier

Abstract-level multi-label classification for 12 study types.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Dict

# Study type labels
STUDY_TYPES = [
    'RCT', 'clinical_trial', 'observational', 'case_report',
    'in_vitro', 'in_vivo', 'cell_line', 'patient_derived',
    'meta_analysis', 'review', 'biomarker_study', 'pharmacokinetics'
]

STUDY_TYPE_TO_ID = {label: i for i, label in enumerate(STUDY_TYPES)}
ID_TO_STUDY_TYPE = {i: label for i, label in enumerate(STUDY_TYPES)}


class StudyTypeClassifier(nn.Module):
    """
    Multi-label classifier for study types.
    
    Architecture:
        Input (abstract) → BioBERT Encoder → [CLS] → Classifier → Sigmoid
    """
    
    def __init__(
        self,
        model_name: str = "michiyasunaga/BioLinkBERT-base",
        num_labels: int = 12,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # Load config and encoder
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        
        hidden_size = self.config.hidden_size
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Classification head (multi-label)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        self.num_labels = num_labels
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            labels: [batch_size, num_labels] (multi-hot encoding)
        
        Returns:
            Dict with logits and optionally loss
        """
        # Encode
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get [CLS] representation
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        pooled_output = self.dropout(pooled_output)
        
        # Classification
        logits = self.classifier(pooled_output)  # [batch_size, num_labels]
        
        output_dict = {'logits': logits}
        
        # Compute loss if labels provided
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())
            output_dict['loss'] = loss
        
        return output_dict
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        threshold: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        Predict study types.
        
        Returns:
            Dict with predictions and probabilities
        """
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            probs = torch.sigmoid(outputs['logits'])  # [batch_size, num_labels]
            preds = (probs >= threshold).long()  # [batch_size, num_labels]
        
        return {
            'predictions': preds,
            'probabilities': probs
        }

