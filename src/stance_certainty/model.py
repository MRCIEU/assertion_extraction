"""
Stance & Certainty Multi-task Classifier

Multi-task model with shared encoder and 4 classification heads:
1. Stance (2-class): SUPPORTS, NEUTRAL
2. Certainty (3-class): HIGH, MEDIUM, LOW
3. Negation (binary): True/False  
4. Speculation (binary): True/False
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Dict, Optional


class StanceCertaintyClassifier(nn.Module):
    """
    Multi-task classifier for stance, certainty, negation, and speculation.
    
    Architecture:
        Input → BioBERT Encoder → [CLS] → 4 classification heads
    """
    
    def __init__(
        self,
        model_name: str = "michiyasunaga/BioLinkBERT-base",
        num_stance_labels: int = 2,  # SUPPORTS, NEUTRAL
        num_certainty_labels: int = 3,  # HIGH, MEDIUM, LOW
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Load config and encoder
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        
        hidden_size = self.config.hidden_size
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Classification heads
        self.stance_classifier = nn.Linear(hidden_size, num_stance_labels)
        self.certainty_classifier = nn.Linear(hidden_size, num_certainty_labels)
        self.negation_classifier = nn.Linear(hidden_size, 2)  # Binary
        self.speculation_classifier = nn.Linear(hidden_size, 2)  # Binary
        
        # Store label counts
        self.num_stance_labels = num_stance_labels
        self.num_certainty_labels = num_certainty_labels
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        stance_labels: Optional[torch.Tensor] = None,
        certainty_labels: Optional[torch.Tensor] = None,
        negation_labels: Optional[torch.Tensor] = None,
        speculation_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            stance_labels: [batch_size] (0=SUPPORTS, 1=NEUTRAL)
            certainty_labels: [batch_size] (0=HIGH, 1=MEDIUM, 2=LOW)
            negation_labels: [batch_size] (0=False, 1=True)
            speculation_labels: [batch_size] (0=False, 1=True)
        
        Returns:
            Dict with logits and optionally losses
        """
        # Encode
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get [CLS] representation
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        pooled_output = self.dropout(pooled_output)
        
        # Classification heads
        stance_logits = self.stance_classifier(pooled_output)
        certainty_logits = self.certainty_classifier(pooled_output)
        negation_logits = self.negation_classifier(pooled_output)
        speculation_logits = self.speculation_classifier(pooled_output)
        
        output_dict = {
            'stance_logits': stance_logits,
            'certainty_logits': certainty_logits,
            'negation_logits': negation_logits,
            'speculation_logits': speculation_logits,
        }
        
        # Compute losses if labels provided
        if stance_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            stance_loss = loss_fct(stance_logits, stance_labels)
            output_dict['stance_loss'] = stance_loss
        
        if certainty_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            certainty_loss = loss_fct(certainty_logits, certainty_labels)
            output_dict['certainty_loss'] = certainty_loss
        
        if negation_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            negation_loss = loss_fct(negation_logits, negation_labels)
            output_dict['negation_loss'] = negation_loss
        
        if speculation_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            speculation_loss = loss_fct(speculation_logits, speculation_labels)
            output_dict['speculation_loss'] = speculation_loss
        
        # Combined loss (if training)
        if all(k in output_dict for k in ['stance_loss', 'certainty_loss', 'negation_loss', 'speculation_loss']):
            # Weighted combination
            # Stance and Certainty are primary tasks
            # Negation and Speculation are auxiliary
            total_loss = (
                1.0 * output_dict['stance_loss'] +
                1.0 * output_dict['certainty_loss'] +
                0.5 * output_dict['negation_loss'] +
                0.5 * output_dict['speculation_loss']
            )
            output_dict['loss'] = total_loss
        
        return output_dict
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Predict labels.
        
        Returns:
            Dict with predicted labels and probabilities
        """
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
        
        # Get predictions
        stance_probs = torch.softmax(outputs['stance_logits'], dim=-1)
        certainty_probs = torch.softmax(outputs['certainty_logits'], dim=-1)
        negation_probs = torch.softmax(outputs['negation_logits'], dim=-1)
        speculation_probs = torch.softmax(outputs['speculation_logits'], dim=-1)
        
        stance_preds = torch.argmax(stance_probs, dim=-1)
        certainty_preds = torch.argmax(certainty_probs, dim=-1)
        negation_preds = torch.argmax(negation_probs, dim=-1)
        speculation_preds = torch.argmax(speculation_probs, dim=-1)
        
        return {
            'stance': stance_preds,
            'certainty': certainty_preds,
            'negation': negation_preds,
            'speculation': speculation_preds,
            'stance_probs': stance_probs,
            'certainty_probs': certainty_probs,
            'negation_probs': negation_probs,
            'speculation_probs': speculation_probs,
        }


class StanceCertaintyLoss(nn.Module):
    """
    Multi-task loss for Stance & Certainty model.
    
    Combines losses from 4 tasks with task-specific weights.
    """
    
    def __init__(
        self,
        stance_weight: float = 1.0,
        certainty_weight: float = 1.0,
        negation_weight: float = 0.5,
        speculation_weight: float = 0.5,
    ):
        super().__init__()
        self.stance_weight = stance_weight
        self.certainty_weight = certainty_weight
        self.negation_weight = negation_weight
        self.speculation_weight = speculation_weight
    
    def forward(
        self,
        stance_logits: torch.Tensor,
        certainty_logits: torch.Tensor,
        negation_logits: torch.Tensor,
        speculation_logits: torch.Tensor,
        stance_labels: torch.Tensor,
        certainty_labels: torch.Tensor,
        negation_labels: torch.Tensor,
        speculation_labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.
        
        Returns:
            Dict with individual losses and combined loss
        """
        loss_fct = nn.CrossEntropyLoss()
        
        stance_loss = loss_fct(stance_logits, stance_labels)
        certainty_loss = loss_fct(certainty_logits, certainty_labels)
        negation_loss = loss_fct(negation_logits, negation_labels)
        speculation_loss = loss_fct(speculation_logits, speculation_labels)
        
        total_loss = (
            self.stance_weight * stance_loss +
            self.certainty_weight * certainty_loss +
            self.negation_weight * negation_loss +
            self.speculation_weight * speculation_loss
        )
        
        return {
            'loss': total_loss,
            'stance_loss': stance_loss,
            'certainty_loss': certainty_loss,
            'negation_loss': negation_loss,
            'speculation_loss': speculation_loss,
        }

