"""
Evidence-Conditioned Relation Extraction (EC-RE) Model

Multi-task model for relation classification based on evidence sentences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Optional


class EvidenceConditionedRE(nn.Module):
    """
    Evidence-Conditioned Relation Extraction model.
    
    Task: Given (head, tail, evidence_pack), predict relation type.
    
    Input format:
        [HEAD:TYPE] head_text [/HEAD] [TAIL:TYPE] tail_text [/TAIL]
        SENT<0>: evidence_sent_0 <SEP> SENT<1>: evidence_sent_1 <SEP> ...
    
    Output:
        - relation_logits: 8-way classification
        - binary_logits: has_relation or not
        - evidence_score: quality of evidence pack
    """
    
    def __init__(
        self,
        model_name: str = "michiyasunaga/BioLinkBERT-base",
        num_labels: int = 8,
        dropout: float = 0.15
    ):
        super().__init__()
        
        # Load encoder (let transformers choose the format)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        
        # Multi-task heads
        self.dropout = nn.Dropout(dropout)
        
        # Main task: relation classification
        self.relation_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 2, num_labels)
        )
        
        # Auxiliary task 1: binary relatedness
        self.binary_head = nn.Linear(self.hidden_size, 2)
        
        # Auxiliary task 2: evidence quality score
        self.evidence_head = nn.Linear(self.hidden_size, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classifier heads"""
        for module in [self.relation_head, self.binary_head, self.evidence_head]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Sequential):
                for m in module:
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
    
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
            relation_logits: [batch, num_labels]
            binary_logits: [batch, 2]
            evidence_score: [batch, 1]
            features: [batch, hidden] (optional)
        """
        # Encode
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # [CLS] representation
        cls_repr = outputs.last_hidden_state[:, 0, :]  # [batch, hidden]
        cls_repr = self.dropout(cls_repr)
        
        # Multi-task predictions
        relation_logits = self.relation_head(cls_repr)  # [batch, num_labels]
        binary_logits = self.binary_head(cls_repr)      # [batch, 2]
        evidence_score = torch.sigmoid(self.evidence_head(cls_repr))  # [batch, 1]
        
        result = {
            'relation_logits': relation_logits,
            'binary_logits': binary_logits,
            'evidence_score': evidence_score
        }
        
        if return_features:
            result['features'] = cls_repr
        
        return result


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # Per-class weights
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch, num_classes]
            targets: [batch]
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        
        # Focal term
        focal_term = (1 - p_t) ** self.gamma
        
        # Alpha weighting
        if self.alpha is not None:
            # Move alpha to same device as targets
            alpha = self.alpha.to(targets.device)
            alpha_t = alpha[targets]
            focal_loss = alpha_t * focal_term * ce_loss
        else:
            focal_loss = focal_term * ce_loss
        
        return focal_loss.mean()


class ECRELoss(nn.Module):
    """
    Multi-task loss for EC-RE:
    L = L_relation + λ_bin * L_binary + λ_evi * L_evidence
    """
    
    def __init__(
        self,
        num_labels: int = 8,
        class_weights: Optional[Dict[str, float]] = None,
        lambda_binary: float = 0.2,
        lambda_evidence: float = 0.5,
        focal_gamma: float = 2.0
    ):
        super().__init__()
        
        self.lambda_binary = lambda_binary
        self.lambda_evidence = lambda_evidence
        
        # Relation loss (focal loss for imbalance)
        if class_weights:
            alpha = torch.tensor([class_weights.get(i, 1.0) for i in range(num_labels)])
        else:
            alpha = None
        
        self.relation_loss = FocalLoss(gamma=focal_gamma, alpha=alpha)
        self.binary_loss = nn.CrossEntropyLoss()
        self.evidence_loss = nn.MSELoss()
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        relation_labels: torch.Tensor,
        binary_labels: torch.Tensor,
        evidence_targets: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs: Model outputs
            relation_labels: [batch] - relation type
            binary_labels: [batch] - 0/1 has relation
            evidence_targets: [batch] - target evidence score (optional)
        """
        # Main task: relation classification
        l_rel = self.relation_loss(outputs['relation_logits'], relation_labels)
        
        # Auxiliary 1: binary relatedness
        l_bin = self.binary_loss(outputs['binary_logits'], binary_labels)
        
        # Auxiliary 2: evidence quality (if provided)
        if evidence_targets is not None:
            l_evi = self.evidence_loss(
                outputs['evidence_score'].squeeze(-1),
                evidence_targets
            )
        else:
            l_evi = torch.tensor(0.0, device=l_rel.device)
        
        # Total loss
        total_loss = l_rel + self.lambda_binary * l_bin + self.lambda_evidence * l_evi
        
        return {
            'loss': total_loss,
            'loss_relation': l_rel,
            'loss_binary': l_bin,
            'loss_evidence': l_evi
        }


def format_ecre_input(
    head_text: str,
    head_type: str,
    tail_text: str,
    tail_type: str,
    evidence_pack_text: str,
    max_length: int = 512
) -> str:
    """
    Format input for EC-RE model.
    
    Format:
        [HEAD:TYPE] text [/HEAD] [TAIL:TYPE] text [/TAIL]
        SENT<0>: evidence_0 <SEP> SENT<1>: evidence_1 <SEP> ...
    """
    input_text = (
        f"[HEAD:{head_type}] {head_text} [/HEAD] "
        f"[TAIL:{tail_type}] {tail_text} [/TAIL] "
        f"{evidence_pack_text}"
    )
    
    # Truncate if too long
    if len(input_text) > max_length * 5:  # Rough char limit
        markers = f"[HEAD:{head_type}] {head_text} [/HEAD] [TAIL:{tail_type}] {tail_text} [/TAIL] "
        remaining = max_length * 5 - len(markers)
        input_text = markers + evidence_pack_text[:remaining]
    
    return input_text


# For compatibility
__all__ = ['EvidenceConditionedRE', 'ECRELoss', 'FocalLoss', 'format_ecre_input']

