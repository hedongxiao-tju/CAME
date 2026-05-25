#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalMoE(nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_experts_per_modality: int = 6,
                 num_selected_experts: int = 2,
                 dropout_rate: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_experts_per_modality = num_experts_per_modality
        self.num_selected_experts = num_selected_experts

        self._track_experts = False
        self._expert_stats = None
        self._router_selected_counts = None  # used by expert_aux_loss()

        self.text_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_experts_per_modality)
        ])

        self.image_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_experts_per_modality)
        ])

        self.text_expert_router = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_experts_per_modality),
        )

        self.image_expert_router = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_experts_per_modality),
        )

        self.modality_fusion_router = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 2),
        )

        self.fusion_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_experts_per_modality)
        ])

        self.fusion_expert_router = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim , num_experts_per_modality),
        )

        self.modality_router = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 3),

        )

        # Router output layers: use Xavier init (larger than PyTorch default ~0.018)
        # This gives logits std ~0.86 vs ~0.34, preventing early collapse
        for name in ["text_expert_router", "image_expert_router", "fusion_expert_router"]:
            router = getattr(self, name)
            nn.init.xavier_uniform_(router[-1].weight)
            nn.init.zeros_(router[-1].bias)

    def set_track_experts(self, enabled: bool):
        """Enable/disable expert activation tracking."""
        self._track_experts = enabled
        if not enabled:
            self._expert_stats = None

    def get_expert_stats(self):
        """Return expert activation statistics. Call after forward."""
        return self._expert_stats

    def reset_expert_stats(self):
        self._expert_stats = None

    def expert_aux_loss(self, weight: float = 0.01) -> torch.Tensor:
        """
        Load-balancing auxiliary loss for all three expert routers.
        Adapted from GShard / Switch Transformer.
        Loss = sum_over_modalities [ alpha * sum_over_experts [ (selected_fraction - 1/K)^2 ] ]
        where selected_fraction = count(expert_i_is_selected) / (batch_size * num_selected)
        The router bias trick encourages uniform selection without gradient starvation.
        """
        if self._router_selected_counts is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        total_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        T = self.num_experts_per_modality

        for mod_key in ["text", "image", "fusion"]:
            counts = self._router_selected_counts[mod_key]  # [T]
            if counts.sum() == 0:
                continue
            # Fraction of selections routed to each expert
            frac = counts.float() / counts.sum()
            # Target: uniform 1/T each → variance penalty
            loss_mod = T * (frac - 1.0 / T).pow(2).mean()
            total_loss = total_loss + loss_mod

        return weight * total_loss

    def reset_router_counts(self):
        """Reset per-step router selection counters."""
        T = self.num_experts_per_modality
        self._router_selected_counts = {
            "text": torch.zeros(T, dtype=torch.long, device=next(self.parameters()).device),
            "image": torch.zeros(T, dtype=torch.long, device=next(self.parameters()).device),
            "fusion": torch.zeros(T, dtype=torch.long, device=next(self.parameters()).device),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        param_dtype = next(self.parameters()).dtype
        device = x.device
        if x.dtype != param_dtype:
            x = x.to(dtype=param_dtype)

        batch_size = x.shape[0]

        if x.shape[-1] == self.hidden_dim * 3:
            x_fusion, x_text, x_image = torch.split(x, self.hidden_dim, dim=-1)
        elif x.shape[-1] == self.hidden_dim * 2:
            x_text, x_image = torch.split(x, self.hidden_dim, dim=-1)
            x_fusion = 0.5 * (x_text + x_image)
        else:
            raise ValueError(f"Unexpected input dim {x.shape[-1]}, expected 2*hidden or 3*hidden ({self.hidden_dim})")

        modality_input = torch.cat([x_fusion, x_text, x_image], dim=-1)
        modality_weights = F.softmax(self.modality_router(modality_input), dim=-1)
        text_weight, image_weight, fusion_weight = modality_weights.split(1, dim=-1)

        text_expert_logits = self.text_expert_router(x_text)
        image_expert_logits = self.image_expert_router(x_image)
        fusion_expert_logits = self.fusion_expert_router(x_fusion)

        text_topk_values, text_topk_indices = torch.topk(text_expert_logits, self.num_selected_experts, dim=-1)
        image_topk_values, image_topk_indices = torch.topk(image_expert_logits, self.num_selected_experts, dim=-1)
        fusion_topk_values, fusion_topk_indices = torch.topk(fusion_expert_logits, self.num_selected_experts, dim=-1)

        # Count router selections for auxiliary load-balancing loss
        if self._router_selected_counts is not None:
            for eid in range(self.num_experts_per_modality):
                self._router_selected_counts["text"][eid] += (text_topk_indices == eid).sum()
                self._router_selected_counts["image"][eid] += (image_topk_indices == eid).sum()
                self._router_selected_counts["fusion"][eid] += (fusion_topk_indices == eid).sum()

        text_expert_weights = F.softmax(text_topk_values, dim=-1)
        image_expert_weights = F.softmax(image_topk_values, dim=-1)
        fusion_expert_weights = F.softmax(fusion_topk_values, dim=-1)

        def compute_expert_outputs_selected(experts, topk_indices, inputs):
            batch_size_local, num_selected = topk_indices.shape
            num_experts = len(experts)

            expert_indices = topk_indices.flatten()
            sample_indices = torch.arange(batch_size_local, device=device).repeat_interleave(num_selected)

            selected_inputs = inputs[sample_indices]

            all_expert_outputs = []
            for expert_id in range(num_experts):
                mask = (expert_indices == expert_id)
                if mask.any():
                    expert_inputs = selected_inputs[mask]
                    out = experts[expert_id](expert_inputs)
                    all_expert_outputs.append(out)
                else:
                    all_expert_outputs.append(torch.empty(0, self.hidden_dim, device=device, dtype=param_dtype))

            output_tensor = torch.zeros(batch_size_local * num_selected, self.hidden_dim, device=device, dtype=param_dtype)
            for expert_id in range(num_experts):
                expert_output = all_expert_outputs[expert_id]
                if expert_output.numel() > 0:
                    expert_positions = (expert_indices == expert_id).nonzero().squeeze(-1)
                    output_tensor[expert_positions] = expert_output.to(dtype=param_dtype)

            return output_tensor.view(batch_size_local, num_selected, self.hidden_dim)

        text_selected_outputs = compute_expert_outputs_selected(self.text_experts, text_topk_indices, x_text)
        image_selected_outputs = compute_expert_outputs_selected(self.image_experts, image_topk_indices, x_image)
        fusion_selected_outputs = compute_expert_outputs_selected(self.fusion_experts, fusion_topk_indices, x_fusion)

        text_fused = torch.sum(text_expert_weights.unsqueeze(-1) * text_selected_outputs, dim=1)
        image_fused = torch.sum(image_expert_weights.unsqueeze(-1) * image_selected_outputs, dim=1)
        fusion_fused = torch.sum(fusion_expert_weights.unsqueeze(-1) * fusion_selected_outputs, dim=1)

        final_fused = text_weight.squeeze(-1).unsqueeze(-1) * text_fused + \
                     image_weight.squeeze(-1).unsqueeze(-1) * image_fused + \
                     fusion_weight.squeeze(-1).unsqueeze(-1) * fusion_fused

        # Track expert activations
        if self._track_experts:
            self._expert_stats = {
                "text_topk_indices": text_topk_indices,          # [B, k]
                "text_topk_weights": text_expert_weights,       # [B, k]
                "image_topk_indices": image_topk_indices,        # [B, k]
                "image_topk_weights": image_expert_weights,      # [B, k]
                "fusion_topk_indices": fusion_topk_indices,      # [B, k]
                "fusion_topk_weights": fusion_expert_weights,   # [B, k]
                "modality_weights": modality_weights,           # [B, 3]
            }

        return final_fused, modality_weights.squeeze(-1)  # [batch, 3]


