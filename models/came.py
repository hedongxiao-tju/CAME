import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

from .fusion_modules import HierarchicalMoE


class MultiLayerGAT(nn.Module):
    def __init__(
        self,
        in_dim: int = 1024,
        out_dim: int = 1024,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        attn_drop: float = 0.1,
    ):
        super().__init__()
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        head_dim = out_dim // num_heads

        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.gat_layers.append(
            dgl.nn.GATConv(
                in_feats=in_dim,
                out_feats=head_dim,
                num_heads=num_heads,
                feat_drop=dropout,
                attn_drop=attn_drop,
                allow_zero_in_degree=True,
            )
        )
        self.norms.append(nn.LayerNorm(out_dim))

        for _ in range(num_layers - 1):
            self.gat_layers.append(
                dgl.nn.GATConv(
                    in_feats=out_dim,
                    out_feats=head_dim,
                    num_heads=num_heads,
                    feat_drop=dropout,
                    attn_drop=attn_drop,
                    allow_zero_in_degree=True,
                )
            )
            self.norms.append(nn.LayerNorm(out_dim))

    @staticmethod
    def _flatten_heads(h: torch.Tensor) -> torch.Tensor:
        return h.flatten(1)

    def forward(self, g: dgl.DGLGraph, x: torch.Tensor):
        h = x
        for gat_layer, norm in zip(self.gat_layers, self.norms):
            h = self._flatten_heads(gat_layer(g, h))
            h = norm(h)
            h = F.gelu(h)
            h = self.dropout(h)
        return h


class TwoLayerGAT(MultiLayerGAT):
    def __init__(self, *args, **kwargs):
        kwargs["num_layers"] = 2
        super().__init__(*args, **kwargs)


def info_nce_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    temperature: float = 0.07,
    symmetric: bool = True,
    batch_size: int = None,
) -> torch.Tensor:
    if z_a.shape[0] != z_b.shape[0]:
        raise ValueError(f"Batch size mismatch: {z_a.shape[0]} vs {z_b.shape[0]}")

    z_a = F.normalize(z_a, dim=-1)
    z_b = F.normalize(z_b, dim=-1)

    num_nodes = z_a.size(0)
    device = z_a.device
    if batch_size is None or batch_size >= num_nodes:
        logits_ab = (z_a @ z_b.t()) / temperature
        labels = torch.arange(num_nodes, device=device)
        loss_ab = F.cross_entropy(logits_ab, labels)

        if not symmetric:
            return loss_ab

        logits_ba = (z_b @ z_a.t()) / temperature
        loss_ba = F.cross_entropy(logits_ba, labels)
        return 0.5 * (loss_ab + loss_ba)

    num_batches = (num_nodes - 1) // batch_size + 1
    indices = torch.arange(0, num_nodes, device=device)

    def compute_batch_loss(z_query, z_key):
        losses = []
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_nodes)
            batch_indices = indices[start_idx:end_idx]
            batch_query = z_query[batch_indices]
            logits = (batch_query @ z_key.t()) / temperature
            loss_batch = F.cross_entropy(logits, batch_indices)
            losses.append(loss_batch)
            del batch_query, logits
        return torch.stack(losses).mean()

    loss_ab = compute_batch_loss(z_a, z_b)
    if not symmetric:
        return loss_ab

    loss_ba = compute_batch_loss(z_b, z_a)
    return 0.5 * (loss_ab + loss_ba)


class CAME(nn.Module):
    def __init__(
        self,
        dim_t5vit: int,
        dim_clip: int,
        hidden_1024: int = 1024,
        out_1024: int = 1024,
        num_heads: int = 4,
        dropout: float = 0.3,
        attn_drop: float = 0.3,
        temperature: float = 0.07,
        w12: float = 1.0,
        w23: float = 1.0,
        w13: float = 1.0,
        symmetric_nce: bool = True,
        loss_batch_size: int = None,
        use_gate: bool = True,
        align_weight: float = 0.0,
        fusion_type: str = "hierarchical_moe",
        use_moe: bool = True,
        num_experts: int = 18,
        num_selected_experts: int = 2,
        text_gat_layers: int = 2,
        vision_gat_layers: int = 2,
        **kwargs,
    ):
        super().__init__()
        if hidden_1024 != out_1024:
            raise ValueError(
                "CAME checkpoints do not contain an output projection; "
                "hidden_1024 and out_1024 must match."
            )

        self.temperature = temperature
        self.hidden_1024 = hidden_1024
        self.out_1024 = out_1024
        self.w12, self.w23, self.w13 = w12, w23, w13
        self.symmetric_nce = symmetric_nce
        self.loss_batch_size = loss_batch_size
        self.use_gate = use_gate
        self.align_weight = float(align_weight)
        self.fusion_type = fusion_type
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.num_selected_experts = num_selected_experts
        self.text_gat_layers = text_gat_layers
        self.vision_gat_layers = vision_gat_layers

        self.gat_t5vit = MultiLayerGAT(
            in_dim=dim_t5vit,
            out_dim=hidden_1024,
            num_heads=num_heads,
            num_layers=text_gat_layers,
            dropout=dropout,
            attn_drop=attn_drop,
        )
        self.gat_clip = MultiLayerGAT(
            in_dim=dim_clip,
            out_dim=hidden_1024,
            num_heads=num_heads,
            num_layers=vision_gat_layers,
            dropout=dropout,
            attn_drop=attn_drop,
        )

        fuse_in = hidden_1024 * (3 if self.use_gate else 2)
        self.fuse_mlp = HierarchicalMoE(
            input_dim=fuse_in,
            hidden_dim=out_1024,
            num_experts_per_modality=self.num_experts // 3,
            num_selected_experts=self.num_selected_experts,
            dropout_rate=dropout,
        )

        if self.use_gate:
            self.fuse_gate = nn.Sequential(
                nn.Linear(hidden_1024 * 2, hidden_1024),
                nn.LayerNorm(hidden_1024),
                nn.GELU(),
                nn.Linear(hidden_1024, hidden_1024),
                nn.Sigmoid(),
            )

    @staticmethod
    def _prep_graph(g: dgl.DGLGraph) -> dgl.DGLGraph:
        return g.remove_self_loop().add_self_loop()

    def forward(
        self,
        graph: dgl.DGLGraph,
        feature_t5vit: torch.Tensor,
        feature_clip: torch.Tensor,
        return_embeddings: bool = False,
        compute_loss: bool = True,
        getz1: bool = False,
        getz2: bool = False,
        getz3: bool = False,
    ):
        graph = self._prep_graph(graph)

        param_dtype = next(self.parameters()).dtype
        feature_t5vit = feature_t5vit.to(dtype=param_dtype)
        feature_clip = feature_clip.to(dtype=param_dtype)

        z1 = self.gat_t5vit(graph, feature_t5vit)
        z2 = self.gat_clip(graph, feature_clip)

        if getattr(self, "use_gate", False) and hasattr(self, "fuse_gate"):
            gate = self.fuse_gate(torch.cat([z1, z2], dim=-1))
            z_fused_gated = gate * z1 + (1.0 - gate) * z2
            z3, modality_weights = self.fuse_mlp(torch.cat([z_fused_gated, z1, z2], dim=-1))
        else:
            z3, modality_weights = self.fuse_mlp(torch.cat([z1, z2], dim=-1))

        expert_stats = None
        if hasattr(self.fuse_mlp, "_expert_stats") and self.fuse_mlp._expert_stats is not None:
            expert_stats = self.fuse_mlp._expert_stats

        if getz1:
            return z1
        if getz2:
            return z2
        if getz3:
            if expert_stats is not None:
                return z3, expert_stats
            return z3

        loss = None
        if compute_loss:
            loss12 = info_nce_loss(
                z1,
                z2,
                temperature=self.temperature,
                symmetric=self.symmetric_nce,
                batch_size=self.loss_batch_size,
            )
            loss23 = info_nce_loss(
                z2,
                z3,
                temperature=self.temperature,
                symmetric=self.symmetric_nce,
                batch_size=self.loss_batch_size,
            )
            loss13 = info_nce_loss(
                z1,
                z3,
                temperature=self.temperature,
                symmetric=self.symmetric_nce,
                batch_size=self.loss_batch_size,
            )
            loss = self.w12 * loss12 + self.w23 * loss23 + self.w13 * loss13

        if return_embeddings:
            out = {
                "z1": z1,
                "z2": z2,
                "z3": z3,
                "modality_weights": modality_weights,
            }
            if compute_loss:
                out.update({"loss12": loss12, "loss23": loss23, "loss13": loss13})
            return loss, out

        return loss


__all__ = ["CAME", "MultiLayerGAT", "TwoLayerGAT", "info_nce_loss"]
