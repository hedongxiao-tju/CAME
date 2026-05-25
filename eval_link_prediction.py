#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CAME 链路预测评估脚本（仅评估）
- 模型：CAME（对齐 text(t5vit) / image(clip) 并融合输出 z3）
- 支持选择 z1、z2 或 z3 作为最终嵌入进行评估
- 正负样本全部从 lp-edge-split.pt 预加载
- 每条正边配有 150 个负样本（shape: [num_edges, 150]）
"""

import argparse
import inspect
import os
from typing import List, Optional, Dict, Tuple

import dgl
import torch
import torch.nn
import torch.nn.functional as F

from data.lp_dataset import LinkPredictionDataset
from models.came import CAME

def load_clusters_from_ppr(ppr_path: str) -> List[torch.Tensor]:
    """从 PPR map 生成簇列表（中心节点放在首位）"""
    print(f"从PPR map加载: {ppr_path}")
    ppr_map = torch.load(ppr_path, map_location="cpu")
    if not isinstance(ppr_map, dict):
        raise ValueError(f"PPR map格式不正确，期望dict，得到{type(ppr_map)}")
    print(f"找到 {len(ppr_map)} 个节点的PPR结果")

    clusters: List[torch.Tensor] = []
    center_nodes = sorted(ppr_map.keys())
    for center in center_nodes:
        neighbors = ppr_map[center]
        if isinstance(neighbors, torch.Tensor):
            if neighbors.numel() > 0 and neighbors[0].item() != center:
                cluster = torch.cat([torch.tensor([center], dtype=neighbors.dtype), neighbors])
            else:
                cluster = neighbors if neighbors.numel() > 0 else torch.tensor([center], dtype=torch.long)
        else:
            cluster = torch.tensor([center], dtype=torch.long)
        clusters.append(cluster)
    print(f"生成了 {len(clusters)} 个簇")
    return clusters

def load_clusters(cluster_path: Optional[str], ppr_path: Optional[str] = None) -> Optional[List[torch.Tensor]]:
    """加载预计算簇（优先使用PPR map）"""
    if ppr_path:
        return load_clusters_from_ppr(ppr_path)
    if not cluster_path:
        return None
    print(f"加载预计算簇: {cluster_path}")
    raw = torch.load(cluster_path, map_location="cpu")
    clusters: List[torch.Tensor] = []

    def _collect(x):
        if x is None:
            return
        if isinstance(x, (list, tuple)):
            for y in x:
                _collect(y)
        else:
            t = torch.as_tensor(x, dtype=torch.long)
            if t.numel() > 0:
                clusters.append(t)

    _collect(raw)
    print(f"载入 {len(clusters)} 个簇")
    return clusters

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CAME 链路预测评估")
    parser.add_argument("--data-dir", required=True, help="数据集根目录路径，需包含 lp-edge-split.pt")
    parser.add_argument("--feat-name", default="image_feature.pt", help="节点特征文件名或前缀（可直接传 image_feature.pt）")
    parser.add_argument("--checkpoint", required=False, default=None, help="Lightning checkpoint 路径 (*.ckpt)，不提供则使用随机初始化权重")

    parser.add_argument("--hidden-dim", type=int, default=1024, help="CAME hidden_1024（GAT输出维度）")
    parser.add_argument("--out-dim", type=int, default=1024, help="CAME fuse 输出维度（即 z3 维度）")
    parser.add_argument("--num-heads", type=int, default=4, help="GAT heads 数")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument("--attn-drop", type=float, default=0.1, help="GAT attention dropout")
    parser.add_argument("--temperature", type=float, default=0.07, help="InfoNCE temperature")
    parser.add_argument("--w12", type=float, default=1.0, help="loss(z1,z2) 权重")
    parser.add_argument("--w23", type=float, default=1.0, help="loss(z2,z3) 权重")
    parser.add_argument("--w13", type=float, default=1.0, help="loss(z1,z3) 权重")
    parser.add_argument("--text-dim", type=int, default=512, help="文本模态输入维度（t5vit）")
    parser.add_argument("--image-dim", type=int, default=512, help="图像模态输入维度（clip）")
    parser.add_argument("--use-gate", action="store_true", help="在模型中启用可学习融合 gate（如果模型支持）")
    parser.add_argument("--fusion-type", choices=["hierarchical_moe"], default="hierarchical_moe", help="融合模块类型：hierarchical_moe（层次化MoE - 三模态：文本、图像、融合）")
    parser.add_argument("--use-moe", action="store_true", help="使用层次化MoE融合（固定为True）")
    parser.add_argument("--num-experts", type=int, default=18, help="总专家数量（平分给文本、图像、融合三个模态）")
    parser.add_argument("--num-selected-experts", type=int, default=2, help="每个模态选择的专家数量")

    parser.add_argument("--device", default="cuda", help="设备: cuda, cpu, 或 cuda:6")
    parser.add_argument("--embed-type", choices=["z1", "z2", "z3"], default="z3", help="使用的嵌入类型：z1（文本GAT输出）、z2（图像GAT输出）、z3（融合输出）")
    parser.add_argument("--mode", choices=["full", "stream"], default="full", help="评估模式：整图/流式子图")
    parser.add_argument("--batch-size", type=int, default=32, help="流式子图模式下的中心节点 batch 大小")
    parser.add_argument("--cluster-path", type=str, default=None, help="预计算节点簇路径 (clusters.pt，stream 模式可选)")
    parser.add_argument("--ppr-path", type=str, default=None, help="PPR map文件路径 (ppr.pt，stream 模式可选，优先于cluster-path)")
    parser.add_argument("--subgraph-cache-path", type=str, default=None, help="预计算子图缓存路径 (.bin)，stream 模式可选")
    parser.add_argument("--subgraph-cache-meta-path", type=str, default=None, help="预计算子图缓存元数据路径 (.pt)，与 --subgraph-cache-path 配套")
    parser.add_argument("--eval-batch-size", type=int, default=500, help="评估批大小（MRR计算）")
    parser.add_argument("--fuse-batch-size", type=int, default=32768, help="z3整图融合分块大小，用于降低MoE推理峰值显存")
    parser.add_argument("--seed", type=int, default=None, help="随机种子（用于复现随机权重初始化）")
    parser.add_argument("--no-normalize", action="store_true", help="不进行L2归一化，使用原始尺度嵌入（测试随机权重表现）")
    parser.add_argument("--random-init", choices=["default", "xavier", "kaiming", "uniform", "normal", "zeros", "ones", "fixed"], default="default", help="随机权重初始化方法")

    return parser.parse_args()

def resolve_device(device_flag: str) -> torch.device:
    return torch.device(device_flag)

def compute_mrr_esci(
    predictor,
    node_emb: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    neg_dst: torch.Tensor,
    device: torch.device,
    batch_size: int = 500,
):
    """计算 MRR / Hits@10 / Hits@1，支持任意数量的负样本（neg_dst: [E, K]）。"""
    node_emb = node_emb.to(device)
    rr = torch.zeros(src.shape[0], device="cpu")
    hits_at_10 = torch.zeros(src.shape[0], device="cpu")
    hits_at_1 = torch.zeros(src.shape[0], device="cpu")

    for start in range(0, src.shape[0], batch_size):
        end = min(start + batch_size, src.shape[0])
        src_batch = src[start:end].to(device)
        dst_batch = dst[start:end].to(device)
        neg_batch = neg_dst[start:end].to(device)
        all_dst = torch.cat([dst_batch[:, None], neg_batch], dim=1)  # [B, 1+K]
        h_src = node_emb[src_batch][:, None, :]  # [B,1,D]
        h_dst = node_emb[all_dst.reshape(-1)].view(*all_dst.shape, -1)  # [B,1+K,D]

        pred = predictor(h_src, h_dst).squeeze(-1)  # [B,1+K]
        y_pred_pos = pred[:, 0:1]     # [B,1]
        y_pred_neg = pred[:, 1:]      # [B,K]

        optimistic_rank = (y_pred_neg >= y_pred_pos).sum(dim=1)
        pessimistic_rank = (y_pred_neg > y_pred_pos).sum(dim=1)
        ranking = 0.5 * (optimistic_rank + pessimistic_rank) + 1

        hits_at_10[start:end] = (ranking <= 10).float().cpu()
        hits_at_1[start:end] = (ranking <= 1).float().cpu()
        rr[start:end] = (1.0 / ranking).float().cpu()

    return rr.mean().item(), hits_at_10.mean().item(), hits_at_1.mean().item()

def get_CAME_embed_dim(model: CAME, embed_type: str = "z3") -> int:
    """
    获取指定嵌入类型的维度。
    z1 和 z2: hidden_1024 维度
    z3: out_1024 维度
    """
    if embed_type in ["z1", "z2"]:
        return int(getattr(model, "hidden_1024", 1024))
    elif embed_type == "z3":
        return int(getattr(model, "out_1024", 1024))
    else:
        raise ValueError(f"不支持的嵌入类型: {embed_type}")

def call_model_without_loss(
    model: CAME,
    graph: dgl.DGLGraph,
    feat_text: torch.Tensor,
    feat_image: torch.Tensor,
    embed_type: str = "z3",
    fuse_batch_size: int = 32768,
) -> torch.Tensor:
    """
    调用模型进行前向传播，不计算损失，直接返回指定类型的嵌入。
    使用 getz1/getz2/getz3 参数直接获取嵌入，避免计算损失和创建字典。
    """
    graph = model._prep_graph(graph)
    param_dtype = next(model.parameters()).dtype

    if embed_type == "z1":
        return model.gat_t5vit(graph, feat_text.to(dtype=param_dtype))
    if embed_type == "z2":
        return model.gat_clip(graph, feat_image.to(dtype=param_dtype))



    feat_text = feat_text.to(dtype=param_dtype)
    feat_image = feat_image.to(dtype=param_dtype)

    z1 = model.gat_t5vit(graph, feat_text)
    z2 = model.gat_clip(graph, feat_image)

    z3_chunks = []
    chunk_size = max(1, int(fuse_batch_size))
    for start in range(0, z1.shape[0], chunk_size):
        end = min(start + chunk_size, z1.shape[0])
        z1_chunk = z1[start:end]
        z2_chunk = z2[start:end]
        if getattr(model, "use_gate", False) and hasattr(model, "fuse_gate"):
            gate = model.fuse_gate(torch.cat([z1_chunk, z2_chunk], dim=-1))
            z_fused_gated = gate * z1_chunk + (1.0 - gate) * z2_chunk
            z3_chunk, _ = model.fuse_mlp(torch.cat([z_fused_gated, z1_chunk, z2_chunk], dim=-1))
        else:
            z3_chunk, _ = model.fuse_mlp(torch.cat([z1_chunk, z2_chunk], dim=-1))
        z3_chunks.append(z3_chunk)

    return torch.cat(z3_chunks, dim=0)

def extract_embeddings(
    model: CAME,
    graph: dgl.DGLGraph,
    feats: Dict[str, torch.Tensor],
    device: torch.device,
    mode: str = "full",
    batch_size: int = 32,
    clusters: Optional[List[torch.Tensor]] = None,
    embed_type: str = "z3",
    subgraph_cache: Optional[tuple] = None,
    fuse_batch_size: int = 32768,
) -> torch.Tensor:
    graph = graph.cpu()
    feats = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in feats.items()}
    if mode == "full":
        return _embed_full_graph(model, graph, feats, device, embed_type, fuse_batch_size)
    if subgraph_cache is not None:
        return _embed_cached_subgraphs(model, feats, device, embed_type, subgraph_cache, fuse_batch_size)
    return _embed_subgraph(model, graph, feats, device, batch_size, clusters, embed_type, fuse_batch_size)

def _embed_full_graph(
    model: CAME,
    graph: dgl.DGLGraph,
    feats: Dict[str, torch.Tensor],
    device: torch.device,
    embed_type: str = "z3",
    fuse_batch_size: int = 32768,
) -> torch.Tensor:
    print(f"整图模式：一次性前向 (CAME)，使用嵌入类型: {embed_type}...")
    graph_gpu = graph.to(device)

    feat_text = feats.get("text", feats.get("t5vit", None))
    feat_img = feats.get("image", feats.get("clip", None))

    if feat_text is None or feat_img is None:
        raise KeyError(
            "CAME expects features with keys ['text','image'] "
            f"(or fallback ['t5vit','clip']), but got: {list(feats.keys())}"
        )

    feat_text_gpu = feat_text.to(device)
    feat_img_gpu = feat_img.to(device)

    with torch.no_grad():
        emb_gpu = call_model_without_loss(model, graph_gpu, feat_text_gpu, feat_img_gpu, embed_type, fuse_batch_size)
        emb = emb_gpu.cpu()

    del graph_gpu, feat_text_gpu, feat_img_gpu, emb_gpu
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"嵌入形状: {emb.shape} ({embed_type})")
    return emb

def _batch(arr: torch.Tensor, bs: int):
    for i in range(0, len(arr), bs):
        yield arr[i: i + bs]

def _collect_nodes(center_ids: torch.Tensor, clusters: List[torch.Tensor]) -> torch.Tensor:
    node_sets = [clusters[int(i)] for i in center_ids.tolist()]
    return torch.unique(torch.cat(node_sets))

def load_subgraph_cache(cache_path: str, meta_path: str):
    print(f"加载预计算子图缓存: {cache_path}")
    graphs, _ = dgl.load_graphs(cache_path)
    meta = torch.load(meta_path, map_location="cpu")
    batch_units = meta.get("batch_units")
    batch_centers = meta.get("batch_centers")
    sub_nodes = meta.get("sub_nodes")
    if batch_units is None or batch_centers is None or sub_nodes is None:
        raise ValueError("子图缓存元数据缺少 batch_units、batch_centers 或 sub_nodes")
    if len(graphs) != len(batch_units) or len(graphs) != len(batch_centers) or len(graphs) != len(sub_nodes):
        raise ValueError("子图缓存图数量与元数据数量不一致")
    print(f"共加载 {len(graphs)} 个缓存子图")
    return graphs, batch_units, batch_centers, sub_nodes

def _embed_subgraph(
    model: CAME,
    graph: dgl.DGLGraph,
    feats: Dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
    clusters: List[torch.Tensor],
    embed_type: str = "z3",
    fuse_batch_size: int = 32768,
) -> torch.Tensor:
    print(f"流式子图模式：按簇前向 (CAME)，使用嵌入类型: {embed_type}...")
    N = graph.num_nodes()
    D = get_CAME_embed_dim(model, embed_type)
    emb = torch.zeros(N, D, device="cpu")
    units = torch.arange(len(clusters), dtype=torch.long)

    feat_text = feats.get("text", feats.get("t5vit", None))
    feat_img = feats.get("image", feats.get("clip", None))

    if feat_text is None or feat_img is None:
        raise KeyError(
            "CAME expects features with keys ['text','image'] "
            f"(or fallback ['t5vit','clip']), but got: {list(feats.keys())}"
        )

    for batch_idx, batch_units in enumerate(_batch(units, batch_size)):
        sub_nodes = _collect_nodes(batch_units, clusters)
        sub_g = dgl.node_subgraph(graph, sub_nodes)
        sub_g = sub_g.remove_self_loop().add_self_loop()

        sub_nodes_long = sub_nodes.long()
        sub_feat_text = feat_text[sub_nodes_long]
        sub_feat_img = feat_img[sub_nodes_long]

        sub_g_gpu = sub_g.to(device)
        sub_feat_text_gpu = sub_feat_text.to(device)
        sub_feat_img_gpu = sub_feat_img.to(device)

        with torch.no_grad():
            h_gpu = call_model_without_loss(model, sub_g_gpu, sub_feat_text_gpu, sub_feat_img_gpu, embed_type, fuse_batch_size)

        mapping = {nid.item(): i for i, nid in enumerate(sub_nodes)}
        for idx in batch_units.tolist():
            cluster = clusters[idx]
            if cluster.numel() == 0:
                continue
            center = int(cluster[0].item())
            if center in mapping:
                emb[center] = h_gpu[mapping[center]].detach().cpu()

        del sub_g_gpu, sub_feat_text_gpu, sub_feat_img_gpu, h_gpu
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if (batch_idx + 1) % 100 == 0:
            print(f"已处理 {batch_idx + 1} 个 batch...")

    print(f"嵌入形状: {emb.shape} ({embed_type}, 流式)")
    return emb

def _embed_cached_subgraphs(
    model: CAME,
    feats: Dict[str, torch.Tensor],
    device: torch.device,
    embed_type: str,
    subgraph_cache: tuple,
    fuse_batch_size: int = 32768,
) -> torch.Tensor:
    print(f"流式子图模式：使用预计算子图缓存 (CAME)，使用嵌入类型: {embed_type}...")
    cached_graphs, _, cached_batch_centers, cached_sub_nodes = subgraph_cache
    N = feats["text"].shape[0]
    D = get_CAME_embed_dim(model, embed_type)
    emb = torch.zeros(N, D, device="cpu")

    feat_text = feats.get("text", feats.get("t5vit", None))
    feat_img = feats.get("image", feats.get("clip", None))

    if feat_text is None or feat_img is None:
        raise KeyError(
            "CAME expects features with keys ['text','image'] "
            f"(or fallback ['t5vit','clip']), but got: {list(feats.keys())}"
        )

    for batch_idx, (sub_g, batch_centers, sub_nodes) in enumerate(
        zip(cached_graphs, cached_batch_centers, cached_sub_nodes)
    ):
        sub_nodes_long = sub_nodes.long()
        sub_feat_text = feat_text[sub_nodes_long]
        sub_feat_img = feat_img[sub_nodes_long]

        sub_g_gpu = sub_g.to(device)
        sub_feat_text_gpu = sub_feat_text.to(device)
        sub_feat_img_gpu = sub_feat_img.to(device)

        with torch.no_grad():
            h_gpu = call_model_without_loss(model, sub_g_gpu, sub_feat_text_gpu, sub_feat_img_gpu, embed_type, fuse_batch_size)

        mapping = {nid.item(): i for i, nid in enumerate(sub_nodes)}
        for center in batch_centers.tolist():
            if center in mapping:
                emb[center] = h_gpu[mapping[center]].detach().cpu()

        del sub_g_gpu, sub_feat_text_gpu, sub_feat_img_gpu, h_gpu
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if (batch_idx + 1) % 100 == 0:
            print(f"已处理 {batch_idx + 1} 个缓存 batch...")

    print(f"嵌入形状: {emb.shape} ({embed_type}, 缓存流式)")
    return emb

class DotPredictor(torch.nn.Module):
    """对 (h_src, h_dst) 做元素乘积后求和，输出标量 logits。"""
    def forward(self, h_src: torch.Tensor, h_dst: torch.Tensor) -> torch.Tensor:
        return (h_src * h_dst).sum(dim=-1, keepdim=True)

def apply_random_init(model: CAME, init_method: str, seed: int = None) -> None:
    """对模型应用不同的随机初始化方法"""
    if seed is not None:
        torch.manual_seed(seed)

    def init_weights(module):
        if isinstance(module, torch.nn.Linear):
            if init_method == "xavier":
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif init_method == "kaiming":
                torch.nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif init_method == "uniform":
                torch.nn.init.uniform_(module.weight, -0.1, 0.1)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif init_method == "normal":
                torch.nn.init.normal_(module.weight, 0, 0.1)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif init_method == "zeros":
                torch.nn.init.zeros_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif init_method == "ones":
                torch.nn.init.ones_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif init_method == "fixed":
                torch.nn.init.constant_(module.weight, 0.01)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.LayerNorm):
            if init_method == "zeros":
                torch.nn.init.zeros_(module.weight)
                torch.nn.init.zeros_(module.bias)
            elif init_method == "ones":
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)
            elif init_method == "fixed":
                torch.nn.init.constant_(module.weight, 0.01)
                torch.nn.init.zeros_(module.bias)

    if init_method != "default":
        model.apply(init_weights)
        print(f"使用 {init_method} 初始化方法重新初始化模型权重")

def build_model(args: argparse.Namespace) -> CAME:
    return CAME(
        dim_t5vit=args.text_dim,
        dim_clip=args.image_dim,
        hidden_1024=args.hidden_dim,
        out_1024=args.out_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
        attn_drop=args.attn_drop,
        temperature=args.temperature,
        w12=args.w12,
        w23=args.w23,
        w13=args.w13,
        use_gate=getattr(args, "use_gate", False),
        fusion_type=getattr(args, "fusion_type", "hierarchical_moe"),
        use_moe=getattr(args, "use_moe", True),
        num_experts=getattr(args, "num_experts", 9),
        num_selected_experts=getattr(args, "num_selected_experts", 2),
    )

def load_checkpoint_weights(model: CAME, checkpoint_path: str, device: torch.device) -> None:
    print(f"加载 checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = None
    for candidate_key in ("state_dict", "model_state", "model_state_dict", "model", "state"):
        if isinstance(ckpt, dict) and candidate_key in ckpt:
            state = ckpt[candidate_key]
            print(f"从 checkpoint 提取 state dict（键: '{candidate_key}'）")
            break
    if state is None:
        state = ckpt

    ckpt_args = ckpt.get("args", {})
    ckpt_fusion_type = ckpt_args.get("fusion_type") or ckpt_args.get("fusion-type", "mlp")

    current_fusion_type = "hierarchical_moe"

    if ckpt_fusion_type != current_fusion_type:
        print(f"融合类型不匹配: checkpoint使用'{ckpt_fusion_type}'，当前模型使用'{current_fusion_type}'")
        print("注意: 当前版本只支持hierarchical_moe融合，旧checkpoint可能无法完全兼容")

    if isinstance(state, dict) and any(not isinstance(v, torch.Tensor) for v in state.values()):
        possible = None
        for k, v in state.items():
            if isinstance(v, dict) and all(isinstance(x, torch.Tensor) for x in v.values()):
                possible = v
                print(f"发现嵌套参数 dict，在键 '{k}' 下提取模型权重")
                break
        if possible is not None:
            state = possible

    model_state = {}
    for k, v in state.items():
        model_state[k.replace("model.", "", 1) if k.startswith("model.") else k] = v

    current_state = model.state_dict()
    filtered_state = {}
    skipped_shape = []
    skipped_missing = []
    for k, v in model_state.items():
        if not isinstance(v, torch.Tensor):
            skipped_missing.append((k, "not_tensor"))
            continue
        if k not in current_state:
            skipped_missing.append((k, "missing_in_model"))
            continue
        if tuple(v.shape) == tuple(current_state[k].shape):
            filtered_state[k] = v
        else:
            skipped_shape.append((k, tuple(v.shape), tuple(current_state[k].shape)))

    try:
        missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    except Exception as e:
        print(f"加载过滤后 state_dict 时出错: {e}")
        raise

    if missing:
        print(f"缺失的键 ({len(missing)}): {missing[:10]}")
    if unexpected:
        print(f"意外的键 ({len(unexpected)}): {unexpected[:10]}")

    if skipped_shape:
        print(f"跳过 {len(skipped_shape)} 个因形状不匹配的参数，示例: {skipped_shape[:8]}")
    if skipped_missing:
        print(f"跳过 {len(skipped_missing)} 个因在模型中缺失或非 Tensor 的键，示例: {skipped_missing[:8]}")

    print(f"checkpoint 加载完成，实际加载参数数: {len(filtered_state)}")

def main():
    args = parse_args()
    device = resolve_device(args.device)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        import numpy as np
        np.random.seed(args.seed)
        print(f"使用随机种子: {args.seed}")

    print(f"使用设备: {device}")

    print("加载链路预测数据集...")
    dataset = LinkPredictionDataset(
        root=args.data_dir,
        feat_name=args.feat_name,
        edge_split_type="hard",
        verbose=True,
        device="cpu",
    )
    g = dataset.graph
    edge_split = dataset.get_edge_split()

    text_feat_path = args.feat_name if args.feat_name.endswith(".pt") else f"{args.feat_name}_feat.pt"
    if not os.path.isabs(text_feat_path):
        text_feat_path = os.path.join(args.data_dir, text_feat_path)
    def _extract_feat(t):
        if isinstance(t, dict):
            if "features" in t:
                return t["features"]
            for key in ["feat", "embedding", "embeddings"]:
                if key in t:
                    return t[key]
            raise ValueError(f"特征文件缺少可用键，已有: {list(t.keys())}")
        return t

    feats: Dict[str, torch.Tensor] = {"text": _extract_feat(torch.load(text_feat_path, map_location="cpu"))}

    clip_path = os.path.join(args.data_dir, "text_feature.pt")
    if os.path.exists(clip_path):
        image_data = torch.load(clip_path, map_location="cpu", weights_only=False)
        feats["image"] = _extract_feat(image_data)
        print(f"已加载图像特征: {clip_path}")
    else:
        feats["image"] = feats["text"]

    repeats = max(1, int(getattr(args, "repeats", 1)))
    all_results = {"valid": [], "test": []}
    for run_idx in range(repeats):
        if repeats > 1:
            if args.seed is not None:
                torch.manual_seed(args.seed + run_idx)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(args.seed + run_idx)
            else:
                import random as _rnd
                s = _rnd.randint(0, 2**31 - 1)
                torch.manual_seed(s)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(s)
            print(f"\n--- Run {run_idx+1}/{repeats} (seed used) ---")

        model = build_model(args).to(device)
        if args.checkpoint:
            load_checkpoint_weights(model, args.checkpoint, device)
        else:
            current_seed = args.seed + run_idx if args.seed is not None and repeats > 1 else args.seed
            apply_random_init(model, args.random_init, current_seed)
            if args.random_init == "default":
                print("未提供 checkpoint，使用默认随机初始化权重")
            else:
                print(f"未提供 checkpoint，使用 {args.random_init} 随机初始化权重")
        model.eval()

        subgraph_cache = None
        if args.mode == "stream":
            if args.subgraph_cache_path is not None or args.subgraph_cache_meta_path is not None:
                if not args.subgraph_cache_path or not args.subgraph_cache_meta_path:
                    raise ValueError("使用子图缓存时必须同时提供 --subgraph-cache-path 和 --subgraph-cache-meta-path")
                subgraph_cache = load_subgraph_cache(args.subgraph_cache_path, args.subgraph_cache_meta_path)
                clusters = None
            else:
                if args.ppr_path is None and args.cluster_path is None:
                    raise ValueError("流式模式需提供 --ppr-path、--cluster-path，或子图缓存参数")
                clusters = load_clusters(args.cluster_path, args.ppr_path)
                if clusters is None:
                    raise ValueError("无法加载簇数据")
            node_emb = extract_embeddings(
                model, g, feats, device,
                mode="stream", batch_size=args.batch_size, clusters=clusters,
                embed_type=args.embed_type, subgraph_cache=subgraph_cache,
                fuse_batch_size=args.fuse_batch_size,
            )
        else:
            node_emb = extract_embeddings(
                model, g, feats, device,
                mode="full", batch_size=args.batch_size, clusters=None,
                embed_type=args.embed_type, subgraph_cache=None,
                fuse_batch_size=args.fuse_batch_size,
            )

        if not args.no_normalize:
            node_emb = F.normalize(node_emb, p=2, dim=1)
            print("已对 embedding 做 L2 归一化")
        else:
            print("未对 embedding 做 L2 归一化（使用原始尺度）")

        predictor = DotPredictor().to(device)

        def eval_split(split_name: str):
            if split_name not in edge_split:
                return None
            es = edge_split[split_name]
            if "target_node_neg" not in es:
                raise ValueError(f"{split_name} 缺少 target_node_neg，需预生成负样本")
            return compute_mrr_esci(
                predictor,
                node_emb,
                es["source_node"],
                es["target_node"],
                es["target_node_neg"],
                device=device,
                batch_size=args.eval_batch_size,
            )

        if "valid" in edge_split:
            mrr, h10, h1 = eval_split("valid")
            all_results["valid"].append((mrr, h10, h1))
            print(f"VALID  MRR={mrr:.4f}, Hits@10={h10:.4f}, Hits@1={h1:.4f}")
        if "test" in edge_split:
            mrr, h10, h1 = eval_split("test")
            all_results["test"].append((mrr, h10, h1))
            print(f"TEST   MRR={mrr:.4f}, Hits@10={h10:.4f}, Hits@1={h1:.4f}")

    import numpy as _np
    print("=" * 60)
    print(f"链路预测评估完成 (CAME -> {args.embed_type}, 预加载150负样本)")

    for split_name in ("valid", "test"):
        vals = all_results.get(split_name, [])
        if not vals:
            continue
        arr = _np.array(vals)
        mean_mrr, mean_h10, mean_h1 = arr.mean(axis=0)
        std_mrr, std_h10, std_h1 = arr.std(axis=0)
        print(f"{split_name.upper()}: MRR={mean_mrr:.4f} +/- {std_mrr:.4f}, Hits@10={mean_h10:.4f} +/- {std_h10:.4f}, Hits@1={mean_h1:.4f} +/- {std_h1:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
