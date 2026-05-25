#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import copy
import inspect
import os
import random
import gc
import time
from typing import List, Optional, Tuple

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.nc_dataset import NodeClassificationDataset

from models.came import CAME

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CAME 节点分类评估")
    parser.add_argument("--data-dir", required=True, help="数据集根目录路径")
    parser.add_argument("--feat-name", default="image_feature.pt", help="文本特征文件名或前缀")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Lightning checkpoint 路径 (*.ckpt)（如果为 None，则使用随机初始化的权重）",
    )

    parser.add_argument("--hidden-dim", type=int, default=1024, help="CAME hidden_1024（投影+GAT维度）")
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
    parser.add_argument("--fusion-type", choices=["hierarchical_moe"], default="hierarchical_moe", help="融合模块类型：hierarchical_moe（层次化MoE - 三模态：文本、图像、融合）")
    parser.add_argument("--use-moe", action="store_true", help="使用层次化MoE融合（固定为True）")
    parser.add_argument("--num-experts", type=int, default=18, help="总专家数量（平分给文本、图像、融合三个模态）")
    parser.add_argument("--num-selected-experts", type=int, default=2, help="每个模态选择的专家数量")

    parser.add_argument("--device", default="cuda", help="设备: cuda, cpu, 或 cuda:6")
    parser.add_argument("--clip", action="store_true", help="使用原始特征进行评估，不进入CAME模型")
    parser.add_argument("--clip-feat-type", choices=["text", "image", "both"], default="both", help="当使用clip模式时，选择使用的特征类型：text（文本特征）、image（图像特征）、both（拼接两者）")
    parser.add_argument("--embed-type", choices=["z1", "z2", "z3"], default="z3", help="使用的嵌入类型：z1（文本GAT输出）、z2（图像GAT输出）、z3（融合输出），仅在非clip模式下有效")
    parser.add_argument("--use-gate", action="store_true", help="评估时使用模型的可学习融合 gate（如果模型支持）")
    parser.add_argument("--classifier", choices=["linear", "mlp"], default="linear", help="分类器类型：linear 或 mlp")
    parser.add_argument("--classifier-epochs", type=int, default=2000, help="分类器训练轮数")
    parser.add_argument("--classifier-lr", type=float, default=1e-2, help="分类器学习率")
    parser.add_argument("--classifier-weight-decay", type=float, default=5e-4, help="分类器权重衰减")
    parser.add_argument("--mlp-hidden-dim", type=int, default=512, help="MLP隐藏层维度")
    parser.add_argument("--mlp-num-layers", type=int, default=2, help="MLP层数（不包括输出层）")
    parser.add_argument("--mlp-dropout", type=float, default=0.5, help="MLP dropout率")
    parser.add_argument("--log-every", type=int, default=50, help="每 N 轮打印一次日志")

    parser.add_argument("--mode", choices=["full", "stream"], default="full", help="评估模式")
    parser.add_argument("--batch-size", type=int, default=32, help="流式子图模式下的中心节点 batch 大小（对于大型图建议使用更小的值，如16或8）")
    parser.add_argument("--cluster-path", type=str, default=None, help="预计算节点簇路径 (clusters.pt，stream 模式可选)")
    parser.add_argument("--ppr-path", type=str, default=None, help="PPR map文件路径 (ppr.pt，stream 模式可选，优先于cluster-path)")
    parser.add_argument("--subgraph-cache-path", type=str, default=None, help="预计算子图缓存路径 (.bin)，stream 模式可选")
    parser.add_argument("--subgraph-cache-meta-path", type=str, default=None, help="预计算子图缓存元数据路径 (.pt)，与 --subgraph-cache-path 配套")

    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="可传入多个种子用于多次评估，输出均值方差")
    parser.add_argument("--seed", type=int, default=None, help="随机种子（用于随机初始化的模型权重），与 --seeds 二选一")
    parser.add_argument("--repeats", type=int, default=1, help="重复评估次数以计算平均和方差（每个 seed 内部重复次数，默认1）")

    return parser.parse_args()

def resolve_device(device_flag: str) -> torch.device:
    return torch.device(device_flag)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"已设置随机种子: {seed}")

def print_embedding_properties(emb: torch.Tensor, prefix: str = "嵌入"):
    print(f"\n {prefix} 性质统计:")
    print(f"形状: {emb.shape}")
    print(f"数据类型: {emb.dtype}")
    print(f"设备: {emb.device}")
    print(f"均值: {emb.mean().item():.6f} | 标准差: {emb.std().item():.6f} | 最小: {emb.min().item():.6f} | 最大: {emb.max().item():.6f}")
    num_nan = torch.isnan(emb).sum().item()
    num_inf = torch.isinf(emb).sum().item()
    if num_nan:
        print(f"NaN 数量: {num_nan}")
    if num_inf:
        print(f"Inf 数量: {num_inf}")
    l2 = emb.norm(p=2, dim=1)
    print(f"L2 范数: mean={l2.mean().item():.6f} std={l2.std().item():.6f} min={l2.min().item():.6f} max={l2.max().item():.6f}")
    zeros = (l2 < 1e-6).sum().item()
    if zeros:
        print(f"零向量数量: {zeros} ({zeros/emb.shape[0]*100:.2f}%)")
    else:
        print("无零向量")

def get_CAME_embed_dim(model: CAME, embed_type: str = "z3") -> int:
    """
    获取指定嵌入类型的维度。
    由于删除了投影层，现在GAT直接接受输入维度。
    z1 和 z2: hidden_1024 维度（GAT输出维度）
    z3: out_1024 维度（融合输出维度）
    """
    if embed_type in ["z1", "z2"]:
        base = getattr(model, 'hidden_1024', 1024)
        return base
    elif embed_type == "z3":
        try:
            if hasattr(model, 'fuse_mlp'):
                if hasattr(model.fuse_mlp, 'modality_router'):  # HierarchicalMoE
                    return getattr(model, 'out_1024', 1024)
                elif hasattr(model.fuse_mlp, 'weight_net'):  # AdaptiveFusion3
                    return getattr(model, 'out_1024', 1024)
                elif hasattr(model.fuse_mlp, 'experts'):  # MoE
                    expert = model.fuse_mlp.experts[0]
                    if hasattr(expert, 'net'):  # MLP结构
                        last_layer = expert.net[-1]
                        if isinstance(last_layer, nn.Linear):
                            return int(last_layer.out_features)
                    elif isinstance(expert[-1], nn.Linear):  # Sequential结构
                        return int(expert[-1].out_features)
                elif hasattr(model.fuse_mlp, 'net'):  # MLP
                    last_layer = model.fuse_mlp.net[-1]
                    if isinstance(last_layer, nn.Linear):
                        return int(last_layer.out_features)
        except Exception:
            pass
        return getattr(model, 'out_1024', 1024)
    else:
        raise ValueError(f"不支持的嵌入类型: {embed_type}")

def call_model_without_loss(model: CAME, graph: dgl.DGLGraph, feat_text: torch.Tensor, feat_image: torch.Tensor, embed_type: str = "z3") -> torch.Tensor:
    """
    调用模型进行前向传播，不计算损失，直接返回指定类型的嵌入。
    使用 getz1/getz2/getz3 参数直接获取嵌入，避免计算损失和创建字典。
    """
    getz1 = (embed_type == "z1")
    getz2 = (embed_type == "z2")
    getz3 = (embed_type == "z3")

    return model(graph, feat_text, feat_image, getz1=getz1, getz2=getz2, getz3=getz3)

def build_model(args: argparse.Namespace) -> CAME:
    return CAME(
        dim_t5vit=args.text_dim,
        dim_clip=args.image_dim,
        hidden_1024=args.hidden_dim,
        out_1024=args.out_dim,   # z3 维度
        num_heads=args.num_heads,
        dropout=args.dropout,
        attn_drop=args.attn_drop,
        temperature=args.temperature,
        w12=args.w12,
        w23=args.w23,
        w13=args.w13,
        use_gate=getattr(args, "use_gate", False),
        fusion_type=getattr(args, "fusion_type", "hierarchical_moe"),
        use_moe=getattr(args, "use_moe", True),  # 固定使用MoE
        num_experts=getattr(args, "num_experts", 12),  # 默认12个专家（4模态×3专家）
        num_selected_experts=getattr(args, "num_selected_experts", 2),
    )

def load_checkpoint_weights(model: nn.Module, checkpoint_path: str, device: torch.device) -> None:
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

    if isinstance(state, dict) and any(not isinstance(v, torch.Tensor) for v in state.values()):
        possible = None
        for k, v in state.items():
            if isinstance(v, dict) and all(isinstance(x, torch.Tensor) for x in v.values()):
                possible = v
                print(f"发现嵌套参数 dict，在键 '{k}' 下提取模型权重")
                break
        if possible is not None:
            state = possible

    ckpt_args = ckpt.get("args", {})
    ckpt_fusion_type = ckpt_args.get("fusion_type") or ckpt_args.get("fusion-type", "mlp")

    current_fusion_type = "hierarchical_moe"

    if ckpt_fusion_type != current_fusion_type:
        print(f"融合类型不匹配: checkpoint使用'{ckpt_fusion_type}'，当前模型使用'{current_fusion_type}'")
        print("注意: 当前版本只支持hierarchical_moe融合，旧checkpoint可能无法完全兼容")

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
        print(f"缺失的键 ({len(missing)} 个)，示例: {missing[:10]}")
    if unexpected:
        print(f"意外的键 ({len(unexpected)} 个)，示例: {unexpected[:10]}")

    if skipped_shape:
        print(f"跳过 {len(skipped_shape)} 个因形状不匹配的参数，示例: {skipped_shape[:8]}")
    if skipped_missing:
        print(f"跳过 {len(skipped_missing)} 个因在模型中缺失或非 Tensor 的键，示例: {skipped_missing[:8]}")

    print(f"checkpoint 加载完成，实际加载参数数: {len(filtered_state)}")

def load_clusters_from_ppr(ppr_path: str) -> List[torch.Tensor]:
    print(f"从PPR map加载: {ppr_path}")
    ppr_map = torch.load(ppr_path, map_location="cpu")
    if not isinstance(ppr_map, dict):
        raise ValueError(f"PPR map格式不正确，期望dict，得到{type(ppr_map)}")

    clusters: List[torch.Tensor] = []
    for center in sorted(ppr_map.keys()):
        neigh = ppr_map[center]
        if isinstance(neigh, torch.Tensor):
            if neigh.numel() == 0:
                clusters.append(torch.tensor([center], dtype=torch.long))
            else:
                if int(neigh[0].item()) != int(center):
                    clusters.append(torch.cat([torch.tensor([center], dtype=torch.long), neigh.long()]))
                else:
                    clusters.append(neigh.long())
        else:
            clusters.append(torch.tensor([center], dtype=torch.long))

    print(f"生成了 {len(clusters)} 个簇")
    return clusters

def load_clusters(cluster_path: Optional[str], ppr_path: Optional[str] = None) -> Optional[List[torch.Tensor]]:
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
    print(f"共 {len(clusters)} 个簇")
    return clusters

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

def extract_embeddings(
    model: Optional[CAME],
    graph: dgl.DGLGraph,
    feats: dict,
    device: torch.device,
    mode: str = "full",
    batch_size: int = 32,
    clusters: Optional[List[torch.Tensor]] = None,
    embed_type: str = "z3",
    use_clip: bool = False,
    clip_feat_type: str = "both",
    subgraph_cache: Optional[tuple] = None,
) -> torch.Tensor:
    graph = graph.cpu()
    feats = {k: v if isinstance(v, torch.Tensor) else v for k, v in feats.items()}
    feats_preloaded_to_gpu = False
    if device.type == "cuda":
        try:
            if mode == "stream":
                for k, v in list(feats.items()):
                    if isinstance(v, torch.Tensor):
                        feats[k + "_gpu"] = v.to(device, non_blocking=True)
                feats_preloaded_to_gpu = True
                print("已将全部特征预加载到 GPU，用于流式子图加速（注意显存占用）")
        except Exception as e:
            print(f"无法将完整特征预加载到 GPU（回退到按需拷贝），错误: {e}")
            for k in list(feats.keys()):
                if k.endswith("_gpu"):
                    del feats[k]
            feats_preloaded_to_gpu = False

    if use_clip:
        if mode == "full":
            return _embed_full_graph_clip(graph, feats, device, clip_feat_type)
        return _embed_subgraph_clip(graph, feats, device, batch_size, clusters, clip_feat_type)
    else:
        if mode == "full":
            return _embed_full_graph(model, graph, feats, device, embed_type)
        if subgraph_cache is not None:
            return _embed_cached_subgraphs(model, feats, device, embed_type, feats_preloaded_to_gpu, subgraph_cache)
        return _embed_subgraph(model, graph, feats, device, batch_size, clusters, embed_type, feats_preloaded_to_gpu)

def _embed_full_graph_clip(graph: dgl.DGLGraph, feats: dict, device: torch.device, clip_feat_type: str = "both") -> torch.Tensor:
    """使用原始特征（clip模式），不通过CAME模型"""
    print(f"整图模式：使用原始特征 (clip模式)，特征类型: {clip_feat_type}...")

    if clip_feat_type == "text":
        emb = feats["text"]
    elif clip_feat_type == "image":
        emb = feats["image"]
    elif clip_feat_type == "both":
        emb = torch.cat([feats["text"], feats["image"]], dim=-1)
    else:
        raise ValueError(f"不支持的 clip_feat_type: {clip_feat_type}")

    emb = emb.to(device, non_blocking=True)

    print(f"嵌入形状: {emb.shape} (原始特征: {clip_feat_type})")
    print_embedding_properties(emb, prefix=f"原始特征({clip_feat_type})")
    return emb.cpu()

def _embed_subgraph_clip(
    graph: dgl.DGLGraph,
    feats: dict,
    device: torch.device,
    batch_size: int,
    clusters: List[torch.Tensor],
    clip_feat_type: str = "both",
) -> torch.Tensor:
    """使用原始特征（clip模式），流式子图模式"""
    print(f"流式子图模式：使用原始特征 (clip模式)，特征类型: {clip_feat_type}...")

    if clip_feat_type == "text":
        feat_dim = feats["text"].shape[1]
    elif clip_feat_type == "image":
        feat_dim = feats["image"].shape[1]
    elif clip_feat_type == "both":
        feat_dim = feats["text"].shape[1] + feats["image"].shape[1]
    else:
        raise ValueError(f"不支持的 clip_feat_type: {clip_feat_type}")

    N = graph.num_nodes()
    emb = torch.zeros(N, feat_dim, device="cpu")

    units = torch.arange(len(clusters), dtype=torch.long)
    for batch_idx, batch_units in enumerate(_batch(units, batch_size)):
        sub_nodes = _collect_nodes(batch_units, clusters)
        sub_nodes_long = sub_nodes.long()

        def _index_feat(src_feat, idx_cpu):
            if src_feat.device.type == "cpu":
                return src_feat[idx_cpu]
            return src_feat[idx_cpu.to(src_feat.device)]

        if clip_feat_type == "text":
            sub_feat = _index_feat(feats.get("text_gpu", feats["text"]), sub_nodes_long)
        elif clip_feat_type == "image":
            sub_feat = _index_feat(feats.get("image_gpu", feats["image"]), sub_nodes_long)
        elif clip_feat_type == "both":
            t_src = feats.get("text_gpu", feats["text"])
            i_src = feats.get("image_gpu", feats["image"])
            t_part = _index_feat(t_src, sub_nodes_long)
            i_part = _index_feat(i_src, sub_nodes_long)
            sub_feat = torch.cat([t_part, i_part], dim=-1)

        mapping = {nid.item(): i for i, nid in enumerate(sub_nodes)}
        for idx in batch_units.tolist():
            cluster = clusters[idx]
            if cluster.numel() == 0:
                continue
            center = int(cluster[0].item())
            if center in mapping:
                emb[center] = sub_feat[mapping[center]].detach().cpu()

        try:
            del sub_feat, sub_nodes, sub_nodes_long, mapping, cluster
        except Exception:
            pass
        gc.collect()

        if (batch_idx + 1) % 100 == 0:
            print(f"已处理 {batch_idx + 1} 个 batch...")

    print(f"嵌入形状: {emb.shape} (原始特征: {clip_feat_type}, 流式)")
    return emb

def _embed_full_graph(model: CAME, graph: dgl.DGLGraph, feats: dict, device: torch.device, embed_type: str = "z3") -> torch.Tensor:
    print(f"整图模式：一次性在全图上前向 (CAME)，使用嵌入类型: {embed_type}...")

    feats_gpu = {}
    for k, v in feats.items():
        feats_gpu[k] = v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v

    graph = graph.remove_self_loop().add_self_loop()
    graph_gpu = graph.to(device)

    with torch.no_grad():
        emb_gpu = call_model_without_loss(model, graph_gpu, feats_gpu["text"], feats_gpu["image"], embed_type)
        emb = emb_gpu.cpu()

    del graph_gpu, feats_gpu, emb_gpu
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"嵌入形状: {emb.shape} ({embed_type})")
    print_embedding_properties(emb, prefix=f"提取的嵌入({embed_type})")
    return emb

def _embed_subgraph(
    model: CAME,
    graph: dgl.DGLGraph,
    feats: dict,
    device: torch.device,
    batch_size: int,
    clusters: List[torch.Tensor],
    embed_type: str = "z3",
    feats_preloaded_to_gpu: bool = False,
) -> torch.Tensor:
    print(f"流式子图模式：按预计算簇批次前向 (CAME)，使用嵌入类型: {embed_type}...")
    N = graph.num_nodes()
    D = get_CAME_embed_dim(model, embed_type)
    emb = torch.zeros(N, D, device="cpu")

    units = torch.arange(len(clusters), dtype=torch.long)
    for batch_idx, batch_units in enumerate(_batch(units, batch_size)):
        sub_nodes = _collect_nodes(batch_units, clusters)
        sub_g = dgl.node_subgraph(graph, sub_nodes)
        sub_g = sub_g.remove_self_loop().add_self_loop()

        sub_nodes_long = sub_nodes.long()
        if feats_preloaded_to_gpu and ("text_gpu" in feats or "image_gpu" in feats):
            text_src = feats.get("text_gpu", feats["text"])
            img_src = feats.get("image_gpu", feats["image"])
            idx_dev = sub_nodes_long.to(text_src.device)
            sub_text = text_src[idx_dev]
            sub_image = img_src[idx_dev]
        else:
            sub_text = feats["text"][sub_nodes_long]
            sub_image = feats["image"][sub_nodes_long]

        sub_feats = {"text": sub_text, "image": sub_image}

        sub_g_gpu = sub_g.to(device)
        sub_feats_gpu = {k: t.to(device, non_blocking=True) for k, t in sub_feats.items()}

        with torch.no_grad():
            h_gpu = call_model_without_loss(model, sub_g_gpu, sub_feats_gpu["text"], sub_feats_gpu["image"], embed_type)

        mapping = {nid.item(): i for i, nid in enumerate(sub_nodes)}
        for idx in batch_units.tolist():
            cluster = clusters[idx]
            if cluster.numel() == 0:
                continue
            center = int(cluster[0].item())
            if center in mapping:
                emb[center] = h_gpu[mapping[center]].detach().cpu()

        del sub_g_gpu, sub_feats_gpu, h_gpu
        if device.type == "cuda":
            torch.cuda.empty_cache()

        try:
            del sub_g, sub_nodes, sub_nodes_long, sub_feats, mapping, cluster, sub_text, sub_image
        except Exception:
            pass
        gc.collect()

        if (batch_idx + 1) % 100 == 0:
            print(f"已处理 {batch_idx + 1} 个 batch...")

    print(f"嵌入形状: {emb.shape} ({embed_type}, 流式)")
    print_embedding_properties(emb, prefix=f"提取的嵌入({embed_type}, 流式)")
    return emb

def _embed_cached_subgraphs(
    model: CAME,
    feats: dict,
    device: torch.device,
    embed_type: str,
    feats_preloaded_to_gpu: bool,
    subgraph_cache: tuple,
) -> torch.Tensor:
    print(f"流式子图模式：使用预计算子图缓存 (CAME)，使用嵌入类型: {embed_type}...")
    cached_graphs, cached_batch_units, cached_batch_centers, cached_sub_nodes = subgraph_cache
    N = feats["text"].shape[0]
    D = get_CAME_embed_dim(model, embed_type)
    emb = torch.zeros(N, D, device="cpu")

    for batch_idx, (sub_g, batch_units, batch_centers, sub_nodes) in enumerate(
        zip(cached_graphs, cached_batch_units, cached_batch_centers, cached_sub_nodes)
    ):
        sub_nodes_long = sub_nodes.long()
        if feats_preloaded_to_gpu and ("text_gpu" in feats or "image_gpu" in feats):
            text_src = feats.get("text_gpu", feats["text"])
            img_src = feats.get("image_gpu", feats["image"])
            idx_dev = sub_nodes_long.to(text_src.device)
            sub_text = text_src[idx_dev]
            sub_image = img_src[idx_dev]
        else:
            sub_text = feats["text"][sub_nodes_long]
            sub_image = feats["image"][sub_nodes_long]

        sub_g_gpu = sub_g.to(device)
        sub_feats_gpu = {
            "text": sub_text.to(device, non_blocking=True),
            "image": sub_image.to(device, non_blocking=True),
        }

        with torch.no_grad():
            h_gpu = call_model_without_loss(model, sub_g_gpu, sub_feats_gpu["text"], sub_feats_gpu["image"], embed_type)

        mapping = {nid.item(): i for i, nid in enumerate(sub_nodes)}
        for center in batch_centers.tolist():
            if center in mapping:
                emb[center] = h_gpu[mapping[center]].detach().cpu()

        del sub_g_gpu, sub_feats_gpu, h_gpu, mapping, sub_text, sub_image
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        if (batch_idx + 1) % 100 == 0:
            print(f"已处理 {batch_idx + 1} 个缓存 batch...")

    print(f"嵌入形状: {emb.shape} ({embed_type}, 缓存流式)")
    print_embedding_properties(emb, prefix=f"提取的嵌入({embed_type}, 缓存流式)")
    return emb

class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 512, num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        layers = []
        layers += [nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout)]
        layers += [nn.Linear(hidden_dim, num_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def train_classifier(
    emb: torch.Tensor,
    labels: torch.Tensor,
    train_m: torch.Tensor,
    val_m: torch.Tensor,
    test_m: torch.Tensor,
    classifier_type: str = "linear",
    epochs: int = 5000,
    lr: float = 1e-2,
    weight_decay: float = 5e-4,
    mlp_hidden_dim: int = 512,
    mlp_num_layers: int = 2,
    mlp_dropout: float = 0.5,
    log_every: int = 50,
) -> Tuple[float, float]:
    name = "MLP" if classifier_type == "mlp" else "线性"
    print(f"训练{name}分类器...")

    C = int(labels.max().item() + 1)

    if torch.isnan(emb).any():
        print("Embedding中包含NaN，将替换为零向量")
        emb = torch.where(torch.isnan(emb), torch.zeros_like(emb), emb)
    if torch.isinf(emb).any():
        print("Embedding中包含Inf，将替换为零向量")
        emb = torch.where(torch.isinf(emb), torch.zeros_like(emb), emb)

    if emb.dtype != torch.float32:
        print(f"将 embedding 从 {emb.dtype} 转为 float32 以避免与分类器权重 dtype 不匹配")
        emb = emb.to(torch.float32)

    if classifier_type == "mlp":
        clf = MLPClassifier(emb.size(1), C, hidden_dim=mlp_hidden_dim, num_layers=mlp_num_layers, dropout=mlp_dropout).to(emb.device)
    else:
        clf = nn.Linear(emb.size(1), C).to(emb.device)

    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=weight_decay)

    best_val, best_test = 0.0, 0.0
    best_model = None

    for epoch in range(1, epochs + 1):
        clf.train()
        logits = clf(emb)
        loss = F.cross_entropy(logits[train_m], labels[train_m])
        opt.zero_grad()
        loss.backward()
        opt.step()

        clf.eval()
        with torch.no_grad():
            logits = clf(emb)
            acc_train = (logits[train_m].argmax(dim=1) == labels[train_m]).float().mean().item()
            acc_val = (logits[val_m].argmax(dim=1) == labels[val_m]).float().mean().item()
            acc_test = (logits[test_m].argmax(dim=1) == labels[test_m]).float().mean().item()

        if acc_val > best_val:
            best_val, best_test = acc_val, acc_test
            best_model = copy.deepcopy(clf)

        if log_every > 0 and epoch % log_every == 0:
            print(f"[Eval] Epoch {epoch:04d} | Train {acc_train:.4f} | Val {acc_val:.4f} | Test {acc_test:.4f}")

    if best_model is not None:
        best_model.eval()
        with torch.no_grad():
            logits = best_model(emb)
            estp_test = (logits[test_m].argmax(dim=1) == labels[test_m]).float().mean().item()
        print(f"[最佳模型] 早停测试准确率: {estp_test:.4f} (对应验证准确率: {best_val:.4f})")
        return best_val, estp_test

    return best_val, best_test

def main():
    args = parse_args()
    overall_start = time.perf_counter()

    device = resolve_device(args.device)
    print(f"使用设备: {device}")
    if args.seeds:
        print(f"🔁 将使用多个 seeds 进行评估: {args.seeds}（每个 seed 内部重复 {args.repeats} 次）")
    else:
        if args.seed is not None:
            set_seed(args.seed)
        elif args.checkpoint is None:
            print("使用随机初始化的模型权重，但未指定 --seed 或 --seeds 参数，结果可能不可重复")

    print("加载数据集...")
    dataset = NodeClassificationDataset(
        root=args.data_dir,
        feat_name=args.feat_name,
        verbose=True,
        device="cpu",
    )

    graph = dataset.graph
    labels = graph.ndata["label"]
    train_m = graph.ndata["train_mask"].bool()
    val_m = graph.ndata["val_mask"].bool()
    test_m = graph.ndata["test_mask"].bool()

    feats = {"text": dataset.features}
    image_path = os.path.join(args.data_dir, "text_feature.pt")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"找不到图像特征文件: {image_path}")
    image_data = torch.load(image_path, map_location="cpu", weights_only=False)
    if isinstance(image_data, dict):
        if "features" in image_data:
            feats["image"] = image_data["features"]
        else:
            for k in ["feat", "image", "embeddings"]:
                if k in image_data:
                    feats["image"] = image_data[k]
                    break
            else:
                raise ValueError(f"图像特征字典中未找到特征数据，可用键: {list(image_data.keys())}")
    elif isinstance(image_data, torch.Tensor):
        feats["image"] = image_data
    else:
        raise TypeError(f"不支持的图像特征文件格式: {type(image_data)}")

    model = None
    if not args.clip:
        model = build_model(args).to(device)
        if args.checkpoint:
            load_checkpoint_weights(model, args.checkpoint, device)
        else:
            print("未指定 checkpoint，使用随机初始化的模型权重")
        model.eval()
    else:
        print("使用 clip 模式：直接使用原始特征，不加载 CAME 模型")

    subgraph_cache = None
    if args.mode == "stream":
        if args.subgraph_cache_path is not None or args.subgraph_cache_meta_path is not None:
            if not args.subgraph_cache_path or not args.subgraph_cache_meta_path:
                raise ValueError("使用子图缓存时必须同时提供 --subgraph-cache-path 和 --subgraph-cache-meta-path")
            subgraph_cache = load_subgraph_cache(args.subgraph_cache_path, args.subgraph_cache_meta_path)
            clusters = None
        else:
            if args.ppr_path is None and args.cluster_path is None:
                raise ValueError("流式模式必须提供 --ppr-path、--cluster-path，或子图缓存参数")
            clusters = load_clusters(args.cluster_path, args.ppr_path)
            if clusters is None:
                raise ValueError("无法加载簇数据")

            if args.batch_size > 64 and len(clusters) > 10000:
                print(f"警告: batch_size={args.batch_size} 对于大型图可能过大，建议使用 16-32 的较小值")
                print("提示: 如果遇到 CUDA out of memory，可以尝试 --batch-size 16 或 --batch-size 8")
    else:
        clusters = None

    embed_start = time.perf_counter()
    emb = extract_embeddings(
        model, graph, feats, device,
        mode=args.mode, batch_size=args.batch_size,
        clusters=clusters, embed_type=args.embed_type,
        use_clip=args.clip, clip_feat_type=args.clip_feat_type,
        subgraph_cache=subgraph_cache,
    )
    embed_elapsed = time.perf_counter() - embed_start
    print(f"⏱ 嵌入提取耗时: {embed_elapsed:.2f} 秒")

    num_classes = int(labels.max().item() + 1)
    print(f"📋 类别数量: {num_classes}")
    print(f"训练节点数: {train_m.sum().item()} | 验证节点数: {val_m.sum().item()} | 测试节点数: {test_m.sum().item()}")

    repeats = max(1, int(getattr(args, "repeats", 1)))
    seed_list = args.seeds if getattr(args, "seeds", None) else ([args.seed] if args.seed is not None else [None])
    all_vals = []
    all_tests = []
    run_counter = 0
    classifier_start = time.perf_counter()
    for s in seed_list:
        for r in range(repeats):
            run_counter += 1
            if s is not None:
                seed_used = int(s) + r
            else:
                seed_used = random.randint(0, 2**31 - 1)
            print(f"\n--- Run {run_counter} (seed {seed_used}) ---")
            set_seed(seed_used)

            best_val, best_test = train_classifier(
                emb.to(device),
                labels.to(device),
                train_m.to(device),
                val_m.to(device),
                test_m.to(device),
                classifier_type=args.classifier,
                epochs=args.classifier_epochs,
                lr=args.classifier_lr,
                weight_decay=args.classifier_weight_decay,
                mlp_hidden_dim=args.mlp_hidden_dim,
                mlp_num_layers=args.mlp_num_layers,
                mlp_dropout=args.mlp_dropout,
                log_every=args.log_every,
            )
            all_vals.append(best_val)
            all_tests.append(best_test)
    classifier_elapsed = time.perf_counter() - classifier_start
    overall_elapsed = time.perf_counter() - overall_start

    mean_val = float(np.mean(all_vals)) if all_vals else 0.0
    std_val = float(np.std(all_vals)) if all_vals else 0.0
    mean_test = float(np.mean(all_tests)) if all_tests else 0.0
    std_test = float(np.std(all_tests)) if all_tests else 0.0

    print("=" * 60)
    if args.clip:
        print(f"节点分类评估完成 (原始特征: {args.clip_feat_type})")
    else:
        print(f"节点分类评估完成 (CAME -> {args.embed_type})")
    if repeats == 1:
        print(f"最佳验证集准确率: {mean_val:.4f}")
        print(f"对应测试集准确率: {mean_test:.4f}")
    else:
        print(f"验证集准确率: {mean_val:.4f} +/- {std_val:.4f} (runs={repeats})")
        print(f"测试集准确率: {mean_test:.4f} +/- {std_test:.4f} (runs={repeats})")
    print(f"嵌入提取耗时: {embed_elapsed:.2f} 秒")
    print(f"下游分类器耗时: {classifier_elapsed:.2f} 秒")
    print(f"下游评测总耗时: {overall_elapsed:.2f} 秒")
    print("=" * 60)

if __name__ == "__main__":
    main()
