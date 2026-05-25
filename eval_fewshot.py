#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Few-shot 评估脚本

使用 CAME 模型优化 features_clip 嵌入，然后进行 Few-shot 评估。
Few-shot 评估采用原型网络（Prototypical Network）方法，无需训练分类器。

用法:
    python evaluate_fewshot.py \
        --dataset FB15K237 \
        --data-dir ./data/FB15K237 \
        --checkpoint checkpoints/epoch=10-train_loss=0.00.ckpt \
        --hidden-dim 1024 \
        --text-dim 512 \
        --image-dim 512 \
        --embed-type z3 \
        --eval-num-label 2 \
        --eval-num-support 3 \
        --eval-num-query 3 \
        --eval-tasks 500 \
        --seeds 0 1 2 \
        --device cuda
"""

import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
import dgl
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
from models.came import CAME


# =========================

# =========================
class PrototypicalNetwork:
    """原型网络用于 Few-shot 学习"""

    def __init__(self):
        pass

    def compute_prototypes(self, support_embeddings: torch.Tensor, support_labels: torch.Tensor) -> torch.Tensor:
        """计算每个类别的原型（质心）"""
        unique_labels = torch.unique(support_labels)
        prototypes = []

        for label in unique_labels:
            class_embeddings = support_embeddings[support_labels == label]
            prototype = class_embeddings.mean(dim=0)
            prototypes.append(prototype)

        return torch.stack(prototypes)  # (num_classes, emb_dim)

    def predict(self, query_embeddings: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        """使用原型进行预测（余弦相似度）"""

        norm_query_emb = query_embeddings / torch.norm(query_embeddings, dim=1, keepdim=True)
        norm_prototypes = prototypes / torch.norm(prototypes, dim=1, keepdim=True)


        similarities = torch.matmul(norm_query_emb, norm_prototypes.T)  # (num_queries, num_classes)


        predictions = torch.argmax(similarities, dim=1)
        return predictions

    def evaluate_episode(self, support_embeddings: torch.Tensor, support_labels: torch.Tensor,
                        query_embeddings: torch.Tensor, query_labels: torch.Tensor) -> float:
        """评估单个 episode 的准确率"""

        prototypes = self.compute_prototypes(support_embeddings, support_labels)


        predictions = self.predict(query_embeddings, prototypes)


        accuracy = (predictions == query_labels).float().mean().item()
        return accuracy


# =========================

# =========================
def load_graph(graph_path: str) -> dgl.DGLGraph:
    """加载图文件（自动检测格式：dgl.save_graphs 或 torch.save）"""
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"图文件不存在: {graph_path}")
    
    try:
        graphs, _ = dgl.load_graphs(graph_path)
        return graphs[0]
    except:
        graph = torch.load(graph_path, map_location="cpu", weights_only=False)
        if not isinstance(graph, dgl.DGLGraph):
            raise TypeError(f"文件不是 DGLGraph: {type(graph)}")
        return graph


def load_triplets(file_path: str, entity2id: dict = None, relation2id: dict = None):
    """加载三元组文件，返回 (node_pairs, labels)
    
    Args:
        file_path: 三元组文件路径
        entity2id: 实体名称到ID的映射（如果为None，则尝试直接解析整数ID）
        relation2id: 关系名称到ID的映射（如果为None，则尝试直接解析整数ID或自动分配）
    """
    if not os.path.exists(file_path):
        return [], []
    
    node_pairs = []
    labels = []
    

    auto_relation2id = {} if entity2id else None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            
            h_str, r_str, t_str = parts[0], parts[1], parts[2]
            
            if entity2id is not None:

                h_id = entity2id.get(h_str)
                t_id = entity2id.get(t_str)
                if h_id is None or t_id is None:
                    continue
                

                if relation2id is not None:
                    r_id = relation2id.get(r_str)
                    if r_id is None:
                        continue
                else:

                    try:
                        r_id = int(r_str)
                    except ValueError:
                        if auto_relation2id is not None:
                            if r_str not in auto_relation2id:
                                auto_relation2id[r_str] = len(auto_relation2id)
                            r_id = auto_relation2id[r_str]
                        else:
                            continue
            else:

                try:
                    h_id = int(h_str)
                    t_id = int(t_str)
                    try:
                        r_id = int(r_str)
                    except ValueError:

                        if auto_relation2id is not None:
                            if r_str not in auto_relation2id:
                                auto_relation2id[r_str] = len(auto_relation2id)
                            r_id = auto_relation2id[r_str]
                        else:
                            continue
                except ValueError:
                    continue
            
            node_pairs.append([h_id, t_id])
            labels.append(r_id)
    
    return node_pairs, labels


def load_kg_data(data_dir: str):
    """加载 KG 数据（图和三元组）"""

    graph_path = os.path.join(data_dir, "graph.bin")
    graph = load_graph(graph_path)
    print(f" 图: {graph.num_nodes()} 节点, {graph.num_edges()} 边")
    

    entity2id = None
    relation2id = None
    mapping_path = os.path.join(data_dir, "entity_mapping.pt")
    if os.path.exists(mapping_path):
        print(f"📋 加载映射文件: {mapping_path}")
        try:
            mapping_data = torch.load(mapping_path, map_location="cpu", weights_only=False)
            entity2id = mapping_data.get("entity2id")
            relation2id = mapping_data.get("relation2id")
            if entity2id:
                print(f"   实体映射: {len(entity2id)} 个")
            if relation2id:
                print(f"   关系映射: {len(relation2id)} 个")
            if not entity2id and not relation2id:
                print(f"    映射文件中未找到 entity2id 或 relation2id，将尝试直接解析整数ID")
        except Exception as e:
            print(f"    加载映射文件失败: {e}，将尝试直接解析整数ID")
    else:
        print(f"   ℹ 未找到映射文件，将尝试直接解析整数ID")
    

    train_path = os.path.join(data_dir, "train.txt")
    valid_path = os.path.join(data_dir, "valid.txt")
    test_path = os.path.join(data_dir, "test.txt")
    
    train_pairs, train_labels = load_triplets(train_path, entity2id, relation2id)
    valid_pairs, valid_labels = load_triplets(valid_path, entity2id, relation2id)
    test_pairs, test_labels = load_triplets(test_path, entity2id, relation2id)
    

    link = {
        "train": [train_pairs, train_labels],
        "valid": [valid_pairs, valid_labels],
        "test": [test_pairs, test_labels]
    }
    

    split_idx = {}
    count = 0
    for name, pairs in [("train", train_pairs), ("valid", valid_pairs), ("test", test_pairs)]:
        num_triplets = len(pairs)
        split_idx[name] = torch.arange(count, count + num_triplets)
        count += num_triplets
    

    all_labels = train_labels + valid_labels + test_labels
    labels = torch.LongTensor(all_labels)
    
    print(f"   训练三元组: {len(train_pairs)}, 验证三元组: {len(valid_pairs)}, 测试三元组: {len(test_pairs)}")
    
    return graph, link, labels, split_idx


def load_node_classification_data_light(data_dir: str):
    """轻量级加载节点分类数据（只加载标签和分割信息，跳过图数据）"""

    labels_path = os.path.join(data_dir, "labels.pt")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"标签文件不存在: {labels_path}")
    labels = torch.load(labels_path, map_location="cpu", weights_only=False)
    if not isinstance(labels, torch.Tensor):
        raise TypeError(f"标签文件格式错误: {type(labels)}")
    print(f"   标签形状: {labels.shape}, 类别数: {len(torch.unique(labels))}")


    train_mask_path = os.path.join(data_dir, "train_mask.pt")
    val_mask_path = os.path.join(data_dir, "val_mask.pt")
    test_mask_path = os.path.join(data_dir, "test_mask.pt")

    if not os.path.exists(train_mask_path):
        raise FileNotFoundError(f"训练掩码文件不存在: {train_mask_path}")
    if not os.path.exists(val_mask_path):
        raise FileNotFoundError(f"验证掩码文件不存在: {val_mask_path}")
    if not os.path.exists(test_mask_path):
        raise FileNotFoundError(f"测试掩码文件不存在: {test_mask_path}")

    train_mask = torch.load(train_mask_path, map_location="cpu", weights_only=False)
    val_mask = torch.load(val_mask_path, map_location="cpu", weights_only=False)
    test_mask = torch.load(test_mask_path, map_location="cpu", weights_only=False)

    if not isinstance(train_mask, torch.Tensor) or not isinstance(val_mask, torch.Tensor) or not isinstance(test_mask, torch.Tensor):
        raise TypeError("掩码文件格式错误")

    print(f"   训练节点: {train_mask.sum().item()}, 验证节点: {val_mask.sum().item()}, 测试节点: {test_mask.sum().item()}")


    split_idx = {
        'train': train_mask.nonzero().squeeze(),
        'valid': val_mask.nonzero().squeeze(),
        'test': test_mask.nonzero().squeeze()
    }


    graph = None
    link = None

    return graph, link, labels, split_idx


def load_node_classification_data(data_dir: str):
    """加载节点分类数据（图和标签、掩码）"""

    graph_path = os.path.join(data_dir, "graph.bin")
    graph = load_graph(graph_path)
    print(f" 图: {graph.num_nodes()} 节点, {graph.num_edges()} 边")
    

    labels_path = os.path.join(data_dir, "labels.pt")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"标签文件不存在: {labels_path}")
    labels = torch.load(labels_path, map_location="cpu", weights_only=False)
    if not isinstance(labels, torch.Tensor):
        raise TypeError(f"标签文件格式错误: {type(labels)}")
    print(f"   标签形状: {labels.shape}, 类别数: {len(torch.unique(labels))}")
    

    train_mask_path = os.path.join(data_dir, "train_mask.pt")
    val_mask_path = os.path.join(data_dir, "val_mask.pt")
    test_mask_path = os.path.join(data_dir, "test_mask.pt")
    
    if not os.path.exists(train_mask_path):
        raise FileNotFoundError(f"训练掩码文件不存在: {train_mask_path}")
    if not os.path.exists(val_mask_path):
        raise FileNotFoundError(f"验证掩码文件不存在: {val_mask_path}")
    if not os.path.exists(test_mask_path):
        raise FileNotFoundError(f"测试掩码文件不存在: {test_mask_path}")
    
    train_mask = torch.load(train_mask_path, map_location="cpu", weights_only=False)
    val_mask = torch.load(val_mask_path, map_location="cpu", weights_only=False)
    test_mask = torch.load(test_mask_path, map_location="cpu", weights_only=False)
    

    if train_mask.dtype != torch.bool:
        train_mask = train_mask.bool()
    if val_mask.dtype != torch.bool:
        val_mask = val_mask.bool()
    if test_mask.dtype != torch.bool:
        test_mask = test_mask.bool()
    

    train_idx = torch.where(train_mask)[0]
    val_idx = torch.where(val_mask)[0]
    test_idx = torch.where(test_mask)[0]
    
    split_idx = {
        "train": train_idx,
        "valid": val_idx,
        "test": test_idx
    }
    
    print(f"   训练节点: {len(train_idx)}, 验证节点: {len(val_idx)}, 测试节点: {len(test_idx)}")
    

    link = None
    
    return graph, link, labels, split_idx


# =========================

# =========================
def build_model(args: argparse.Namespace) -> CAME:
    return CAME(
        dim_t5vit=args.text_dim,
        dim_clip=args.image_dim,
        hidden_1024=args.hidden_dim,
        out_1024=args.hidden_dim,
        num_heads=4,
        dropout=0.1,
        attn_drop=0.1,
        temperature=0.07,
        w12=1.0,
        w23=1.0,
        w13=1.0,
        symmetric_nce=True,
    )


def load_checkpoint_weights(model: CAME, checkpoint_path: str, device: torch.device) -> None:
    print(f"加载 checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    state = None
    if isinstance(ckpt, dict):
        if "model_state" in ckpt and isinstance(ckpt["model_state"], dict):
            state = ckpt["model_state"]
            print("    使用 ckpt['model_state'] 作为 state dict")
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state = ckpt["state_dict"]
            print("    使用 ckpt['state_dict'] 作为 state dict")


    if state is None:
        for candidate_key in ("model_state", "state_dict", "model_state_dict", "model", "state"):
            if isinstance(ckpt, dict) and candidate_key in ckpt:
                state = ckpt[candidate_key]
                print(f"    从 checkpoint 提取 state dict（键: '{candidate_key}'）")
                break
        if state is None:
            state = ckpt


    if isinstance(state, dict) and any(not isinstance(v, torch.Tensor) for v in state.values()):
        possible = None
        for k, v in state.items():
            if isinstance(v, dict) and all(isinstance(x, torch.Tensor) for x in v.values()):
                possible = v
                print(f"    发现嵌套参数 dict（键: '{k}'），将其作为 state dict")
                break
        if possible is not None:
            state = possible


    model_state = {}
    if isinstance(state, dict):
        for k, v in state.items():
            model_state[k.replace("model.", "", 1) if k.startswith("model.") else k] = v
    else:
        raise RuntimeError("无法解析 checkpoint 中的 state_dict（格式不支持）")


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
        print(f" 加载过滤后 state_dict 时出错: {e}")
        raise

    if missing:
        print(f"    缺失的键 ({len(missing)}): {missing[:10]}")
    if unexpected:
        print(f"    意外的键 ({len(unexpected)}): {unexpected[:10]}")

    if skipped_shape:
        print(f"    跳过 {len(skipped_shape)} 个因形状不匹配的参数，示例: {skipped_shape[:8]}")
    if skipped_missing:
        print(f"    跳过 {len(skipped_missing)} 个因在模型中缺失或非 Tensor 的键，示例: {skipped_missing[:8]}")

    print(f" checkpoint 加载完成，实际加载参数数: {len(filtered_state)}")


def call_model_without_loss(model: CAME, graph: dgl.DGLGraph, feat_text: torch.Tensor, feat_image: torch.Tensor, embed_type: str = "z3") -> torch.Tensor:
    """
    调用模型进行前向传播，不计算损失，直接返回指定类型的嵌入。
    使用 getz1/getz2/getz3 参数直接获取嵌入，避免计算损失和创建字典。
    """

    getz1 = (embed_type == "z1")
    getz2 = (embed_type == "z2")
    getz3 = (embed_type == "z3")
    
    return model(graph, feat_text, feat_image, getz1=getz1, getz2=getz2, getz3=getz3)


def extract_embeddings(
    model: CAME,
    graph: dgl.DGLGraph,
    feats: dict,
    device: torch.device,
    mode: str = "full",
    batch_size: int = 32,
    clusters: Optional[List[torch.Tensor]] = None,
    embed_type: str = "z3",
) -> torch.Tensor:
    """提取节点 embedding

    Args:
        embed_type: 
            - "z1": 使用文本GAT输出（文本模态的嵌入）
            - "z2": 使用图像GAT输出（图像模态的嵌入）
            - "z3": 使用融合输出（默认，融合后的嵌入）
    """

    graph = graph.cpu()
    feats = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in feats.items()}
    
    if mode == "full":
        return _embed_full_graph(model, graph, feats, device, embed_type=embed_type)
    return _embed_subgraph(model, graph, feats, device, batch_size, clusters, embed_type=embed_type)


def _embed_full_graph(
    model: CAME,
    graph: dgl.DGLGraph,
    feats: dict,
    device: torch.device,
    embed_type: str = "z3",
) -> torch.Tensor:
    """整图模式：一次性在全图上前向"""
    print(f" 整图模式：一次性在全图上前向 (CAME)，使用嵌入类型: {embed_type}...")
    

    feats_gpu = {}
    for k, v in feats.items():
        if isinstance(v, torch.Tensor):

            if v.device != device:
                feats_gpu[k] = v.to(device, non_blocking=True)
            else:
                feats_gpu[k] = v
        else:
            feats_gpu[k] = v
    
    graph_gpu = graph.to(device)
    

    feature_t5vit = feats_gpu.get("text", feats_gpu.get("image"))
    feature_clip = feats_gpu.get("image", feats_gpu.get("text"))
    
    with torch.no_grad():
        emb_gpu = call_model_without_loss(model, graph_gpu, feature_t5vit, feature_clip, embed_type)
        emb = emb_gpu.cpu()
    

    del feats_gpu, emb_gpu
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    print(f" 嵌入形状: {emb.shape} ({embed_type})")
    return emb


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


def _embed_subgraph(
    model: CAME,
    graph: dgl.DGLGraph,
    feats: dict,
    device: torch.device,
    batch_size: int,
    clusters: List[torch.Tensor],
    embed_type: str = "z3",
) -> torch.Tensor:
    """流式子图模式：按预计算簇批次前向"""
    print(f" 流式子图模式：按预计算簇批次前向 (CAME)，使用嵌入类型: {embed_type}...")
    N = graph.num_nodes()
    expected_dim = get_CAME_embed_dim(model, embed_type)
    emb = torch.zeros(N, expected_dim, device="cpu")
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
            h_gpu = call_model_without_loss(model, sub_g_gpu, sub_feat_text_gpu, sub_feat_img_gpu, embed_type)
        

        if emb.shape[1] != h_gpu.shape[1]:
            expected_dim = h_gpu.shape[1]
            emb = torch.zeros(N, expected_dim, device="cpu")
            print(f"    更新嵌入维度: {emb.shape} ({embed_type})")
        
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
            print(f"   已处理 {batch_idx + 1} 个 batch...")
    

    covered = torch.zeros(N, dtype=torch.bool)
    for idx in range(len(clusters)):
        cluster = clusters[idx]
        if cluster.numel() > 0:
            center = int(cluster[0].item())
            if center < N:
                covered[center] = True
    
    missing = (~covered).sum().item()
    if missing > 0:
        print(f"    警告：{missing} 个节点未出现在任何簇中，其embedding保持为零")
    
    print(f" 优化后的嵌入形状: {emb.shape}")
    return emb


def _batch(arr: torch.Tensor, bs: int):
    """批处理生成器"""
    for i in range(0, len(arr), bs):
        yield arr[i : i + bs]


def _collect_nodes(
    center_ids: torch.Tensor,
    clusters: List[torch.Tensor],
) -> torch.Tensor:
    """从预计算簇中收集节点"""
    node_sets = [clusters[int(i)] for i in center_ids.tolist()]
    return torch.unique(torch.cat(node_sets))


def load_clusters_from_ppr(ppr_path: str) -> List[torch.Tensor]:
    """从PPR map生成簇列表"""
    print(f" 从PPR map加载: {ppr_path}")
    ppr_map = torch.load(ppr_path, map_location="cpu")
    
    if not isinstance(ppr_map, dict):
        raise ValueError(f"PPR map格式不正确，期望dict，得到{type(ppr_map)}")
    
    print(f"   找到 {len(ppr_map)} 个节点的PPR结果")
    

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
    
    print(f"   生成了 {len(clusters)} 个簇")
    return clusters


def load_clusters(cluster_path: Optional[str], ppr_path: Optional[str] = None) -> Optional[List[torch.Tensor]]:
    """加载预计算簇（优先使用PPR map）"""

    if ppr_path:
        return load_clusters_from_ppr(ppr_path)
    

    if not cluster_path:
        return None
    
    print(f" 加载预计算簇: {cluster_path}")
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
    print(f"   共 {len(clusters)} 个簇")
    return clusters
    print(f"   嵌入统计（归一化前）: mean={emb.mean().item():.4f}, std={emb.std().item():.4f}, "
          f"min={emb.min().item():.4f}, max={emb.max().item():.4f}")
    

    l2_norms = emb.norm(p=2, dim=1)
    print(f"   L2 范数统计（归一化前）: mean={l2_norms.mean().item():.6f}, std={l2_norms.std().item():.6f}, "
          f"min={l2_norms.min().item():.6f}, max={l2_norms.max().item():.6f}")
    




    per_dim_std = emb.std(dim=0)
    print(f"   各维度标准差统计: mean={per_dim_std.mean().item():.6f}, std={per_dim_std.std().item():.6f}, "
          f"min={per_dim_std.min().item():.6f}, max={per_dim_std.max().item():.6f}")
    

    constant_dims = (per_dim_std < 1e-6).sum().item()
    if constant_dims > 0:
        print(f"    警告: {constant_dims} 个维度的标准差 < 1e-6（可能退化）")
    

    num_nan = torch.isnan(emb).sum().item()
    num_inf = torch.isinf(emb).sum().item()
    if num_nan > 0:
        print(f"    警告: 嵌入中包含 {num_nan} 个 NaN")
    if num_inf > 0:
        print(f"    警告: 嵌入中包含 {num_inf} 个 Inf")
    
    return emb


# =========================

# =========================
class InContextDataset(torch.utils.data.IterableDataset):
    """Few-shot 任务数据集"""

    def __init__(
        self,
        emb: torch.Tensor,
        link: Dict,
        labels: torch.Tensor,
        split_idx: Dict[str, torch.Tensor],
        eval_tasks: int = 500,
        num_label: int = 2,
        num_support: int = 3,
        num_query: int = 3,
        random_seed: int = 42,
    ):
        super(InContextDataset, self).__init__()
        self.num_label = num_label
        self.num_support = num_support
        self.num_query = num_query
        self.total_steps = eval_tasks
        self.link = link
        self.random_seed = random_seed
        

        if link is not None:

            all_pairs_list = []
            for split_name in ["train", "valid", "test"]:
                if split_name in link and len(link[split_name]) > 0:
                    pairs = link[split_name][0]
                    if isinstance(pairs, torch.Tensor):
                        all_pairs_list.append(pairs)
                    elif isinstance(pairs, list):
                        all_pairs_list.append(torch.LongTensor(pairs))

            if all_pairs_list:
                all_pairs = torch.cat(all_pairs_list, dim=0) if len(all_pairs_list) > 1 else all_pairs_list[0]
            else:
                all_pairs = []
            if len(all_pairs) == 0:

                node_pairs = torch.empty((0, 2), dtype=torch.long)

                if emb.dim() == 2:
                    self.emb = torch.empty((0, emb.shape[1] * 2), dtype=emb.dtype)
                else:
                    raise ValueError(f"期望 emb 是2维张量，但得到 {emb.dim()} 维，形状: {emb.shape}")
            else:

                if not isinstance(all_pairs, torch.Tensor):
                    all_pairs = torch.LongTensor(all_pairs)
                node_pairs = all_pairs

                if emb.dim() == 1:
                    raise ValueError(f"emb 是1维张量（形状: {emb.shape}），无法用于索引。期望2维张量 [num_nodes, embedding_dim]")
                self.emb = torch.cat([emb[node_pairs[:, 0]], emb[node_pairs[:, 1]]], dim=1)
            print(f"   KG 任务：使用节点对拼接嵌入，形状: {self.emb.shape}")
        else:
            self.emb = emb
            print(f"   节点分类任务：使用节点嵌入，形状: {self.emb.shape}")
        
        self.labels = labels
        self.split_idx = split_idx
        


        self.label_dict = {}
        for split in ["train", "valid", "test"]:
            for idx in split_idx[split]:

                idx_val = idx.item() if isinstance(idx, torch.Tensor) else idx
                label = labels[idx_val].item() if isinstance(labels[idx_val], torch.Tensor) else labels[idx_val]
                

                if not np.isnan(label):
                    if label not in self.label_dict:
                        self.label_dict[label] = {"train": [], "valid": [], "test": []}

                    self.label_dict[label][split].append(idx_val)
        
        self.total_labels = len(self.label_dict)
        print(f"   总类别数: {self.total_labels}")
    
    def generate_batch(self, batch_type="mt"):
        """生成一个 few-shot 任务 batch"""
        def sample(sample_list, size):
            if len(sample_list) >= size:
                return np.random.choice(sample_list, size=size, replace=False).tolist()
            return np.random.choice(sample_list, size=size, replace=True).tolist()

        m = self.num_label
        k = self.num_support
        n = self.num_query



        if not hasattr(self, '_batch_counter'):
            self._batch_counter = 0
        self._batch_counter += 1


        current_seed = hash((self.random_seed, self._batch_counter, batch_type)) % 2**32
        np.random.seed(current_seed)



        current_labels = np.random.choice(range(self.total_labels), m, replace=False)
        

        while True:
            flag = True
            for label in current_labels:

                actual_label = list(self.label_dict.keys())[label]
                if len(self.label_dict[actual_label]["train"]) == 0 or \
                   len(self.label_dict[actual_label]["test"]) == 0:
                    flag = False
                    break
            if flag:
                break
            else:
                current_labels = np.random.choice(range(self.total_labels), m, replace=False)
        

        support_examples = []
        query_examples = []
        support_labels = []
        query_labels = []
        
        for idx, label in enumerate(current_labels):


            actual_label = list(self.label_dict.keys())[label]


            train_samples = self.label_dict[actual_label]["train"]
            test_samples = self.label_dict[actual_label]["test"]


            support_sample_ids = sample(train_samples, k)

            query_sample_ids = sample(test_samples, n)


            for sample_id in support_sample_ids:
                support_examples.append(self.emb[sample_id])
                support_labels.append(idx)


            for sample_id in query_sample_ids:
                query_examples.append(self.emb[sample_id])
                query_labels.append(idx)
            

            if self.link is None:
                train_set = set(support_sample_ids)
                test_set = set(query_sample_ids)
                overlap = train_set & test_set
                if len(overlap) > 0:
                    print(f" 警告: 类别 {actual_label} 的 support 和 query 有重叠: {overlap}")
        
        batch = {
            "support_examples": torch.stack(support_examples, dim=0),  # [m*k, dim]
            "query_examples": torch.stack(query_examples, dim=0),       # [m*n, dim]
            "support_labels": torch.LongTensor(support_labels),         # [m*k]
            "query_labels": torch.LongTensor(query_labels),             # [m*n]
        }
        return batch
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = max(worker_info.num_workers, 1) if worker_info else 1
        for _ in range(self.total_steps // num_workers):
            yield self.generate_batch(batch_type="mt")
    
    def __len__(self):
        return self.total_steps


def setup_incontext_dataloader(
    emb: torch.Tensor,
    link: Dict,
    labels: torch.Tensor,
    split_idx: Dict[str, torch.Tensor],
    eval_tasks: int = 500,
    num_label: int = 2,
    num_support: int = 3,
    num_query: int = 3,
    num_workers: int = 4,
    random_seed: int = 42,
):
    """设置 Few-shot 数据加载器"""
    eval_dataset = InContextDataset(
        emb, link, labels, split_idx,
        eval_tasks=eval_tasks,
        num_label=num_label,
        num_support=num_support,
        num_query=num_query,
        random_seed=random_seed,
    )
    dataloader = DataLoader(eval_dataset, batch_size=None, num_workers=num_workers)
    return dataloader


def set_random_seed(seed: int):
    """设置随机种子"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def incontext_evaluate(
    emb: torch.Tensor,
    link: Dict,
    labels: torch.Tensor,
    split_idx: Dict[str, torch.Tensor],
    dataset_name: str,
    eval_tasks: int = 500,
    num_label: int = 2,
    num_support: int = 3,
    num_query: int = 3,
    seeds: List[int] = [0, 1, 2],
    num_workers: int = 4,
):
    """执行 Few-shot 评估
    
    Args:
        emb: 节点嵌入 [num_nodes, hidden_dim]
        link: 链接数据（KG 任务）或 None（节点分类任务）
        labels: 标签
        split_idx: 分割索引
        dataset_name: 数据集名称
        eval_tasks: 评估任务数
        num_label: 每个任务的类别数
        num_support: 每个类别的 support 样本数（K-shot）
        num_query: 每个类别的 query 样本数
        seeds: 随机种子列表
        num_workers: 数据加载器工作进程数
    """
    print("\n" + "=" * 60)
    print(" 开始 Few-shot 评估")
    print("=" * 60)
    print(f"   配置: {num_label}-way {num_support}-shot")
    print(f"   任务数: {eval_tasks}")
    print(f"   随机种子: {seeds}")
    
    all_task_accuracies = []
    acc_list = []

    for seed_idx, seed in enumerate(seeds):
        print(f"\n####### Run seed {seed} for In-Context Evaluation...")


        eval_dataset = InContextDataset(
            emb=emb,
            link=link,
            labels=labels,
            split_idx=split_idx,
            eval_tasks=eval_tasks,
            num_label=num_label,
            num_support=num_support,
            num_query=num_query,
            random_seed=seed,
        )

        dataloader = DataLoader(eval_dataset, batch_size=None, num_workers=num_workers)

        print(f"    种子 {seed}: 创建数据加载器，开始评估...")


        model = PrototypicalNetwork()
        task_accuracies = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"评估任务 (种子 {seed})"):
                support_examples = batch["support_examples"]
                query_examples = batch["query_examples"]
                support_labels = batch["support_labels"]
                query_labels = batch["query_labels"]


                acc = model.evaluate_episode(support_examples, support_labels, query_examples, query_labels)
                task_accuracies.append(acc)


        if task_accuracies:
            seed_mean = np.mean(task_accuracies)
            seed_std = np.std(task_accuracies)
            seed_var = np.var(task_accuracies)
            print(f"    种子 {seed}: 准确率 = {seed_mean:.4f}  {seed_std:.4f} task_var={seed_var:.6f} ({len(task_accuracies)} 个任务)")
        else:
            seed_mean = 0.0
            seed_std = 0.0
            print(f"    种子 {seed}: 没有成功评估任何任务")


        all_task_accuracies.extend(task_accuracies)


        acc_list.append(seed_mean)

    if not all_task_accuracies:
        raise RuntimeError("没有成功创建任何 Few-shot 任务")


    final_mean = np.mean(all_task_accuracies)
    final_std = np.std(all_task_accuracies)
    final_var = np.var(all_task_accuracies)
    seed_mean_std = np.std(acc_list) if acc_list else 0.0

    print(f"\n 最终结果:")
    print(f"   平均准确率: {final_mean:.4f}  {final_std:.4f} (task-level)")
    print(f"   task方差: {final_var:.6f}")
    print(f"   seed均值标准差: {seed_mean_std:.4f}")
    print(f"   总任务数: {len(all_task_accuracies)}")
    print(f"   种子数: {len(seeds)}")
    print(f"   每个种子任务数: {eval_tasks}")

    return final_mean, final_std, final_var


# =========================

# =========================
def main():
    parser = argparse.ArgumentParser(description="Few-shot 评估脚本")
    

    parser.add_argument("--dataset", type=str, required=True,
                        help="数据集名称（KG任务: FB15K237, WN18RR; 节点分类任务: ele-fashion, ogbn-arxiv 等）")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="数据集目录（KG任务: 包含 graph.bin, train.txt, valid.txt, test.txt; 节点分类任务: 包含 graph.bin, labels.pt, train_mask.pt, val_mask.pt, test_mask.pt）")
    

    parser.add_argument("--checkpoint", type=str, default=None,
                        help="CAME checkpoint 路径（如果为 None，则使用随机初始化的权重）")
    parser.add_argument("--hidden-dim", type=int, default=1024, help="隐藏层维度")
    parser.add_argument("--text-dim", type=int, default=512, help="文本模态输入维度")
    parser.add_argument("--image-dim", type=int, default=512, help="图像模态输入维度")
    

    parser.add_argument(
        "--embed-type",
        choices=["z1", "z2", "z3"],
        default="z3",
        help="使用的嵌入类型：z1（文本GAT输出）、z2（图像GAT输出）、z3（融合输出，默认）",
    )
    

    parser.add_argument("--feat-name", type=str, default="text_feature.pt",
                        help="主特征文件名（用于 baseline 对比）")
    parser.add_argument("--text-feat-name", type=str, default=None,
                        help="文本特征文件名（如果为 None，则自动检测或使用 feat-name）")
    parser.add_argument("--image-feat-name", type=str, default=None,
                        help="图像特征文件名（如果为 None，则自动检测或使用 feat-name）")
    parser.add_argument("--compare-baseline", action="store_true",
                        help="对比原始 CLIP 特征和 CAME 优化后的嵌入")
    parser.add_argument("--baseline-merge", choices=["mean", "concat", "text", "image"],
                        default="mean", help="baseline 特征合并方式：mean(默认)/concat/text/image")
    

    parser.add_argument("--eval-num-label", type=int, default=2,
                        help="每个任务的类别数（N-way）")
    parser.add_argument("--eval-num-support", type=int, default=3,
                        help="每个类别的 support 样本数（K-shot）")
    parser.add_argument("--eval-num-query", type=int, default=3,
                        help="每个类别的 query 样本数")
    parser.add_argument("--eval-tasks", type=int, default=500,
                        help="评估任务数")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                        help="随机种子列表")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="数据加载器工作进程数")
    

    parser.add_argument("--device", type=str, default="cuda",
                        help="设备 (cuda/cpu)")
    

    parser.add_argument("--model-seed", type=int, default=42,
                        help="模型权重初始化的随机种子（当没有 checkpoint 时使用，默认 42）")
    

    parser.add_argument("--mode", choices=["full", "stream"], default="full",
                        help="评估模式：整图/流式子图")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="流式子图模式下的中心节点 batch 大小")
    parser.add_argument("--cluster-path", type=str, default=None,
                        help="预计算节点簇路径 (clusters.pt，stream 模式可选)")
    parser.add_argument("--ppr-path", type=str, default=None,
                        help="PPR map文件路径 (ppr.pt，stream 模式可选，优先于cluster-path)")
    parser.add_argument("--normalize-emb", action="store_false",
                       help="在评估前对嵌入进行 L2 归一化（默认关闭，评估函数内部会归一化）")


    parser.add_argument("--clip", action="store_true", help="使用原始 CLIP 特征进行评估，不进入 CAME 模型")
    parser.add_argument("--clip-feat-type", choices=["text", "image", "both", "custom"], default="image",
                        help="当使用--clip时，选择使用的特征类型：text（文本特征）、image（图像特征）、both（均值）、custom（使用feat-name指定的文件）")
    
    args = parser.parse_args()
    device = torch.device(args.device)
    
    print("=" * 60)
    print("Few-shot 评估脚本")
    print("=" * 60)
    print(f" 使用设备: {device}")
    print(f" 数据集: {args.dataset}")
    print(f"📁 数据目录: {args.data_dir}")
    

    print("\n 加载数据...")


    kg_datasets = ["FB15K237", "WN18RR"]
    is_kg_task = args.dataset in kg_datasets

    if is_kg_task:
        print("   任务类型: KG 链接预测")
        graph, link, labels, split_idx = load_kg_data(args.data_dir)
    else:
        print("   任务类型: 节点分类")
        if args.clip:
            print("    CLIP 模式：跳过图数据加载，只加载标签和分割信息")
            graph, link, labels, split_idx = load_node_classification_data_light(args.data_dir)
        else:
            graph, link, labels, split_idx = load_node_classification_data(args.data_dir)
    

    print("\n 加载特征...")
    
    def load_feature_file(file_path: str, file_name: str) -> torch.Tensor:
        """加载特征文件"""
        full_path = os.path.join(args.data_dir, file_name)
        if not os.path.exists(full_path):
            return None
        data = torch.load(full_path, map_location="cpu", weights_only=False)
        if isinstance(data, dict):
            feat = data.get("features", data.get("feat", None))
            if feat is None:
                raise ValueError(f"特征字典中未找到特征数据，可用键: {list(data.keys())}")
            return feat
        elif isinstance(data, torch.Tensor):
            return data
        else:
            raise TypeError(f"不支持的特征文件格式: {type(data)}")
    

    if args.text_feat_name:
        text_feat_path = args.text_feat_name
    else:

        if os.path.exists(os.path.join(args.data_dir, "image_feature.pt")):
            text_feat_path = "image_feature.pt"
            print(f"   自动检测到文本特征: {text_feat_path}")
        else:
            text_feat_path = args.feat_name
    

    if args.image_feat_name:
        image_feat_path = args.image_feat_name
    else:

        if os.path.exists(os.path.join(args.data_dir, "text_feature.pt")):
            image_feat_path = "text_feature.pt"
            print(f"   自动检测到图像特征: {image_feat_path}")
        else:
            image_feat_path = args.feat_name
    

    text_feat = load_feature_file(args.data_dir, text_feat_path)
    if text_feat is None:
        raise FileNotFoundError(f"文本特征文件不存在: {os.path.join(args.data_dir, text_feat_path)}")
    print(f"   文本特征: {text_feat_path}, 形状: {text_feat.shape}")
    

    if image_feat_path == text_feat_path:

        image_feat = text_feat.clone()
        print(f"   图像特征: 使用文本特征作为占位（{image_feat_path}）")
    else:
        image_feat = load_feature_file(args.data_dir, image_feat_path)
        if image_feat is None:
            image_feat = text_feat.clone()
            print(f"    图像特征文件不存在，使用文本特征作为占位")
        else:
            print(f"   图像特征: {image_feat_path}, 形状: {image_feat.shape}")
    
    feats = {"text": text_feat, "image": image_feat}
    

    num_nodes = graph.num_nodes() if graph is not None else labels.shape[0]
    for modality in ['text', 'image']:
        if modality in feats and isinstance(feats[modality], torch.Tensor):
            feat_nodes = feats[modality].shape[0]
            if feat_nodes != num_nodes:
                if feat_nodes > num_nodes:
                    feats[modality] = feats[modality][:num_nodes]
                else:
                    padding = torch.zeros(num_nodes - feat_nodes, feats[modality].shape[1],
                                         dtype=feats[modality].dtype)
                    feats[modality] = torch.cat([feats[modality], padding], dim=0)
    
    print(f"   文本特征形状: {feats['text'].shape}")
    print(f"   图像特征形状: {feats['image'].shape}")
    

    print("\n🤖 构建并加载模型...")
    
    if args.checkpoint:

        model = build_model(args).to(device)
        load_checkpoint_weights(model, args.checkpoint, device)
        print("    使用训练后的模型权重")
    else:

        print(f"    未指定 checkpoint，使用随机初始化的模型权重（种子: {args.model_seed}）")
        print("    这将用于对比随机初始化 vs 训练后的模型性能")
        

        torch.manual_seed(args.model_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.model_seed)
        

        model = build_model(args).to(device)
        
        print(f"    模型权重已使用固定种子 {args.model_seed} 初始化（可复现）")
    

    model.eval()
    print(f"    评估模式设置: model.eval() (CAME 的 dropout 在 GAT 和 MLP 内部，已自动禁用)")
    

    if args.checkpoint:


        has_nonzero = False
        for name, param in model.named_parameters():
            if any(key in name for key in ['gat_t5vit', 'gat_clip', 'fuse_mlp', 'proj_t5vit', 'proj_clip']):
                if param.data.abs().max().item() > 1e-6:
                    has_nonzero = True
                    break
        if not has_nonzero:
            print("    警告: 模型关键层参数可能全为零，请检查 checkpoint 加载是否正确")
        else:
            print("    模型参数检查: 关键层参数已加载（非零）")
    

    if args.compare_baseline:
        print("\n" + "=" * 60)
        print(" Baseline: 使用原始 CLIP 特征")
        print("=" * 60)
        

        merge_mode = args.baseline_merge
        if "text" in feats and "image" in feats:
            if merge_mode == "mean":
                baseline_emb = (feats["text"] + feats["image"]) / 2.0
                print(f"    Baseline 使用: text 和 image 特征的平均")
            elif merge_mode == "concat":
                baseline_emb = torch.cat([feats["text"], feats["image"]], dim=1)
                print(f"    Baseline 使用: text 和 image 特征的拼接")
            elif merge_mode == "text":
                baseline_emb = feats["text"].clone()
                print(f"    Baseline 使用: text 特征（忽略 image）")
            elif merge_mode == "image":
                baseline_emb = feats["image"].clone()
                print(f"    Baseline 使用: image 特征（忽略 text）")
            else:
                raise ValueError(f"未知的 baseline 合并方式: {merge_mode}")
            print(f"    特征文件: text={args.feat_name}, image={image_feat_path}")
        elif "image" in feats:
            baseline_emb = feats["image"].clone()
            print(f"    Baseline 使用: image 特征（text 特征不可用）")
        elif "text" in feats:
            baseline_emb = feats["text"].clone()
            print(f"    Baseline 使用: text 特征（image 特征不可用）")
        else:
            raise ValueError("未找到可用的特征（text 或 image）")
        

        print(f"    Baseline 特征统计: mean={baseline_emb.mean().item():.6f}, "
              f"std={baseline_emb.std().item():.6f}, "
              f"min={baseline_emb.min().item():.6f}, max={baseline_emb.max().item():.6f}")
        

        if baseline_emb.shape[0] != graph.num_nodes():
            if baseline_emb.shape[0] > graph.num_nodes():
                baseline_emb = baseline_emb[:graph.num_nodes()]
            else:
                padding = torch.zeros(graph.num_nodes() - baseline_emb.shape[0], 
                                     baseline_emb.shape[1], dtype=baseline_emb.dtype)
                baseline_emb = torch.cat([baseline_emb, padding], dim=0)
        

        if args.normalize_emb:
            print(f"    对 baseline 嵌入进行 L2 归一化...")
            baseline_emb = baseline_emb / torch.norm(baseline_emb, dim=1, keepdim=True)
            print(f"    Baseline 归一化完成，L2 范数均值: {torch.norm(baseline_emb, dim=1).mean().item():.6f}")
            print(f"   原始 CLIP 特征形状: {baseline_emb.shape}")
            print(f"   特征范围（归一化后）: [{baseline_emb.min().item():.4f}, {baseline_emb.max().item():.4f}]")
        else:


            print(f"   原始 CLIP 特征形状: {baseline_emb.shape}")
            print(f"   特征范围（归一化前）: [{baseline_emb.min().item():.4f}, {baseline_emb.max().item():.4f}]")
            print("    注意：不在输入前归一化，评估函数内部会统一归一化")
        

        print(f"\n Baseline: 使用原始特征进行 Few-shot 评估")
        baseline_mean_acc, baseline_std_acc, baseline_var_acc = incontext_evaluate(
            emb=baseline_emb,
            link=link,
            labels=labels,
            split_idx=split_idx,
            dataset_name=args.dataset,
            eval_tasks=args.eval_tasks,
            num_label=args.eval_num_label,
            num_support=args.eval_num_support,
            num_query=args.eval_num_query,
            seeds=args.seeds,
            num_workers=args.num_workers,
        )
    else:

        baseline_mean_acc = None
        baseline_std_acc = None
        baseline_var_acc = None

    if args.clip:
        print("\n" + "=" * 60)
        print(" 使用原始 CLIP 特征（跳过模型前向传播）")
        print("=" * 60)


        if args.clip_feat_type == "image":
            if "image" not in feats:
                raise ValueError("未找到 CLIP 图像特征，请确保数据目录中包含 text_feature.pt 文件")
            node_emb = feats["image"].clone()
            print(f" 使用 CLIP 图像特征作为节点嵌入，形状: {node_emb.shape}")
        elif args.clip_feat_type == "text":
            if "text" not in feats:
                raise ValueError("未找到 CLIP 文本特征，请确保数据目录中包含 image_feature.pt 文件")
            node_emb = feats["text"].clone()
            print(f" 使用 CLIP 文本特征作为节点嵌入，形状: {node_emb.shape}")
        elif args.clip_feat_type == "both":
            if "text" not in feats or "image" not in feats:
                raise ValueError("未找到 CLIP 文本或图像特征，请确保数据目录中包含相应特征文件")
            node_emb = (feats["text"] + feats["image"]) / 2.0
            print(f" 使用 CLIP 文本和图像特征的均值作为节点嵌入，形状: {node_emb.shape}")
        elif args.clip_feat_type == "custom":

            custom_feat_path = os.path.join(args.data_dir, args.feat_name)
            if not os.path.exists(custom_feat_path):
                raise ValueError(f"未找到自定义特征文件: {custom_feat_path}")
            custom_feat = load_feature_file(args.data_dir, args.feat_name)
            if custom_feat is None:
                raise ValueError(f"无法加载自定义特征文件: {custom_feat_path}")
            node_emb = custom_feat.clone()
            print(f" 使用自定义特征文件作为节点嵌入，形状: {node_emb.shape}")
        else:
            raise ValueError(f"未知的特征类型: {args.clip_feat_type}")
    else:
        print("\n" + "=" * 60)
        print(" CAME: 使用优化后的嵌入")
        print("=" * 60)
        print(" 提取节点嵌入...")

        if args.mode == "stream":
            if args.ppr_path is None and args.cluster_path is None:
                raise ValueError("流式模式必须提供 --ppr-path 或 --cluster-path 参数")
            clusters = load_clusters(args.cluster_path, args.ppr_path)
            if clusters is None:
                raise ValueError("无法加载簇数据")
            node_emb = extract_embeddings(
                model, graph, feats, device,
                mode="stream", batch_size=args.batch_size, clusters=clusters,
                embed_type=args.embed_type,
            )
        else:
            node_emb = extract_embeddings(
                model, graph, feats, device,
                mode="full", batch_size=args.batch_size, clusters=None,
                embed_type=args.embed_type,
            )
    

    expected_nodes = graph.num_nodes() if graph is not None else labels.shape[0]
    if node_emb.shape[0] != expected_nodes:
        raise ValueError(f"节点嵌入数量 ({node_emb.shape[0]}) 与节点数 ({expected_nodes}) 不一致！")


    if args.normalize_emb:
        print(f"    对嵌入进行 L2 归一化...")
        node_emb = node_emb / torch.norm(node_emb, dim=1, keepdim=True)
        print(f"    归一化完成，L2 范数均值: {torch.norm(node_emb, dim=1).mean().item():.6f}")
        print(f"   节点嵌入形状: {node_emb.shape}, 范围（归一化后）: [{node_emb.min().item():.4f}, {node_emb.max().item():.4f}]")
    else:


        print(f"   节点嵌入形状: {node_emb.shape}, 范围（归一化前）: [{node_emb.min().item():.4f}, {node_emb.max().item():.4f}]")
        print("    注意：不在输入前归一化，评估函数内部会统一归一化")
    

    node_norms = node_emb.norm(p=2, dim=1)
    print(f"   L2 范数统计（归一化前）: mean={node_norms.mean().item():.6f}, std={node_norms.std().item():.6f}, "
          f"min={node_norms.min().item():.6f}, max={node_norms.max().item():.6f}")
    

    node_per_dim_std = node_emb.std(dim=0)
    print(f"   各维度标准差统计: mean={node_per_dim_std.mean().item():.6f}, std={node_per_dim_std.std().item():.6f}, "
          f"min={node_per_dim_std.min().item():.6f}, max={node_per_dim_std.max().item():.6f}")
    

    if args.compare_baseline and 'baseline_emb' in locals():
        baseline_per_dim_std = baseline_emb.std(dim=0)
        print(f"\n    对比分析:")
        print(f"      CLIP 值范围: [{baseline_emb.min().item():.4f}, {baseline_emb.max().item():.4f}], "
              f"各维度标准差均值: {baseline_per_dim_std.mean().item():.6f}")
        print(f"      CAME 值范围: [{node_emb.min().item():.4f}, {node_emb.max().item():.4f}], "
              f"各维度标准差均值: {node_per_dim_std.mean().item():.6f}")
        

        clip_range = baseline_emb.max().item() - baseline_emb.min().item()
        CAME_range = node_emb.max().item() - node_emb.min().item()
        if CAME_range < clip_range * 0.5:
            print(f"       警告: CAME 的值范围 ({CAME_range:.4f}) 明显小于 CLIP ({clip_range:.4f})")
            print(f"         这可能说明 CAME 的嵌入被压缩或退化，导致信息丢失")
    

    if args.clip:
        print(f"\n CLIP: 使用原始 CLIP 特征进行 Few-shot 评估")
    else:
        print(f"\n CAME: 使用优化后的嵌入进行 Few-shot 评估")

    try:
        CAME_mean_acc, CAME_std_acc, CAME_var_acc = incontext_evaluate(
            emb=node_emb,
            link=link,
            labels=labels,
            split_idx=split_idx,
            dataset_name=args.dataset,
            eval_tasks=args.eval_tasks,
            num_label=args.eval_num_label,
            num_support=args.eval_num_support,
            num_query=args.eval_num_query,
            seeds=args.seeds,
            num_workers=args.num_workers,
        )
        print(f"    评估结果: {CAME_mean_acc:.4f}{CAME_std_acc:.4f} task_var={CAME_var_acc:.6f}")
    except Exception as e:
        print(f"    评估失败: {e}")
        CAME_mean_acc = 0.0
        CAME_std_acc = 0.0
        CAME_var_acc = 0.0
    
    print("\n" + "=" * 60)
    if args.clip:
        print(" Few-shot 评估完成（使用原始 CLIP 特征）")
    else:
        print(" Few-shot 评估完成（CAME 优化后）")
    print("=" * 60)
    

    random_baseline = 1.0 / args.eval_num_label
    
    if args.clip:

        print(f"   最终准确率: {CAME_mean_acc:.4f}{CAME_std_acc:.4f} task_var={CAME_var_acc:.6f}")
        improvement = (CAME_mean_acc - random_baseline) / random_baseline * 100
        print(f"\n   随机猜测基线: {random_baseline:.4f} ({random_baseline*100:.1f}%)")
        print(f"   相对提升: {improvement:+.1f}%")
    elif args.compare_baseline:

        print(f"\n 对比结果:")
        print(f"   {'方法':<20} {'准确率':<20} {'相对随机提升':<20}")
        print(f"   {'-'*60}")
        print(f"   {'随机猜测':<20} {random_baseline:.4f} ({random_baseline*100:.1f}%){'':<15}")
        baseline_improvement = (baseline_mean_acc - random_baseline) / random_baseline * 100
        print(f"   {'原始 CLIP':<20} {baseline_mean_acc:.4f}{baseline_std_acc:.4f} task_var={baseline_var_acc:.6f} {baseline_improvement:+.1f}%")
        CAME_improvement = (CAME_mean_acc - random_baseline) / random_baseline * 100
        print(f"   {'CAME':<20} {CAME_mean_acc:.4f}{CAME_std_acc:.4f} task_var={CAME_var_acc:.6f} {CAME_improvement:+.1f}%")


        relative_improvement = ((CAME_mean_acc - baseline_mean_acc) / baseline_mean_acc) * 100
        absolute_improvement = CAME_mean_acc - baseline_mean_acc
        print(f"\n    CAME 相对原始 CLIP:")
        print(f"      绝对提升: {absolute_improvement:+.4f}")
        print(f"      相对提升: {relative_improvement:+.2f}%")

        if relative_improvement > 0:
            print(f"    CAME 提升了嵌入质量！")
        elif relative_improvement < -1:
            print(f"    CAME 可能降低了嵌入质量，建议检查模型训练")
        else:
            print(f"   ➡ CAME 与原始 CLIP 性能相近")
    else:

        print(f"   最终准确率: {CAME_mean_acc:.4f}{CAME_std_acc:.4f} task_var={CAME_var_acc:.6f}")
        improvement = (CAME_mean_acc - random_baseline) / random_baseline * 100
        print(f"\n   随机猜测基线: {random_baseline:.4f} ({random_baseline*100:.1f}%)")
        print(f"   相对提升: {improvement:+.1f}%")
    
    print(f"\n   配置: {args.eval_num_label}-way {args.eval_num_support}-shot")
    print(f"   任务数: {args.eval_tasks}")
    print(f"   随机种子数: {len(args.seeds)}")
    

    if args.compare_baseline:
        if CAME_mean_acc < baseline_mean_acc - 0.01:
            print(f"\n    建议:")
            print(f"      - CAME 性能低于原始 CLIP，可能模型训练不充分")
            print(f"      - 检查 checkpoint 是否来自充分训练的模型")
            print(f"      - 检查训练 loss 是否收敛")
    elif CAME_mean_acc < 0.5:
        print(f"\n    建议:")
        if args.eval_num_support == 1:
            print(f"      - 1-shot 可能太少，建议尝试 3-shot 或 5-shot")
        print(f"      - 检查模型训练是否充分（checkpoint epoch）")
        print(f"      - 检查嵌入质量（是否已归一化，是否有异常值）")
        print(f"      - 尝试更少的类别数（如 2-way 或 3-way）")
        print(f"      - 使用 --compare-baseline 对比原始 CLIP 特征")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
