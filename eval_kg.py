#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CAME 知识图谱数据集评估脚本（FB15K237 / WN18RR）
- 加载 CAME checkpoint
- 提取节点嵌入（支持选择 z1、z2 或 z3）
- 评估 ACC（准确率）（KG 边分类）
"""

import argparse
import os
import copy
from collections import defaultdict

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from models.came import CAME
from utils.utils import LogisticRegression, accuracy, create_optimizer

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
    return _embed_full_graph(model, graph, feats, device, embed_type=embed_type)

def _embed_full_graph(
    model: CAME,
    graph: dgl.DGLGraph,
    feats: dict,
    device: torch.device,
    embed_type: str = "z3",
) -> torch.Tensor:
    """整图模式：一次性在全图上前向"""
    print(f"整图模式：一次性在全图上前向 (CAME)，使用嵌入类型: {embed_type}...")

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

    print(f"嵌入形状: {emb.shape} ({embed_type})")
    return emb

def linear_probing_for_transductive_node_classification(
    model, graph, num_nodes, feat, labels, split_idx, optimizer, max_epoch, device, mute=False
):
    """线性探测训练函数（用于边分类）"""
    criterion = torch.nn.CrossEntropyLoss()

    x = feat.to(device)
    labels = labels.to(device)

    train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

    if not torch.is_tensor(train_idx):
        train_idx = torch.as_tensor(train_idx).to(device)
        val_idx = torch.as_tensor(val_idx).to(device)
        test_idx = torch.as_tensor(test_idx).to(device)
    else:
        train_idx = train_idx.to(device)
        val_idx = val_idx.to(device)
        test_idx = test_idx.to(device)

    train_mask = torch.full((num_nodes,), False, device=device).index_fill_(0, train_idx, True)
    val_mask = torch.full((num_nodes,), False, device=device).index_fill_(0, val_idx, True)
    test_mask = torch.full((num_nodes,), False, device=device).index_fill_(0, test_idx, True)

    best_val_acc = 0
    best_val_epoch = 0
    best_model = None

    if not mute:
        epoch_iter = tqdm(range(max_epoch))
    else:
        epoch_iter = range(max_epoch)

    for epoch in epoch_iter:
        model.train()
        out = model(x)
        loss = criterion(out[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            pred = model(x)
            val_acc = accuracy(pred[val_mask], labels[val_mask])
            val_loss = criterion(pred[val_mask], labels[val_mask])
            test_acc = accuracy(pred[test_mask], labels[test_mask])
            test_loss = criterion(pred[test_mask], labels[test_mask])

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            best_model = copy.deepcopy(model)

        if not mute:
            epoch_iter.set_description(
                f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, val_loss:{val_loss.item(): .4f}, "
                f"val_acc:{val_acc:.4f}, test_loss:{test_loss.item(): .4f}, test_acc:{test_acc:.4f}"
            )

    best_model.eval()
    with torch.no_grad():
        pred = best_model(x)
        estp_test_acc = accuracy(pred[test_mask], labels[test_mask])

    if mute:
        print(
            f"# IGNORE: --- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, "
            f"Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- "
        )
    else:
        print(
            f"--- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, "
            f"Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- "
        )

    return test_acc, estp_test_acc, best_val_acc

def evaluate_kg_edge_classification(
    node_emb: torch.Tensor,
    train_triplets: torch.Tensor,  # [N_train, 3] (h, r, t)
    valid_triplets: torch.Tensor,  # [N_valid, 3] (h, r, t)
    test_triplets: torch.Tensor,   # [N_test, 3] (h, r, t)
    graph: dgl.DGLGraph,
    device: torch.device,
    lr_f: float = 0.01,
    weight_decay_f: float = 0.0,
    max_epoch_f: int = 30,
    mute: bool = False,
):
    """
    评估 KG 边分类的 ACC（准确率）

    Args:
        node_emb: 节点嵌入 [num_nodes, hidden_dim]
        train_triplets: 训练三元组 [N_train, 3] (h, r, t)
        valid_triplets: 验证三元组 [N_valid, 3] (h, r, t)
        test_triplets: 测试三元组 [N_test, 3] (h, r, t)
        graph: DGL 图
        device: 设备
        lr_f: 学习率
        weight_decay_f: 权重衰减
        max_epoch_f: 最大训练轮数
        mute: 是否静默模式
    """
    print("构建边对和标签...")

    all_triplets = torch.cat([train_triplets, valid_triplets, test_triplets], dim=0)

    node_pairs = all_triplets[:, [0, 2]]  # [N, 2] (h, t)

    labels = all_triplets[:, 1]  # [N] (r)

    n_train = len(train_triplets)
    n_valid = len(valid_triplets)
    n_test = len(test_triplets)

    split_idx = {
        "train": torch.arange(0, n_train),
        "valid": torch.arange(n_train, n_train + n_valid),
        "test": torch.arange(n_train + n_valid, n_train + n_valid + n_test),
    }

    print(f"训练边数: {n_train}")
    print(f"验证边数: {n_valid}")
    print(f"测试边数: {n_test}")
    print(f"总边数: {len(node_pairs)}")

    if len(labels) == 0:
        raise ValueError("没有三元组数据，无法进行评估。请检查三元组文件是否正确加载。")

    print(f"关系类别数: {int(labels.max().item() + 1)}")

    print("构建边特征...")

    node_pairs = node_pairs.to(device)
    labels = labels.to(device)
    node_emb = node_emb.to(device)

    max_node_id = node_emb.size(0) - 1
    valid_mask = (node_pairs[:, 0] <= max_node_id) & (node_pairs[:, 1] <= max_node_id)

    if valid_mask.sum() < len(node_pairs):
        print(f"警告：{len(node_pairs) - valid_mask.sum()} 条边的节点ID超出范围，将被过滤")
        node_pairs = node_pairs[valid_mask]
        labels = labels[valid_mask]
        n_train = (valid_mask[:n_train]).sum().item()
        n_valid = (valid_mask[n_train:n_train+n_valid]).sum().item()
        n_test = (valid_mask[n_train+n_valid:]).sum().item()
        split_idx = {
            "train": torch.arange(0, n_train),
            "valid": torch.arange(n_train, n_train + n_valid),
            "test": torch.arange(n_train + n_valid, n_train + n_valid + n_test),
        }

    x = torch.cat([node_emb[node_pairs[:, 0]], node_emb[node_pairs[:, 1]]], dim=1)  # [N, 2*hidden_dim]

    print(f"边特征形状: {x.shape}")
    print(f"split_idx: train={split_idx['train'].shape}, valid={split_idx['valid'].shape}, test={split_idx['test'].shape}")

    train_labels = labels[split_idx["train"]]
    unique, counts = torch.unique(train_labels, return_counts=True)
    print("训练集类别分布:")
    for u, c in zip(unique[:5], counts[:5]):
        print(f"类别 {u.item()}: {c.item()} 条 ({c.item()/len(train_labels)*100:.1f}%)")
    if len(unique) > 5:
        print(f"... (共 {len(unique)} 个类别)")

    if not mute:
        print("诊断嵌入质量（随机采样 100 对边）...")
        sample_idx = torch.randint(0, len(x), (min(100, len(x)),))
        sample_x = x[sample_idx]
        sample_labels = labels[sample_idx]

        cos_sim = F.cosine_similarity(sample_x.unsqueeze(1), sample_x.unsqueeze(0), dim=2)
        same_class_mask = (sample_labels.unsqueeze(1) == sample_labels.unsqueeze(0))
        diff_class_mask = ~same_class_mask
        same_class_mask.fill_diagonal_(False)

        if same_class_mask.sum() > 0:
            same_sim = cos_sim[same_class_mask].mean().item()
            print(f"同类边平均余弦相似度: {same_sim:.4f}")
        if diff_class_mask.sum() > 0:
            diff_sim = cos_sim[diff_class_mask].mean().item()
            print(f"不同类边平均余弦相似度: {diff_sim:.4f}")
            if same_class_mask.sum() > 0:
                print(f"区分度 (同类-不同类): {same_sim - diff_sim:.4f} (越大越好)")

    num_classes = int(labels.max().item() + 1)
    in_feat = x.shape[1]

    encoder = LogisticRegression(in_feat, num_classes)
    encoder.to(device)

    num_finetune_params = [p.numel() for p in encoder.parameters() if p.requires_grad]
    if not mute:
        print(f"微调参数数量: {sum(num_finetune_params)}")

    optimizer_f = create_optimizer("adam", encoder, lr_f, weight_decay_f)

    num_edge_nodes = x.shape[0]
    virtual_graph = dgl.graph(([], []), num_nodes=num_edge_nodes).to(device)

    print("训练边分类器...")
    final_acc, estp_acc, best_val_acc = linear_probing_for_transductive_node_classification(
        encoder, virtual_graph, num_edge_nodes, x, labels, split_idx,
        optimizer_f, max_epoch_f, device, mute
    )

    results = {
        "ACC": final_acc,
        "EarlyStop_ACC": estp_acc,
        "BestVal_ACC": best_val_acc,
    }

    return results

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

def load_kg_data(data_dir: str, dataset_name: str, use_mapping: bool = True):
    """加载 KG 数据集

    逻辑：
    - 如果 use_mapping=True: 使用映射文件将实体名/关系名转换为ID
    - 如果 use_mapping=False: 直接解析三元组中的整数ID（假设已对齐）
    """
    print(f"加载 KG 数据集: {dataset_name}")

    graph_path = os.path.join(data_dir, "graph.bin")
    graph = load_graph(graph_path)
    print(f"图: {graph.num_nodes()} 节点, {graph.num_edges()} 边")

    entity2id = None
    relation2id = None
    if use_mapping:
        mapping_path = os.path.join(data_dir, "entity_mapping.pt")
        if os.path.exists(mapping_path):
            print(f"📋 加载映射: {mapping_path}")
            mapping_data = torch.load(mapping_path, map_location="cpu", weights_only=False)
            entity2id = mapping_data.get("entity2id")
            relation2id = mapping_data.get("relation2id")
            if entity2id:
                print(f"实体映射: {len(entity2id)} 个")
            if relation2id:
                print(f"关系映射: {len(relation2id)} 个")
        else:
            print("未找到映射文件，将直接解析整数ID")
    else:
        print("不使用映射（直接解析整数ID）")

    auto_relation2id = {} if not use_mapping else None

    def load_triplets(file_path: str) -> torch.Tensor:
        """加载三元组文件

        逻辑：
        - 如果有映射：使用映射转换实体名/关系名 -> ID
        - 如果无映射：直接解析整数ID（实体和关系都是整数，或关系字符串自动分配ID）
        """
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return torch.empty((0, 3), dtype=torch.long)

        triplets = []

        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue

                h_str, r_str, t_str = parts[0], parts[1], parts[2]

                if use_mapping and entity2id:
                    h_id = entity2id.get(h_str)
                    t_id = entity2id.get(t_str)
                    if h_id is None or t_id is None:
                        continue

                    if relation2id:
                        r_id = relation2id.get(r_str)
                        if r_id is None:
                            continue
                    else:
                        r_id = int(r_str) if r_str.isdigit() else hash(r_str) % (2**31)
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

                triplets.append([h_id, r_id, t_id])

        return torch.tensor(triplets, dtype=torch.long) if triplets else torch.empty((0, 3), dtype=torch.long)

    train_path = os.path.join(data_dir, "train.txt")
    valid_path = os.path.join(data_dir, "valid.txt")
    test_path = os.path.join(data_dir, "test.txt")

    train_triplets = load_triplets(train_path)
    valid_triplets = load_triplets(valid_path)
    test_triplets = load_triplets(test_path)

    if auto_relation2id and len(auto_relation2id) > 0:
        print(f"自动处理了 {len(auto_relation2id)} 个关系类型")

    print(f"训练三元组: {len(train_triplets)}")
    print(f"验证三元组: {len(valid_triplets)}")
    print(f"测试三元组: {len(test_triplets)}")

    if len(train_triplets) == 0 and len(valid_triplets) == 0 and len(test_triplets) == 0:
        raise ValueError("所有三元组文件都为空！请检查文件格式。")

    return graph, train_triplets, valid_triplets, test_triplets

def infer_model_cfg_from_checkpoint(checkpoint_path: str) -> dict:
    """从 checkpoint 推断模型配置（hidden_dim, out_dim, use_gate）以便重建兼容模型。"""
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
    except Exception as e:
        print(f"无法读取 checkpoint {checkpoint_path}: {e}")
        return {}

    state = None
    for candidate_key in ("state_dict", "model_state", "model_state_dict", "model", "state"):
        if isinstance(ckpt, dict) and candidate_key in ckpt:
            state = ckpt[candidate_key]
            break
    if state is None:
        state = ckpt

    if isinstance(state, dict) and any(not isinstance(v, torch.Tensor) for v in state.values()):
        for k, v in state.items():
            if isinstance(v, dict) and all(isinstance(x, torch.Tensor) for x in v.values()):
                state = v
                break

    model_state = {k.replace("model.", "", 1) if k.startswith("model.") else k: v for k, v in state.items() if isinstance(v, torch.Tensor)}

    inferred = {}
    for proj_key in ("proj_t5vit.0.weight", "proj_clip.0.weight"):
        if proj_key in model_state:
            inferred["hidden_dim"] = int(model_state[proj_key].shape[0])
            break

    fuse_key = "fuse_mlp.net.0.weight"
    if fuse_key in model_state:
        fuse_w = model_state[fuse_key]
        in_dim = int(fuse_w.shape[1])
        out_dim = int(fuse_w.shape[0])
        inferred["out_dim"] = out_dim
        if "hidden_dim" in inferred:
            inferred["use_gate"] = (in_dim == 3 * inferred["hidden_dim"])
            if not inferred["use_gate"] and in_dim == 2 * inferred["hidden_dim"]:
                inferred["use_gate"] = False
    return inferred

def build_model(args) -> CAME:
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
        use_gate=getattr(args, "use_gate", False),
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
        if isinstance(v, torch.Tensor):
            key = k.replace("model.", "", 1) if k.startswith("model.") else k
            model_state[key] = v

    current_state = model.state_dict()
    filtered_state = {}
    skipped_shape = []
    skipped_missing = []
    for k, v in model_state.items():
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
        print(f"加载过滤后的 state_dict 时出错: {e}")
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
    parser = argparse.ArgumentParser(description="CAME KG 数据集评估")

    parser.add_argument("--data-dir", required=True, help="数据集根目录（包含 graph.bin, train.txt, valid.txt, test.txt）")
    parser.add_argument("--dataset", choices=["FB15K237", "WN18RR"], required=True, help="数据集名称")
    parser.add_argument("--feat-name", default="image_feature.pt", help="特征文件名（如 image_feature.pt）")

    parser.add_argument("--checkpoint", type=str, default=None,
                       help="CAME checkpoint 路径（如果为 None，则使用随机初始化的权重）")
    parser.add_argument("--hidden-dim", type=int, default=1024, help="隐藏层维度")
    parser.add_argument("--num-layers", type=int, default=2, help="GNN 层数")
    parser.add_argument("--num-experts", type=int, default=6, help="专家数量")
    parser.add_argument("--num-selected-experts", type=int, default=2, help="选择的专家数量")
    parser.add_argument("--text-dim", type=int, default=512, help="文本模态输入维度")
    parser.add_argument("--image-dim", type=int, default=512, help="图像模态输入维度")
    parser.add_argument("--use-gate", action="store_true", help="在模型中启用可学习融合 gate（如果模型支持）")

    parser.add_argument(
        "--embed-type",
        choices=["z1", "z2", "z3"],
        default="z3",
        help="使用的嵌入类型：z1（文本GAT输出）、z2（图像GAT输出）、z3（融合输出，默认）",
    )

    parser.add_argument("--device", default="cuda", help="设备: cuda, cpu, 或 cuda:0")
    parser.add_argument("--eval-batch-size", type=int, default=500, help="评估批大小")

    parser.add_argument("--lr-f", type=float, default=0.001, help="边分类器学习率")
    parser.add_argument("--weight-decay-f", type=float, default=0.0, help="边分类器权重衰减")
    parser.add_argument("--lp-epochs", type=int, default=5000, help="线性探测训练轮数")

    parser.add_argument("--no-mapping", action="store_true",
                       help="不使用映射（graph和feature已对齐，三元组文件中的ID为整数）")

    parser.add_argument("--clip", action="store_true", help="使用原始 CLIP 特征进行评估，不进入 CAME 模型")

    args = parser.parse_args()
    device = torch.device(args.device)

    print(f"使用设备: {device}")

    graph, train_triplets, valid_triplets, test_triplets = load_kg_data(
        args.data_dir, args.dataset, use_mapping=not args.no_mapping
    )

    print("加载节点特征...")
    feat_path = os.path.join(args.data_dir, args.feat_name)
    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"特征文件不存在: {feat_path}")

    feat_data = torch.load(feat_path, map_location="cpu", weights_only=False)
    if isinstance(feat_data, dict):
        if "features" in feat_data:
            text_feat = feat_data["features"]
        else:
            text_feat = feat_data.get("feat", feat_data.get("text", None))
            if text_feat is None:
                raise ValueError(f"特征字典中未找到特征数据，可用键: {list(feat_data.keys())}")
    elif isinstance(feat_data, torch.Tensor):
        text_feat = feat_data
    else:
        raise TypeError(f"不支持的特征文件格式: {type(feat_data)}")

    feats = {"text": text_feat}

    image_path = os.path.join(args.data_dir, "text_feature.pt")
    if os.path.exists(image_path):
        image_data = torch.load(image_path, map_location="cpu", weights_only=False)
        if isinstance(image_data, dict):
            feats["image"] = image_data.get("features", image_data.get("feat", None))
        elif isinstance(image_data, torch.Tensor):
            feats["image"] = image_data
        if feats.get("image") is not None:
            print(f"已加载图像特征: {image_path}")

    if "image" not in feats:
        feats["image"] = feats["text"]
        print("未找到图像特征，使用文本特征作为占位")

    print(f"文本特征形状: {feats['text'].shape}")
    print(f"图像特征形状: {feats['image'].shape}")

    num_nodes = graph.num_nodes()
    print(f"图节点数: {num_nodes}")

    for modality in ['text', 'image']:
        if modality in feats and isinstance(feats[modality], torch.Tensor):
            feat_nodes = feats[modality].shape[0]
            if feat_nodes != num_nodes:
                print(f"{modality} 特征节点数 ({feat_nodes}) 与图节点数 ({num_nodes}) 不一致")
                if feat_nodes > num_nodes:
                    print(f"截断 {modality} 特征到 {num_nodes} 个节点")
                    feats[modality] = feats[modality][:num_nodes]
                else:
                    print(f"{modality} 特征节点数少于图节点数，使用零向量填充（可能导致性能下降）")
                    padding = torch.zeros(num_nodes - feat_nodes, feats[modality].shape[1],
                                         dtype=feats[modality].dtype, device=feats[modality].device)
                    feats[modality] = torch.cat([feats[modality], padding], dim=0)
                print(f"调整后 {modality} 特征形状: {feats[modality].shape}")

    if args.clip:
        print("使用原始 CLIP 特征进行评估（跳过模型前向传播）...")
        if "image" not in feats:
            raise ValueError("未找到 CLIP 图像特征，请确保数据目录中包含 text_feature.pt 文件")

        node_emb = feats["image"].clone()
        print(f"使用 CLIP 特征作为节点嵌入，形状: {node_emb.shape}")
    else:
        if args.checkpoint:
            inferred = infer_model_cfg_from_checkpoint(args.checkpoint)
            if inferred:
                print(f"🔎 从 checkpoint 推断到模型配置: {inferred}")
                if "hidden_dim" in inferred and (not hasattr(args, "hidden_dim") or args.hidden_dim == 512):
                    args.hidden_dim = inferred["hidden_dim"]
                if "out_dim" in inferred and (not hasattr(args, "out_dim") or getattr(args, "out_dim", None) is None):
                    args.out_dim = inferred["out_dim"]
                if "use_gate" in inferred and inferred["use_gate"]:
                    args.use_gate = True

        model = build_model(args).to(device)
        if args.checkpoint:
            load_checkpoint_weights(model, args.checkpoint, device)
        else:
            print("未指定 checkpoint，使用随机初始化的模型权重")
        model.eval()

        node_emb = extract_embeddings(
            model,
            graph,
            feats,
            device,
            embed_type=args.embed_type,
        )

    node_emb = F.normalize(node_emb, p=2, dim=1)
    print("已对 embedding 做 L2 归一化")

    print("\n" + "=" * 60)
    if args.clip:
        print("开始 KG 边分类评估（ACC），使用原始 CLIP 特征")
    else:
        print(f"开始 KG 边分类评估（ACC），使用嵌入类型: {args.embed_type}")
    print("=" * 60)

    print("\n 评估边分类...")
    results = evaluate_kg_edge_classification(
        node_emb, train_triplets, valid_triplets, test_triplets,
        graph, device,
        lr_f=args.lr_f,
        weight_decay_f=args.weight_decay_f,
        max_epoch_f=args.lp_epochs,
        mute=False,
    )
    print("边分类结果:")
    for k, v in results.items():
        print(f"{k}: {v:.6f}")

    print("\n" + "=" * 60)
    if args.clip:
        print("KG 评估完成（使用原始 CLIP 特征）")
    else:
        print(f"KG 评估完成（嵌入类型: {args.embed_type}）")
    print("=" * 60)

if __name__ == "__main__":
    main()
