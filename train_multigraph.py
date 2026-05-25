#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CAME multi-graph pretraining.
实现多图循环训练（每个 epoch 依次在各图上前向、反向、更新），支持:
 - AMP 混合精度
 - 可选负样本队列（MoCo-style）
 - 可学习门、模态对齐权重
 - 按需上卡每个图以节省显存
 - checkpoint 保存/恢复
"""
import argparse
import json
import os
import time
import copy
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import dgl
from models.came import CAME
from data.nc_dataset import NodeClassificationDataset

# AMP helpers across torch versions
try:
    AMP_AUTOCAST = torch.amp.autocast
except Exception:
    AMP_AUTOCAST = torch.cuda.amp.autocast


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-graph pretraining for CAME")
    parser.add_argument("--multi-graph-config", type=str, required=True, help="JSON config with datasets list")
    parser.add_argument("--device", type=str, default="cuda", help="device e.g. cuda:0 or cpu")
    parser.add_argument("--epochs", type=int, default=8)

    # model
    parser.add_argument("--text-dim", type=int, default=512, help="文本特征维度（输入），优先于从 batch 推断")
    parser.add_argument("--image-dim", type=int, default=512, help="图像特征维度（输入），优先于从 batch 推断")
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--out-dim", type=int, default=1024)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--attn-drop", type=float, default=0.3)
    parser.add_argument("--temperature", type=float, default=0.07)

    # losses and weights
    parser.add_argument("--w-txt-img", type=float, default=1)
    parser.add_argument("--w-txt-fused", type=float, default=0.1)
    parser.add_argument("--w-img-fused", type=float, default=0.5)

    # training
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--loss-batch-size", type=int, default=1000, help="InfoNCE inner batch (None=full)")
    parser.add_argument("--use-gate", action="store_true")
    parser.add_argument("--align-weight", type=float, default=0.1)
    # 融合模块选择
    parser.add_argument("--fusion-type", choices=["hierarchical_moe"], default="hierarchical_moe", help="融合模块类型：hierarchical_moe（层次化MoE - 三模态：文本、图像、融合）")
    parser.add_argument("--use-moe", action="store_true", help="使用层次化MoE融合（固定为True）")
    parser.add_argument("--num-experts", type=int, default=18, help="总专家数量（平分给文本、图像、融合三个模态）")
    parser.add_argument("--num-selected-experts", type=int, default=2, help="每个模态选择的专家数量")

    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--embed-log-interval", type=int, default=100)

    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--save-every", type=int, default=8)
    parser.add_argument("--run-name", type=str, default="came_multigraph_pretrain")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset_entry(entry: Dict) -> Dict:
    data_dir = entry["data_dir"]
    feat_name = entry.get("feat_name", "image_feature.pt")

    # Load graph (prefer graph.bin)
    graph_path = os.path.join(data_dir, "graph.bin")
    if os.path.exists(graph_path):
        graphs, _ = dgl.load_graphs(graph_path)
        graph = graphs[0]
    else:
        # fallback: try NodeClassificationDataset to build graph
        ds = NodeClassificationDataset(root=data_dir, feat_name=feat_name, verbose=False, device="cpu")
        graph = ds.graph

    # load features (CPU)
    if feat_name.endswith(".pt"):
        text_path = os.path.join(data_dir, feat_name)
    else:
        text_path = os.path.join(data_dir, f"features_{feat_name}.pt")
    if not os.path.exists(text_path):
        raise FileNotFoundError(f"text features not found: {text_path}")
    tdata = torch.load(text_path, map_location="cpu", weights_only=False)
    text_feat = tdata.get("features", next(iter(tdata.values()))) if isinstance(tdata, dict) else tdata

    image_path = os.path.join(data_dir, entry.get("image_feat", "text_feature.pt"))
    image_feat = None
    if os.path.exists(image_path):
        idata = torch.load(image_path, map_location="cpu", weights_only=False)
        image_feat = idata.get("features", next(iter(idata.values()))) if isinstance(idata, dict) else idata
    else:
        image_feat = text_feat

    return {"name": entry.get("name", data_dir), "graph": graph, "feats": {"text": text_feat, "image": image_feat}, "weight": float(entry.get("weight", 1.0))}


def _normalize_run_name(run_name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in run_name.strip())
    return safe or "came_multigraph_pretrain"


def save_ckpt(state: dict, save_dir: str, epoch: int, run_name: str):
    try:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{_normalize_run_name(run_name)}_epoch{epoch}.pt"

        path = os.path.abspath(os.path.join(save_dir, filename))
        torch.save(state, path)
        print(f" saved checkpoint: {path}")
    except Exception as e:
        print(f" failed to save checkpoint to {save_dir}: {e}")
        raise


def main():
    args = parse_args()
    if args.seed is not None:
        set_seed(args.seed)
    device = torch.device(args.device)
    print(f"device: {device}")

    with open(args.multi_graph_config, "r") as f:
        cfg = json.load(f)
    datasets = cfg.get("datasets", cfg if isinstance(cfg, list) else [])

    # 不预加载所有图到内存，直接使用MultiGraphDataModule（按需加载）
    print(f"配置了 {len(datasets)} 个数据集:")
    for e in datasets:
        data_dir = e.get("data_dir", "")
        print(f" - {data_dir}")

    # build MultiGraphDataModule and train loader (will provide precomputed batches)
    from data.multi_graph_datamodule import MultiGraphDataModule
    mgdm = MultiGraphDataModule(dataset_configs=datasets, batch_size=1, num_workers=0, shuffle=True, seed=42)
    mgdm.setup()
    train_loader = mgdm.train_dataloader()
    print(f"使用 MultiGraphDataModule，多图共 {len(datasets)} 个数据集，train loader batches: {len(train_loader.dataset)}")

    # Determine input dims: prefer user-provided args, otherwise infer from a sample batch
    if args.text_dim and args.image_dim:
        dim_t5vit = args.text_dim
        dim_clip = args.image_dim
    else:
        try:
            sample_batch = next(iter(train_loader))
        except StopIteration:
            raise RuntimeError("train_loader is empty; no precomputed batches found")
        sample_feats = sample_batch["features"]
        dim_t5vit = sample_feats["text"].shape[1]
        dim_clip = sample_feats.get("image", sample_feats["text"]).shape[1]

    model = CAME(
        dim_t5vit=dim_t5vit,
        dim_clip=dim_clip,
        hidden_1024=args.hidden_dim,
        out_1024=args.out_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
        attn_drop=args.attn_drop,
        temperature=args.temperature,
        w12=args.w_txt_img,
        w23=args.w_txt_fused,
        w13=args.w_img_fused,
        loss_batch_size=args.loss_batch_size,
        use_gate=args.use_gate,
        fusion_type=getattr(args, "fusion_type", "hierarchical_moe"),
        use_moe=getattr(args, "use_moe", True),  # 固定使用MoE
        num_experts=getattr(args, "num_experts", 18),
        num_selected_experts=getattr(args, "num_selected_experts", 2),
        **({"align_weight": args.align_weight} if "align_weight" in CAME.__init__.__code__.co_varnames else {})
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # GradScaler helper across torch versions
    try:
        scaler = torch.amp.GradScaler(device_type="cuda", enabled=args.amp)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    start_epoch = 1
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        state = ck.get("model_state", ck.get("state_dict", ck))
        model.load_state_dict(state, strict=False)
        if "optimizer" in ck:
            try:
                opt.load_state_dict(ck["optimizer"])
            except ValueError as e:
                print(f" 跳过 optimizer 状态恢复: {e}")
        start_epoch = ck.get("epoch", 1) + 1
        print(f"resumed from {args.resume}, start_epoch={start_epoch}")

    # Use MultiGraphDataModule to iterate precomputed batches across datasets (if available)
    from data.multi_graph_datamodule import MultiGraphDataModule
    mgdm = MultiGraphDataModule(dataset_configs=datasets, batch_size=1, num_workers=0, shuffle=True, seed=42)
    mgdm.setup()
    train_loader = mgdm.train_dataloader()

    model.train()
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        epoch_loss = 0.0
        total_weight = 0.0
        step = 0
        for batch in train_loader:
            # batch is collated by MultiGraphDataModule -> contains 'graph' and 'features'
            g_gpu = batch["graph"].to(device)
            feats_gpu = {k: v.to(device, non_blocking=True) for k, v in batch["features"].items()}

            opt.zero_grad()
            # debug: time the forward/backward to detect hanging/slow batches
            batch_start = time.time()
            print(f"   -> Processing batch {step+1} / {len(train_loader.dataset)} ...", flush=True)
            with AMP_AUTOCAST(device_type="cuda", enabled=args.amp):
                loss = model(g_gpu, feats_gpu["text"], feats_gpu.get("image", feats_gpu["text"]), compute_loss=True)
                if isinstance(loss, (tuple, list)):
                    loss_val = loss[0]
                else:
                    loss_val = loss
                if loss_val is None:
                    raise RuntimeError("model returned None loss")
                weighted = loss_val / max(1, args.grad_accum)
            scaler.scale(weighted).backward()
            batch_end = time.time()
            dt = batch_end - batch_start
            if dt > 5.0:
                print(f"    长批次耗时 {dt:.1f}s (可能包含大子图或 I/O)。", flush=True)
            if (step + 1) % args.grad_accum == 0:
                if args.grad_clip and args.grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()

            epoch_loss += float(loss_val.item())
            total_weight += 1.0
            step += 1

            # free memory
            try:
                del g_gpu, feats_gpu, loss, weighted
            except Exception:
                pass
            if device.type == "cuda":
                torch.cuda.empty_cache()
                # 每10个batch强制清理一次
                if step % 10 == 0:
                    torch.cuda.synchronize()

        avg_loss = epoch_loss / max(1.0, step)
        t1 = time.time()
        print(f"Epoch {epoch:03d} | avg_loss:{avg_loss:.6f} | time:{t1-t0:.1f}s")
        if args.save_every > 0 and (epoch % args.save_every == 0 or epoch == args.epochs):
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer": opt.state_dict(),
                "args": vars(args),
            }
            save_ckpt(ckpt, args.save_dir, epoch, args.run_name)

    print("training finished")


if __name__ == "__main__":
    main()
