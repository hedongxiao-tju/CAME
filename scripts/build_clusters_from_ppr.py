#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import time
from typing import List

import torch

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a large ppr.pt map into a reusable clusters.pt file."
    )
    parser.add_argument("--ppr-path", required=True, help="Path to input ppr.pt")
    parser.add_argument("--output-path", required=True, help="Path to output clusters.pt")
    parser.add_argument(
        "--topk",
        type=int,
        default=None,
        help="Keep at most top-k nodes per cluster including center node. "
             "If unset, keep the full cluster from PPR.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only convert the first N centers after sorting. Mainly for debugging.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50000,
        help="Print progress every N centers.",
    )
    return parser.parse_args()

def to_cluster(center: int, neighbors, topk: int = None) -> torch.Tensor:
    if isinstance(neighbors, torch.Tensor):
        neigh = neighbors.long().view(-1)
    else:
        neigh = torch.as_tensor(neighbors, dtype=torch.long).view(-1)

    if neigh.numel() == 0:
        cluster = torch.tensor([center], dtype=torch.long)
    elif int(neigh[0].item()) == int(center):
        cluster = neigh
    else:
        cluster = torch.cat([torch.tensor([center], dtype=torch.long), neigh], dim=0)

    if cluster.numel() > 1:
        uniq = []
        seen = set()
        for nid in cluster.tolist():
            if nid not in seen:
                seen.add(nid)
                uniq.append(nid)
        cluster = torch.tensor(uniq, dtype=torch.long)

    if topk is not None and topk > 0 and cluster.numel() > topk:
        cluster = cluster[:topk]

    return cluster

def main() -> None:
    args = parse_args()
    start = time.perf_counter()

    print(f"Loading PPR map from: {args.ppr_path}")
    ppr_map = torch.load(args.ppr_path, map_location="cpu")
    if not isinstance(ppr_map, dict):
        raise TypeError(f"Expected dict from ppr.pt, got: {type(ppr_map)}")

    centers = sorted(ppr_map.keys())
    if args.limit is not None:
        centers = centers[: args.limit]

    print(f"Found {len(centers)} centers")
    if args.topk is not None:
        print(f"Using topk={args.topk} per cluster (including center)")

    clusters: List[torch.Tensor] = []
    for idx, center in enumerate(centers, start=1):
        cluster = to_cluster(center, ppr_map[center], topk=args.topk)
        clusters.append(cluster)

        if args.log_every > 0 and idx % args.log_every == 0:
            print(f"Processed {idx}/{len(centers)} centers")

    output_dir = os.path.dirname(os.path.abspath(args.output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    torch.save(clusters, args.output_path)
    elapsed = time.perf_counter() - start

    lengths = torch.tensor([c.numel() for c in clusters], dtype=torch.long)
    print(f"Saved clusters to: {args.output_path}")
    print(f"num_clusters={len(clusters)}")
    print(
        "cluster_size_stats="
        f"min={int(lengths.min().item())} "
        f"mean={lengths.float().mean().item():.2f} "
        f"max={int(lengths.max().item())}"
    )
    print(f"elapsed_sec={elapsed:.2f}")

if __name__ == "__main__":
    main()
