#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import time
from typing import List

import dgl
import torch

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build reusable batch-level subgraph cache for stream evaluation."
    )
    parser.add_argument("--graph-path", required=True, help="Path to full graph.bin")
    parser.add_argument("--cluster-path", required=True, help="Path to clusters.pt")
    parser.add_argument("--output-graph-path", required=True, help="Path to output cached subgraphs .bin")
    parser.add_argument("--output-meta-path", required=True, help="Path to output cache metadata .pt")
    parser.add_argument("--batch-size", type=int, required=True, help="Number of clusters per stream batch")
    parser.add_argument("--log-every", type=int, default=100, help="Log every N built subgraphs")
    parser.add_argument("--limit-batches", type=int, default=None, help="Only build the first N batches for debugging")
    return parser.parse_args()

def load_clusters(cluster_path: str) -> List[torch.Tensor]:
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
    return clusters

def batched_indices(total: int, batch_size: int):
    indices = torch.arange(total, dtype=torch.long)
    for start in range(0, total, batch_size):
        yield indices[start : start + batch_size]

def collect_nodes(batch_units: torch.Tensor, clusters: List[torch.Tensor]) -> torch.Tensor:
    return torch.unique(torch.cat([clusters[int(i)] for i in batch_units.tolist()]))

def main() -> None:
    args = parse_args()
    start_time = time.perf_counter()

    print(f"Loading graph: {args.graph_path}")
    graph = dgl.load_graphs(args.graph_path)[0][0].cpu()

    print(f"Loading clusters: {args.cluster_path}")
    clusters = load_clusters(args.cluster_path)
    print(f"Loaded {len(clusters)} clusters")

    subgraphs = []
    batch_ranges = []
    batch_centers = []
    sub_nodes_list = []

    for batch_idx, batch_units in enumerate(batched_indices(len(clusters), args.batch_size)):
        if args.limit_batches is not None and batch_idx >= args.limit_batches:
            break

        sub_nodes = collect_nodes(batch_units, clusters)
        subgraph = dgl.node_subgraph(graph, sub_nodes)
        subgraph = subgraph.remove_self_loop().add_self_loop()
        centers = torch.tensor(
            [int(clusters[int(i)][0].item()) for i in batch_units.tolist()],
            dtype=torch.long,
        )

        subgraphs.append(subgraph)
        batch_ranges.append(batch_units.clone())
        batch_centers.append(centers)
        sub_nodes_list.append(sub_nodes.clone())

        if args.log_every > 0 and (batch_idx + 1) % args.log_every == 0:
            print(
                f"Built {batch_idx + 1} subgraphs | "
                f"last_nodes={subgraph.num_nodes()} last_edges={subgraph.num_edges()}"
            )

    output_graph_dir = os.path.dirname(os.path.abspath(args.output_graph_path))
    output_meta_dir = os.path.dirname(os.path.abspath(args.output_meta_path))
    if output_graph_dir:
        os.makedirs(output_graph_dir, exist_ok=True)
    if output_meta_dir:
        os.makedirs(output_meta_dir, exist_ok=True)

    print(f"Saving subgraphs to: {args.output_graph_path}")
    dgl.save_graphs(args.output_graph_path, subgraphs)

    meta = {
        "graph_path": args.graph_path,
        "cluster_path": args.cluster_path,
        "batch_size": args.batch_size,
        "num_subgraphs": len(subgraphs),
        "batch_units": batch_ranges,
        "batch_centers": batch_centers,
        "sub_nodes": sub_nodes_list,
    }
    torch.save(meta, args.output_meta_path)

    elapsed = time.perf_counter() - start_time
    print(f"Saved metadata to: {args.output_meta_path}")
    print(f"num_subgraphs={len(subgraphs)}")
    print(f"elapsed_sec={elapsed:.2f}")

if __name__ == "__main__":
    main()
