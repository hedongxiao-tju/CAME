# CAME
Official implementation of [A Graph Foundation Model with Cross-Modal Alignment and Modality-Aware Expert Fusion for Multi-Modal Graphs], accepted by ICML 2026.
# Environment
Python3.9.23,PyTorch2.8.0(CUDA12.8),and DGL1.1.3.

# How to use CAME
```bash
python train_multigraph.py --multi-graph-config configs/multi_graph_example.json
python eval_node_classification.py --data-dir /path/to/dataset --checkpoint checkpoints/came_multigraph_pretrain_epoch8.pt
python eval_link_prediction.py --data-dir /path/to/dataset --checkpoint checkpoints/came_multigraph_pretrain_epoch8.pt
```
# Shell Runners
Reusable shell runners are available under [scripts](scripts/):

```bash
scripts/run_node_classification.sh product arxiv ele-fashion wikics books-nc
scripts/run_link_prediction.sh books-lp amazon-cloth amazon-sports
scripts/run_kg.sh wn18rr fb15k237
scripts/run_all_downstream.sh
```
