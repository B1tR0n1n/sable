#!/usr/bin/env python3
"""
Project PARALLAX — GNN Benchmark
===================================
Runs four baselines against the real CORTEX test split and compares
to SableGNN's known scores. One graph, one test, honest numbers.

Usage:
    python benchmark.py                           # Default
    python benchmark.py --data cortex_graph.pt    # Custom data path
    python benchmark.py --with-gnn                # Also run GNN (needs GPU)

Requires: ml-env with torch, torch_geometric, numpy
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ── Configuration ──────────────────────────────────────────────────────────

RELATION_TYPES = [
    "supports", "contradicts", "elaborates",
    "depends_on", "caused_by", "related", "supersedes",
]

C_GOLD = "\033[38;2;201;162;39m"
C_DIM = "\033[38;2;138;130;114m"
C_TEXT = "\033[38;2;200;189;160m"
C_BRIGHT = "\033[38;2;232;221;196m"
C_DANGER = "\033[38;2;201;74;58m"
C_SUCCESS = "\033[38;2;58;173;110m"
C_INFO = "\033[38;2;58;142;201m"
C_RESET = "\033[0m"
C_BOLD = "\033[1m"


# ── Metrics ────────────────────────────────────────────────────────────────


def compute_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC-ROC via trapezoidal rule."""
    if labels.sum() == 0 or labels.sum() == len(labels):
        return 0.5
    sorted_idx = np.argsort(scores)[::-1]
    sorted_labels = labels[sorted_idx]
    tps = np.cumsum(sorted_labels)
    fps = np.cumsum(1 - sorted_labels)
    tpr = tps / labels.sum()
    fpr = fps / (1 - labels).sum()
    auc = np.abs(np.trapz(tpr, fpr))
    return float(auc)


def compute_f1_binary(scores: np.ndarray, labels: np.ndarray) -> float:
    """Best F1 across thresholds."""
    thresholds = np.percentile(scores, np.arange(5, 96, 5))
    best = 0.0
    for t in thresholds:
        preds = (scores >= t).astype(float)
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        best = max(best, f1)
    return float(best)


def compute_mrr_hits(pos_scores: np.ndarray, neg_scores: np.ndarray) -> dict:
    """MRR and Hits@K — rank each positive against all negatives."""
    ranks = []
    for ps in pos_scores:
        rank = 1 + (neg_scores >= ps).sum()
        ranks.append(rank)
    ranks = np.array(ranks, dtype=float)
    return {
        "mrr": float(np.mean(1.0 / ranks)),
        "hits_1": float(np.mean(ranks <= 1)),
        "hits_5": float(np.mean(ranks <= 5)),
        "hits_10": float(np.mean(ranks <= 10)),
    }


def compute_macro_f1(preds: np.ndarray, targets: np.ndarray, n_classes: int) -> float:
    """Macro F1 across classes."""
    f1s = []
    for c in range(n_classes):
        tp = ((preds == c) & (targets == c)).sum()
        fp = ((preds == c) & (targets != c)).sum()
        fn = ((preds != c) & (targets == c)).sum()
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        f1s.append(f1)
    return float(np.mean(f1s))


def compute_per_class_f1(preds: np.ndarray, targets: np.ndarray, n_classes: int) -> list:
    """Per-class F1."""
    f1s = []
    for c in range(n_classes):
        tp = ((preds == c) & (targets == c)).sum()
        fp = ((preds == c) & (targets != c)).sum()
        fn = ((preds != c) & (targets == c)).sum()
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        f1s.append(float(f1))
    return f1s


def precision_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Precision@K — fraction of top-K that are positive."""
    if labels.sum() == 0:
        return 0.0
    ranked = np.argsort(scores)[::-1][:k]
    return float(labels[ranked].sum() / min(k, labels.sum()))


# ── Baselines ──────────────────────────────────────────────────────────────


def baseline_random(data, seed=42):
    """Random scores. The floor."""
    rng = np.random.default_rng(seed)
    n_pos = data.test_edge_index.size(1)
    n_neg = data.test_neg_edge_index.size(1)
    n_all = data.edge_index.size(1)
    return {
        "link_scores": rng.random(n_pos + n_neg),
        "type_preds": rng.integers(0, len(RELATION_TYPES), size=n_pos),
        "contra_scores": rng.random(n_all),
    }


def baseline_cosine(data):
    """Cosine similarity on raw node embeddings (first 1024 dims)."""
    # Extract embeddings (strip the type one-hot suffix)
    emb = data.x[:, :1024].numpy()
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb_norm = emb / np.maximum(norms, 1e-8)

    def cos_sim(i, j):
        return float(np.dot(emb_norm[i], emb_norm[j]))

    # Link prediction: cosine as existence score
    link_scores = []
    test_ei = data.test_edge_index.numpy()
    for k in range(test_ei.shape[1]):
        link_scores.append(cos_sim(test_ei[0, k], test_ei[1, k]))
    neg_ei = data.test_neg_edge_index.numpy()
    for k in range(neg_ei.shape[1]):
        link_scores.append(cos_sim(neg_ei[0, k], neg_ei[1, k]))

    # Type prediction: high sim → supports, low → depends_on
    type_preds = []
    for k in range(test_ei.shape[1]):
        sim = cos_sim(test_ei[0, k], test_ei[1, k])
        if sim > 0.7:
            type_preds.append(0)  # supports
        elif sim > 0.5:
            type_preds.append(5)  # related
        elif sim > 0.3:
            type_preds.append(2)  # elaborates
        else:
            type_preds.append(3)  # depends_on

    # Contradiction: low similarity → high contradiction score
    contra_scores = []
    all_ei = data.edge_index.numpy()
    for k in range(all_ei.shape[1]):
        contra_scores.append(1.0 - cos_sim(all_ei[0, k], all_ei[1, k]))

    return {
        "link_scores": np.array(link_scores),
        "type_preds": np.array(type_preds),
        "contra_scores": np.array(contra_scores),
    }


def baseline_graph_heuristics(data):
    """Jaccard + Common Neighbors on training graph topology."""
    import networkx as nx

    # Build graph from training edges only
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    train_ei = data.train_edge_index.numpy()
    for k in range(train_ei.shape[1]):
        G.add_edge(train_ei[0, k], train_ei[1, k])

    def score_pair(u, v):
        nu = set(G.neighbors(u)) if G.has_node(u) else set()
        nv = set(G.neighbors(v)) if G.has_node(v) else set()
        union = nu | nv
        cn = len(nu & nv)
        jaccard = cn / len(union) if union else 0.0
        # Adamic-Adar
        aa = 0.0
        for w in nu & nv:
            deg = G.degree(w)
            if deg > 1:
                aa += 1.0 / np.log(deg)
        aa_norm = min(aa / (cn + 1e-8), 1.0) if cn else 0.0
        max_cn = max(len(nu), len(nv), 1)
        return (cn / max_cn + jaccard + aa_norm) / 3.0

    # Link prediction
    link_scores = []
    test_ei = data.test_edge_index.numpy()
    for k in range(test_ei.shape[1]):
        link_scores.append(score_pair(test_ei[0, k], test_ei[1, k]))
    neg_ei = data.test_neg_edge_index.numpy()
    for k in range(neg_ei.shape[1]):
        link_scores.append(score_pair(neg_ei[0, k], neg_ei[1, k]))

    # Type: always predict most common from training
    train_labels = data.train_edge_labels.numpy()
    counts = np.bincount(train_labels, minlength=len(RELATION_TYPES))
    most_common = int(counts.argmax())
    type_preds = np.full(test_ei.shape[1], most_common)

    # Contradiction: low Jaccard with existing edge → suspicious
    contra_scores = []
    all_ei = data.edge_index.numpy()
    for k in range(all_ei.shape[1]):
        u, v = all_ei[0, k], all_ei[1, k]
        nu = set(G.neighbors(u)) if G.has_node(u) else set()
        nv = set(G.neighbors(v)) if G.has_node(v) else set()
        union = nu | nv
        jaccard = len(nu & nv) / len(union) if union else 0.0
        contra_scores.append(1.0 - jaccard)

    return {
        "link_scores": np.array(link_scores),
        "type_preds": type_preds,
        "contra_scores": np.array(contra_scores),
    }


def baseline_type_frequency(data):
    """Always predict the most common type. Tests majority-class bias."""
    train_labels = data.train_edge_labels.numpy()
    counts = np.bincount(train_labels, minlength=len(RELATION_TYPES))
    most_common = int(counts.argmax())
    n_test = data.test_edge_index.size(1)
    return {
        "type_preds": np.full(n_test, most_common),
    }


def run_gnn(data, device="cuda"):
    """Run SableGNN on the test split."""
    sys.path.insert(0, str(Path(__file__).parent))
    from cortex_gnn_model import SableGNN

    ckpt = torch.load("checkpoints/best_model.pt", weights_only=False)
    mc = ckpt["config"]
    model = SableGNN(
        in_dim=mc["in_dim"], hidden_dim=mc["hidden_dim"],
        edge_dim=mc["edge_dim"], num_layers=mc["num_layers"],
        heads=mc["heads"], dropout=mc["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        # Encode using ALL edges for message passing (eval mode)
        node_emb = model.encode(
            data.x.to(device),
            data.edge_index.to(device),
            data.edge_attr.to(device),
        )

        # Link prediction
        test_ei = data.test_edge_index.to(device)
        neg_ei = data.test_neg_edge_index.to(device)
        pos_logits = model.predict_link(node_emb, test_ei).cpu()
        neg_logits = model.predict_link(node_emb, neg_ei).cpu()
        link_scores = torch.cat([
            torch.sigmoid(pos_logits),
            torch.sigmoid(neg_logits),
        ]).numpy()

        # Type prediction
        type_logits = model.predict_link_type(node_emb, test_ei)
        type_preds = type_logits.argmax(dim=-1).cpu().numpy()

        # Contradiction
        all_ei = data.edge_index.to(device)
        contra_logits = model.predict_contradiction(node_emb, all_ei)
        contra_scores = torch.sigmoid(contra_logits).cpu().numpy()

    return {
        "link_scores": link_scores,
        "type_preds": type_preds,
        "contra_scores": contra_scores,
    }


# ── Evaluation ─────────────────────────────────────────────────────────────


def evaluate(name: str, results: dict, data, elapsed_ms: float) -> dict:
    """Evaluate a baseline's results against ground truth."""
    metrics = {"name": name, "time_ms": round(elapsed_ms, 1)}

    n_pos = data.test_edge_index.size(1)
    n_neg = data.test_neg_edge_index.size(1)
    test_labels = data.test_edge_labels.numpy()
    contra_labels = data.edge_labels.numpy() == 1  # contradicts = index 1

    # Link prediction
    if "link_scores" in results:
        scores = results["link_scores"]
        labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
        metrics["link_auc"] = round(compute_auc(scores, labels), 4)
        metrics["link_f1"] = round(compute_f1_binary(scores, labels), 4)
        rr = compute_mrr_hits(scores[:n_pos], scores[n_pos:])
        metrics["mrr"] = round(rr["mrr"], 4)
        metrics["hits_1"] = round(rr["hits_1"], 4)
        metrics["hits_5"] = round(rr["hits_5"], 4)
        metrics["hits_10"] = round(rr["hits_10"], 4)

    # Type prediction
    if "type_preds" in results:
        preds = results["type_preds"]
        metrics["type_macro_f1"] = round(
            compute_macro_f1(preds, test_labels, len(RELATION_TYPES)), 4
        )
        metrics["type_per_class"] = [
            round(f, 3) for f in compute_per_class_f1(preds, test_labels, len(RELATION_TYPES))
        ]

    # Contradiction detection
    if "contra_scores" in results:
        scores = results["contra_scores"]
        metrics["contra_auc"] = round(compute_auc(scores, contra_labels.astype(float)), 4)
        metrics["contra_p5"] = round(precision_at_k(scores, contra_labels.astype(float), 5), 4)
        metrics["contra_p10"] = round(precision_at_k(scores, contra_labels.astype(float), 10), 4)

    return metrics


# ── Report ─────────────────────────────────────────────────────────────────


def print_report(all_metrics: list):
    """Print comparison table."""
    print(f"\n{C_GOLD}{C_BOLD}  PROJECT PARALLAX — GNN Benchmark{C_RESET}")
    print(f"  {C_DIM}{'═' * 55}{C_RESET}\n")

    # Link Prediction
    print(f"  {C_INFO}{C_BOLD}Link Prediction{C_RESET}")
    print(f"  {C_DIM}{'─' * 75}{C_RESET}")
    print(f"  {C_DIM}{'Method':<22s} {'AUC':>7s} {'F1':>7s} {'MRR':>7s} {'H@1':>7s} {'H@5':>7s} {'H@10':>7s} {'ms':>6s}{C_RESET}")
    for m in all_metrics:
        if "link_auc" not in m:
            continue
        is_gnn = m["name"] == "sable_gnn"
        c = C_SUCCESS if is_gnn else C_TEXT
        b = C_BOLD if is_gnn else ""
        print(f"  {c}{b}{m['name']:<22s}{C_RESET} "
              f"{c}{m['link_auc']:7.4f} {m['link_f1']:7.4f} "
              f"{m['mrr']:7.4f} {m['hits_1']:7.4f} "
              f"{m['hits_5']:7.4f} {m['hits_10']:7.4f} "
              f"{m['time_ms']:6.0f}{C_RESET}")

    # Type Prediction
    print(f"\n  {C_INFO}{C_BOLD}Link Type Prediction (Macro F1){C_RESET}")
    print(f"  {C_DIM}{'─' * 75}{C_RESET}")
    header = f"  {C_DIM}{'Method':<22s} {'Macro':>7s}"
    for rt in RELATION_TYPES:
        header += f" {rt[:6]:>7s}"
    header += f"{C_RESET}"
    print(header)
    for m in all_metrics:
        if "type_macro_f1" not in m:
            continue
        is_gnn = m["name"] == "sable_gnn"
        c = C_SUCCESS if is_gnn else C_TEXT
        b = C_BOLD if is_gnn else ""
        line = f"  {c}{b}{m['name']:<22s}{C_RESET} {c}{m['type_macro_f1']:7.4f}"
        if "type_per_class" in m:
            for f1 in m["type_per_class"]:
                line += f" {f1:7.3f}"
        line += f"{C_RESET}"
        print(line)

    # Contradiction Detection
    print(f"\n  {C_INFO}{C_BOLD}Contradiction Detection{C_RESET}")
    print(f"  {C_DIM}{'─' * 55}{C_RESET}")
    print(f"  {C_DIM}{'Method':<22s} {'AUC':>7s} {'P@5':>7s} {'P@10':>7s}{C_RESET}")
    for m in all_metrics:
        if "contra_auc" not in m:
            continue
        is_gnn = m["name"] == "sable_gnn"
        c = C_SUCCESS if is_gnn else C_TEXT
        b = C_BOLD if is_gnn else ""
        print(f"  {c}{b}{m['name']:<22s}{C_RESET} "
              f"{c}{m['contra_auc']:7.4f} {m['contra_p5']:7.4f} "
              f"{m['contra_p10']:7.4f}{C_RESET}")

    # Delta vs cosine
    cosine = next((m for m in all_metrics if m["name"] == "cosine_similarity"), None)
    gnn = next((m for m in all_metrics if m["name"] == "sable_gnn"), None)
    if cosine and gnn:
        print(f"\n  {C_GOLD}{C_BOLD}GNN vs Cosine Similarity (the bar to clear){C_RESET}")
        print(f"  {C_DIM}{'─' * 55}{C_RESET}")
        deltas = {
            "Link AUC": (gnn.get("link_auc", 0) - cosine.get("link_auc", 0), 0.03),
            "Type Macro F1": (gnn.get("type_macro_f1", 0) - cosine.get("type_macro_f1", 0), 0.05),
            "Contra AUC": (gnn.get("contra_auc", 0) - cosine.get("contra_auc", 0), 0.05),
        }
        wins = 0
        for metric, (delta, threshold) in deltas.items():
            passed = delta >= threshold
            if passed:
                wins += 1
            marker = f"{C_SUCCESS}PASS{C_RESET}" if passed else f"{C_DANGER}FAIL{C_RESET}"
            sign = "+" if delta >= 0 else ""
            print(f"    {C_TEXT}{metric:<18s}{C_RESET} {sign}{delta:.4f} (need {'+' if threshold >= 0 else ''}{threshold})  {marker}")

        verdict = f"{C_SUCCESS}{C_BOLD}GNN ADDS VALUE" if wins >= 2 else f"{C_DANGER}{C_BOLD}GNN DOES NOT JUSTIFY COMPLEXITY"
        print(f"\n  {verdict} ({wins}/3 tests passed){C_RESET}")

    print()


# ── Main ───────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="SABLE GNN Benchmark")
    parser.add_argument("--data", type=str, default="cortex_graph.pt")
    parser.add_argument("--with-gnn", action="store_true", help="Also run SableGNN (needs GPU)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    data = torch.load(args.data, weights_only=False)
    print(f"\n  {C_DIM}Loaded: {data.num_nodes} nodes, {data.edge_index.size(1)} edges")
    print(f"  Test: {data.test_edge_index.size(1)} pos, {data.test_neg_edge_index.size(1)} neg{C_RESET}")

    all_metrics = []

    # Run baselines
    baselines = [
        ("random", lambda: baseline_random(data)),
        ("cosine_similarity", lambda: baseline_cosine(data)),
        ("graph_heuristics", lambda: baseline_graph_heuristics(data)),
        ("type_frequency", lambda: baseline_type_frequency(data)),
    ]

    for name, fn in baselines:
        t0 = time.time()
        results = fn()
        elapsed = (time.time() - t0) * 1000
        metrics = evaluate(name, results, data, elapsed)
        all_metrics.append(metrics)

    # Run GNN
    if args.with_gnn:
        t0 = time.time()
        gnn_results = run_gnn(data, args.device)
        elapsed = (time.time() - t0) * 1000
        gnn_metrics = evaluate("sable_gnn", gnn_results, data, elapsed)
        all_metrics.append(gnn_metrics)

    print_report(all_metrics)


if __name__ == "__main__":
    main()
