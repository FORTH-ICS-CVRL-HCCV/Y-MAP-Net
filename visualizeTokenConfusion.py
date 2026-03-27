#!/usr/bin/python3
"""
visualizeTokenConfusion.py — Visualise evaluation_results_*_token_confusion.json

Produces a multi-panel overview figure and a dedicated confusion-pair heatmap:

  Panel 1 – Top-N classes by ground-truth frequency (TP+FN), stacked-bar TP/FP/FN
  Panel 2 – F1 score distribution histogram across all active classes
  Panel 3 – Precision vs Recall scatter (point size = GT frequency)
  Panel 4 – Top confusion pairs table with GloVe cosine similarity column
             (high cosine = semantically similar pair, e.g. "snow" vs "snowy")

  Separate figure – Confusion-pair heatmap matrix for the top-K pairs by count,
                    cells annotated with GloVe cosine similarity

Usage:
    python3 visualizeTokenConfusion.py [confusion_json] [options]

    confusion_json           Path to evaluation_results_*_token_confusion.json
                             (defaults to first match of that pattern in cwd)
    --top N                  Number of classes/pairs to highlight (default: 30)
    --output PREFIX          Filename prefix for saved PNGs (default: token_confusion)
    --no-show                Do not call plt.show(), only save files
    --heatmap-top N          Number of top pairs for the heatmap (default: 50)
    --embeddings PATH        Path to GloVe_D300.embeddings
                             (defaults to 2d_pose_estimation/GloVe_D300.embeddings)
"""

import os
import sys
import json
import glob
import argparse
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import defaultdict


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(description='Visualise token confusion JSON')
    p.add_argument('confusion_json', nargs='?', default=None,
                   help='Path to token_confusion JSON (auto-detected if omitted)')
    p.add_argument('--top',         type=int, default=30,
                   help='Number of top classes to display (default: 30)')
    p.add_argument('--heatmap-top', type=int, default=50,
                   help='Number of top confusion pairs for heatmap (default: 50)')
    p.add_argument('--output',      default='token_confusion',
                   help='Output filename prefix (default: token_confusion)')
    p.add_argument('--no-show',     action='store_true',
                   help='Save only, do not call plt.show()')
    p.add_argument('--embeddings',  default='2d_pose_estimation/GloVe_D300.embeddings',
                   help='Path to GloVe embeddings file (default: 2d_pose_estimation/GloVe_D300.embeddings)')
    return p


def find_confusion_json():
    """Auto-detect the most recent token_confusion JSON in the current directory."""
    candidates = sorted(glob.glob('evaluation_results_*_token_confusion.json'))
    if candidates:
        print(f'Auto-detected: {candidates[-1]}')
        return candidates[-1]
    raise FileNotFoundError(
        'No evaluation_results_*_token_confusion.json found. '
        'Pass the path explicitly as the first argument.')


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load(path):
    with open(path) as f:
        return json.load(f)


def load_glove_embeddings(file_path):
    """Load GloVe embeddings from the custom project format.

    File layout (all values newline-separated):
        D              — embedding dimension
        N              — number of entries
        offset         — (unused here)
        scaling        — (unused here)
        "start"
        For each entry:
            key        — vocabulary index as string ("0", "1", ...)
            D floats   — one per line
            "next_token"

    Returns:
        embeddings : dict {str_key: np.ndarray(D,)}
        D          : int  embedding dimension
    """
    embeddings = {}
    with open(file_path, 'r') as f:
        D = int(f.readline().strip())
        N = int(f.readline().strip())
        f.readline()   # offset  (unused)
        f.readline()   # scaling (unused)
        f.readline()   # "start"
        for _ in range(N):
            key = f.readline().strip()
            vec = np.array([float(f.readline()) for _ in range(D)], dtype=np.float32)
            f.readline()  # "next_token"
            embeddings[key] = vec
    print(f'Loaded {len(embeddings)} GloVe embeddings (D={D}) from {file_path}')
    return embeddings, D


def cosine_sim(vec_a, vec_b):
    """Cosine similarity in [-1, 1]. Returns np.nan if either vector is zero."""
    na = np.linalg.norm(vec_a)
    nb = np.linalg.norm(vec_b)
    if na < 1e-9 or nb < 1e-9:
        return float('nan')
    return float(np.dot(vec_a, vec_b) / (na * nb))


def enrich_confusion_pairs_with_similarity(confusion_pairs, embeddings):
    """Add a 'glove_cos' field to every confusion pair dict (in-place).

    Pairs where one or both embeddings are missing get glove_cos=None.
    """
    for p in confusion_pairs:
        ti = str(p.get('true_idx', ''))
        pi = str(p.get('pred_idx', ''))
        if ti in embeddings and pi in embeddings:
            p['glove_cos'] = round(cosine_sim(embeddings[ti], embeddings[pi]), 3)
        else:
            p['glove_cos'] = None


def per_class_arrays(data):
    """Return numpy arrays extracted from the per_class list."""
    pc = data['per_class']
    names     = [e['class']     for e in pc]
    tp        = np.array([e['TP']        for e in pc], dtype=np.int64)
    fp        = np.array([e['FP']        for e in pc], dtype=np.int64)
    fn        = np.array([e['FN']        for e in pc], dtype=np.int64)
    precision = np.array([e['precision'] for e in pc], dtype=np.float32)
    recall    = np.array([e['recall']    for e in pc], dtype=np.float32)
    f1        = np.array([e['f1']        for e in pc], dtype=np.float32)
    gt_freq   = tp + fn   # ground-truth occurrences per class
    return names, tp, fp, fn, precision, recall, f1, gt_freq


# ---------------------------------------------------------------------------
# Panel 1: stacked-bar of top-N classes by GT frequency
# ---------------------------------------------------------------------------
def plot_top_classes(ax, names, tp, fp, fn, f1, gt_freq, top_n):
    """Stacked horizontal bar: TP (green) | FP (orange) | FN (red), sorted by gt_freq."""
    # Already sorted by gt_freq descending from evaluateYMAPNet; take first top_n
    idx   = np.arange(min(top_n, len(names)))
    lbls  = [names[i] for i in idx]
    tp_v  = tp[idx];  fp_v = fp[idx];  fn_v = fn[idx]
    f1_v  = f1[idx]
    total = tp_v + fp_v + fn_v
    total = np.where(total == 0, 1, total)   # avoid /0

    y = np.arange(len(idx))
    h = 0.6

    ax.barh(y, tp_v, height=h, color='#2ecc71', label='TP')
    ax.barh(y, fp_v, height=h, left=tp_v, color='#e67e22', label='FP')
    ax.barh(y, fn_v, height=h, left=tp_v + fp_v, color='#e74c3c', label='FN')

    # Annotate with F1 score
    for i, (t, p_v, n, f) in enumerate(zip(tp_v, fp_v, fn_v, f1_v)):
        ax.text(t + p_v + n + 0.5, i, f'F1={f:.2f}', va='center', fontsize=6.5)

    ax.set_yticks(y)
    ax.set_yticklabels(lbls, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel('Count (TP + FP + FN)')
    ax.set_title(f'Top {len(idx)} classes by GT frequency (TP+FN)', fontsize=9)
    ax.legend(fontsize=7, loc='lower right')


# ---------------------------------------------------------------------------
# Panel 2: F1 distribution histogram
# ---------------------------------------------------------------------------
def plot_f1_histogram(ax, f1, n_active):
    """Histogram of F1 scores across all active classes."""
    # Separate zero-F1 classes (not predicted at all) from non-zero
    nonzero_f1 = f1[f1 > 0]
    zero_count  = int(np.sum(f1 == 0))

    bins = np.linspace(0, 1, 41)
    ax.hist(nonzero_f1, bins=bins, color='#3498db', edgecolor='white', linewidth=0.4)
    ax.axvline(float(np.mean(f1)),        color='red',    linestyle='--', linewidth=1.2,
               label=f'mean F1={np.mean(f1):.3f}')
    ax.axvline(float(np.median(nonzero_f1)) if len(nonzero_f1) else 0,
               color='orange', linestyle='--', linewidth=1.2,
               label=f'median (non-zero)={np.median(nonzero_f1):.3f}' if len(nonzero_f1) else 'median=0')

    ax.set_xlabel('F1 score')
    ax.set_ylabel('Number of classes')
    ax.set_title(
        f'F1 distribution — {n_active} active classes\n'
        f'({zero_count} with F1=0, {len(nonzero_f1)} with F1>0)',
        fontsize=9)
    ax.legend(fontsize=7)


# ---------------------------------------------------------------------------
# Panel 3: Precision vs Recall scatter
# ---------------------------------------------------------------------------
def plot_precision_recall(ax, names, precision, recall, f1, gt_freq, top_n):
    """Scatter plot of all active classes; point size ∝ GT frequency."""
    # Clamp point sizes to a visible range
    size_raw    = gt_freq.astype(np.float32)
    size_normed = np.clip(size_raw / (size_raw.max() + 1e-6), 0.02, 1.0)
    sizes       = 8 + size_normed * 120

    sc = ax.scatter(recall, precision, s=sizes, c=f1,
                    cmap='RdYlGn', vmin=0, vmax=1, alpha=0.6, linewidths=0)
    plt.colorbar(sc, ax=ax, label='F1', fraction=0.03, pad=0.02)

    # Annotate the top-N most frequent classes by name
    order  = np.argsort(-gt_freq)[:top_n]
    for i in order:
        ax.annotate(names[i], (recall[i], precision[i]),
                    fontsize=5, alpha=0.8,
                    xytext=(2, 2), textcoords='offset points')

    # Draw iso-F1 curves as faint guides
    _r = np.linspace(0.01, 1.0, 200)
    for iso_f1 in [0.1, 0.2, 0.4, 0.6, 0.8]:
        _p = iso_f1 * _r / (2 * _r - iso_f1 + 1e-9)
        _p = np.clip(_p, 0, 1)
        ax.plot(_r, _p, color='grey', linewidth=0.5, linestyle=':', alpha=0.5)
        ax.text(1.01, iso_f1 / (2 - iso_f1 + 1e-9),
                f'F1={iso_f1}', fontsize=5, color='grey', va='center')

    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision vs Recall (point size = GT frequency)', fontsize=9)


# ---------------------------------------------------------------------------
# Panel 4: Top confusion pairs table
# ---------------------------------------------------------------------------
def _cos_color(cos_val):
    """Return a background color hinting at semantic similarity.

    High cosine (≥0.7)  → pale red   — very similar words, likely a synonym/plural
    Mid  cosine (0.4–0.7)→ pale amber — related concepts
    Low  cosine (<0.4)   → pale blue  — unrelated; a genuine confusion
    None                 → light grey — embedding unavailable
    """
    if cos_val is None:
        return '#e0e0e0'
    if cos_val >= 0.7:
        return '#ffd6d6'   # pale red
    if cos_val >= 0.4:
        return '#fff4cc'   # pale amber
    return '#d6eaff'       # pale blue


def plot_confusion_table(ax, confusion_pairs, top_n):
    """Render the top-N confusion pairs as a tidy text table.

    Columns: #  |  True label  |  Predicted label  |  Count  |  CosSim
    The CosSim column is colour-coded:
        pale-red   ≥ 0.70 — nearly synonymous (e.g. 'snow' → 'snowy')
        pale-amber  0.40–0.69 — semantically related
        pale-blue  < 0.40 — unrelated; model is genuinely confused
    """
    ax.axis('off')
    top = confusion_pairs[:top_n]
    if not top:
        ax.text(0.5, 0.5, 'No confusion pairs', ha='center', va='center')
        return

    has_cos = any(p.get('glove_cos') is not None for p in top)
    col_labels = ['#', 'True label', 'Predicted label', 'Count']
    if has_cos:
        col_labels.append('CosSim')

    rows = []
    cos_vals = []
    for i, p in enumerate(top):
        cos = p.get('glove_cos')
        row = [str(i + 1), p['true_label'], p['pred_label'], str(p['count'])]
        if has_cos:
            row.append(f'{cos:.3f}' if cos is not None else 'n/a')
        rows.append(row)
        cos_vals.append(cos)

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc='center',
        cellLoc='left',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.auto_set_column_width(list(range(len(col_labels))))

    # Style header row
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor('#2c3e50')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    # Zebra stripe for data rows; colour CosSim cell separately
    for i, cos in enumerate(cos_vals):
        base_color = '#f2f2f2' if i % 2 == 0 else 'white'
        for j in range(len(col_labels)):
            if has_cos and j == len(col_labels) - 1:
                # CosSim column: semantic-similarity colour hint
                table[(i + 1, j)].set_facecolor(_cos_color(cos))
            else:
                table[(i + 1, j)].set_facecolor(base_color)

    title_suffix = '  |  CosSim: ■ ≥0.7 synonym  ■ 0.4–0.7 related  ■ <0.4 unrelated' if has_cos else ''
    ax.set_title(
        f'Top {len(top)} confusion pairs (true label missed, pred label wrongly fired){title_suffix}',
        fontsize=8, pad=12)


# ---------------------------------------------------------------------------
# Figure 2: Confusion-pair heatmap
# ---------------------------------------------------------------------------
def plot_confusion_heatmap(confusion_pairs, top_n, serial, output_prefix, no_show):
    """
    Build a matrix of the top-N confusion pairs and display it as a heatmap.

    Rows = true labels (ground-truth classes that were missed / FN).
    Cols = predicted labels (classes wrongly fired / FP at the same time).
    Cell value = number of co-occurrence samples.

    Cell annotation (when matrix is small enough):
        top line    — co-occurrence count
        bottom line — GloVe cosine similarity (if available)
                      colour-coded: green=high (synonyms), yellow=mid, blue=low
    """
    top = confusion_pairs[:top_n]
    if not top:
        print('[warn] No confusion pairs to plot heatmap.')
        return

    # Collect unique true / pred labels (preserve order of first appearance)
    true_labels_ordered = []
    pred_labels_ordered = []
    seen_true = set()
    seen_pred = set()
    for p in top:
        tl, pl = p['true_label'], p['pred_label']
        if tl not in seen_true:
            true_labels_ordered.append(tl)
            seen_true.add(tl)
        if pl not in seen_pred:
            pred_labels_ordered.append(pl)
            seen_pred.add(pl)

    # Sort both axes alphabetically for easier reading
    true_labels_ordered.sort()
    pred_labels_ordered.sort()
    t_idx = {l: i for i, l in enumerate(true_labels_ordered)}
    p_idx = {l: i for i, l in enumerate(pred_labels_ordered)}

    n_true = len(true_labels_ordered)
    n_pred = len(pred_labels_ordered)
    count_matrix = np.zeros((n_true, n_pred), dtype=np.float32)
    # cos_matrix: NaN where no pair exists, float otherwise
    cos_matrix = np.full((n_true, n_pred), np.nan, dtype=np.float32)
    for p in top:
        ti = t_idx[p['true_label']]
        pi = p_idx[p['pred_label']]
        count_matrix[ti, pi] = p['count']
        c = p.get('glove_cos')
        if c is not None:
            cos_matrix[ti, pi] = c

    has_cos = not np.all(np.isnan(cos_matrix))

    # Figure size scales with matrix dimensions, capped for readability
    cell_px = 0.35          # inches per cell
    fig_w   = max(10, min(28, n_pred * cell_px + 5))
    fig_h   = max(6,  min(24, n_true * cell_px + 3))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    # Use a sequential colormap; log scale if range is large
    vmax = count_matrix.max()
    norm = matplotlib.colors.LogNorm(vmin=0.5, vmax=max(vmax, 1)) if vmax > 5 else None

    im = ax.imshow(count_matrix, aspect='auto',
                   cmap='YlOrRd', norm=norm,
                   interpolation='nearest')
    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label('Co-occurrence count', color='white', fontsize=8)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white', fontsize=7)

    # Axis labels
    ax.set_xticks(np.arange(n_pred))
    ax.set_xticklabels(pred_labels_ordered, rotation=60, ha='right',
                       fontsize=max(4, min(7, 120 // n_pred)))
    ax.set_yticks(np.arange(n_true))
    ax.set_yticklabels(true_labels_ordered,
                       fontsize=max(4, min(7, 120 // n_true)))
    ax.tick_params(colors='white')

    # Annotate cells when matrix is small enough to be readable.
    # Each occupied cell shows: count (top), cosine similarity (bottom, colour-coded).
    if n_true * n_pred <= 900:
        fsize = max(4, min(6, 60 // max(n_true, n_pred)))
        for ti in range(n_true):
            for pi in range(n_pred):
                v = int(count_matrix[ti, pi])
                if v == 0:
                    continue
                bright = count_matrix[ti, pi] > vmax * 0.5
                count_color = 'black' if bright else 'white'

                if has_cos and not np.isnan(cos_matrix[ti, pi]):
                    c = float(cos_matrix[ti, pi])
                    # Cosine similarity colour: green=similar, yellow=related, cyan=unrelated
                    if c >= 0.7:
                        cos_color = '#88ff88'    # green — likely synonym/plural
                    elif c >= 0.4:
                        cos_color = '#ffee66'    # yellow — semantically related
                    else:
                        cos_color = '#88ddff'    # cyan — genuinely unrelated confusion
                    # Two-line annotation: count then cosine
                    ax.text(pi, ti - 0.18, str(v),
                            ha='center', va='center', fontsize=fsize,
                            color=count_color, fontweight='bold')
                    ax.text(pi, ti + 0.22, f'{c:.2f}',
                            ha='center', va='center', fontsize=max(3, fsize - 1),
                            color=cos_color)
                else:
                    ax.text(pi, ti, str(v),
                            ha='center', va='center', fontsize=fsize,
                            color=count_color)

    cos_legend = ('  |  cell bottom: GloVe cosine  '
                  '■green ≥0.7 synonym  ■yellow 0.4–0.7 related  ■cyan <0.4 unrelated'
                  if has_cos else '')
    ax.set_xlabel('Predicted label (false positive)', color='white', fontsize=9)
    ax.set_ylabel('True label (false negative / missed)', color='white', fontsize=9)
    ax.set_title(
        f'Token confusion heatmap — v{serial}  '
        f'(top {len(top)} pairs, rows=missed GT, cols=wrong prediction){cos_legend}',
        color='white', fontsize=9, pad=10)

    fig.tight_layout()
    out_path = f'{output_prefix}_heatmap.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'Saved: {out_path}')
    if not no_show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Panel 5: GloVe cosine similarity distribution across all confusion pairs
# ---------------------------------------------------------------------------
def plot_glove_cos_histogram(ax, confusion_pairs):
    """Histogram of GloVe cosine similarity values across all confusion pairs.

    Each pair is weighted by its co-occurrence count so that high-frequency
    confusions dominate.  Three vertical bands mark the semantic categories:
        ≥ 0.70  synonym / plural / inflection  (pale red)
        0.40–0.69  semantically related          (pale amber)
        < 0.40  genuinely unrelated             (pale blue)
    """
    cos_vals   = []
    cos_wtd    = []   # count-weighted
    for p in confusion_pairs:
        c = p.get('glove_cos')
        if c is not None:
            cos_vals.append(c)
            cos_wtd.append((c, p['count']))

    if not cos_vals:
        ax.text(0.5, 0.5, 'No GloVe similarity data', ha='center', va='center')
        return

    cos_arr = np.array(cos_vals, dtype=np.float32)
    counts  = np.array([w for _, w in cos_wtd], dtype=np.float32)

    # Shade semantic regions as background bands
    ax.axvspan(-1.0,  0.4,  alpha=0.10, color='#3498db', label='_nolegend_')
    ax.axvspan( 0.4,  0.7,  alpha=0.10, color='#f39c12', label='_nolegend_')
    ax.axvspan( 0.7,  1.01, alpha=0.10, color='#e74c3c', label='_nolegend_')

    bins = np.linspace(-1.0, 1.0, 41)

    # Unweighted (pair count) histogram
    ax.hist(cos_arr, bins=bins, color='#95a5a6', alpha=0.5,
            edgecolor='white', linewidth=0.3, label='# pairs')

    # Count-weighted histogram (shows where model spends most confusion)
    ax.hist(cos_arr, bins=bins, weights=counts,
            color='#e74c3c', alpha=0.7, edgecolor='white', linewidth=0.3,
            label='weighted by count')

    # Threshold lines
    ax.axvline(0.4, color='#f39c12', linestyle='--', linewidth=1.0)
    ax.axvline(0.7, color='#e74c3c', linestyle='--', linewidth=1.0)
    ax.axvline(float(np.mean(cos_arr)), color='white', linestyle=':',
               linewidth=1.2, label=f'mean={np.mean(cos_arr):.3f}')

    # Category counts and weighted totals
    n_syn  = int(np.sum(cos_arr >= 0.7))
    n_rel  = int(np.sum((cos_arr >= 0.4) & (cos_arr < 0.7)))
    n_unr  = int(np.sum(cos_arr < 0.4))
    w_syn  = float(np.sum(counts[cos_arr >= 0.7]))
    w_rel  = float(np.sum(counts[(cos_arr >= 0.4) & (cos_arr < 0.7)]))
    w_unr  = float(np.sum(counts[cos_arr < 0.4]))
    w_tot  = w_syn + w_rel + w_unr

    legend_text = (
        f'Synonym ≥0.70:  {n_syn} pairs  ({100*w_syn/w_tot:.1f}% of confusion samples)\n'
        f'Related 0.4–0.7: {n_rel} pairs  ({100*w_rel/w_tot:.1f}%)\n'
        f'Unrelated <0.40: {n_unr} pairs  ({100*w_unr/w_tot:.1f}%)'
    )
    ax.text(0.02, 0.97, legend_text,
            transform=ax.transAxes, va='top', fontsize=7,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#2c3e50',
                      edgecolor='#7f8c8d', alpha=0.85),
            color='white', family='monospace')

    ax.set_xlabel('GloVe cosine similarity (true label vs predicted label)')
    ax.set_ylabel('Number of confusion pairs / weighted count')
    ax.set_title('GloVe cosine similarity distribution across confusion pairs\n'
                 '(red=count-weighted, grey=pair count)', fontsize=9)
    ax.legend(fontsize=7, loc='upper left',
              bbox_to_anchor=(0.0, 0.72))   # below the stats box


# ---------------------------------------------------------------------------
# Panel 6: Count vs cosine scatter — are high-frequency confusions semantic?
# ---------------------------------------------------------------------------
def plot_count_vs_cosine(ax, confusion_pairs, top_n):
    """Scatter of co-occurrence count vs GloVe cosine for each confusion pair.

    Point colour encodes semantic category; point size is proportional to count.
    The top-N highest-count pairs are annotated with "true→pred" labels.
    """
    pairs_with_cos = [(p['count'], p.get('glove_cos'),
                       p['true_label'], p['pred_label'])
                      for p in confusion_pairs
                      if p.get('glove_cos') is not None]
    if not pairs_with_cos:
        ax.text(0.5, 0.5, 'No GloVe similarity data', ha='center', va='center')
        return

    counts = np.array([x[0] for x in pairs_with_cos], dtype=np.float32)
    cos    = np.array([x[1] for x in pairs_with_cos], dtype=np.float32)
    labels = [(x[2], x[3]) for x in pairs_with_cos]

    # Colour by semantic category
    colors = []
    for c in cos:
        if c >= 0.7:
            colors.append('#e74c3c')   # synonym — red
        elif c >= 0.4:
            colors.append('#f39c12')   # related — amber
        else:
            colors.append('#3498db')   # unrelated — blue

    sizes = 10 + (counts / (counts.max() + 1e-6)) * 120

    ax.scatter(cos, counts, c=colors, s=sizes, alpha=0.6, linewidths=0)

    # Annotate top-N by count
    order = np.argsort(-counts)[:top_n]
    for i in order:
        tl, pl = labels[i]
        # Abbreviate long labels to keep annotations readable
        tl_s = tl[:12] + '…' if len(tl) > 13 else tl
        pl_s = pl[:12] + '…' if len(pl) > 13 else pl
        ax.annotate(f'{tl_s}→{pl_s}',
                    (cos[i], counts[i]),
                    fontsize=5, alpha=0.85,
                    xytext=(3, 2), textcoords='offset points')

    # Threshold lines
    ax.axvline(0.4, color='#f39c12', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.axvline(0.7, color='#e74c3c', linestyle='--', linewidth=0.8, alpha=0.7)

    # Legend patches
    import matplotlib.patches as mpatches
    ax.legend(handles=[
        mpatches.Patch(color='#e74c3c', label='synonym ≥ 0.70'),
        mpatches.Patch(color='#f39c12', label='related  0.40–0.69'),
        mpatches.Patch(color='#3498db', label='unrelated < 0.40'),
    ], fontsize=7, loc='upper right')

    ax.set_xlabel('GloVe cosine similarity')
    ax.set_ylabel('Co-occurrence count')
    ax.set_title('Confusion frequency vs semantic similarity\n'
                 '(high-right = frequent synonymic confusion)', fontsize=9)


# ---------------------------------------------------------------------------
# Main overview figure (6 panels, 3 rows × 2 cols)
# ---------------------------------------------------------------------------
def plot_overview(data, args):
    names, tp, fp, fn, precision, recall, f1, gt_freq = per_class_arrays(data)
    serial  = data.get('serial', '?')
    top_n   = args.top

    has_cos = any(p.get('glove_cos') is not None for p in data['confusion_pairs'])

    # Expand to 3 rows when GloVe cosine data is available
    n_rows = 3 if has_cos else 2
    fig_h  = 24 if has_cos else 16

    fig = plt.figure(figsize=(20, fig_h))
    fig.suptitle(
        f'Token evaluation — model v{serial}  |  '
        f'{data["n_active_classes"]} active classes  |  '
        f'{data["n_confusion_pairs"]} confusion pairs  |  '
        f'{data["total_samples"]} validation samples',
        fontsize=11, fontweight='bold', y=0.99)

    gs = fig.add_gridspec(n_rows, 2, hspace=0.40, wspace=0.30,
                          left=0.07, right=0.97, top=0.96, bottom=0.04)

    ax_bar   = fig.add_subplot(gs[0, 0])
    ax_hist  = fig.add_subplot(gs[0, 1])
    ax_pr    = fig.add_subplot(gs[1, 0])
    ax_table = fig.add_subplot(gs[1, 1])

    plot_top_classes(ax_bar,   names, tp, fp, fn, f1, gt_freq, top_n)
    plot_f1_histogram(ax_hist, f1, data['n_active_classes'])
    plot_precision_recall(ax_pr, names, precision, recall, f1, gt_freq, top_n)
    plot_confusion_table(ax_table, data['confusion_pairs'], top_n)

    if has_cos:
        ax_cos_hist  = fig.add_subplot(gs[2, 0])
        ax_cos_scat  = fig.add_subplot(gs[2, 1])
        plot_glove_cos_histogram(ax_cos_hist, data['confusion_pairs'])
        plot_count_vs_cosine(ax_cos_scat, data['confusion_pairs'], top_n)

    out_path = f'{args.output}_overview.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_path}')
    if not args.no_show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    args = build_parser().parse_args()

    if args.confusion_json is None:
        args.confusion_json = find_confusion_json()

    if not os.path.exists(args.confusion_json):
        print(f'ERROR: file not found: {args.confusion_json}')
        sys.exit(1)

    print(f'Loading {args.confusion_json} ...')
    data = load(args.confusion_json)

    print(f'  serial={data["serial"]}  '
          f'samples={data["total_samples"]}  '
          f'active_classes={data["n_active_classes"]}  '
          f'confusion_pairs={data["n_confusion_pairs"]}')

    # Load GloVe embeddings and annotate confusion pairs with cosine similarity.
    # Pairs where one/both indices are missing from the embedding table get glove_cos=None.
    if os.path.exists(args.embeddings):
        embeddings, _ = load_glove_embeddings(args.embeddings)
        enrich_confusion_pairs_with_similarity(data['confusion_pairs'], embeddings)
        n_enriched = sum(1 for p in data['confusion_pairs'] if p.get('glove_cos') is not None)
        print(f'  enriched {n_enriched}/{len(data["confusion_pairs"])} pairs with GloVe cosine similarity')
    else:
        print(f'[warn] GloVe embeddings not found at {args.embeddings!r} — '
              f'cosine similarity column will be omitted. '
              f'Pass --embeddings PATH to specify the location.')

    # Use non-interactive backend if --no-show
    if args.no_show:
        matplotlib.use('Agg')

    plot_overview(data, args)
    plot_confusion_heatmap(data['confusion_pairs'], args.heatmap_top,
                           data.get('serial', '?'), args.output, args.no_show)

    print('Done.')


if __name__ == '__main__':
    main()
