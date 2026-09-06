"""
SVGLogger – Keras callback that writes a status.svg training-progress
dashboard on every epoch end.  Can also be used standalone by calling
_update_svg(epoch, logs) directly.

Usage in a training script:
    from ymapnet.reporting.plotTrainingProgressToSVG import SVGLogger
    svg_logger = SVGLogger(monitor='val_hm_hdm', total_epochs=cfg['epochs'])
    model.fit(..., callbacks=[..., svg_logger])
"""

import re
import math
from collections import defaultdict, OrderedDict

# ─── Layout constants ─────────────────────────────────────────────────────────
CANVAS_W = 1600
COLS = 3
GAP_X = 16
GAP_Y = 20
MARGIN = 20
HEADER_H = 86

PLOT_W = (CANVAS_W - 2 * MARGIN - (COLS - 1) * GAP_X) // COLS
PLOT_H = 250
PAD_L = 64  # left  – y-axis labels
PAD_R = 12  # right
PAD_T = 32  # top   – plot title
PAD_B = 36  # bottom – x-axis labels
IW = PLOT_W - PAD_L - PAD_R
IH = PLOT_H - PAD_T - PAD_B

PALETTE = [
    '#4e9af1',
    '#f07060',
    '#50d890',
    '#f0c050',
    '#b07af5',
    '#f06fa0',
    '#40d4d4',
    '#f0a050',
    '#90d040',
    '#7070f0',
    '#e05050',
    '#50f0b0',
]

BG_DARK = '#1a1a2e'
BG_PLOT = '#16213e'
BG_HEADER = '#0f3460'
FG_MAIN = '#e0e0f0'
FG_DIM = '#8090aa'
GRID_COL = '#252545'
AXIS_COL = '#404060'

# ─── SVG helpers ──────────────────────────────────────────────────────────────


def _xe(s):
    return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')


def _fv(v):
    """Compact float label for axis ticks."""
    a = abs(v)
    if a == 0: return '0'
    if a >= 10000: return '%.0fk' % (v / 1000)
    if a >= 1000: return '%.1fk' % (v / 1000)
    if a >= 100: return '%.0f' % v
    if a >= 10: return '%.1f' % v
    if a >= 1: return '%.2f' % v
    if a >= 0.01: return '%.3f' % v
    return '%.2e' % v


def _nice_range(vmin, vmax):
    if vmin == vmax:
        delta = max(abs(vmin) * 0.1, 1e-6)
        return vmin - delta, vmax + delta
    span = vmax - vmin
    pad = span * 0.05
    return vmin - pad, vmax + pad


def _yticks(lo, hi, n=5):
    span = hi - lo
    if span == 0:
        return [lo]
    raw = span / n
    mag = 10**math.floor(math.log10(raw)) if raw > 0 else 1
    step = mag * min([1, 2, 2.5, 5, 10], key=lambda s: abs(s * mag - raw))
    start = math.ceil(lo / step) * step
    ticks, v = [], start
    while v <= hi + step * 0.01:
        ticks.append(v)
        v += step
    return ticks


# ─── Previous-serial comparison (text, for status.txt) ────────────────────────


def baseline_comparison_lines(logs, current_serial=None,
                              experiments_csv="knowledge/experiments.csv",
                              config_path="configuration.json"):
    """Build at-a-glance comparison lines for status.txt: every val_* metric in
    `logs` versus the previous experiment's final value pulled from experiments.csv.

    The baseline is the last data row in experiments.csv whose serial differs from
    the current one (read from configuration.json unless `current_serial` is given).
    Returns a list of text lines (with a header), or [] if no baseline is available.
    Best-effort: any I/O or parse failure yields [] so the caller can ignore it.
    """
    import csv, json

    if current_serial is None:
        try:
            with open(config_path) as f:
                current_serial = str(json.load(f).get("serial"))
        except Exception:
            current_serial = None

    try:
        with open(experiments_csv, newline="") as f:
            rows = list(csv.reader(f))
    except Exception:
        return []
    if len(rows) < 2:
        return []

    header = rows[0]
    serial_idx = header.index("serial") if "serial" in header else None

    base_row = None
    for row in reversed(rows[1:]):
        if serial_idx is not None and serial_idx < len(row) and row[serial_idx] == current_serial:
            continue
        base_row = row
        break
    if base_row is None:
        return []

    base_serial = base_row[serial_idx] if serial_idx is not None and serial_idx < len(base_row) else "?"

    baseline = {}
    for i, col in enumerate(header):
        if col.startswith("val_") and i < len(base_row):
            try:
                baseline[col] = float(base_row[i])
            except (TypeError, ValueError):
                pass  # empty / nan -> no baseline for this metric
    if not baseline:
        return []

    lines = ["Compared to serial %s (val metrics, current - baseline):" % base_serial]
    for k in sorted(logs):
        if not k.startswith("val_"):
            continue
        try:
            cur = float(logs[k])
        except (TypeError, ValueError):
            continue
        base = baseline.get(k)
        if base is None:
            lines.append("  %-30s %.6f   (new, no baseline)" % (k, cur))
        else:
            lines.append("  %-30s %.6f   %s=%.6f   %+.6f" % (k, cur, base_serial, base, cur - base))
    return lines


# ─── Metric grouping ──────────────────────────────────────────────────────────


def _detect_groups(base_keys):
    """
    Cluster base metric names into logical groups.
    Returns list of (group_title, [key, ...]).
    """
    groups = OrderedDict()

    token_cs = sorted(k for k in base_keys if re.fullmatch(r't\d+_cossim', k))
    token_ls = sorted(k for k in base_keys if re.fullmatch(r't\d+_loss', k))
    hm_subs = sorted(k for k in base_keys if re.fullmatch(r'hm_hdm_\w+', k))
    tok_multi = sorted(k for k in base_keys if k.startswith('tokens_multihot'))
    hm_16b = sorted(k for k in base_keys if k.startswith('hm_16b'))
    used = set(token_cs) | set(token_ls) | set(hm_subs) | set(tok_multi) | set(hm_16b)
    rest = sorted(k for k in base_keys if k not in used)

    if token_cs: groups['Token Cosine Similarities'] = token_cs
    if token_ls: groups['Token Losses'] = token_ls
    if hm_subs: groups['HDM Sub-metrics'] = hm_subs
    if tok_multi: groups['Tokens Multihot'] = tok_multi
    if hm_16b: groups['HM 16b'] = hm_16b
    for k in rest:
        groups[k] = [k]

    return list(groups.items())


# ─── Single-plot renderer ─────────────────────────────────────────────────────


def _render_plot(title, series_list, epochs):
    """
    Render one plot as an SVG <g> element (positioned at origin).

    series_list : [(label, color, [float|None, ...], is_val), ...]
    epochs      : [epoch_index, ...]  (x-axis, 0-based)
    """
    all_vals = [v for _, _, vals, _ in series_list for v in vals if v is not None and math.isfinite(v)]

    out = ['<g>']
    out.append('<rect width="%d" height="%d" rx="6" fill="%s"/>' % (PLOT_W, PLOT_H, BG_PLOT))

    # Title
    out.append('<text x="%d" y="%d" font-size="11" font-weight="bold" fill="%s" '
               'text-anchor="middle">%s</text>' % (PAD_L + IW // 2, PAD_T - 9, FG_MAIN, _xe(title)))

    if not all_vals or not epochs:
        out.append('<text x="%d" y="%d" font-size="10" fill="%s" '
                   'text-anchor="middle">No data yet</text>' % (PLOT_W // 2, PLOT_H // 2, FG_DIM))
        out.append('</g>')
        return '\n'.join(out)

    ylo, yhi = _nice_range(min(all_vals), max(all_vals))
    xlo = epochs[0]
    xhi = epochs[-1] if len(epochs) > 1 else epochs[0] + 1
    xspan = max(xhi - xlo, 1)
    yspan = yhi - ylo

    def px(ep):
        return PAD_L + (ep - xlo) / xspan * IW

    def py(v):
        return PAD_T + IH - (v - ylo) / yspan * IH

    # Y grid + labels
    for yv in _yticks(ylo, yhi, n=5):
        yp = py(yv)
        if PAD_T - 1 <= yp <= PAD_T + IH + 1:
            out.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" '
                       'stroke="%s" stroke-width="0.5"/>' % (PAD_L, yp, PAD_L + IW, yp, GRID_COL))
            out.append('<text x="%d" y="%.1f" font-size="8.5" fill="%s" '
                       'text-anchor="end" dominant-baseline="middle">%s</text>' % (PAD_L - 4, yp, FG_DIM, _xe(_fv(yv))))

    # X grid + labels
    n_ep = len(epochs)
    tick_every = max(1, n_ep // 6)
    for i, ep in enumerate(epochs):
        if i == 0 or i == n_ep - 1 or i % tick_every == 0:
            xp = px(ep)
            out.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" '
                       'stroke="%s" stroke-width="0.5"/>' % (xp, PAD_T, xp, PAD_T + IH, GRID_COL))
            out.append('<text x="%.1f" y="%d" font-size="8.5" fill="%s" '
                       'text-anchor="middle">%d</text>' % (xp, PAD_T + IH + 14, FG_DIM, ep + 1))

    # Axes
    out.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" stroke-width="1"/>' %
               (PAD_L, PAD_T, PAD_L, PAD_T + IH, AXIS_COL))
    out.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" stroke-width="1"/>' %
               (PAD_L, PAD_T + IH, PAD_L + IW, PAD_T + IH, AXIS_COL))

    # Data lines
    for label, color, vals, is_val in series_list:
        opacity = '0.65' if is_val else '1.0'
        dash = 'stroke-dasharray="4,3" ' if is_val else ''
        pts = []
        for ep, v in zip(epochs, vals):
            if v is not None and math.isfinite(v):
                pts.append('%.1f,%.1f' % (px(ep), py(v)))
            else:
                if len(pts) >= 2:
                    out.append('<polyline points="%s" fill="none" stroke="%s" '
                               'stroke-width="1.5" stroke-opacity="%s" %s/>' % (' '.join(pts), color, opacity, dash))
                pts = []
        if len(pts) >= 2:
            out.append('<polyline points="%s" fill="none" stroke="%s" '
                       'stroke-width="1.5" stroke-opacity="%s" %s/>' % (' '.join(pts), color, opacity, dash))
        elif len(pts) == 1:
            cx, cy = pts[0].split(',')
            out.append('<circle cx="%s" cy="%s" r="2.5" fill="%s" fill-opacity="%s"/>' % (cx, cy, color, opacity))

    # Legend
    # For grouped plots show unique base names; for single-metric plots show train/val
    has_val = any(is_v for _, _, _, is_v in series_list)
    unique_colors = list(dict.fromkeys((c, lbl, is_v) for lbl, c, _, is_v in series_list))

    # Determine legend area – right side inside plot
    max_label_chars = 20
    lx0 = PAD_L + IW - 4  # right edge of inner area
    ly0 = PAD_T + 6
    lh = 12

    shown = []
    for color, label, is_val in unique_colors:
        short = label if len(label) <= max_label_chars else label[:max_label_chars - 1] + '…'
        shown.append((color, short, is_val))

    # If too many legend rows would overflow, truncate
    max_rows = max(1, (IH - 10) // lh)
    shown = shown[:max_rows]

    for i, (color, short, is_val) in enumerate(shown):
        ty = ly0 + i * lh + 9
        lx1 = lx0 - 28
        lx2 = lx0 - 6
        dash = 'stroke-dasharray="3,2" ' if is_val else ''
        op = '0.65' if is_val else '1.0'
        out.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" '
                   'stroke="%s" stroke-width="1.5" stroke-opacity="%s" %s/>' %
                   (lx1, ty - 4, lx2, ty - 4, color, op, dash))
        out.append('<text x="%.1f" y="%.1f" font-size="7.5" fill="%s" '
                   'text-anchor="end">%s</text>' % (lx1 - 3, ty, FG_DIM, _xe(short)))

    # "dashed = val" hint for grouped plots
    if has_val and len(shown) > 2:
        out.append('<text x="%d" y="%d" font-size="7" fill="%s" text-anchor="middle">'
                   'solid=train · dashed=val</text>' % (PAD_L + IW // 2, PAD_T + IH + 28, FG_DIM))

    out.append('</g>')
    return '\n'.join(out)


# ─── SVGLogger class ──────────────────────────────────────────────────────────

import tensorflow as tf


class SVGLogger(tf.keras.callbacks.Callback):
    """
    Keras callback that produces status.svg after each epoch.

    Parameters
    ----------
    output_path   : path to write the SVG file
    monitor       : metric name to track as primary monitor
    total_epochs  : total number of training epochs (for ETA / progress)
    mode          : 'max' or 'min' for the monitor metric
    """

    def __init__(self, output_path='status.svg', monitor='val_hm_hdm', total_epochs=0, mode='max'):
        super().__init__()
        self.output_path = output_path
        self.monitor = monitor
        self.total_epochs = total_epochs
        self.mode = mode

        self._history = defaultdict(list)  # metric_name -> [float|None, ...]
        self._epochs = []  # list of epoch indices seen
        self._epoch_times = []  # [(epoch_idx, timestamp), ...]
        self._best_epoch = None
        self._best_val = None
        self._preamble = None

    # ── Keras callback hooks ─────────────────────────────────────────────────

    def on_epoch_end(self, epoch, logs=None):
        self._update_svg(epoch, logs or {})

    # ── Core ─────────────────────────────────────────────────────────────────

    def _update_svg(self, epoch, logs, preamble=None):
        import datetime
        now = datetime.datetime.now()
        self._epoch_times.append((epoch, now.timestamp()))
        if preamble:
            self._preamble = preamble

        # Record history for every metric in logs
        self._epochs.append(epoch)
        n = len(self._epochs)
        for k, v in logs.items():
            try:
                self._history[k].append(float(v))
            except (TypeError, ValueError):
                self._history[k].append(None)
        # Pad keys that were absent this epoch
        for k in list(self._history):
            while len(self._history[k]) < n:
                self._history[k].append(None)

        # Track best epoch
        mon_vals = self._history.get(self.monitor, [])
        valid = [(i, v) for i, v in enumerate(mon_vals) if v is not None]
        if valid:
            fn = max if self.mode == 'max' else min
            bi, bv = fn(valid, key=lambda x: x[1])
            self._best_epoch = self._epochs[bi]
            self._best_val = bv

        # Timing
        eta_str = epoch_time_str = 'N/A'
        if len(self._epoch_times) >= 2:
            ep_span = self._epoch_times[-1][0] - self._epoch_times[0][0]
            if ep_span > 0:
                elapsed = self._epoch_times[-1][1] - self._epoch_times[0][1]
                spe = elapsed / ep_span
                mins, secs = divmod(int(spe), 60)
                epoch_time_str = ('%dm %02ds' % (mins, secs)) if mins else ('%ds' % secs)
                if self.total_epochs > 0:
                    rem = self.total_epochs - (epoch + 1)
                    if rem > 0:
                        eta_ts = now + datetime.timedelta(seconds=spe * rem)
                        eta_str = eta_ts.strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        eta_str = 'Done'

        svg = self._build_svg(epoch, now, eta_str, epoch_time_str)
        try:
            with open(self.output_path, 'w') as fh:
                fh.write(svg)
        except Exception as e:
            print('SVGLogger: could not write %s: %s' % (self.output_path, e))

    # ── SVG assembly ─────────────────────────────────────────────────────────

    def _build_svg(self, epoch, now, eta_str, epoch_time_str):
        import datetime

        # Split train / val keys; build set of base names
        all_keys = set(self._history)
        val_keys = {k for k in all_keys if k.startswith('val_')}
        trn_keys = all_keys - val_keys
        base_keys = trn_keys | {k[4:] for k in val_keys}

        groups = _detect_groups(base_keys)
        n_plots = len(groups)
        n_rows = max(1, math.ceil(n_plots / COLS))
        total_h = MARGIN + HEADER_H + GAP_Y + n_rows * (PLOT_H + GAP_Y) + MARGIN

        lines = []
        lines.append('<?xml version="1.0" encoding="UTF-8"?>')
        lines.append('<svg xmlns="http://www.w3.org/2000/svg" '
                     'width="%d" height="%d" '
                     'style="background:%s;font-family:\'Courier New\',monospace,sans-serif">' %
                     (CANVAS_W, total_h, BG_DARK))

        # ── Header block ──────────────────────────────────────────────────────
        hx, hy = MARGIN, MARGIN
        hw = CANVAS_W - 2 * MARGIN
        lines.append('<rect x="%d" y="%d" width="%d" height="%d" rx="8" fill="%s"/>' %
                     (hx, hy, hw, HEADER_H, BG_HEADER))

        epoch_label = ('Epoch %d / %d' % (epoch + 1, self.total_epochs)
                       if self.total_epochs > 0 else 'Epoch %d' % (epoch + 1))
        if self._preamble:
            epoch_label = self._preamble + '  |  ' + epoch_label

        best_label = ''
        if self._best_epoch is not None:
            best_label = 'Best epoch %d  |  %s = %.6f' % (self._best_epoch + 1, self.monitor, self._best_val)

        tx = hx + 18
        lines.append('<text x="%d" y="%d" font-size="16" font-weight="bold" fill="%s">%s</text>' %
                     (tx, hy + 26, FG_MAIN, _xe(epoch_label)))
        lines.append('<text x="%d" y="%d" font-size="12" fill="%s">'
                     'Updated: %s  |  Epoch time: %s  |  ETA: %s</text>' %
                     (tx, hy + 48, FG_DIM, _xe(now.strftime('%Y-%m-%d %H:%M:%S')), _xe(epoch_time_str), _xe(eta_str)))
        lines.append('<text x="%d" y="%d" font-size="12" fill="%s">'
                     'Monitor: %s  |  %s</text>' % (tx, hy + 68, FG_DIM, _xe(self.monitor), _xe(best_label)))

        # ── Plots ─────────────────────────────────────────────────────────────
        plot_y0 = MARGIN + HEADER_H + GAP_Y
        for idx, (group_title, group_keys) in enumerate(groups):
            row = idx // COLS
            col = idx % COLS
            ox = MARGIN + col * (PLOT_W + GAP_X)
            oy = plot_y0 + row * (PLOT_H + GAP_Y)

            # Build series list for this group
            series = []
            for ki, key in enumerate(group_keys):
                color = PALETTE[ki % len(PALETTE)]
                if key in self._history:
                    series.append((key, color, self._history[key], False))
                val_key = 'val_' + key
                if val_key in self._history:
                    series.append((val_key, color, self._history[val_key], True))

            lines.append('<g transform="translate(%d,%d)">' % (ox, oy))
            lines.append(_render_plot(group_title, series, self._epochs))
            lines.append('</g>')

        lines.append('</svg>')
        return '\n'.join(lines)


# ─── Placeholder SVG ──────────────────────────────────────────────────────────


def writeStatusSVGPlaceholder(output_path='status.svg'):
    """
    Write a branded Y-Map-Net splash screen to *output_path* (default
    'status.svg').  Call this once before training starts so the status page
    shows something meaningful while waiting for the first epoch to complete.
    """
    import datetime

    W, H = CANVAS_W, 520

    # ── COCO-style stick figure (17 keypoints), centred in a 200×320 box ──────
    # Keypoint indices: nose=0, leye=1, reye=2, lear=3, rear=4,
    #   lsho=5, rsho=6, lelb=7, relb=8, lwri=9, rwri=10,
    #   lhip=11, rhip=12, lkne=13, rkne=14, lank=15, rank=16
    kp = {
        0: (0, -110),  # nose
        1: (-14, -122),  # left eye
        2: (14, -122),  # right eye
        3: (-26, -116),  # left ear
        4: (26, -116),  # right ear
        5: (-44, -74),  # left shoulder
        6: (44, -74),  # right shoulder
        7: (-64, -24),  # left elbow
        8: (64, -24),  # right elbow
        9: (-68, 28),  # left wrist
        10: (68, 28),  # right wrist
        11: (-30, 30),  # left hip
        12: (30, 30),  # right hip
        13: (-34, 104),  # left knee
        14: (34, 104),  # right knee
        15: (-32, 178),  # left ankle
        16: (32, 178),  # right ankle
    }
    # COCO skeleton connectivity
    bones = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),  # head
        (5, 6),
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),  # arms
        (5, 11),
        (6, 12),
        (11, 12),  # torso
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),  # legs
    ]

    # Colour by body part
    def _bone_color(a, b):
        pair = frozenset((a, b))
        if pair <= {0, 1, 2, 3, 4}: return '#f07060'  # head
        if pair & {5, 6, 7, 8, 9, 10}: return '#4e9af1'  # arms
        if pair & {11, 12} and pair & {5, 6}: return '#50d890'  # torso
        if pair == frozenset({11, 12}): return '#50d890'
        return '#f0c050'  # legs

    # Centre of figure area
    cx, cy = W // 2, 280

    lines = ['<?xml version="1.0" encoding="UTF-8"?>']
    lines.append('<svg xmlns="http://www.w3.org/2000/svg" '
                 'width="%d" height="%d" '
                 'style="background:%s;font-family:\'Courier New\',monospace,sans-serif">' % (W, H, BG_DARK))

    # ── CSS animation ─────────────────────────────────────────────────────────
    lines.append('''<defs>
  <style>
    .pulse { animation: pulse 2.4s ease-in-out infinite; }
    @keyframes pulse {
      0%,100% { opacity: 0.55; r: 5; }
      50%      { opacity: 1.00; r: 7; }
    }
    .dots { animation: dots 1.8s steps(3, end) infinite; }
    @keyframes dots {
      0%   { opacity: 0; }
      100% { opacity: 1; }
    }
  </style>
  <filter id="glow">
    <feGaussianBlur stdDeviation="3.5" result="blur"/>
    <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>
</defs>''')

    # ── Header ────────────────────────────────────────────────────────────────
    hx, hy, hw = MARGIN, MARGIN, W - 2 * MARGIN
    lines.append('<rect x="%d" y="%d" width="%d" height="%d" rx="8" fill="%s"/>' % (hx, hy, hw, HEADER_H, BG_HEADER))
    lines.append('<text x="%d" y="%d" font-size="22" font-weight="bold" fill="%s">'
                 'Y-Map-Net  –  2D Pose Estimation</text>' % (hx + 18, hy + 32, FG_MAIN))
    lines.append('<text x="%d" y="%d" font-size="12" fill="%s">'
                 'Preparing training pipeline…  |  %s</text>' %
                 (hx + 18, hy + 58, FG_DIM, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    lines.append('<text x="%d" y="%d" font-size="11" fill="%s">'
                 'Waiting for epoch 1 results – this view refreshes automatically</text>' % (hx + 18, hy + 76, FG_DIM))

    # ── Background grid (heatmap flavour) ────────────────────────────────────
    grid_x0, grid_y0 = MARGIN, MARGIN + HEADER_H + GAP_Y
    grid_w = W - 2 * MARGIN
    grid_h = H - grid_y0 - MARGIN
    lines.append('<rect x="%d" y="%d" width="%d" height="%d" rx="6" fill="%s"/>' %
                 (grid_x0, grid_y0, grid_w, grid_h, BG_PLOT))

    dot_step = 28
    import random
    rng = random.Random(42)
    for gy in range(grid_y0 + dot_step, grid_y0 + grid_h, dot_step):
        for gx in range(grid_x0 + dot_step, grid_x0 + grid_w, dot_step):
            op = rng.uniform(0.04, 0.14)
            r = rng.uniform(1.5, 3.0)
            lines.append('<circle cx="%d" cy="%d" r="%.1f" fill="%s" opacity="%.2f"/>' %
                         (gx, gy, r, PALETTE[rng.randint(0,
                                                         len(PALETTE) - 1)], op))

    # ── Skeleton bones ────────────────────────────────────────────────────────
    for (a, b) in bones:
        x1, y1 = cx + kp[a][0], cy + kp[a][1]
        x2, y2 = cx + kp[b][0], cy + kp[b][1]
        col = _bone_color(a, b)
        lines.append('<line x1="%d" y1="%d" x2="%d" y2="%d" '
                     'stroke="%s" stroke-width="3.5" stroke-linecap="round" '
                     'opacity="0.85" filter="url(#glow)"/>' % (x1, y1, x2, y2, col))

    # ── Keypoint circles (pulsing) ────────────────────────────────────────────
    for idx, (kx, ky) in kp.items():
        ax, ay = cx + kx, cy + ky
        # Stagger animation delay so joints don't all pulse in sync
        delay = '%.2fs' % (idx * 0.14)
        color = '#f07060' if idx <= 4 else ('#4e9af1' if idx <= 10 else '#f0c050')
        lines.append('<circle cx="%d" cy="%d" r="5" fill="%s" '
                     'style="animation:pulse 2.4s ease-in-out %s infinite" '
                     'filter="url(#glow)"/>' % (ax, ay, color, delay))

    # ── Head circle ───────────────────────────────────────────────────────────
    lines.append('<circle cx="%d" cy="%d" r="20" fill="none" '
                 'stroke="#f07060" stroke-width="2.5" opacity="0.7" '
                 'filter="url(#glow)"/>' % (cx, cy - 140))

    # ── "Initializing" label ──────────────────────────────────────────────────
    lines.append('<text x="%d" y="%d" font-size="14" fill="%s" text-anchor="middle" '
                 'font-weight="bold">Initializing</text>' % (cx, cy + 215, FG_MAIN))
    lines.append('<text x="%d" y="%d" font-size="11" fill="%s" text-anchor="middle">'
                 'Training has not started yet – charts will appear after epoch 1</text>' % (cx, cy + 236, FG_DIM))

    lines.append('</svg>')
    svg = '\n'.join(lines)
    try:
        with open(output_path, 'w') as fh:
            fh.write(svg)
    except Exception as e:
        print('writeStatusSVGPlaceholder: could not write %s: %s' % (output_path, e))
