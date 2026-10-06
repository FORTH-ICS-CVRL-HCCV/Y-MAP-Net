#!/usr/bin/python3
"""
Author : "Ammar Qammaz"
Copyright : "2024 Foundation of Research and Technology, Computer Science Department Greece, See license.txt"
License : "FORTH"

Benchmark skeleton resolvers with the official COCO keypoint AP (OKS) on val2017.

Resolution is evaluated in isolation from the network: the model is run once
(letterboxed, like --borders) on every val2017 image that has an annotated person
and its joint / PAF / person_lr_bridge channels are cached as int8.  Resolvers
then run on the cache in seconds, so they can be compared and tuned offline.

  # 1) cache model outputs (needs TF + the model, ~2 images/s)
  python3 -m ymapnet.skeletons.benchmarkSkeletons --cache 1000 --dir cache_skeletons

  # 2) compare resolvers (COCO keypoint AP, ymapnet/evaluation/cocoKeypointAP.py)
  python3 -m ymapnet.skeletons.benchmarkSkeletons --dir cache_skeletons --resolver old new new+aux

Ground-truth mode caches the C DataLoader's own training targets (no augmentation, built from
configuration.json and the compiled datasets/DataLoader switches) instead of model outputs.  It
measures how well the resolver can group PERFECT heatmaps, i.e. what a training-target change
(PAF encoding, limb topology, bridge encoding, ...) is worth before spending a training run on it;
'oracle' is the ceiling.  The archive must contain joint tables (cocoVal.pzpd does).

  # 3) cache + evaluate the DataLoader's ground truth (seconds, no TF)
  python3 -m ymapnet.skeletons.benchmarkSkeletons --gt /path/to/cocoVal/cocoVal.pzpd --cache 1000 \
      --dir cache_gt --resolver new+aux oracle --set seed_leftovers=true min_joints=1 min_person_score=0 depth_weight=0

--breakdown splits what a cache loses with grouping for free (oracle) into joint positions, missed joints
and person ranking, with the position-error and miss statistics behind it.  --flip (model outputs only)
caches the joint channels averaged with a mirrored pass (test-time flip), to compare against a plain cache.

  python3 -m ymapnet.skeletons.benchmarkSkeletons --dir cache_skeletons --breakdown
"""
import os
import sys
import json
import argparse
import numpy as np

ANNOTATIONS = 'datasets/coco/cache/annotations/person_keypoints_val2017.json'
IMAGES = 'datasets/coco/cache/coco/val2017/'
BRIDGE = 'person_lr_bridge'
# cached besides the 17 joints + 12 PAFs (plus the last channel, what convertIO hands YMAPNet.py as `depthmap`)
EXTRA_CHANNELS = (BRIDGE, 'rightjoints', 'leftjoints', 'depthmap', 'Person', 'Face', 'Hand', 'Foot', 'normalX', 'normalY',
                  'normalZ')
CFG_KEYS = ('keypoint_names', 'keypoint_parents', 'keypoint_children', 'paf_parents', 'heatmapPAFEncoding',
            'heatmapBridgeEncoding')
SIGMAS = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0  # COCO OKS


def personImageIds(count):
    ann = json.load(open(ANNOTATIONS))
    return sorted({a['image_id'] for a in ann['annotations'] if a['num_keypoints'] > 0})[:count], ann


def keptChannels(names):
    return list(range(29)) + [names.index(n) for n in EXTRA_CHANNELS if n in names] + [len(names) - 1]


def predict(est, frame):
    est.process(frame, borders=True)
    raw = np.asarray(est.keypoints_predictions)
    return np.array(raw[0] if raw.ndim == 4 else raw, np.float32)


def flipAveragedJoints(raw, est, frame):
    """Test-time flip: run the mirrored image, mirror its 17 joint channels back (left/right swapped) and
    average them with raw's.  Other channels are left as they are (signed PAFs are not mirror-symmetric)."""
    import cv2
    mir = predict(est, cv2.flip(frame, 1))
    names = est.cfg['keypoint_names']
    swap = [names.index(n.replace('left_', 'right_') if n.startswith('left_') else n.replace('right_', 'left_'))
            for n in names]
    # letterbox content columns [xo, xo+nw) (resize_image_with_borders); mirrored column i shows column 2xo+nw-1-i
    h, w = frame.shape[:2]
    W = raw.shape[1]
    nw = W if w / h > 1.0 else int(W * (w / h))
    src = 2 * ((W - nw) // 2) + nw - 1 - np.arange(W)
    ok = (src >= 0) & (src < W)
    back = raw[:, :, :17].copy()
    back[:, ok] = mir[:, src[ok]][:, :, swap]
    raw[:, :, :17] = 0.5 * (raw[:, :, :17] + back)
    return raw


def cacheOutputs(count, outDir, modelPath, flip=False):
    import cv2
    from ymapnet.core.YMAPNet import YMAPNet
    os.makedirs(outDir, exist_ok=True)
    ids, ann = personImageIds(count)
    files = {i['id']: i['file_name'] for i in ann['images']}

    est = YMAPNet(modelPath=modelPath, threshold=0, keypoint_threshold=50.0, engine='tensorflow', profiling=False,
                  compileModel=False, show=False, resolve_skeleton=False, estimate_person_id=False)
    names = est.cfg['heatmaps']
    keep = keptChannels(names)
    sizes = {}
    for k, iid in enumerate(ids):
        frame = cv2.imread(IMAGES + files[iid])
        raw = predict(est, frame)
        if flip:
            raw = flipAveragedJoints(raw, est, frame)
        np.savez_compressed('%s/%d.npz' % (outDir, iid),
                            hm=np.clip(np.rint(raw[:, :, keep]), -127, 127).astype(np.int8))
        sizes[iid] = (frame.shape[1], frame.shape[0])
        if k % 50 == 0:
            print('cached', k, '/', len(ids), flush=True)
    json.dump({'ids': ids, 'sizes': sizes, 'channels': [names[c] for c in keep], 'flip': flip,
               'cfg': {k: est.cfg[k] for k in CFG_KEYS if k in est.cfg}}, open(outDir + '/meta.json', 'w'))


def cacheGroundTruth(count, outDir, archive, configPath, batch=32):
    """Cache the C DataLoader's ground truth for the same images cacheOutputs uses.  The DataLoader is
    built like trainYMAPNet's (from configPath) with augmentations off, so the targets are letterboxed
    exactly as the model's --borders input; samples are matched to COCO ids by file name."""
    import re
    sys.path.append('datasets/DataLoader')
    from DataLoader import DataLoader
    os.makedirs(outDir, exist_ok=True)
    cfg = json.load(open(configPath))
    want, ann = personImageIds(count)
    sizeOf = {i['id']: (i['width'], i['height']) for i in ann['images']}
    db = DataLoader((cfg['inputHeight'], cfg['inputWidth'], cfg['inputChannels']),
                    (cfg['outputHeight'], cfg['outputWidth'], cfg['outputChannels']),
                    output16BitChannels=cfg['output16BitChannels'], streamData=1, batchSize=batch, numberOfThreads=4,
                    gradientSize=cfg['heatmapGradientSize'], PAFSize=cfg['heatmapPAFSize'], doAugmentations=0,
                    addPAFs=int(cfg['heatmapAddPAFs']), addBackground=int(cfg['heatmapGenerateSkeletonBkg']),
                    addDepthMap=int(cfg['heatmapAddDepthmap']), addDepthLevelsHeatmaps=int(cfg['heatmapAddDepthLevels']),
                    addNormals=int(cfg['heatmapAddNormals']), addSegmentation=int(cfg['heatmapAddSegmentation']),
                    addInstanceDetection=int(cfg.get('heatmapAddInstanceDetection', False)),
                    addSuperpoint=int(cfg.get('heatmapAddSuperpoint', False)),
                    superpointChannels=int(cfg.get('superpointChannels', 0)),
                    superpointPcaPath=cfg.get('superpointPcaFile', ''),
                    addGeolocation=int(cfg.get('heatmapAddGeolocation', False)), datasets=[['enabled', archive]],
                    vocabularyPath='ymapnet_model/vocabulary.json', synonymPath=cfg.get('synonymPath', None),
                    embeddingsPath=cfg.get('embeddingsPath', None), libraryPath='datasets/DataLoader/libDataLoader.so')
    db.updateJointDifficulty(cfg['keypoint_difficulty'])  # per-joint cone radius, as trainYMAPNet's dbTrain
    names = cfg['heatmaps']
    keep = keptChannels(names)
    wanted, got = set(want), []
    for start in range(0, db.numberOfSamples, batch):
        end = min(start + batch, db.numberOfSamples)
        _, out, _ = db.get_partial_update_IO_array(start, end)
        for k in range(end - start):
            digits = re.findall(r'\d{6,12}', str(db.get_filename_of_sample(start + k)))
            iid = int(digits[-1]) if digits else -1
            if iid in wanted:
                np.savez_compressed('%s/%d.npz' % (outDir, iid), hm=np.asarray(out[k])[:, :, keep].astype(np.int8))
                got.append(iid)
    got.sort()
    print('cached ground truth of', len(got), 'of', len(want), 'images', flush=True)
    json.dump({'ids': got, 'sizes': {iid: sizeOf[iid] for iid in got}, 'channels': [names[c] for c in keep],
               'cfg': {k: cfg[k] for k in CFG_KEYS if k in cfg}}, open(outDir + '/meta.json', 'w'))


def letterboxToImage(x, y, w, h, size=256):
    """Inverse of resize_image_with_borders: heatmap pixel -> original image pixel."""
    if w > h:
        nw, nh = size, int(size * h / w)
    else:
        nw, nh = int(size * w / h), size
    xo, yo = (size - nw) // 2, (size - nh) // 2
    return (x - xo) * w / nw, (y - yo) * h / nh


def imageToLetterbox(x, y, w, h, size=256):
    if w > h:
        nw, nh = size, int(size * h / w)
    else:
        nw, nh = int(size * w / h), size
    return x * nw / w + (size - nw) // 2, y * nh / h + (size - nh) // 2


def matchPeaksToGT(kp, threshold, anns, w, h, radius=6.0):
    """Every GT person takes, per annotated joint, the nearest unused detected peak within `radius`
    heatmap pixels.  -> (peaks per joint, [(annotation, [(joint, gt x, gt y, peak index or -1)])])"""
    from ymapnet.skeletons.peaks import findPeaks
    W = kp.shape[1]
    peaks = [findPeaks(kp[:, :, j], threshold - 120.0) for j in range(17)]
    used = [set() for _ in range(17)]
    people = []
    for a in anns:
        if a['iscrowd'] or a['num_keypoints'] == 0:
            continue
        g = np.array(a['keypoints'], np.float32).reshape(-1, 3)
        joints = []
        for j in range(17):
            if g[j, 2] == 0:
                continue
            gx, gy = imageToLetterbox(g[j, 0], g[j, 1], w, h, W)
            i = -1
            if len(peaks[j]):
                d = np.hypot(peaks[j][:, 0] + 0.5 - gx, peaks[j][:, 1] + 0.5 - gy)
                d[list(used[j])] = np.inf
                k = int(np.argmin(d))
                if d[k] <= radius:
                    used[j].add(k)
                    i = k
            joints.append((j, gx, gy, i))
        people.append((a, joints))
    return peaks, people


def oracleSkeletons(kp, threshold, anns, w, h, radius=6.0):
    """Upper bound for any resolver: the detected peaks grouped by ground truth (matchPeaksToGT)."""
    H, W = kp.shape[:2]
    peaks, people = matchPeaksToGT(kp, threshold, anns, w, h, radius)
    out = []
    for _, joints in people:
        sk = [0.0] * 51
        for j, _, _, i in joints:
            if i >= 0:
                sk[j * 3:j * 3 + 3] = [peaks[j][i, 0] / W, peaks[j][i, 1] / H, float(peaks[j][i, 2])]
        out.append(sk)
    return out


def runResolver(name, hm, channels, cfg, threshold, anns, w, h, params=None):
    kp = hm[:, :, :17]
    if name == 'oracle':
        return oracleSkeletons(kp, threshold, anns, w, h)
    paf = [hm[:, :, 17 + c] for c in range(12)]
    depth = np.clip(hm[:, :, -1] + 120.0, 0, 255).astype(np.uint8)  # as convertIO produces it
    args = (kp, paf, depth, cfg['keypoint_names'], cfg['keypoint_parents'], cfg['keypoint_children'],
            cfg['paf_parents'])
    if name == 'old':
        from ymapnet.utils.resolveJointHierarchy import resolveJointHierarchyNew
        return resolveJointHierarchyNew(*args, sanity_check=True, person_label_map=None, threshold=threshold)
    from ymapnet.skeletons import resolveSkeletons
    aux = name == 'new+aux'
    bridge = hm[:, :, channels.index(BRIDGE)] if (aux and BRIDGE in channels) else None
    sides = None
    if aux and 'rightjoints' in channels and 'leftjoints' in channels:
        sides = (hm[:, :, channels.index('rightjoints')], hm[:, :, channels.index('leftjoints')])
    depth = hm[:, :, channels.index('depthmap')] if (aux and 'depthmap' in channels) else None
    parts = {m: hm[:, :, channels.index(m)] for m in ('Hand', 'Foot', 'Face') if aux and m in channels}
    params = dict({'paf_encoding': cfg.get('heatmapPAFEncoding', 'signed'),
                   'bridge_encoding': cfg.get('heatmapBridgeEncoding', 'signed')}, **(params or {}))
    return resolveSkeletons(*args, threshold=threshold, bridge_heatmap=bridge, side_heatmaps=sides,
                            depth_heatmap=depth, part_heatmaps=parts, **params)


def openCache(cacheDir, limit):
    from ymapnet.evaluation.cocoKeypointAP import CocoKeypointGT
    meta = json.load(open(cacheDir + '/meta.json'))
    ids = meta['ids'][:limit] if limit else meta['ids']
    return CocoKeypointGT(ANNOTATIONS), meta, ids


def detections(iid, skeletons, W, H, w, h, scores=None):
    """skeletons (heatmap-normalised x, y, confidence per joint) -> COCO keypoint results in image pixels;
    score = mean joint confidence * joints / 17 unless `scores` (one per skeleton) is given"""
    dets = []
    for n, sk in enumerate(skeletons):
        kps, vs = [], []
        for j in range(17):
            x, y, v = sk[j * 3:j * 3 + 3]
            if v > 0:
                # +0.5: the DataLoader splats joints at int(coord), so a peak pixel covers [i, i+1)
                ix, iy = letterboxToImage(x * W + 0.5, y * H + 0.5, w, h, W)
                kps += [ix, iy, 1]
                vs.append(v / 240.0 if v > 1.0 else v)
            else:
                kps += [0, 0, 0]
        if vs:
            dets.append({'image_id': iid, 'category_id': 1, 'keypoints': kps,
                         'score': scores[n] if scores is not None else float(np.mean(vs)) * len(vs) / 17.0})
    return dets


def cocoStats(gt, dets, ids):
    """COCOeval keypoint stats: AP, AP50, AP75, APm, APl, AR, AR50, AR75, ARm, ARl"""
    from ymapnet.evaluation.cocoKeypointAP import cocoKeypointStats
    return cocoKeypointStats(gt, ids, dets)


def evaluate(cacheDir, resolvers, threshold, limit, params=None):
    gt, meta, ids = openCache(cacheDir, limit)
    cfg, channels = meta['cfg'], meta['channels']
    maps = {iid: np.load('%s/%d.npz' % (cacheDir, iid))['hm'] for iid in ids}  # int8: float32 needs ~11GB

    for name in resolvers:
        dets = []
        for iid in ids:
            hm = maps[iid].astype(np.float32)
            H, W = hm.shape[:2]
            w, h = meta['sizes'][str(iid)]
            dets += detections(iid, runResolver(name, hm, channels, cfg, threshold, gt.imgToAnns[iid], w, h, params),
                               W, H, w, h)
        print('\n=== resolver: %s  (%d images, %d skeletons) ===' % (name, len(ids), len(dets)))
        if not dets:
            print('no detections')
            continue
        from ymapnet.evaluation.cocoKeypointAP import printStats
        printStats(cocoStats(gt, dets, ids))


def breakdown(cacheDir, threshold, limit, radius=6.0):
    """Group the detected peaks by ground truth (oracle) and measure what each remaining error costs:
    found joints moved to their exact GT position, missed joints added at GT, people ranked by true OKS.
    Then the position error of the found joints and why the others were missed."""
    gt, meta, ids = openCache(cacheDir, limit)
    names = meta['cfg']['keypoint_names']
    rows = ('as detected (= oracle)', 'found joints at exact GT position', 'missed joints added at GT',
            'people ranked by true OKS')
    dets = {r: [] for r in rows}
    MISS = ('found', 'no activation', 'merged (peak taken by another person)',
            'displaced %g-%gpx' % (radius, 2 * radius), 'weak (no peak of its own)')
    err, miss = [], []  # err: (gt x, gt y, dx, dy, person size, person); miss: (joint, visibility, MISS index)
    person = 0
    for iid in ids:
        kp = np.load('%s/%d.npz' % (cacheDir, iid))['hm'][:, :, :17].astype(np.float32)
        H, W = kp.shape[:2]
        w, h = meta['sizes'][str(iid)]
        peaks, people = matchPeaksToGT(kp, threshold, gt.imgToAnns[iid], w, h, radius)
        found, exact, filled, oks = [], [], [], []
        for a, joints in people:
            person += 1
            g = np.array(a['keypoints'], np.float32).reshape(-1, 3)
            size = np.sqrt(a['area']) * W / max(w, h)  # heatmap px
            sf, se, sm, o = [0.0] * 51, [0.0] * 51, [0.0] * 51, 0.0
            for j, gx, gy, i in joints:
                at = [(gx - 0.5) / W, (gy - 0.5) / H]  # the GT position in skeleton coordinates (see detections)
                if i >= 0:
                    px, py, c = (float(v) for v in peaks[j][i])
                    sf[j * 3:j * 3 + 3] = [px / W, py / H, c]
                    sm[j * 3:j * 3 + 3] = [px / W, py / H, c]
                    se[j * 3:j * 3 + 3] = at + [c]
                    ix, iy = letterboxToImage(px + 0.5, py + 0.5, w, h, W)
                    o += np.exp(-((ix - g[j, 0]) ** 2 + (iy - g[j, 1]) ** 2) /
                                (2.0 * a['area'] * (2.0 * SIGMAS[j]) ** 2 + np.spacing(1)))
                    err.append((gx, gy, px + 0.5 - gx, py + 0.5 - gy, size, person))
                    cls = 0
                else:
                    sm[j * 3:j * 3 + 3] = at + [0.5]
                    d = np.hypot(peaks[j][:, 0] + 0.5 - gx, peaks[j][:, 1] + 0.5 - gy).min() if len(peaks[j]) else np.inf
                    v = kp[min(H - 1, int(gy)), min(W - 1, int(gx)), j]
                    cls = 1 if (v <= threshold - 120.0 or d == np.inf) else 2 if d <= radius else 3 if d <= 2 * radius else 4
                miss.append((j, g[j, 2], cls))
            found.append(sf)
            exact.append(se)
            filled.append(sm)
            oks.append(o / len(joints))
        for r, sks, scores in zip(rows, (found, exact, filled, found), (None, None, None, oks)):
            dets[r] += detections(iid, sks, W, H, w, h, scores)

    print('\n=== breakdown: %s (%d images), peaks grouped by ground truth ===' % (cacheDir, len(ids)))
    print('%-40s %6s %6s %6s' % ('', 'AP', 'APm', 'APl'))
    for r in rows:
        stats = cocoStats(gt, dets[r], ids)
        print('%-40s %6.3f %6.3f %6.3f' % (r, stats[0], stats[3], stats[4]))

    e = np.array(err)
    dist = np.hypot(e[:, 2], e[:, 3])
    print('\nposition error of %d found joints (heatmap px): mean %.2f  median %.2f  p90 %.2f  bias x %+.2f y %+.2f' %
          (len(e), dist.mean(), np.median(dist), np.percentile(dist, 90), e[:, 2].mean(), e[:, 3].mean()))
    sizes = (('<16', 0, 16), ('16-32', 16, 32), ('32-64', 32, 64), ('>=64', 64, np.inf))
    print('  mean by person size (sqrt area, heatmap px): ' + '  '.join(
        '%s %.2f' % (s, dist[(e[:, 4] >= lo) & (e[:, 4] < hi)].mean()) for s, lo, hi in sizes if
        ((e[:, 4] >= lo) & (e[:, 4] < hi)).any()))
    sx, sy = np.polyfit(e[:, 0] - W / 2, e[:, 2], 1)[0], np.polyfit(e[:, 1] - H / 2, e[:, 3], 1)[0]
    shift, jitter = [], []
    for p in np.unique(e[:, 5]):
        v = e[e[:, 5] == p, 2:4]
        if len(v) >= 5:
            shift.append((v.mean(0) ** 2).sum())
            jitter.append(((v - v.mean(0)) ** 2).sum(1).mean())
    print('  letterbox scale error x %+.2f%% y %+.2f%%;  squared error: whole-person shift %.2f, per joint %.2f px^2' %
          (100 * sx, 100 * sy, np.mean(shift), np.mean(jitter)))

    m = np.array(miss)
    print('\n%d annotated joints: ' % len(m) + ' | '.join('%s %.1f%%' % (s, 100 * np.mean(m[:, 2] == c))
                                                          for c, s in enumerate(MISS)))
    print('  found: visible %.3f, occluded %.3f' % tuple(np.mean(m[m[:, 1] == v, 2] == 0) for v in (2, 1)))
    print('  found per joint: ' + ' '.join('%s %.2f' % (names[j], np.mean(m[m[:, 0] == j, 2] == 0)) for j in range(17)))


if __name__ == '__main__':
    try:  # long, memory-hungry offline job: be the OOM killer's first victim, not a system service
        with open('/proc/self/oom_score_adj', 'w') as f:
            f.write('1000')
    except OSError:
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', default='cache_skeletons', help='cache directory')
    ap.add_argument('--cache', type=int, default=0, help='run the model on N val2017 person images first')
    ap.add_argument('--model', default='ymapnet_model')
    ap.add_argument('--gt', metavar='ARCHIVE', help='with --cache: cache the C DataLoader ground truth from this '
                    '.pzpd archive instead of model outputs (no TF needed)')
    ap.add_argument('--config', default='configuration.json', help='DataLoader configuration for --gt')
    ap.add_argument('--flip', action='store_true', help='with --cache (model outputs): average the joint channels '
                    'with a mirrored pass (test-time flip)')
    ap.add_argument('--breakdown', action='store_true', help='oracle AP lost to joint positions / missed joints / '
                    'person ranking, plus position-error and miss statistics (no resolvers unless --resolver)')
    ap.add_argument('--resolver', nargs='+', default=None,
                    help='old = utils.resolveJointHierarchy, new = ymapnet.skeletons (PAFs only), '
                         'new+aux = + person_lr_bridge, left/right joint aggregates, depth and Hand/Foot/Face, '
                         'oracle = GT grouping of the detected peaks (resolver upper bound); '
                         'default old new new+aux')
    ap.add_argument('--threshold', type=float, default=50.0, help='joint peak threshold on the 0..240 scale')
    ap.add_argument('--limit', type=int, default=0, help='evaluate only the first N cached images')
    ap.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE',
                    help='override resolveSkeletons keyword arguments, e.g. --set min_joints=1 owned_bonus=0.3')
    a = ap.parse_args()
    params = {}
    for kv in a.set:
        k, v = kv.split('=', 1)
        params[k] = json.loads(v) if v[:1] in '0123456789-[{tf' else v  # numbers / true / false, else string
    if a.cache and a.gt:
        cacheGroundTruth(a.cache, a.dir, a.gt, a.config)
    elif a.cache:
        cacheOutputs(a.cache, a.dir, a.model, a.flip)
    resolvers = a.resolver if a.resolver is not None else ([] if a.breakdown else ['old', 'new', 'new+aux'])
    if resolvers:
        evaluate(a.dir, resolvers, a.threshold, a.limit, params)
    if a.breakdown:
        breakdown(a.dir, a.threshold, a.limit)
