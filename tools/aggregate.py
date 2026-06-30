#!/usr/bin/env python3
"""
aggregate.py — aggregate ALL submissions in data/submissions/ into an `outcome/` folder:
per-task result summaries across participants.

  outcome/
    SUMMARY.txt                  headline results across every task
    draw_boundary_scores.csv     one row per participant x stimulus x method
    draw_summary.txt             mean Boundary-F / P / R / VI per method (overall + by dataset/density)
    bws_long.csv                 one row per participant x BWS item (decoded best/worst method)
    bws_summary.txt              best/worst tallies + Best-Worst score per method (overall + by dataset)
    questionnaire_long.csv       one row per participant (all Likert items + open text)
    questionnaire_summary.txt    n / mean / SD / 1-5 distribution per item + open-ended answers

DRAW task -> Boundary-F (BSDS protocol): does a method's partition put its dividing lines
where the human drew theirs?  Needs tools/boundaries_075.json (run export_boundaries.py once).

Usage:
    python tools/aggregate.py                 # all real participants (skips CC-TEST*)
    python tools/aggregate.py --include-test  # also include test-token submissions
    python tools/aggregate.py --df 0.75       # boundary manifest tag (default 0.75)

Only needs numpy / scipy / PIL — no moe-crowd-monitoring dependency at analysis time.
"""
import os, sys, csv, json, argparse
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, label as cc_label

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SUBS = os.path.join(REPO, "data", "submissions")
OUTDIR = os.path.join(REPO, "outcome")
STIMDIR = os.path.join(REPO, "stimuli_075")
METHODS = ["Voronoi", "Grid", "Quad"]          # draw manifest keys (capitalized)
BWS_METHODS = ["voronoi", "grid", "quad"]      # BWS data values (lowercase)
TOL_FRAC, WALL_W, MAXD = 0.0075, 3, 1024       # BSDS tolerance, human-wall width, raster cap


# ============================================================== Boundary-F (BSDS)
def _scale_of(W, H, maxd=MAXD):
    s = min(1.0, float(maxd) / max(W, H))
    return s, max(1, round(W * s)), max(1, round(H * s))

def raster_polylines(polylines, Ws, Hs, s, width=1):
    im = Image.new("L", (Ws, Hs), 0); dr = ImageDraw.Draw(im)
    for p in polylines:
        p = np.asarray(p, float)
        if len(p) < 2: continue
        dr.line([(float(x * s), float(y * s)) for x, y in p], fill=255, width=width, joint="curve")
    return np.asarray(im) > 0

def raster_polygon_edges(polys, Ws, Hs, s, width=1):
    rings = []
    for poly in polys:
        poly = np.asarray(poly, float)
        if len(poly) < 2: continue
        rings.append(np.vstack([poly, poly[:1]]))      # close the ring
    return raster_polylines(rings, Ws, Hs, s, width=width)

def method_label_map(polys, Ws, Hs, s):
    im = Image.new("I", (Ws, Hs), 0); dr = ImageDraw.Draw(im)
    for i, poly in enumerate(polys):
        poly = np.asarray(poly, float)
        if len(poly) < 3: continue
        dr.polygon([(float(x * s), float(y * s)) for x, y in poly], fill=i + 1)
    return np.asarray(im, dtype=np.int32)

def human_label_map(polylines, Ws, Hs, s):
    walls = raster_polylines(polylines, Ws, Hs, s, width=WALL_W)
    lab, _ = cc_label(~walls)
    return lab.astype(np.int32)

def extend_endpoints(polylines, W, H, margin_frac=0.04):
    """Snap a stroke's end/start onto the nearest image border when it stops just short of it,
    so a line drawn ACROSS the image actually separates it (closed loops are untouched —
    their endpoints sit in the interior, not near a border)."""
    m = margin_frac * min(W, H)
    out = []
    for p in polylines:
        p = np.asarray(p, float).copy()
        if len(p) < 2:
            out.append(p); continue
        for idx in (0, -1):
            x, y = p[idx]
            if x <= m: x = 0.0
            elif x >= W - m: x = float(W)
            if y <= m: y = 0.0
            elif y >= H - m: y = float(H)
            p[idx] = (x, y)
        out.append(p)
    return out

def close_loops(polylines, W, H, close_frac=0.08):
    """Close a stroke whose start and end land near each other (a hand-drawn 'circle the group'
    loop that didn't quite meet) by appending the start point — so its interior becomes a region.
    Spanning lines (ends far apart, e.g. snapped to opposite borders) are left open."""
    thr = close_frac * float(np.hypot(W, H))
    out = []
    for p in polylines:
        p = np.asarray(p, float)
        if len(p) >= 3 and np.hypot(p[0, 0] - p[-1, 0], p[0, 1] - p[-1, 1]) <= thr:
            p = np.vstack([p, p[:1]])
        out.append(p)
    return out

def boundary_prf(gt_bd, pr_bd, tol):
    g, p = gt_bd.sum(), pr_bd.sum()
    if g == 0 or p == 0:
        return dict(precision=0.0, recall=0.0, f=0.0)
    dt_gt = distance_transform_edt(~gt_bd)
    dt_pr = distance_transform_edt(~pr_bd)
    precision = float((dt_gt[pr_bd] <= tol).mean())
    recall = float((dt_pr[gt_bd] <= tol).mean())
    f = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return dict(precision=precision, recall=recall, f=f)

def _entropy(counts):
    p = counts / counts.sum(); p = p[p > 0]
    return float(-(p * np.log(p)).sum())

def variation_of_information(a, b):
    a, b = a.ravel(), b.ravel()
    _, ia = np.unique(a, return_inverse=True)
    _, ib = np.unique(b, return_inverse=True)
    n = a.size
    joint = np.zeros((ia.max() + 1, ib.max() + 1)); np.add.at(joint, (ia, ib), 1.0); joint /= n
    pa, pb = joint.sum(1), joint.sum(0); nz = joint > 0
    mi = float((joint[nz] * np.log(joint[nz] / np.outer(pa, pb)[nz])).sum())
    return _entropy(pa * n) + _entropy(pb * n) - 2 * mi

def region_metrics(human_lab, method_lab):
    """IoU-based region agreement, robust to differing region counts (best-match Covering).
      cover_h2m : area-weighted mean over HUMAN regions of best IoU with a method region
      cover_m2h : area-weighted mean over METHOD regions of best IoU with a human region
      group_iou : mean best-IoU over human FOREGROUND regions (drop largest=background) —
                  'is each circled group captured by one method region?'  (nan if <2 regions)
    """
    hl, ml = human_lab.ravel(), method_lab.ravel()
    valid = (hl > 0) & (ml > 0)                      # ignore wall pixels & method gaps
    hl, ml = hl[valid], ml[valid]
    if hl.size == 0:
        return dict(cover_h2m=float("nan"), cover_m2h=float("nan"),
                    group_iou=float("nan"), n_human=0, n_method=0)
    h_ids, h_inv = np.unique(hl, return_inverse=True)
    m_ids, m_inv = np.unique(ml, return_inverse=True)
    cont = np.zeros((len(h_ids), len(m_ids)))
    np.add.at(cont, (h_inv, m_inv), 1.0)
    area_h, area_m = cont.sum(1), cont.sum(0)
    union = area_h[:, None] + area_m[None, :] - cont
    iou = np.where(union > 0, cont / union, 0.0)
    best_h, best_m = iou.max(1), iou.max(0)
    cover_h2m = float((area_h * best_h).sum() / area_h.sum())
    cover_m2h = float((area_m * best_m).sum() / area_m.sum())
    group_iou = float(best_h[np.argsort(area_h)[:-1]].mean()) if len(area_h) > 1 else float("nan")
    return dict(cover_h2m=cover_h2m, cover_m2h=cover_m2h, group_iou=group_iou,
                n_human=int(len(h_ids)), n_method=int(len(m_ids)))


def score_partition(strokes_native, method_bnds, W, H):
    s, Ws, Hs = _scale_of(W, H)
    tol = TOL_FRAC * float(np.hypot(Ws, Hs))
    human_bd = raster_polylines(strokes_native, Ws, Hs, s, width=1)
    human_lab = human_label_map(close_loops(extend_endpoints(strokes_native, W, H), W, H), Ws, Hs, s)
    out = {}
    for m in METHODS:
        polys = method_bnds[m]
        prf = boundary_prf(human_bd, raster_polygon_edges(polys, Ws, Hs, s, width=1), tol)
        m_lab = method_label_map(polys, Ws, Hs, s); mask = m_lab > 0
        prf["vi"] = variation_of_information(human_lab[mask], m_lab[mask]) if mask.any() else float("nan")
        prf.update(region_metrics(human_lab, m_lab))
        out[m] = prf
    return out


# ============================================================== small helpers
def jload(s):
    try: return json.loads(s) if s else {}
    except Exception: return {}

def is_practice(r):
    return str(r.get("practice", "")).strip().lower() == "true"

def to_int(v):
    try: return int(float(v))
    except Exception: return None

def fstats(vals):
    a = np.array([v for v in vals if v is not None], float)
    if not len(a): return None
    return dict(n=len(a), mean=float(a.mean()),
                sd=float(a.std(ddof=1)) if len(a) > 1 else 0.0)

def parse_strokes(row, W, H):
    loops = jload(row.get("strokes"))
    if not isinstance(loops, list): return []
    cw = float(row.get("canvas_width") or W); ch = float(row.get("canvas_height") or H)
    sx, sy = W / cw, H / ch
    out = []
    for loop in loops:
        if not loop: continue
        pts = ([(pt["x"] * sx, pt["y"] * sy) for pt in loop] if isinstance(loop[0], dict)
               else [(pt[0] * sx, pt[1] * sy) for pt in loop])
        if len(pts) >= 2: out.append(np.asarray(pts, float))
    return out


# ============================================================== TASK 1: DRAW
def safemean(x):
    a = np.array(x, float); a = a[~np.isnan(a)]
    return float(a.mean()) if a.size else float("nan")

def render_montage(sid, strokes, W, H, method_bnds, sc, out_path):
    """3-panel montage: raw photo + human lines (red) + each method's edges (blue), metrics on top."""
    raw = os.path.join(STIMDIR, f"{sid}_raw.jpg")
    if not os.path.isfile(raw): return
    PW = 380; panels = []
    for m in METHODS:
        img = Image.open(raw).convert("RGB").resize((W, H)); d = ImageDraw.Draw(img)
        for poly in method_bnds[m]:
            d.line([(x, y) for x, y in poly] + [tuple(poly[0])], fill=(0, 140, 255), width=max(2, W // 350))
        for st in strokes:
            d.line([(float(x), float(y)) for x, y in st], fill=(235, 30, 30), width=max(3, W // 220))
        panels.append((m, img.resize((PW, max(1, round(PW * H / W))))))
    ph = max(p.size[1] for _, p in panels); lab = 34
    mont = Image.new("RGB", (PW * len(panels), ph + lab), (255, 255, 255)); dd = ImageDraw.Draw(mont)
    for i, (m, p) in enumerate(panels):
        mont.paste(p, (i * PW, lab)); s = sc[m]
        gi = "nan" if s["group_iou"] != s["group_iou"] else f"{s['group_iou']:.2f}"
        dd.text((i * PW + 5, 4), f"{m}  gIoU={gi} cov={s['cover_h2m']:.2f} F={s['f']:.2f}", fill=(0, 0, 0))
    os.makedirs(os.path.dirname(out_path), exist_ok=True); mont.save(out_path)

def aggregate_draw(subs, bmap, overlays=False):
    KEYS = ["group_iou", "cover_h2m", "cover_m2h", "f", "vi"]
    rows, by_method = [], {m: {k: [] for k in KEYS} for m in METHODS}
    by_ds, by_dens = {}, {}
    n_scored = 0
    for tok, data in subs.items():
        for r in [x for x in data if x.get("task") == "draw" and not is_practice(x)]:
            sid = r.get("stim_id"); mb = bmap.get(sid)
            if mb is None:
                print(f"  !! {tok}: {sid} not in manifest — skipped"); continue
            strokes = parse_strokes(r, mb["W"], mb["H"])
            if not strokes:
                print(f"  .. {tok}: {sid} no usable strokes — skipped"); continue
            sc = score_partition(strokes, mb["methods"], mb["W"], mb["H"]); n_scored += 1
            ds, dens = mb["dataset"], mb["density"]
            if overlays:
                render_montage(sid, strokes, mb["W"], mb["H"], mb["methods"], sc,
                               os.path.join(OUTDIR, "overlays", f"{tok}_{sid}.png"))
            valid = sc[METHODS[0]]["n_human"] >= 2   # a real partition (>=2 regions); else region metrics degenerate
            for m in METHODS:
                s = sc[m]
                by_method[m]["f"].append(s["f"])     # boundary-F is a line metric: defined for every drawing
                by_method[m]["group_iou"].append(s["group_iou"])                    # already nan if <2 regions
                by_method[m]["cover_h2m"].append(s["cover_h2m"] if valid else np.nan)
                by_method[m]["cover_m2h"].append(s["cover_m2h"] if valid else np.nan)
                by_method[m]["vi"].append(s["vi"] if valid else np.nan)
                by_ds.setdefault(ds, {mm: [] for mm in METHODS})[m].append(s["group_iou"])
                by_dens.setdefault(dens, {mm: [] for mm in METHODS})[m].append(s["group_iou"])
                gi = s["group_iou"]
                rows.append(dict(token=tok, stim_id=sid, dataset=ds, density=dens, method=m,
                                 group_iou=round(gi, 4) if gi == gi else "",
                                 cover_h2m=round(s["cover_h2m"], 4), cover_m2h=round(s["cover_m2h"], 4),
                                 boundary_f=round(s["f"], 4), vi=round(s["vi"], 4) if s["vi"] == s["vi"] else "",
                                 n_human=s["n_human"], n_method=s["n_method"], n_strokes=len(strokes)))
    with open(os.path.join(OUTDIR, "draw_scores.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["token", "stim_id", "dataset", "density", "method",
            "group_iou", "cover_h2m", "cover_m2h", "boundary_f", "vi", "n_human", "n_method", "n_strokes"])
        w.writeheader(); w.writerows(rows)

    n_grp = sum(1 for m in [METHODS[0]] for v in by_method[m]["group_iou"] if v == v)
    L = ["DRAW TASK — region agreement (IoU-based) + boundary-F",
         "=" * 72,
         f"{n_scored} drawings scored;  {n_grp} actually divide the image (>=2 regions) -> region metrics",
         "(near-border line endpoints are snapped to the edge, so a line drawn across the image counts)",
         "PRIMARY  group_IoU/cover_h2m computed on the {0} dividing drawings only".format(n_grp),
         "         group_IoU = mean best-IoU of each human region vs a method region (higher=better)",
         "         cover_h2m = area-weighted best-IoU (method covers human regions)",
         "SECOND.  boundary_F = line overlap, over ALL drawings · VI = region info-distance (lower=better)",
         "overlays/: red = human lines, blue = method edges", "",
         f"{'method':<10}{'group_IoU':>11}{'cover_h2m':>11}{'cover_m2h':>11}{'bound_F':>9}{'VI↓':>8}"]
    rank = []
    for m in METHODS:
        d = by_method[m]; cov = safemean(d["cover_h2m"]); rank.append((cov, m))
        L.append(f"{m:<10}{safemean(d['group_iou']):>11.3f}{cov:>11.3f}{safemean(d['cover_m2h']):>11.3f}"
                 f"{safemean(d['f']):>9.3f}{safemean(d['vi']):>8.3f}")
    rank.sort(reverse=True)
    winner = rank[0][1] if rank and rank[0][0] == rank[0][0] else None
    vi_order = sorted(METHODS, key=lambda m: safemean(by_method[m]["vi"]))
    L += ["", "  PRIMARY = cover_h2m (Covering); group_IoU is granularity-biased, diagnostic only",
          f"  -> best region match (cover_h2m): {winner}",
          f"  -> VI cross-check (lower=better): {' < '.join(vi_order)}"]
    def block(title, grp):
        out = ["", title, "-" * 72, f"  {'group':<12}" + "".join(f"{m:>10}" for m in METHODS)]
        for g in sorted(grp):
            out.append(f"  {g:<12}" + "".join(f"{safemean(grp[g][m]):>10.3f}" for m in METHODS))
        return out
    L += block("Mean group_IoU by DATASET", by_ds)
    L += block("Mean group_IoU by DENSITY", by_dens)
    open(os.path.join(OUTDIR, "draw_summary.txt"), "w").write("\n".join(L) + "\n")
    return dict(n=n_scored, winner=winner,
                means={m: safemean(by_method[m]["cover_h2m"]) for m in METHODS})


# ============================================================== TASK 2: BWS
def aggregate_bws(subs):
    long_rows, best, worst, appear = [], {m: 0 for m in BWS_METHODS}, {m: 0 for m in BWS_METHODS}, {m: 0 for m in BWS_METHODS}
    by_ds = {}
    n_items = 0
    for tok, data in subs.items():
        items = {}
        for r in [x for x in data if x.get("task") == "bws" and not is_practice(x)]:
            sid = r.get("stim_id")
            it = items.setdefault(sid, {"best": "", "worst": "", "ds": r.get("dataset", "")})
            if r.get("best_method"): it["best"] = r["best_method"]
            if r.get("worst_method"): it["worst"] = r["worst_method"]
        for sid, it in items.items():
            n_items += 1
            long_rows.append(dict(token=tok, stim_id=sid, dataset=it["ds"],
                                  best_method=it["best"], worst_method=it["worst"]))
            for m in BWS_METHODS: appear[m] += 1            # all 3 shown each item
            if it["best"] in best: best[it["best"]] += 1
            if it["worst"] in worst: worst[it["worst"]] += 1
            d = by_ds.setdefault(it["ds"], {m: [0, 0, 0] for m in BWS_METHODS})  # [best,worst,appear]
            for m in BWS_METHODS: d[m][2] += 1
            if it["best"] in d: d[it["best"]][0] += 1
            if it["worst"] in d: d[it["worst"]][1] += 1
    with open(os.path.join(OUTDIR, "bws_long.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["token", "stim_id", "dataset", "best_method", "worst_method"])
        w.writeheader(); w.writerows(long_rows)
    # Best-Worst score = (best - worst) / appearances  in [-1, +1]; higher = preferred
    def bw(m, b, wo, ap): return (b[m] - wo[m]) / ap[m] if ap[m] else float("nan")
    L = ["BWS TASK — most / least meaningful division",
         "=" * 72, f"{n_items} judged items across {len(subs)} participant(s)", "",
         f"{'method':<10}{'best':>7}{'worst':>7}{'BW score':>11}{'win%':>8}"]
    rank = []
    for m in BWS_METHODS:
        sc = bw(m, best, worst, appear); rank.append((sc, m))
        winp = 100 * best[m] / appear[m] if appear[m] else 0
        L.append(f"{m:<10}{best[m]:>7}{worst[m]:>7}{sc:>+11.3f}{winp:>7.0f}%")
    L += ["", "  BW score = (#best − #worst) / #appearances, range −1..+1 (higher = more preferred)",
          f"  -> most preferred: {max(rank)[1]}   least: {min(rank)[1]}", "",
          "Best-Worst score by DATASET", "-" * 72,
          f"  {'dataset':<10}" + "".join(f"{m:>10}" for m in BWS_METHODS)]
    for ds in sorted(by_ds):
        d = by_ds[ds]
        L.append(f"  {ds:<10}" + "".join(f"{((d[m][0]-d[m][1])/d[m][2] if d[m][2] else float('nan')):>+10.2f}" for m in BWS_METHODS))
    open(os.path.join(OUTDIR, "bws_summary.txt"), "w").write("\n".join(L) + "\n")
    rank.sort(reverse=True)
    return dict(n=n_items, order=[m for _, m in rank],
                scores={m: bw(m, best, worst, appear) for m in BWS_METHODS})


# ============================================================== TASK 3: QUESTIONNAIRE
QFIELDS = [  # (column, task, json-key, label)
    ("confidence", "post_reflection", "confidence", "Confidence in own decisions"),
    ("depend",     "post_scenario",   "depend",     "Accuracy of B depends on A's quality"),
    ("trust",      "post_scenario",   "trust",      "Trust depends on interpretability"),
    ("useful",     "post_scenario",   "useful",     "Usefulness for safety monitoring"),
    # legacy single-page questionnaire:
    ("clarity",    "post",            "clarity",    "[legacy] Clarity of divisions"),
    ("trust_old",  "post",            "trust",      "[legacy] Trust in regional counts"),
    ("pref",       "post",            "pref",       "[legacy] Usefulness for monitoring"),
]

def aggregate_questionnaire(subs):
    long_rows, opens = [], []
    cols = {c: [] for c, *_ in QFIELDS}
    for tok, data in subs.items():
        resp = {}
        for task in ("post_reflection", "post_scenario", "post"):
            resp[task] = next((jload(r.get("response")) for r in data if r.get("task") == task), {})
        row = {"token": tok}
        for col, task, key, _ in QFIELDS:
            v = to_int(resp[task].get(key))
            row[col] = v if v is not None else ""
            if v is not None: cols[col].append(v)
        otxt = (resp["post_reflection"].get("open") or "").strip()
        row["open"] = otxt
        if otxt: opens.append((tok, otxt))
        long_rows.append(row)
    with open(os.path.join(OUTDIR, "questionnaire_long.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["token"] + [c for c, *_ in QFIELDS] + ["open"])
        w.writeheader(); w.writerows(long_rows)
    L = ["QUESTIONNAIRE — Likert items (1–5)", "=" * 72,
         f"{'item':<42}{'n':>4}{'mean':>7}{'SD':>7}   dist 1..5"]
    head = {}
    for col, task, key, label in QFIELDS:
        st = fstats(cols[col])
        if not st: continue
        dist = "".join(f"{sum(1 for v in cols[col] if v == k):>3}" for k in (1, 2, 3, 4, 5))
        L.append(f"{label:<42}{st['n']:>4}{st['mean']:>7.2f}{st['sd']:>7.2f}   [{dist} ]")
        head[col] = st["mean"]
    L += ["", "OPEN-ENDED — what made a division clear / confusing:", "-" * 72]
    L += [f"  • [{tok}] {txt}" for tok, txt in opens] or ["  (none)"]
    open(os.path.join(OUTDIR, "questionnaire_summary.txt"), "w").write("\n".join(L) + "\n")
    return head


# ============================================================== main
def ensure_gitignore():
    gi = os.path.join(REPO, ".gitignore")
    lines = open(gi).read().splitlines() if os.path.isfile(gi) else []
    if "/outcome/" not in lines:
        with open(gi, "a") as f:
            f.write(("" if not lines or lines[-1] == "" else "\n") + "/outcome/\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-test", action="store_true", help="include CC-TEST* submissions")
    ap.add_argument("--df", default="0.75", help="boundary manifest tag (boundaries_<tag>.json)")
    ap.add_argument("--overlays", action="store_true",
                    help="write per-drawing human-vs-method overlays to outcome/overlays/")
    a = ap.parse_args()

    bpath = os.path.join(HERE, f"boundaries_{a.df.replace('.', '')}.json")
    bmap = json.load(open(bpath)) if os.path.isfile(bpath) else {}
    if not bmap:
        print(f"WARNING: {bpath} missing — run export_boundaries.py; draw scoring will be empty.")

    subs = {}
    for tok in sorted(os.listdir(SUBS)) if os.path.isdir(SUBS) else []:
        if tok.startswith("CC-TEST") and not a.include_test: continue
        fp = os.path.join(SUBS, tok, f"{tok}.csv")
        if os.path.isfile(fp):
            subs[tok] = list(csv.DictReader(open(fp, encoding="utf-8")))
    if not subs:
        sys.exit("no submissions found (did you run fetch_submissions.py?)")

    os.makedirs(OUTDIR, exist_ok=True); ensure_gitignore()
    print(f"aggregating {len(subs)} participant(s): {', '.join(subs)}")
    draw = aggregate_draw(subs, bmap, overlays=a.overlays) if bmap else {"n": 0, "winner": None, "means": {}}
    bws = aggregate_bws(subs)
    q = aggregate_questionnaire(subs)

    # master summary
    S = ["CROWD-PARTITION USER STUDY — AGGREGATE OUTCOME", "=" * 72,
         f"participants: {len(subs)}   ({', '.join(subs)})", "",
         "DRAW (cover_h2m = IoU-based Covering, higher = method regions match human regions):"]
    if draw["n"]:
        S += ["   " + "  ".join(f"{m} {draw['means'].get(m, float('nan')):.3f}" for m in METHODS),
              f"   -> best: {draw['winner']}   (n={draw['n']} drawings)"]
    else:
        S.append("   (no draw scores — boundary manifest missing?)")
    S += ["", "BWS (Best-Worst score, higher = more preferred):",
          "   " + "  ".join(f"{m} {bws['scores'][m]:+.3f}" for m in BWS_METHODS),
          f"   -> order: {' > '.join(bws['order'])}   (n={bws['n']} items)",
          "", "QUESTIONNAIRE (mean, 1–5):"]
    for col, _, _, label in QFIELDS:
        if col in q: S.append(f"   {label:<42}{q[col]:.2f}")
    S += ["", "files: draw_*, bws_*, questionnaire_* (+ this SUMMARY) in outcome/"]
    open(os.path.join(OUTDIR, "SUMMARY.txt"), "w").write("\n".join(S) + "\n")
    print("wrote ->", os.path.relpath(OUTDIR, REPO) + "/  (SUMMARY.txt + per-task csv/txt)")


if __name__ == "__main__":
    main()
