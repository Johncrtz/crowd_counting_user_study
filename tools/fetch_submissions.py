#!/usr/bin/env python3
"""
fetch_submissions.py — pull user-study submissions from OSF and lay them out locally.

For every submission (one <token>.csv in the OSF data component) this builds:

    data/submissions/<token>/
        <token>.csv             raw data exactly as downloaded
        results.txt             human-readable summary of the whole submission
        draw/<sid>_drawn.png    each drawn image with the participant's lines on it

Usage
-----
    export OSF_TOKEN=...                         # read-only OSF personal access token
    python tools/fetch_submissions.py            # pull + process ALL submissions
    python tools/fetch_submissions.py CC-TEST00  # only this token (still pulls it)
    python tools/fetch_submissions.py --local    # skip OSF; re-process CSVs already on disk

Notes
-----
* data/ is git-ignored — participant data must never be pushed to the public repo.
* The token is read from the environment, never stored. Revoke it on OSF when done.
"""
import os, sys, csv, json
from collections import Counter
from datetime import datetime, timezone

OSF_NODE = "a76qk"   # OSF data component (DataPipe writes <token>.csv here)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STIM = os.path.join(REPO, "stimuli_075")          # rendered stimulus images (<sid>_raw.jpg)
OUT  = os.path.join(REPO, "data", "submissions")


# --------------------------------------------------------------------------- OSF
def osf_get(url, token, binary=False, tries=5):
    """GET with bearer auth, re-attaching the header across OSF's cross-host redirects.
    Retries transient 5xx / network errors with backoff (the OSF API 500s intermittently)."""
    import requests, time
    H = {"Authorization": "Bearer " + token}
    last = None
    for attempt in range(tries):
        try:
            r = requests.get(url, headers=H, allow_redirects=False, timeout=60)
            hops = 0
            while r.status_code in (301, 302, 303, 307, 308) and hops < 6:
                r = requests.get(r.headers["Location"], headers=H, allow_redirects=False, timeout=60)
                hops += 1
            if r.status_code >= 500:
                last = f"HTTP {r.status_code} {r.reason}"
                time.sleep(2 * (attempt + 1)); continue
            r.raise_for_status()
            return r.content if binary else r.json()
        except requests.exceptions.RequestException as e:
            last = str(e)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"OSF request failed after {tries} attempts: {url}\n  last error: {last}")


def list_osf_csvs(token):
    url = f"https://api.osf.io/v2/nodes/{OSF_NODE}/files/osfstorage/?page[size]=100"
    out = []
    while url:
        d = osf_get(url, token)
        for f in d.get("data", []):
            a = f["attributes"]
            if a["kind"] == "file" and a["name"].endswith(".csv"):
                out.append((a["name"], f["links"]["download"]))
        url = (d.get("links") or {}).get("next")
    return out


# --------------------------------------------------------------------- draw overlay
def render_draw(row, out_path):
    """Resize the raw photo to the canvas the participant drew on, then stroke their
    open dividing lines onto it (white casing + dark core, exactly as they saw)."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return False, "Pillow not installed"
    sid = row.get("stim_id", "")
    raw = os.path.join(STIM, f"{sid}_raw.jpg")
    if not os.path.isfile(raw):
        return False, f"raw image missing ({sid}_raw.jpg)"
    try:
        cw, ch = int(float(row["canvas_width"])), int(float(row["canvas_height"]))
        strokes = json.loads(row.get("strokes") or "[]")
    except Exception as e:
        return False, f"bad stroke data ({e})"
    img = Image.open(raw).convert("RGB").resize((cw, ch))
    d = ImageDraw.Draw(img, "RGBA")
    for s in strokes:
        pts = [(p["x"], p["y"]) for p in s if "x" in p and "y" in p]
        if len(pts) < 2:
            continue
        d.line(pts, fill=(255, 255, 255, 235), width=6, joint="curve")  # white casing
        d.line(pts, fill=(17, 17, 17, 255),   width=3, joint="curve")   # dark core
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)
    return True, None


# --------------------------------------------------------------------- summary text
def jload(s):
    try:
        return json.loads(s) if s else {}
    except Exception:
        return {}


def is_practice(r):
    return str(r.get("practice", "")).strip().lower() == "true"


def write_results(token, rows, folder):
    L = []
    add = L.append
    bar, sub = "=" * 72, "-" * 72

    lang = next((r.get("language") for r in rows if r.get("language")), "?")
    subj = next((r.get("subject_id") for r in rows if r.get("subject_id")), "?")
    times = [float(r["time_elapsed"]) for r in rows
             if str(r.get("time_elapsed", "")).strip() not in ("", "null")]
    dur = f"{max(times) / 60000:.1f} min" if times else "?"

    add(bar); add(" CROWD-PARTITION USER STUDY — SUBMISSION"); add(bar)
    add(f" Token:      {token}")
    add(f" Subject ID: {subj}")
    add(f" Language:   {lang}")
    add(f" Trials:     {len(rows)}")
    add(f" Duration:   ~{dur}")
    add(f" Pulled:     {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}")
    add(sub); add("")

    # ---- demographics ----
    demo = next((jload(r.get("response")) for r in rows if r.get("task") == "demographics"), {})
    add("PARTICIPANT")
    if demo:
        for k, lab in [("age", "Age"), ("gender", "Gender"), ("role", "Role"),
                       ("vision", "Vision"), ("experience", "Crowd-counting experience"),
                       ("truthful", "Confirmed honest")]:
            if k in demo:
                add(f"  {lab + ':':<30}{demo[k]}")
    else:
        add("  (no demographics recorded)")
    add("")

    # ---- TASK 1: draw ----
    draws = [r for r in rows if r.get("task") == "draw"]
    add(f"TASK 1 — DRAW (divide the image with lines)   [{len(draws)} image(s)]")
    for r in draws:
        sid = r.get("stim_id", "?"); prac = is_practice(r)
        out = os.path.join(folder, "draw", f"{sid}{'_practice' if prac else ''}_drawn.png")
        ok, err = render_draw(r, out)
        status = os.path.relpath(out, folder) if ok else f"(overlay skipped: {err})"
        tag = " [practice]" if prac else ""
        add(f"  • {sid:<5}{tag:<11} {r.get('dataset', ''):<5} {r.get('density', ''):<7} "
            f"{r.get('n_strokes', '?')} line(s)   {status}")
    add("")

    # ---- TASK 2: BWS (handles both 1-row-per-item and legacy best/worst rows) ----
    items, order = {}, []
    for r in [x for x in rows if x.get("task") == "bws"]:
        sid = r.get("stim_id", "?")
        if sid not in items:
            items[sid] = {"best": "", "worst": "", "prac": is_practice(r),
                          "ds": r.get("dataset", "")}
            order.append(sid)
        if r.get("best_method"):
            items[sid]["best"] = r["best_method"]
        if r.get("worst_method"):
            items[sid]["worst"] = r["worst_method"]
    add(f"TASK 2 — BWS (best / worst division)   [{len(order)} item(s)]")
    add("  (A/B/C positions were randomized & blinded; method decoded below)")
    for sid in order:
        it = items[sid]; tag = " [practice]" if it["prac"] else ""
        add(f"  • {sid:<5}{tag:<11} {it['ds']:<5} "
            f"best={it['best'] or '—':<9} worst={it['worst'] or '—'}")
    cb, cw = Counter(), Counter()
    for sid in order:
        it = items[sid]
        if it["prac"]:
            continue
        if it["best"]:  cb[it["best"]] += 1
        if it["worst"]: cw[it["worst"]] += 1
    add("")
    add("  Method tally (excl. practice):")
    add(f"    {'method':<10}{'best':>6}{'worst':>7}{'net':>6}")
    for m in ["voronoi", "grid", "quad"]:
        add(f"    {m:<10}{cb[m]:>6}{cw[m]:>7}{cb[m] - cw[m]:>+6}")
    add("")

    # ---- questionnaire (new two-page; falls back to legacy single page) ----
    add("QUESTIONNAIRE")
    refl = next((jload(r.get("response")) for r in rows if r.get("task") == "post_reflection"), {})
    scen = next((jload(r.get("response")) for r in rows if r.get("task") == "post_scenario"), {})
    oldp = next((jload(r.get("response")) for r in rows if r.get("task") == "post"), {})
    if refl:
        add(f"  Confidence in own decisions (1-5):  {refl.get('confidence', '—')}")
        add("  What made a division clear / confusing:")
        add(f"    \"{(refl.get('open', '') or '').strip()}\"")
    if scen:
        add("  Scenario (System A divides  →  System B counts per region):")
        add(f"    Accuracy of B depends on A's quality (1-5):  {scen.get('depend', '—')}")
        add(f"    Trust depends on interpretability   (1-5):  {scen.get('trust', '—')}")
        add(f"    Usefulness for safety monitoring    (1-5):  {scen.get('useful', '—')}")
    if oldp and not (refl or scen):
        add("  (legacy questionnaire)")
        add(f"    Clarity (1-5):    {oldp.get('clarity', '—')}")
        add(f"    Trust (1-5):      {oldp.get('trust', '—')}")
        add(f"    Usefulness (1-5): {oldp.get('pref', '—')}")
    if not (refl or scen or oldp):
        add("  (no questionnaire recorded)")
    add(""); add(bar)

    with open(os.path.join(folder, "results.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


# --------------------------------------------------------------------------- main
def ensure_gitignore():
    """Keep participant data out of the public repo."""
    gi = os.path.join(REPO, ".gitignore")
    lines = []
    if os.path.isfile(gi):
        lines = open(gi, encoding="utf-8").read().splitlines()
    if "/data/" not in lines:
        with open(gi, "a", encoding="utf-8") as f:
            f.write(("" if not lines or lines[-1] == "" else "\n")
                    + "# participant submissions — never commit\n/data/\n")


def main():
    argv = sys.argv[1:]
    local = "--local" in argv
    wanted = [a for a in argv if not a.startswith("-")]
    os.makedirs(OUT, exist_ok=True)
    ensure_gitignore()

    if local:
        toks = sorted(d for d in os.listdir(OUT) if os.path.isdir(os.path.join(OUT, d)))
        if wanted:
            toks = [t for t in toks if t in wanted]
    else:
        token = os.environ.get("OSF_TOKEN", "").strip()
        if not token:
            sys.exit("ERROR: set OSF_TOKEN (read-only OSF personal access token), or use --local")
        csvs = list_osf_csvs(token)
        if wanted:
            csvs = [(n, u) for (n, u) in csvs if n[:-4] in wanted]
        if not csvs:
            print(f"no matching CSVs in OSF component {OSF_NODE}")
        toks = []
        for name, dl in csvs:
            tok = name[:-4]
            folder = os.path.join(OUT, tok)
            os.makedirs(folder, exist_ok=True)
            content = osf_get(dl, token, binary=True)
            with open(os.path.join(folder, name), "wb") as f:
                f.write(content)
            toks.append(tok)
            print(f"pulled {name}  ({len(content)} bytes)")

    for tok in sorted(set(toks)):
        folder = os.path.join(OUT, tok)
        csvpath = os.path.join(folder, f"{tok}.csv")
        if not os.path.isfile(csvpath):
            print(f"skip {tok}: no {tok}.csv on disk"); continue
        rows = list(csv.DictReader(open(csvpath, encoding="utf-8")))
        write_results(tok, rows, folder)
        n_draw = len([r for r in rows if r.get("task") == "draw"])
        print(f"  -> {tok}/results.txt  +  draw/ ({n_draw} overlay(s))   [{len(rows)} rows]")


if __name__ == "__main__":
    main()
