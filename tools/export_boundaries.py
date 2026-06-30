#!/usr/bin/env python3
"""
export_boundaries.py — ONE-TIME export of the method partition geometry the participants
were shown (Voronoi / Grid / Quad, df=0.75), so the study repo can score Boundary-F without
depending on moe-crowd-monitoring / shapely / the render caches at analysis time.

Run with the env that has moe-crowd-monitoring set up (shapely etc.):

    /net/.../venv_crowd_monitoring/bin/python tools/export_boundaries.py

Writes tools/boundaries_075.json:
    { "<stimulus_id>": {"W":int,"H":int,"dataset":str,"density":str,
                        "methods": {"Voronoi":[[[x,y],...],...], "Grid":[...], "Quad":[...]}}, ... }
Coordinates are in the ORIGINAL image frame (same frame the partitions were computed in).
Re-run only if the stimuli / df / partition logic change.
"""
import os, sys, json, csv
import numpy as np

MOE = "/net/vid-ssd1/storage/deeplearning/users/joh04637/moe-crowd-monitoring/paper/studie"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "boundaries_075.json")
sys.path.insert(0, MOE)
import render_partitions as rp   # __main__-guarded import

rp.DF = "0.75"           # the partitions the participants actually saw (stimuli_075)
rp._cache.clear()

sel = list(csv.DictReader(open(os.path.join(rp.BASE, "stimuli_selected.csv"))))
# group by (dataset, split) so we can free each split cache after use (memory)
from collections import defaultdict
by = defaultdict(list)
for r in sel:
    by[(r["dataset"], r["split"])].append(r)

out, missing = {}, []
for (ds, sp), rows in by.items():
    for r in rows:
        sid = r["stimulus_id"]
        e = rp.get_entry(ds, sp, r["img_id"])
        if e is None:
            missing.append(sid); continue
        H, W = e["original_img_shape"]
        bnds = rp.boundaries(e)
        out[sid] = {
            "W": int(W), "H": int(H), "dataset": ds, "density": r.get("density_bin", ""),
            "methods": {m: [np.asarray(p, float).round(2).tolist() for p in bnds[m]]
                        for m in ("Voronoi", "Grid", "Quad")},
        }
    rp._cache.clear()

json.dump(out, open(OUT, "w"))
print(f"exported {len(out)} stimuli -> {OUT}  ({os.path.getsize(OUT)/1e6:.2f} MB)")
if missing:
    print(f"MISSING ({len(missing)}):", missing[:10], "..." if len(missing) > 10 else "")
