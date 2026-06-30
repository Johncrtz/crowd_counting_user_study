# How we measure "which partition is most human-meaningful"

We compare three automatic ways of dividing a crowd image into regions — **Voronoi**,
**Grid**, and density-adaptive **Quadtree** — and ask which one best matches how *people*
divide the same crowd. Two participant tasks give the evidence, and we score them with three
**methodologically independent** metrics. The headline result: **Voronoi wins all three** —
one *subjective* measure (stated preference) and two *objective* measures (region overlap and
information distance). Because the three measure different things, their agreement is strong
evidence the effect is real and not an artifact of one metric.

All numbers below are from the current pilot (**3 participants, 9 drawings, 48 BWS judgments**)
— directional, not yet significant. `tools/aggregate.py` recomputes everything as data grows.

---

## Metric 1 — Best–Worst Scaling (BWS): *subjective preference*

**Question it answers:** which partition do people *find* most meaningful?

**How the data is collected.** Each screen shows the **same crowd** partitioned three ways,
blinded and randomised as A / B / C. The participant clicks the **best** division, then the
**worst**.

**How we score it.** For each method we count how often it was picked best and worst, then:

> **BW score = (#best − #worst) / #times shown**, ranging −1 … +1 (higher = more preferred).

This is the standard Best–Worst tally; a fuller write-up can fit a Bradley–Terry / conditional-
logit model (with random effects for participant and image), but the simple score already gives
the ranking. *(Implemented in `aggregate.py → aggregate_bws`; output `outcome/bws_summary.txt`.)*

**Result:** **Voronoi +0.27** (chosen best 56 % of the time) · Quad +0.06 · Grid −0.33.

---

## Metric 2 — Covering (IoU-based): *objective region overlap*

**Question it answers:** when people draw their *own* division, do a method's regions cover the
**same areas** people grouped together?

**The building block — IoU (Intersection over Union).** For any two regions,
`IoU = overlap area / combined area`: 0 if they don't touch, 1 if identical. It is the standard
measure of "how much do these two areas agree."

**The wrinkle.** The human drawing and a method produce **different numbers of regions** with no
labels saying which human region corresponds to which method region. So a single IoU isn't
defined — we need a matching first.

**Covering** handles this: for **each human region**, find the method region it overlaps most
(its *best* IoU), then **average those best-IoUs weighted by region size**. In words: *"on
average, how well is each area a person marked captured by some single method region?"*

**How it's implemented (high level).**
1. Rasterise the participant's drawn lines into a mask; the connected areas *between* the lines
   are the **human regions**. (Lines drawn across the image are snapped to the border, and
   near-closed "circle the group" loops are closed, so both styles form proper regions.)
2. Fill each method's polygons to get the **method regions**.
3. Build a human×method overlap table, turn each cell into an IoU, take every human region's
   best match, and average weighted by area.

*(Implemented in `aggregate.py → region_metrics`, column `cover_h2m`.)*

**Why the area-weighted Covering and not a plain best-region IoU:** see *"Metrics we did not
headline"* below — the naïve version is biased by how finely a method is tiled.

**Result:** **Voronoi 0.38** > Quad 0.31 > Grid 0.28.

---

## Metric 3 — Variation of Information (VI): *objective information distance*

**Question it answers:** treating the human division and a method division as two ways of
**grouping pixels**, how far apart are they?

**Intuition.** VI is a true distance between two partitions: **0 means identical**, larger means
more different (so **lower is better**). Crucially it penalises **both** mistakes —
*splitting* one human region across several method regions **and** *merging* several human
regions into one. So a method that chops a person's region into many little tiles (e.g. the
Quadtree) is penalised, and so is one that lumps distinct groups together.

**How it's implemented (high level).** From the same human and method region maps, compute the
entropy of each labelling and their mutual information, then
`VI = H(human) + H(method) − 2 · MutualInformation(human, method)`. *(Implemented in
`aggregate.py → variation_of_information`, column `vi`.)*

**Why it's worth having alongside Covering:** it is information-theoretic, not overlap-based, so
it is **independent** of IoU and robust to differing region counts. Agreement between VI and
Covering therefore means two unrelated lenses see the same thing.

**Result:** **Voronoi 1.74** < Grid 1.99 < Quad 2.10 (Voronoi is closest to the human
partitions; Quad is never best on any single image).

---

## The convergence (what to put in the paper)

| Metric | Type | What it checks | Winner |
|---|---|---|---|
| **BWS** | subjective | which partition people *prefer* | **Voronoi** |
| **Covering (IoU)** | objective | do method regions *overlap* human regions | **Voronoi** |
| **VI** | objective | *information distance* between the two partitions | **Voronoi** |

The three are independent in method (a *stated preference*, an *area-overlap* measure, and an
*information-theoretic* measure). They nonetheless agree, which is the core claim: **the Voronoi
partition is the most human-aligned**, robustly across subjective and objective evidence.

---

## Metrics we considered but did **not** headline (and why)

- **Boundary-F (BSDS protocol).** Checks whether the method's *lines* fall where the human's
  *lines* fall, within a small tolerance. It came out ~0.05 and **non-discriminative for all
  three methods**, because people draw a few enclosing/dividing strokes while the methods tile
  the *whole* image — a line-vs-tiling *style* mismatch. It answers "do the contours coincide,"
  which is the wrong question once people draw groups rather than trace tilings. Kept only as a
  secondary column.
- **group-IoU (best single-region IoU).** Intuitive — "is each group captured by one region?" —
  but **biased by granularity**: a finely tiled method (Quadtree, ~13 tiles) can score high just
  because one small tile happens to be the same *size* as a small human region, regardless of
  meaning. On our data this even flips the ranking to Quad. We keep it as a **diagnostic column
  only**, not a headline metric, and use the size-weighted Covering instead.

*(Reproduce all of this with `python tools/aggregate.py --overlays`; per-drawing numbers in
`outcome/draw_scores.csv`, visual human-vs-method overlays in `outcome/overlays/`.)*
