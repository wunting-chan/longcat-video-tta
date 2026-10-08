# Figure style guide

**Purpose.** Every figure, slide and report in this project uses this guide. The code that implements it is
[`figures/figstyle.py`](figstyle.py). [`figures/style_gallery.py`](style_gallery.py) renders one
example of each template into [`docs/figure_style/`](figure_style/).

**Sources.** The style follows DeepSeek's technical reports, measured on 2026-10-08 from the arXiv PDFs and HTML of:
- DeepSeek-V3 (2412.19437);
- DeepSeek-R1 (2501.12948);
- Janus-Pro (2501.17811);
- DeepSeek-VL2 (2412.10302);
- DeepSeek-OCR (2510.18234);
- DeepSeek-V2 (2405.04434) and DeepSeekMath (2402.03300).

Where DeepSeek has no convention (video, our own definitions), the rule here is ours and says so.

## 1. What to show, and in what order

DeepSeek's reports follow one pattern:

1. **Figure 1 is the headline result**, placed directly under the abstract. It takes one of two forms:
   - grouped bars of "ours against counterparts" on a few named benchmarks (V3, R1);
   - a scatter of performance against cost, with "ours" as red stars (Janus-Pro, VL2, OCR).

   One idea, legible at a glance.
2. **A method or pipeline diagram** (V3 Figure 2, R1 Figure 2).
3. **Training dynamics.** Accuracy and response length against training steps, with a reference line for a meaningful
   baseline (R1 Figure 1: "human participants").
4. **Ablations and breakdowns.** Lines across steps for each variant (R1 Figure 4, PPO against GRPO), lines by an ordered
   factor (R1 Figure 8, difficulty levels), and heatmaps of internal statistics (V3 Figure 9, expert load).
5. **Qualitative results** (Janus-Pro Figures 2 and 4, VL2 Figures 4–10, OCR Figures 7–12):
   - real outputs in clean grids;
   - the prompt or input under each image;
   - the compared method as the column header;
   - "Best viewed on screen" in the caption.

For us this means:
- **Figure 1** is the headline result for the question asked. Today that is "history makes models individual and
  creative".
- **Then** what was done, how it evolved over training or layers, where it lives in the network, and real examples.

Show the result that answers the question, not every number. Ablations and full tables go to an appendix or to
`DECISIONS.md`.

## 2. Chart type for each job

| Job | Template (`fs.use(...)`) | DeepSeek model | Helper |
|---|---|---|---|
| Compare conditions on a few named measures | `"bar"` | V3 Fig. 1, R1 Fig. 10 | `fs.grouped_bars`, `fs.top_legend` |
| Something over steps, lessons or layers | `"line"` | R1 Figs. 1, 4, 8 | `fs.line_series`, `fs.reference_line` |
| Trade-off between two quantities, one point per model | `"scatter"` | Janus-Pro Fig. 1a, VL2 Fig. 1 | `fs.scatter_points` |
| A matrix (condition × unit) | `"heatmap"` | V3 Fig. 9 | `fs.heatmap`, `fs.bottom_colorbar` |
| Real examples side by side | `"image"` | Janus-Pro Fig. 2 | `fs.image_cell` |
| A capability shown by example | `"image"` | Janus-Pro Fig. 4, VL2 Fig. 4 | `fs.title_tab`, `fs.prompt_bubble` |

Do not use pie charts, 3-D plots, dual y-axes (one exception, OCR Fig. 1a, which we do not copy), radar charts or
stacked bars whose parts cannot be read.

## 3. Palette

| Role | Hex | Use |
|---|---|---|
| Lead series | `#4D6BFE` | The condition the figure is about ("ours", our learner, different-history committees), drawn with white `///` hatching on bars. One lead per figure. |
| Closest counterpart | `#AAC1FF` | The previous version or ablation (plain DINO next to DINO + rule). |
| Other baselines | `#BDBDBD`, `#D4D4D4`, `#E8D2A0`, `#F5EBD2` | Further comparisons, in that order. |
| Reference population | `#E0E0E0` | Human experts, chance, a control. |
| Lines | `#1F3FD1`, `#E8222D`, `#2CA02C`, `#7F3FBF`, `#F28E2B`, `#555555` | R1-style primary lines, each with its own marker (circle, square, diamond, …). |
| Ordered categories | `Blues` ramp, light to dark | Difficulty levels, visit 1–3, early to late. |
| Reference line | `#2CA02C`, dashed | "chance", "no effect", "human level". |
| Our points in a scatter | `#E8222D` red star, black edge | Janus-Pro / VL2. |
| Heatmaps | `YlOrRd`; diverging data `RdBu_r` centred at 0 | V3 expert load. |
| Highlight in text answers | `#1AAE6F` green | The key phrase in a model's answer (Janus-Pro / VL2). |
| Text | `#262626` | Never pure black on bars. |

Colour must never be the only cue. The lead also differs by hatching, and lines differ by marker.

## 4. Typography and size

| Element | Rule |
|---|---|
| Bar charts and image captions | DejaVu Serif (V3 / R1 / Janus-Pro). |
| Line, scatter and heatmap plots | DejaVu Sans (R1 / Janus-Pro / V3 heatmap). |
| Report body | Palatino 11 pt (`mathpazo`), A4, 1-inch margins, header logo and rule (as in `ai-progress-report-*`). |
| Figure widths | Full text width 6.27 in (`fs.size("full", aspect)`), half 3.05 in, beamer 4.25 in. Build at print size; never shrink a figure in LaTeX by more than 10%. |
| Font sizes at print size | Base 7.5 pt; tick labels 6–7 pt; value labels 5.6 pt (6.6 pt bold for the lead); never below 5.5 pt. |
| Output | `fs.save(fig, path)` writes vector `.pdf` (Type 42 fonts) and 300-dpi `.png`. |

## 5. Labelling

DeepSeek conventions:
- **Axis titles** are bold in bar and scatter templates and state the unit: "Accuracy / Percentile (%)".
- **Category names under bars** are bold. The metric goes in small italic parentheses underneath: *(Pass@1)*. Use
  `metric_notes`.
- **The legend** is one framed row across the top of the plot (`fs.top_legend`). Inside a line plot it sits in the
  emptiest corner.
- **Values are printed on top of bars.** The lead's values are bold and slightly larger.
- **Scatter points are labelled by name** next to the point, and model families are joined by thin dashed lines.
- **Panel titles** are short and descriptive ("DeepSeek-R1-Zero AIME accuracy during training"). Sub-figures are
  (a), (b), with the sub-caption in LaTeX.

Project rules (from user feedback, 2026-10-06):
- **State what one mark is.** "Each dot is one class", "each line is one model pair".
- **State what each axis measures**, with its definition and unit: "Dissimilarity of class-similarity structure
  (1 − RSA, ×100)". A bare "PC 1" or "1 − cos" is never enough.
- **Name every colour, line style, band and whisker** in the legend or the caption: "whisker = one standard deviation
  over three seeds".
- **Never mix two units on one axis.**
- **No bare abbreviations.** Spell out "k-NN accuracy", "WordNet (Wu–Palmer) similarity" the first time.
- **No notes to ourselves on a figure** ("to-do", "safe claim", coaching). Figures inform the reader.
- **Claims stay inside the data.** Titles and captions say what was measured, on what, with how many seeds.

`fs.save` refuses to write a figure whose data axis lacks a label. Mark schematics and network drawings explicitly
with `fs.exempt(ax)`.

## 6. Captions

The format is DeepSeek's: **"Figure n | Lead sentence in bold." Then how to read it.**

- **The lead sentence states the finding, not the content.** "History reshapes only the late layers", not "RSA by
  layer".
- **The body defines:**
  - what one mark is, the axes and every encoding;
  - the data scope (models, lesson or stage, layer, number of seeds);
  - how error bars or bands are computed;
  - for examples, **the selection rule**: "drawn at random (seed 0) from the 412 qualifying classes".
- **End image figures with "Best viewed on screen."**
- **In LaTeX**, use `\DeclareCaptionLabelSeparator{dspipe}{\ |\ }` and
  `\captionsetup{labelsep=dspipe,font=normalsize,labelfont=normalfont}`.

## 7. Images and other media

These rules follow Janus-Pro and VL2.

- **Tiles** are square, the same size, and borderless on white, with 2–4 pt gaps between tiles and larger gaps between
  groups.
- **The compared condition is a header above each column** (serif, regular weight). The input or prompt is centred
  below, in serif.
- **The lead condition gets a thin `#4D6BFE` frame** (1.6 pt). Never use red frames or arrows drawn on the image.
- **Capability boxes** have a thin black outline with a black title tab at the top-left (`fs.title_tab`). A query or
  prompt goes in a light-blue rounded bubble (`fs.prompt_bubble`).
- **Real data only.** Validation images (`ai-eval-runs/slides/thumbs`, 192 px) for ImageNet classes, labelled with the
  class name. The caption states how the examples were chosen. Never cherry-pick without saying so; prefer a seeded
  random draw from a stated qualifying set.
- **Video** (our rule; DeepSeek shows no video):
  - show a **frame strip**: 4–8 frames, evenly spaced in time, with the timestamp ("t = 2.5 s") under each frame;
  - one row per condition, with the condition as the row label on the left;
  - for motion, add a line plot underneath against the same time axis;
  - in slides, embed the MP4 and use the frame strip as its poster.
- **Diagrams** use the same palette:
  - flat boxes with thin outlines and rounded corners at most 2 pt;
  - the lead component in `#4D6BFE` and supporting components in `#AAC1FF` or gray;
  - arrows 0.8 pt black.

## 8. Tables

- **Rules.** `booktabs` rules only (`\toprule`, `\midrule`, `\bottomrule`), with no vertical lines.
- **Columns.** The first column is the model or condition. Group header rows with `\cmidrule` (V3 Table 3 style).
- **Marking results.** Bold the best result in each column and underline the second best. Mark the direction in the
  header ("Accuracy ↑").
- **Units.** Each column header carries its unit. Values use the same number of decimals within a column.

## 9. Code template

```python
import matplotlib.pyplot as plt
import figstyle as fs

fs.use("bar")
fig, ax = plt.subplots(figsize=fs.size("full", 0.42))
fs.grouped_bars(ax, ["Random", "Relational", "Taxonomy"], ["DINO + rule", "DINO"],
                {"DINO + rule": [172, 186, 223], "DINO": [150, 149, 211]},
                errors={"DINO + rule": [8, 4, 20], "DINO": [22, 17, 10]}, lead="DINO + rule", decimals=0)
ax.set_ylabel("Original and appropriate associations (count)")
ax.set_xlabel("History (curriculum)")
fs.top_legend(fig, ax)
fs.save(fig, "figures/out/creative_yield")      # .pdf + .png, label check
```

```python
fs.use("line")
fig, ax = plt.subplots(figsize=fs.size("half", 0.75))
fs.line_series(ax, layers, [diff, same], ["Different history", "Same history, different seed"],
               bands=[(diff - sd1, diff + sd1), (same - sd2, same + sd2)])
fs.reference_line(ax, 0, "no difference")
ax.set_xlabel("Layer of the vision transformer"); ax.set_ylabel("1 − RSA (×100)")
ax.legend(loc="upper left"); fs.save(fig, "figures/out/layers")
```

```python
fs.use("image")
fig, axes = plt.subplots(2, 3, figsize=fs.size("full", 0.45))
for row, q in enumerate(queries):
    for j, (cls, head) in enumerate([(q, "Query"), (a[q], "Raised on Random"), (b[q], "Raised on Taxonomy")]):
        fs.image_cell(axes[row, j], thumb(cls), header=head if row == 0 else None, caption=name(cls),
                      frame=fs.BLUE if j == 2 else None)
fs.save(fig, "figures/out/examples")
```

## 10. Checklist before delivering

1. Does Figure 1 answer the question being asked?
2. Is there exactly one lead, with the same colour and hatching everywhere in the document?
3. Does every axis say what it measures and in what unit? Is every mark, band and whisker named?
4. Does the caption open with the finding, give the data scope and seeds, and state any example-selection rule?
5. Are all text sizes at least 5.5 pt at print size? No overlapping labels (open the PNG and look)?
6. Are the claims limited to what was measured? Is anything mentioned that the reader does not need?
7. Are both `.pdf` and `.png` saved, and is the generating script kept with the figures (`src/`)?

## Gallery

| Bars (V3 Fig. 1) | Lines (R1) |
|---|---|
| ![bars](figure_style/template_bar.png) | ![lines](figure_style/template_line.png) |
| **Scatter (Janus-Pro / VL2 Fig. 1)** | **Heatmap (V3 Fig. 9)** |
| ![scatter](figure_style/template_scatter.png) | ![heatmap](figure_style/template_heatmap.png) |
| **Image grid (Janus-Pro Fig. 2)** | **Capability boxes (Janus-Pro Fig. 4)** |
| ![grid](figure_style/template_image_grid.png) | ![boxes](figure_style/template_showcase.png) |
