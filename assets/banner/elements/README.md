# Banner elements

Transparent, self-contained SVGs for composing banners. Each file holds one named group
(`<g id="...">`) and no background, so it drops onto either mode. Light and dark versions
carry the colours of that mode.

| Element | Intrinsic size (px) | Used in |
|---|---|---|
| [`matrix-214-light`](matrix-214-light.svg) / [`-dark`](matrix-214-dark.svg) | 214 × 214 | A, C |
| [`matrix-194-light`](matrix-194-light.svg) / [`-dark`](matrix-194-dark.svg) | 194 × 194 | B |
| [`wordmark-light`](wordmark-light.svg) / [`-dark`](wordmark-dark.svg) | 445 × 106 (drawn at size 112) | all; A/C use size 108 |
| [`caption-light`](caption-light.svg) / [`-dark`](caption-dark.svg) | 589 × 73 (23 / 20 px) | A, C; B uses 24 / 21 px |
| [`backends-light`](backends-light.svg) / [`-dark`](backends-dark.svg) | 212 × 239 | all |
| [`toeplitz-square-light`](toeplitz-square-light.svg) / [`-dark`](toeplitz-square-dark.svg) | 214 × 214 | C |
| [`toeplitz-field-light`](toeplitz-field-light.svg) / [`-dark`](toeplitz-field-dark.svg) | 870 × 246 | B |

Notes:

* The wordmark, caption and backends files are cropped toward their ink; the matrix, field
  and square files use a `0 0` local origin. The assembled banner SVGs are the canonical
  arrangements and already have every element in a named group — copy a group from there
  when you need an exact placement rather than re-measuring a cropped element.
* `caption-*.svg` is left-aligned at 23/20 with a 35 px baseline gap. C centres each line
  separately; the positions are in the table below.
* `toeplitz-field-*.svg` is the band field only. Banner B places a background-coloured
  `channel` rectangle over it before the wordmark.

## Layout reference (1600 × 400 canvas)

Coordinates are top-left for blocks and baseline for text, in the assembled coordinate
system.

### A — MATRIX

| Layer | Placement |
|---|---|
| matrix-214 | top-left (84, 93) |
| wordmark | call origin (382, 210), size 108 |
| caption | line 1 at (384, 274), line 2 at (384, 309); 23 / 20 px |
| rule | x = 1306, y = 86 → 312 |
| backends | right edge 1560, top 96 |

### B — TOEPLITZ

| Layer | Placement |
|---|---|
| matrix-194 | top-left (70, 72) |
| toeplitz-field | top-left (310, 46), 870 × 246 |
| channel | rect (350, 86, 506, 166), fill = banner background |
| wordmark | call origin (384, 212), size 112 |
| caption | line 1 at (390, 330), line 2 at (390, 365); 24 / 21 px |
| rule | x = 1306, y = 86 → 312 |
| backends | right edge 1560, top 96 |

### C — SQUARE

| Layer | Placement |
|---|---|
| matrix-214 | top-left (84, 93) |
| toeplitz-square | top-left (1063, 93), 214 × 214 (centre 1170, 200) |
| wordmark | call origin (469.5, 210), size 108 |
| caption | line 1 call x = 392.8, line 2 call x = 485.6, baselines 274 / 309; both centred on x = 680.5 |
| rule | x = 1330, y = 86 → 312 |
| backends | right edge 1560, top 96 |

The rules are 1 px lines in the theme rule colour; they are the only layer without a
separate element file. The B channel is likewise a simple background-coloured rectangle.
