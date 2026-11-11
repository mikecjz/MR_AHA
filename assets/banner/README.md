# README banners

Six wide banners for the top of the repository README: the three directions **A — MATRIX**,
**B — TOEPLITZ** and **C — SQUARE**, each cut for light and dark display. Every banner is
1600 × 400, shipped as an editable SVG and a 2× PNG (3200 × 800).

`concepts-light.png` and `concepts-dark.png` show all three directions in each mode.

| Direction | Light | Dark |
|---|---|---|
| A — MATRIX | [`light/concept-a-matrix`](light/concept-a-matrix.svg) | [`dark/concept-a-matrix`](dark/concept-a-matrix.svg) |
| B — TOEPLITZ | [`light/concept-b-toeplitz`](light/concept-b-toeplitz.svg) | [`dark/concept-b-toeplitz`](dark/concept-b-toeplitz.svg) |
| C — SQUARE | [`light/concept-c-square`](light/concept-c-square.svg) | [`dark/concept-c-square`](dark/concept-c-square.svg) |

## Displaying both modes on GitHub

GitHub honours `prefers-color-scheme` inside `<picture>`, so one snippet serves the right
cut automatically:

```html
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/banner/dark/concept-a-matrix.png">
    <img src="assets/banner/light/concept-a-matrix.png" width="880"
         alt="MR-A^H A — GPU forward normal operator for non-Cartesian MRI">
  </picture>
</p>
```

Swap the file name for B or C.

## The directions

* **A — MATRIX.** Lower-triangular Gram grid beside the lockup. The cells use a five-colour
  spectrum; hue follows the column and the shade deepens down the row, so the diagonal
  steps through all five hues. Light uses the saturated spectrum, dark uses a lifted
  variant built for the dark background.
* **B — TOEPLITZ.** A field of 30 px constant-diagonal columns wraps the wordmark only.
  The channel is flush left with the same 40 px margin as the top and bottom, and the
  caption sits below on the open background. The field is a muted spectrum in both modes:
  dusty mid-tones on light, dark muted tones on dark.
* **C — SQUARE.** The matrix and a 214 px Toeplitz-band square flank a centred lockup. The
  square's centre band is green and lies on the square's diagonal; the outermost bands are
  placed so their outer edges land on the corners, making every edge band a full-height
  triangle rather than a corner sliver. The Toeplitz square uses the matrix spectrum for
  the mode.

## Elements

Every banner is assembled from transparent element SVGs in [`elements/`](elements/), and
every element inside the assembled SVGs is a named group:

`background`, `matrix`, `wordmark`, `caption`, `toeplitz-field`, `channel`,
`toeplitz-square`, `rule`, `backends`.

You can therefore either open an assembled SVG and move/recolour whole groups, or compose
new layouts from the element files. See [`elements/README.md`](elements/README.md) for the
sizes and the layout reference (positions, scales, baselines).

## Palette

| Token | Light | Dark |
|---|---|---|
| background | `#FBFBF8` | `#0B0F15` |
| type / wordmark | `#101418` | `#F4F6F8` |
| matrix outlines | `#C9CBD2` | `#344050` |
| rule | `#E1E2E5` | `#232D3A` |
| caption | `#3C434E` / `#7C838F` | `#B9C1CE` / `#79838F` |
| backends block | `#5E6672` / `#A7ACB4` | `#AEB8C6` / `#6E7988` |
| matrix spectrum | `#D36C12` → `#12D38C` → `#1299D3` → `#122CD3` → `#7312D3` | `#F18322` → `#22F1A5` → `#22B3F1` → `#223EF1` → `#8A22F1` |
| B field | `#8A93A0`, `#A87563`, `#6E9C7C`, `#63989C`, `#6C7DA0`, `#8578A6` | `#1B2634`, `#6F4B3E`, `#3E6F53`, `#3E6A6F`, `#3E4F6F`, `#4D3E6F` |
| C square | matrix light base row | matrix dark base row |
| green centre band | `#12D38C` | `#22F1A5` |

The right-hand block reads **BACKENDS** — CUDA, Apple Silicon; **LANGUAGES SUPPORT** —
MATLAB · Python, C / C++.

## File notes

* All type is outlined to paths — no `<text>`, no font dependency, no rasters or filters.
  The SVGs are self-contained.
* Fonts used for the outlines: Inter (OFL), Latin Modern Roman (GUST Font License),
  DejaVu Sans Mono (Bitstream Vera license).
* The SVGs are 1600 × 400; the PNGs are 2× (3200 × 800). The sheets `concepts-*.png` are
  presentation images, not README assets.
