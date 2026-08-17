"""
paper_style.py — Single source of truth for publication figure style

Every figure in the paper draws its widths, fonts, and colours from here, so a
journal-spec change is a one-line edit rather than a hunt through notebooks and
scripts.

**Why this module exists.** These constants used to be duplicated between
`build_results_paper_nb.py` and `fig_efield_age.py`. The copies drifted, and
the drift was invisible: the figure script never set `font.family` at all, so
it rendered in matplotlib's default DejaVu Sans while every other figure used
Arial, all the while carrying a comment claiming it matched the notebook.
Nothing errored — the figures just came out in the wrong typeface.

**Importing this module applies the style.** That is deliberate. The original
bug was a forgotten setup call, so using any constant from here guarantees the
rcParams that go with it. Call `apply_style()` again only to undo a later
override.

Journal specification (J Neurosci): figures are submitted at final size, with
a maximum width of 8.5 cm (1 column), 11.6 cm (1.5 column), or 17.6 cm
(2 column). The widths below sit at or just inside those limits.

NOTE: the minimum permitted font size has not been confirmed against the
journal's current author guidelines. The smallest type here is 6 pt. If they
specify an 8 pt floor, raise FONT_TICK, FONT_LEGEND, and FONT_ANNOTATION —
which is the whole point of them living in one place.
"""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    'CM_TO_IN', 'WIDTH_1COL', 'WIDTH_1_5COL', 'WIDTH_2COL',
    'FONT_FAMILY', 'FONT_AXIS_TITLE', 'FONT_TICK', 'FONT_PANEL_LABEL',
    'FONT_LEGEND', 'FONT_ANNOTATION', 'FONT_STATS_BOX',
    'SHAM_COLOR', 'ACTIVE_COLOR', 'ACCENT_GREEN', 'ACCENT_RED',
    'NEUTRAL_GRAY', 'REGRESSION_COLOR', 'CI_COLOR', 'CI_ALPHA', 'SESOI_COLOR',
    'AGE_MIN', 'AGE_MAX', 'AGE_YOUNG', 'AGE_OLD', 'AGE_CMAP',
    'age_to_hex', 'style_ax', 'stats_box', 'apply_style',
]

# --- Sizes -------------------------------------------------------------------
CM_TO_IN = 1 / 2.54
WIDTH_1COL = 8.5 * CM_TO_IN       # 8.5 cm  (journal max 8.5)
WIDTH_1_5COL = 11.5 * CM_TO_IN    # 11.5 cm (journal max 11.6)
WIDTH_2COL = 17.5 * CM_TO_IN      # 17.5 cm (journal max 17.6)

# --- Type --------------------------------------------------------------------
FONT_FAMILY = 'Arial'
FONT_AXIS_TITLE = 7
FONT_TICK = 6.5
FONT_PANEL_LABEL = 10
FONT_LEGEND = 6
FONT_ANNOTATION = 6
FONT_STATS_BOX = 6.5

# --- Colours -----------------------------------------------------------------
SHAM_COLOR = '#1565C0'
ACTIVE_COLOR = '#E64A19'
ACCENT_GREEN = '#2E7D32'
ACCENT_RED = '#C62828'
NEUTRAL_GRAY = '#757575'
REGRESSION_COLOR = '#404040'
CI_COLOR = 'gray'
CI_ALPHA = 0.18
SESOI_COLOR = '#BDBDBD'

AGE_MIN, AGE_MAX = 20, 80
AGE_YOUNG, AGE_OLD = '#1565C0', '#FFB300'
AGE_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'blue_gold', [AGE_YOUNG, AGE_OLD])


def apply_style() -> None:
    """
    Push the style into matplotlib's rcParams.

    Applied on import; call again only to undo a later override.
    """
    plt.rcParams.update({
        'font.family': FONT_FAMILY,
        'axes.labelsize': FONT_AXIS_TITLE,
        'xtick.labelsize': FONT_TICK,
        'ytick.labelsize': FONT_TICK,
        'legend.fontsize': FONT_LEGEND,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'savefig.dpi': 300,
        'figure.dpi': 110,

        # Mathtext ignores font.family and falls back to DejaVu, so without
        # these every $r$, $p$, $N$, $\\Delta$ and $\\leq$ renders in a
        # different typeface from the text beside it — inside the same label.
        'mathtext.fontset': 'custom',
        'mathtext.rm': FONT_FAMILY,
        'mathtext.it': f'{FONT_FAMILY}:italic',
        'mathtext.bf': f'{FONT_FAMILY}:bold',

        # Embed text as TrueType rather than converting it to paths, so the
        # publisher can still edit and search it.
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })


def age_to_hex(age):
    """Map an age onto the blue→gold gradient used for age throughout."""
    norm = np.clip((age - AGE_MIN) / (AGE_MAX - AGE_MIN), 0, 1)
    r, g, b, _ = AGE_CMAP(norm)
    return f'#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}'


def style_ax(ax, xlabel=None, ylabel=None, title=None):
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FONT_AXIS_TITLE)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FONT_AXIS_TITLE)
    if title:
        ax.set_title(title, fontsize=FONT_AXIS_TITLE)
    ax.tick_params(labelsize=FONT_TICK)
    return ax


def stats_box(ax, text, x=0.97, y=0.97, ha='right', va='top'):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=FONT_STATS_BOX,
            fontfamily=FONT_FAMILY, color=REGRESSION_COLOR, ha=ha, va=va,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      alpha=0.9, edgecolor='none'))


apply_style()
