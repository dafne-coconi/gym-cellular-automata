"""
The render of the bulldozer consists of four subplots:
1. Local Grid
    + Grid centered at current position, visualizes agent's micromanagment
2. Global Grid
    + Whole grid view, visualizes agent's strategy
3. Gauge
    + Shows time until next CA update
4. Counts
    + Shows Forest vs No Forest cell counts. Translates on how well the agent is doing.
"""
import matplotlib.pyplot as plt
import numpy as np

from gym_cellular_automata.forest_fire.utils.neighbors import moore_n
from gym_cellular_automata.forest_fire.utils.render import (
    EMOJIFONT,
    TITLEFONT,
    align_marker,
    clear_ax,
    get_norm_cmap,
    parse_svg_into_mpl,
    plot_grid,
)

from . import svg_paths
from .config import CONFIG

# Figure Globals
TITLE = "Forest Fire\nBulldozer-v1"
TITLE_SIZE = 42
TITLE_POS = {"x": 0.121, "y": 0.96}
TITLE_ALIGN = "left"

COLOR_EMPTY = "#DDD1D3"  # Gray
COLOR_BURNED = "#DFA4A0"  # Light-Red
COLOR_TREE = "#A9C499"  # Green
COLOR_FIRE = "#E68181"  # Salmon-Red

EMPTY = CONFIG["cell_symbols"]["empty"]
BURNED = CONFIG["cell_symbols"]["burned"]
TREE = CONFIG["cell_symbols"]["tree"]
FIRE = CONFIG["cell_symbols"]["fire"]

COLORS = [COLOR_EMPTY, COLOR_BURNED, COLOR_TREE, COLOR_FIRE]
CELLS = [EMPTY, BURNED, TREE, FIRE]
NORM, CMAP = get_norm_cmap(CELLS, COLORS)

# Local Grid
MARKBULL_SIZE = 52

# Global Grid

# Gauge
HEIGHT_GAUGE = 0.1
COLOR_GAUGE = "#D4CCDB"  # "Gray-Purple"
CYCLE_SYMBOL = "\U0001f504"
CYCLE_SIZE = 32

# Counts
TREE_SYMBOL = "\U0001f332"
BURNED_SYMBOL = "\ue08a"


def render(env):
    grid = env.grid
    ca_params, pos, time = env.context

    local_grid = moore_n(3, pos, grid, EMPTY)

    plt.style.use("seaborn-whitegrid")

    fig_shape = (12, 14)
    fig = plt.figure(figsize=(15, 12))
    fig.suptitle(
        TITLE,
        font=TITLEFONT,
        fontsize=TITLE_SIZE,
        **TITLE_POS,
        color="0.6",
        ha=TITLE_ALIGN
    )

    ax_gauge = plt.subplot2grid(fig_shape, (10, 0), colspan=8, rowspan=2)
    ax_lgrid = plt.subplot2grid(fig_shape, (0, 0), colspan=8, rowspan=10)
    ax_ggrid = plt.subplot2grid(fig_shape, (0, 8), colspan=6, rowspan=6)
    ax_counts = plt.subplot2grid(fig_shape, (6, 8), colspan=6, rowspan=6)

    plot_local(ax_lgrid, local_grid)

    plot_global(ax_ggrid, env)

    plot_gauge(ax_gauge, time)

    d = env.count_cells()
    counts = d[EMPTY], d[BURNED], d[TREE], d[FIRE]
    plot_counts(ax_counts, *counts)

    return plt.gcf()


def plot_local(ax, grid):
    nrows, ncols = grid.shape
    mid_row, mid_col = nrows // 2, nrows // 2

    plot_grid(ax, grid, interpolation="none", cmap=CMAP, norm=NORM)

    markbull = parse_svg_into_mpl(svg_paths.BULLDOZER)
    ax.plot(mid_col, mid_row, marker=markbull, markersize=MARKBULL_SIZE, color="1.0")


def plot_gauge(ax, time):
    ax.barh(0.0, time, height=HEIGHT_GAUGE, color=COLOR_GAUGE, edgecolor="None")

    ax.barh(
        0.0,
        1.0,
        height=0.15,
        color="None",
        edgecolor="0.86",
    )

    # Comment here please!
    ax.set_yticks([0])
    ax.set_xlim(0 - 0.03, 1 + 0.1)
    ax.set_ylim(-0.4, 0.4)

    ax.set_yticklabels(CYCLE_SYMBOL, font=EMOJIFONT, size=CYCLE_SIZE)

    ax.get_yticklabels()[0].set_color("0.74")

    ax.set_xticks([0.0, 1.0])

    clear_ax(ax, yticks=False)


def plot_counts(ax, counts_empty, counts_burned, counts_tree, counts_fire):

    counts_total = sum((counts_empty, counts_burned, counts_tree, counts_fire))

    commons = {"x": [0, 1], "width": 0.1}
    pc = "1.0"  # placeholder color

    lv1y = [counts_tree, counts_empty]
    lv1c = [COLOR_TREE, COLOR_EMPTY]

    lv2y = [0, counts_burned]  # level 2 y axis
    lv2c = [pc, COLOR_BURNED]  # level 2 colors
    lv2b = lv1y  # level 2 bottom

    lv3y = [0, counts_fire]
    lv3c = [pc, COLOR_FIRE]
    lv3b = [lv1y[i] + lv2y[i] for i in range(len(lv1y))]

    # First Level Bars
    ax.bar(height=lv1y, color=lv1c, **commons)

    # Second Level Bars
    ax.bar(height=lv2y, color=lv2c, bottom=lv2b, **commons)

    # Third Level Bars
    ax.bar(height=lv3y, color=lv3c, bottom=lv3b, **commons)

    # Bar Symbols Settings
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels([TREE_SYMBOL, BURNED_SYMBOL], font=EMOJIFONT, size=34)
    # Same colors as bars
    for label, color in zip(ax.get_xticklabels(), [COLOR_TREE, COLOR_BURNED]):
        label.set_color(color)

    # Mess with x,y limits for aethetics reasons
    goff = 2048
    ax.set_ylim(0 - goff, counts_total + goff)  # It gives breathing room for bars
    ax.set_xlim(-1, 2)  # It centers the bars

    # Grid Settings and Tick settings
    # Show marks each quarter
    ax.set_yticks(np.linspace(0, counts_total, 3, dtype=int))
    # Remove clutter
    clear_ax(ax, xticks=False)
    # Add back y marks each quarter
    ax.grid(axis="y", color="0.94")


def plot_global(ax, env):
    size = 17
    from gym_cellular_automata.forest_fire.bulldozer.utils import svg_paths

    grid = env.grid
    __, pos, __ = env.context

    ax.imshow(grid, interpolation="none", cmap=CMAP, norm=NORM)

    # Fire Seed
    svg_fire = parse_svg_into_mpl(svg_paths.FIRE)
    fire_seed = env._fire_seed
    offset = 10

    if fire_seed[0] - offset >= 0:
        # Position the Fire svg for better visualization
        ax.plot(
            fire_seed[1],
            fire_seed[0] - offset,
            marker=svg_fire,
            markersize=size,
            color=COLOR_FIRE,
        )
    else:
        # If offset just put a point
        ax.plot(
            fire_seed[1], fire_seed[0], marker=".", markersize=size, color=COLOR_FIRE
        )

    # Bulldozer Location
    svg_location = parse_svg_into_mpl(svg_paths.LOCATION)

    # Off set
    offset = 15

    if pos[0] - offset >= 0:
        ax.plot(
            pos[1], pos[0] - offset, marker=svg_location, markersize=size, color="1.0"
        )
    else:
        ax.plot(pos[1], pos[0], marker=".", markersize=size, color="1.0")

    clear_ax(ax)
