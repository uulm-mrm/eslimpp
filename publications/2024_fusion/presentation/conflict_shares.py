import itertools

import subjective_logic as sl
import matplotlib.pyplot as plt
import math

plt.rcParams.update({
    'font.size': 8,
    'text.usetex': True,
    'text.latex.preamble': " ".join([
        r'\usepackage{amsmath}',
        r'\usepackage{amssymb}',
        r'\usepackage{amsfonts}',
        r'\usepackage{bm}',
    ])
})
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

draw_flags_opinion = {
    'draw_hypo_texts': True,
    'draw_axis': False,
    'draw_axis_label': False,
    'draw_opinion': True,
    'draw_opinion_label': True,
    'draw_prior': False,
    'draw_prior_label': False,
    'draw_projection': False,
    'draw_projection_label': False,
    'belief_label_position': 0.5,
    'disbelief_label_position': 0.7,
    'uncertainty_label_position': 0.7,
}

default_point_draw_args = {
    "color" : 'tab:gray',
    "zorder": 500,
    "s": 80,
}

default_text_draw_args = {
    "ha": 'left',
    "va": 'top',
    "fontsize": 18,
    "offset": [0.01, 0.00],
}
default_arrow_draw_args = {
    "length_includes_head": True,
    "head_width": 0.02,
    "overhang": 0.2,
    "color": 'tab:gray',
    "length_cap": 0.8,
}

export_args = {
    "format": 'png',
    "dpi": 500,
    "transparent": True,
    "bbox_inches": 'tight',
    "pad_inches": 0,
}
op_str = r"$\omega_{{{l}}}^{{{u}}}$"

def size_figure(fig, ax):
    x_start = -0.15
    x_end = 1.15
    y_start = -0.1
    y_end = 0.95

    x_diff = x_end - x_start
    y_diff = y_end - y_start

    relative_size = x_diff / y_diff

    figure_size_y = 5

    ax.set_xlim(x_start, x_end)
    ax.set_ylim(y_start, y_end)
    fig.set_size_inches(figure_size_y * relative_size, figure_size_y)


def store_figure(fig, ax, filename, export_arguments=None):
    if export_arguments is None:
        export_arguments = {}
    size_figure(fig, ax)
    if STORE_FIGS:
        fig.savefig(filename, **export_arguments)


STORE_FIGS = True

opinion_0 = sl.Opinion(0.65, 0.2)
opinion_1 = sl.Opinion(0.1, 0.75)
opinion_2 = sl.Opinion(0.1, 0.6)
opinions = [opinion_0, opinion_1, opinion_2]

fig, ax = sl.create_triangle_plot(hypo_texts=('disbelief\n(sunny)', 'belief\n(rainy)'))
for idx, opinion in enumerate(opinions):
    point_draw_args = default_point_draw_args.copy()
    text_draw_args = default_text_draw_args.copy()
    if idx == 1:
        text_draw_args['ha'] = 'center'
        text_draw_args['va'] = 'top'
        text_draw_args['offset'] = [0.01, -0.03]
    elif idx == 2:
        text_draw_args['ha'] = 'center'
        text_draw_args['va'] = 'bottom'
        text_draw_args['offset'] = [0.01, 0.03]
    sl.draw_point(opinion, **point_draw_args)
    sl.draw_text_at_point(opinion, text=op_str.format(l="X", u="S_" + str(idx)), **text_draw_args)

arrow_draw_args = default_arrow_draw_args.copy()
arrow_draw_args['color'] = 'tab:green'
arrow_draw_args['y_offset'] = 0.05
sl.draw_arrow(opinions[0], opinions[1], start_offset=0.5, **arrow_draw_args)
arrow_draw_args['y_offset'] = -0.05
sl.draw_arrow(opinions[1], opinions[0], start_offset=0.5, **arrow_draw_args)

store_figure(fig, ax, 'ms_cs_2.png', export_args)

sl.reset_plot()

fig, ax = sl.create_triangle_plot(hypo_texts=('disbelief\n(sunny)', 'belief\n(rainy)'))
for idx, opinion in enumerate(opinions):
    point_draw_args = default_point_draw_args.copy()
    text_draw_args = default_text_draw_args.copy()
    if idx == 1:
        text_draw_args['ha'] = 'center'
        text_draw_args['va'] = 'top'
        text_draw_args['offset'] = [0.01, -0.03]
    elif idx == 2:
        text_draw_args['ha'] = 'center'
        text_draw_args['va'] = 'bottom'
        text_draw_args['offset'] = [0.01, 0.03]
    sl.draw_point(opinion, **point_draw_args)
    sl.draw_text_at_point(opinion, text=op_str.format(l="X", u="S_" + str(idx)), **text_draw_args)

arrow_draw_args = default_arrow_draw_args.copy()
arrow_draw_args['color'] = 'tab:green'
# since opinions circle size is not considered, shorten arrow for close opinions
arrow_draw_args['length_cap'] = 0.6
arrow_draw_args['y_offset'] = 0.05
sl.draw_arrow(opinions[1], opinions[2], start_offset=0.5, **arrow_draw_args)
arrow_draw_args['y_offset'] = -0.05
sl.draw_arrow(opinions[2], opinions[1], start_offset=0.5, **arrow_draw_args)

store_figure(fig, ax, 'ms_cs_0.png', export_args)

# calc revision factors for this specific example
weighted_types_cs_avg = [
    (sl.TrustRevision.TrustRevisionType.CONFLICT_SHARES, sl.Conflict.ConflictType.AVERAGE, 2.),
]

revision_factors = sl.TrustRevision.revision_factors(
    sl.TrustRevision.TrustRevisionType.CONFLICT_SHARES,
    sl.Conflict.ConflictType.AVERAGE,
    [sl.TrustedOpinion(sl.Opinion2d(1.0,0.), op) for op in opinions]
    # [True,True,True]
)
print(revision_factors)

plt.show()
