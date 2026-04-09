import subjective_logic as sl
import matplotlib.pyplot as plt
import numpy as np
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
    'draw_axis': True,
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


export_args = {
    "format": 'png',
    "dpi": 500,
    "transparent": True,
    "bbox_inches": 'tight',
    "pad_inches": 0,
}

opinion_0 = sl.Opinion(0.55, 0.1)
opinion_1 = sl.Opinion(0.55, 0.1)
opinion_2 = sl.Opinion(0.1, 0.55)
# opinion.prior_belief_masses = [0.2, 0.8]
# variable_postfix = '_X^A'

fig_0, ax_0, _= sl.draw_full_opinion_triangle(
    opinion_0,
    '^A_X',
    ('$\overline{x}$\n(sunny)', '$x$\n(rainy)'),
    draw_flags_opinion
)
sl.reset_plot()
fig_1, ax_1, _ = sl.draw_full_opinion_triangle(
    opinion_1,
    '^B_X',
    ('$\overline{x}$\n(sunny)', '$x$\n(rainy)'),
    draw_flags_opinion
)
sl.reset_plot()
fig_2, ax_2, _ = sl.draw_full_opinion_triangle(
    opinion_2,
    '^C_X',
    ('$\overline{x}$\n(sunny)', '$x$\n(rainy)'),
    draw_flags_opinion
)
sl.reset_plot()

fused_opinion = sl.Fusion.fuse_opinions(sl.Fusion.FusionType.CUMULATIVE, [opinion_0, opinion_1, opinion_2])

fig_f, ax_f, _= sl.draw_full_opinion_triangle(
    fused_opinion,
    '^{\diamond[\{A,B,C\}]}_X',
    ('$\overline{x}$\n(sunny)', '$x$\n(rainy)'),
    draw_flags_opinion
)

sl.reset_plot()
draw_flags_opinion['draw_axis'] = False

fig_f2, ax_f2, _= sl.draw_full_opinion_triangle(
    fused_opinion,
    '^{\diamond[\{A,B,C\}]}_X',
    ('$\overline{x}$\n(sunny)', '$x$\n(rainy)'),
    draw_flags_opinion
)

sl.reset_plot()

fig_all, ax_all, _ = sl.draw_full_opinion_triangle(
    opinion_0,
    '^A_X',
    ('$\overline{x}$\n(sunny)', '$x$\n(rainy)'),
    draw_flags_opinion
)
sl.draw_full_opinion_triangle(
    opinion_2,
    '^C_X',
    ('$\overline{x}$\n(sunny)', '$x$\n(rainy)'),
    draw_flags_opinion
)
draw_flags_opinion['draw_opinion'] = False
sl.draw_full_opinion_triangle(
    opinion_1,
    '^B_X',
    ('$\overline{x}$\n(sunny)', '$x$\n(rainy)'),
    draw_flags_opinion
)
text_draw_args = {
    "ha": 'right',
    "va": 'bottom',
    "fontsize": 18,
    "offset": [-0.01, 0.02],
}
text = '$w^B_X$'
sl.draw_text_at_point(opinion_1, text=text, **text_draw_args)

x_start = -0.15
x_end = 1.15
y_start = -0.1
y_end = 0.95

x_diff = x_end - x_start
y_diff = y_end - y_start

relative_size = x_diff / y_diff

figure_size_y = 5

for _ax, _fig in zip([ax_0, ax_1, ax_2, ax_f, ax_f2, ax_all],[fig_0, fig_1, fig_2, fig_f, fig_f2, fig_all]):
    _ax.set_xlim(x_start, x_end)
    _ax.set_ylim(y_start, y_end)
    _fig.set_size_inches(figure_size_y * relative_size,figure_size_y)

plt.show()
fig_0.savefig("sl_fusion_0.png", **export_args)
fig_1.savefig("sl_fusion_1.png", **export_args)
fig_2.savefig("sl_fusion_2.png", **export_args)
fig_f.savefig("sl_fusion_f.png", **export_args)
fig_f2.savefig("sl_fusion_f2.png", **export_args)
fig_all.savefig("sl_fusion_all.png", **export_args)
