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
    'draw_axis_label': True,
    'draw_opinion': True,
    'draw_opinion_label': True,
    'draw_prior': True,
    'draw_prior_label': True,
    'draw_projection': True,
    'draw_projection_label': True,
    'belief_label_position': 0.5,
    'disbelief_label_position': 0.6,
    'uncertainty_label_position': 0.7,
}

export_args = {
    "format": 'pdf',
    "dpi": 500,
    "transparent": True,
    "bbox_inches": 'tight',
    "pad_inches": 0,
}

opinion = sl.Opinion(0.4, 0.2)
opinion.prior_belief_masses = [0.2, 0.8]
variable_postfix = '_X^A'
# fig, ax = sl.create_triangle_plot(hypo_texts=('disbelief\n(sunny)', 'belief\n(rainy)'))
fig, ax = sl.draw_full_opinion_triangle(opinion, '_X', ('disbelief','belief'), draw_flags_opinion)

x_start = -0.15
x_end = 1.15
y_start = -0.1
y_end = 0.95

x_diff = x_end - x_start
y_diff = y_end - y_start

relative_size = x_diff / y_diff

figure_size_y = 5

for _ax, _fig in zip([ax],[fig]):
    _ax.set_xlim(x_start, x_end)
    _ax.set_ylim(y_start, y_end)
    _fig.set_size_inches(figure_size_y * relative_size,figure_size_y)

plt.show()
fig.savefig("02-barycentric_triangle.pdf", **export_args)
