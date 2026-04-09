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
    'draw_axis': False,
    'draw_axis_label': True,
    'draw_opinion': True,
    'draw_opinion_label': True,
    'draw_prior': False,
    'draw_prior_label': True,
    'draw_projection': False,
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

default_draw_flags = {
    'color' : 'tab:blue',
    'sizes': [200],
}

default_text_draw_args = {
    "ha": 'left',
    "va": 'top',
    "fontsize": 20,
    "offset": [0.02, 0.01],
}

mltrue = sl.Opinion(0.9, 0.0)
mlfalse = sl.Opinion(0.0, 0.9)
somewhat = sl.Opinion(0.4, 0.1)
# fig, ax = sl.create_triangle_plot(hypo_texts=('disbelief\n(sunny)', 'belief\n(rainy)'))
fig, ax = sl.create_triangle_plot()
# fig, ax, _ = sl.draw_full_opinion_triangle(mltrue, '_X', ('disbelief','belief'), draw_flags_opinion)

default_draw_flags['color'] = '#77AC30'
sl.draw_point(mltrue, **default_draw_flags)
opinion_text = f'$\\omega_X^1$'
default_text_draw_args['va'] = 'bottom'
sl.draw_text_at_point(mltrue, opinion_text, **default_text_draw_args)

default_draw_flags['color'] = '#D95319'
sl.draw_point(mlfalse, **default_draw_flags)
opinion_text = f'$\\omega_X^2$'
default_text_draw_args['ha'] = 'right'
default_text_draw_args['offset'] = [-0.04, 0.01]
sl.draw_text_at_point(mlfalse, opinion_text, **default_text_draw_args)

default_draw_flags['color'] = '#EDB120'
sl.draw_point(somewhat, **default_draw_flags)
opinion_text = f'$\\omega_X^3$'
sl.draw_text_at_point(somewhat, opinion_text, **default_text_draw_args)

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
fig.savefig("colored_opinions.pdf", **export_args)
