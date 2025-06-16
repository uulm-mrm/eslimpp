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
default_arrow_draw_args = {
    "length_includes_head": True,
    "head_width": 0.02,
    "overhang": 0.2,
    "color": 'tab:gray',
    "length_cap": 0.9,
    # "linestyle": 'dashed',
}

matlab_green = '#77AC30'
matlab_blue = '#0072BD'
matlab_red = '#D95319'
matlab_yellow = '#EDB120'

opinion = sl.Opinion(0.3, 0.3)
opinion.prior_belief_masses = [0.2, 0.8]
opinion_pos = opinion.revise_trust(0.5)
opinion_neg = opinion.revise_trust(-0.5)

variable_postfix = '_X^A'
fig, ax = sl.create_triangle_plot(hypo_texts=('distrust', 'trust'))

default_draw_flags['color'] = matlab_blue
sl.draw_point(opinion, **default_draw_flags)
default_draw_flags['color'] = matlab_red
sl.draw_point(opinion_pos, **default_draw_flags)
default_draw_flags['color'] = matlab_green
sl.draw_point(opinion_neg, **default_draw_flags)

sl.draw_arrow(opinion, opinion_pos, **default_arrow_draw_args)
sl.draw_arrow(opinion, opinion_neg, **default_arrow_draw_args)

opinion_mid_pos = opinion.interpolate(opinion_pos, 0.5)
default_text_draw_args['ha'] = 'right'
default_text_draw_args['va'] = 'bottom'
default_text_draw_args['offset'] = [0.01, 0.01]
default_text_draw_args['fontsize'] = 16
sl.draw_text_at_point(opinion_mid_pos, '$\\textup{RF} > 0$', **default_text_draw_args)

opinion_mid_neg = opinion.interpolate(opinion_neg, 0.5)
default_text_draw_args['ha'] = 'left'
default_text_draw_args['va'] = 'bottom'
default_text_draw_args['offset'] = [-0.01, 0.01]
sl.draw_text_at_point(opinion_mid_neg, '$\\textup{RF} < 0$', **default_text_draw_args)

x_start = -0.1
x_end = 1.1
y_start = -0.05
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
fig.savefig("03-symmetric_tr.pdf", **export_args)
