import subjective_logic as sl
import matplotlib.pyplot as plt

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

draw_flags_opinion_start = {
    'draw_hypo_texts': True,
    'draw_axis': True,
    'draw_axis_label': True,
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

draw_flags_opinion_end = draw_flags_opinion_start.copy()
draw_flags_opinion_end['draw_prior'] = True
draw_flags_opinion_end['draw_prior_label'] = True
draw_flags_opinion_end['draw_projection'] = True
draw_flags_opinion_end['draw_projection_label'] = True

export_args = {
    "format": 'png',
    "dpi": 500,
    "transparent": True,
    "bbox_inches": 'tight',
    "pad_inches": 0,
}

opinion = sl.Opinion(0.4, 0.3)
opinion.prior_belief_masses = [0.2, 0.8]
# variable_postfix = '_X^A'

fig_start, ax_start, _= sl.draw_full_opinion_triangle(
    opinion,
    '_X',
    ('$\overline{x}$\n(disbelief)', '$x$\n(belief)'),
    draw_flags_opinion_start
)
sl.reset_plot()
fig_end, ax_end, _= sl.draw_full_opinion_triangle(
    opinion,
    '_X',
    ('$\overline{x}$\n(disbelief)', '$x$\n(belief)'),
    draw_flags_opinion_end
)

x_start = -0.15
x_end = 1.15
y_start = -0.1
y_end = 0.95

x_diff = x_end - x_start
y_diff = y_end - y_start

relative_size = x_diff / y_diff

figure_size_y = 5

for _ax, _fig in zip([ax_start, ax_end],[fig_start, fig_end]):
    _ax.set_xlim(x_start, x_end)
    _ax.set_ylim(y_start, y_end)
    _fig.set_size_inches(figure_size_y * relative_size,figure_size_y)

plt.show()
fig_start.savefig("barycentric_triangle_start.png", **export_args)
fig_end.savefig("barycentric_triangle_end.png", **export_args)
