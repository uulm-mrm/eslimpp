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
    'disbelief_label_position': 0.7,
    'uncertainty_label_position': 0.7,
    'prior_offset': [-0.02, -0.02],
    'projection_offset': [0.03, -0.02],
}

export_args = {
    "format": 'png',
    "dpi": 500,
    "transparent": True,
    "bbox_inches": 'tight',
    "pad_inches": 0,
}

trust = sl.Opinion([0.3,0.4],[0.85,0.15])
opinion = sl.Opinion([0.4,0.3],[0.2,0.8])

discounted_opinion = opinion.trust_discount(trust)

axs = []
figs = []

fig, ax, _= sl.draw_full_opinion_triangle(
    trust,
    '^A_B',
    ('distrust', 'trust'),
    draw_flags_opinion
)
figs.append(fig)
axs.append(ax)
sl.reset_plot()

fig, ax, _ = sl.draw_full_opinion_triangle(
    opinion,
    '^B_X',
    ('$\overline{x}$\n(sunny)', '$x$\n(rainy)'),
    draw_flags_opinion
)
figs.append(fig)
axs.append(ax)
sl.reset_plot()
draw_flags_opinion['belief_label_position'] = 1.6
draw_flags_opinion['disbelief_label_position'] = 1.8

fig, ax, _ = sl.draw_full_opinion_triangle(
    discounted_opinion,
    '^{A;B}_X',
    ('$\overline{x}$\n(sunny)', '$x$\n(rainy)'),
    draw_flags_opinion
)
lower_left = sl.get_bari_point_between(sl.Opinion(0., 0.), sl.Opinion(0., 1.), position=trust.getBinomialProjection())
lower_right = sl.get_bari_point_between(sl.Opinion(0., 0.), sl.Opinion(1., 0.), position=trust.getBinomialProjection())

# define nodes for triangle and plot empty triangle
nodes = np.asfortranarray([
    [lower_left[0], lower_right[0], 0.5],
    [lower_left[1], lower_right[1], math.sqrt(3) / 2],
])
triangle = bezier.Triangle(nodes, degree=1)
triangle.plot(2, ax=ax, alpha=0.3, color="Black")
figs.append(fig)
axs.append(ax)


x_start = -0.15
x_end = 1.15
y_start = -0.1
y_end = 0.95

x_diff = x_end - x_start
y_diff = y_end - y_start

relative_size = x_diff / y_diff

figure_size_y = 5

for _ax, _fig in zip(axs , figs):
    _ax.set_xlim(x_start, x_end)
    _ax.set_ylim(y_start, y_end)
    _fig.set_size_inches(figure_size_y * relative_size,figure_size_y)

plt.show()

figs[0].savefig("sl_trust_discount_trust.png", **export_args)
figs[1].savefig("sl_trust_discount_opinion.png", **export_args)
figs[2].savefig("sl_trust_discount_discounted.png", **export_args)
