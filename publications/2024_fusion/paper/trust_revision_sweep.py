#!/usr/bin/python3
import numpy as np
import subjective_logic as sl
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 8,
    'text.usetex': True,
    'text.latex.preamble': " ".join([
        r'\usepackage{amsmath}',
        r'\usepackage{amssymb}',
        r'\usepackage{amsfonts}',
    ])
})
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


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
    "length_cap": 0.92,
}
export_args = {
    "format": 'pdf',
    "dpi": 500,
    "transparent": True,
    "bbox_inches": 'tight',
    "pad_inches": 0,
}

SHOW_ACC = False
SHOW_AVG = not SHOW_ACC
SHOW_NO_TR = True
SHOW_NO_TR_FUSION = False

conflict_weights_ours = np.linspace(0.,5.,20)
conflict_weights_josang = np.linspace(0.,5.,20)
# conflict_weights_josang = np.linspace(0.,20.,50)

######################################################
# create opinions
######################################################
meas_s1 = sl.Opinion(0.1, 0.75)
trust_s1 = sl.Opinion2d([0.3, 0.3], [1.0, 0.0])
t_s1 = sl.TrustedOpinion(trust_s1, meas_s1)

meas_s2 = sl.Opinion(0.0, 0.7)
trust_s2 = sl.Opinion2d([0.3, 0.2], [1.0, 0.0])
t_s2 = sl.TrustedOpinion(trust_s2, meas_s2)

meas_s3 = sl.Opinion(0.7, 0.05)
trust_s3 = sl.Opinion2d([0.7, 0.0], [1.0, 0.0])
t_s3 = sl.TrustedOpinion(trust_s3, meas_s3)

t_vec = [t_s1, t_s2, t_s3]

######################################################
# calc normal and trusted fusion
######################################################
trusted_fusion = sl.TrustedFusion.fuse_opinions(sl.FusionType.CUMULATIVE, t_vec)

######################################################
# calc trusted fusion using trust revision
######################################################

t_fusion = []
for c_ours in conflict_weights_ours:
    t_fusion_normal = []
    for c_josan in conflict_weights_josang:
        weighted_types = [
            sl.WeightedTypes(sl.RelationType.CONFLICT, sl.TrustRevisionType.SHARES, sl.ConflictType.AVERAGE, c_ours),
            sl.WeightedTypes(sl.RelationType.CONFLICT, sl.TrustRevisionType.REFERENCE_FUSION, sl.ConflictType.BELIEF_AVERAGE, c_josan),
        ]
        fusion = sl.TrustedFusion.fuse_opinions(sl.FusionType.CUMULATIVE, weighted_types, t_vec)
        t_fusion_normal.append(fusion)

    t_fusion.append(t_fusion_normal)

######################################################
# illustrate results
######################################################
oranges = plt.get_cmap('Oranges')
greens = plt.get_cmap('Greens')

op_str = r"$\omega_{{{l}}}^{{{u}}}$"
op_tr_str = r"$\check{{\omega}}_{{{l}}}^{{{u}}}$"

cs_shift = 2
bc_shift = 0

######################################################
# fusion opinions

sl.reset_plot()
fig, ax = sl.create_triangle_plot(hypo_texts=('disbelief\n(sunny)', 'belief\n(rainy)'))

point_draw_args = default_point_draw_args.copy()
point_draw_args["edgecolor"] = 'none'
for idx1, c_ours in enumerate(conflict_weights_ours):
    for idx2, c_josang in enumerate(conflict_weights_josang):

        color_1 = greens(0.5 - idx2 / len(conflict_weights_josang) / 2.)
        color_2 = oranges(0.5 - idx1 / len(conflict_weights_ours) / 2.)

        div = idx1 + idx2
        if div == 0:
            color = 0.5 * np.array(color_1) + 0.5 * np.array(color_2)
        else:
            color = idx1 / div * np.array(color_1) + idx2 / div * np.array(color_2)

        point_draw_args["color"] = color
        #  marker='D'
        sl.draw_point(t_fusion[idx1][idx2], **point_draw_args)


point_draw_args["color"] = 'tab:blue'
sl.draw_point(trusted_fusion, **point_draw_args)

######################################################
# export results

x_start = -0.05
x_end = 1.05
y_start = -0.175
y_end = 0.95

x_diff = x_end - x_start
y_diff = y_end - y_start

relative_size = x_diff / y_diff

figure_size_y = 5

for _ax, _fig in zip([ax],[fig]):
    _ax.set_xlim(x_start, x_end)
    _ax.set_ylim(y_start, y_end)
    _fig.set_size_inches(figure_size_y * relative_size,figure_size_y)

# fig.savefig("exp_tr_seep_2_opinions.pdf", **export_args)

plt.show()



