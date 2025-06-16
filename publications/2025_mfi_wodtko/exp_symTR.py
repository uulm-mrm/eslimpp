#!/usr/bin/python3
import numpy as np
import subjective_logic as sl
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Rectangle
import math

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

SHOW_NO_TR = True
SHOW_NO_TR_FUSION = False

######################################################
# create opinions
######################################################
meas_s1 = sl.Opinion(0.1, 0.75)
trust_s1 = sl.Opinion2d([0.3, 0.3], [0.5, 0.5])
t_s1 = sl.TrustedOpinion(trust_s1, meas_s1)

meas_s2 = sl.Opinion(0.0, 0.7)
trust_s2 = sl.Opinion2d([0.3, 0.2], [0.5, 0.5])
t_s2 = sl.TrustedOpinion(trust_s2, meas_s2)

meas_s3 = sl.Opinion(0.7, 0.05)
trust_s3 = sl.Opinion2d([0.7, 0.0], [0.5, 0.5])
t_s3 = sl.TrustedOpinion(trust_s3, meas_s3)

t_vec = [t_s1, t_s2, t_s3]
opinions = sl.TrustedOpinion2d.extractOpinions(t_vec)
discounted_opinions = sl.TrustedOpinion2d.extractDiscountedOpinions(t_vec)


normal_fusion = sl.Fusion.fuse_opinions(sl.Fusion.FusionType.CUMULATIVE, opinions)
trusted_fusion = sl.TrustedFusion.fuse_opinions(sl.Fusion.FusionType.CUMULATIVE, t_vec)

######################################################
# calc trusted fusion using trust revision
######################################################
weighted_types_conflict = [
    (sl.TrustRevision.TrustRevisionType.CONFLICT_SHARES, sl.Conflict.ConflictType.AVERAGE, 2.),
]
weighted_types_harmony = [
    (sl.TrustRevision.TrustRevisionType.CONFLICT_SHARES, sl.Conflict.ConflictType.AVERAGE, 2.),
    (sl.TrustRevision.TrustRevisionType.HARMONY_SHARES, sl.Conflict.ConflictType.AVERAGE, 2.),
]

fusion_conflict, t_vec_updated_conflict = sl.TrustedFusion.fuse_opinions_(sl.Fusion.FusionType.CUMULATIVE, weighted_types_conflict, t_vec)
fusion_harmony, t_vec_updated_harmony = sl.TrustedFusion.fuse_opinions_(sl.Fusion.FusionType.CUMULATIVE, weighted_types_harmony, t_vec)

discounted_opinions_conflict = sl.TrustedOpinion2d.extractDiscountedOpinions(t_vec_updated_conflict)
discounted_opinions_harmony = sl.TrustedOpinion2d.extractDiscountedOpinions(t_vec_updated_harmony)

######################################################
# illustrate results
######################################################
oranges = plt.get_cmap('Oranges')
greens = plt.get_cmap('Greens')

op_str = r"$\omega_{{{l}}}^{{{u}}}$"
op_tr_str = r"$\check{{\omega}}_{{{l}}}^{{{u}}}$"

cs_shift = 2
bc_shift = [0,1,2]

######################################################
# source opinions

fig, ax = sl.create_triangle_plot(hypo_texts=('disbelief', 'belief'))
point_draw_args = default_point_draw_args.copy()
text_draw_args = default_text_draw_args.copy()
if SHOW_NO_TR:
    for idx, opinion in enumerate(opinions):
        sl.draw_point(opinion, **point_draw_args)
        # op_str=r"$\omega_X^{S_{{idx}}}$"
        if idx == 2:
            text_draw_args["ha"] = "right"
            text_draw_args["va"] = "top"
            text_draw_args["offset"] = [0.0, -0.02]
            # text = op_str.format(l="X", u="S_" + str(idx)) + " = "  + op_str.format(l="X", u="[A;S_" + str(idx) + "]")
            # sl.draw_text_at_point(opinion, text=text, **text_draw_args)
            # continue
        sl.draw_text_at_point(opinion, text=op_str.format(l="X", u="S_" + str(idx)), **text_draw_args)

point_draw_args = default_point_draw_args.copy()
point_draw_args["color"] = 'tab:blue'
text_draw_args = default_text_draw_args.copy()
for idx, opinion in enumerate(discounted_opinions):
    sl.draw_point(opinion, **point_draw_args)
    # if idx == 2 and SHOW_NO_TR:
    #     continue
    if idx == 1:
        text_draw_args["ha"] = "right"
        text_draw_args["va"] = "bottom"
        text_draw_args["offset"] = [0.04, 0.02]
    elif idx == 0:
        text_draw_args["ha"] = "left"
        text_draw_args["va"] = "bottom"
        text_draw_args["offset"] = [0.04, 0.0]
    else:
        text_draw_args["ha"] = "right"
        text_draw_args["va"] = "top"
        text_draw_args["offset"] = [-0.02, 0.0]

    sl.draw_text_at_point(opinion, text=op_str.format(l="X", u="[A;S_" + str(idx) + "]"), **text_draw_args)

point_draw_args = default_point_draw_args.copy()
point_draw_args["color"] = 'tab:orange'
text_draw_args = default_text_draw_args.copy()
text_draw_args["ha"] = "left"
text_draw_args["va"] = "bottom"
text_draw_args["offset"] = [0.01, 0.01]
for idx, (opinion, opinion_up) in enumerate(zip(discounted_opinions, discounted_opinions_conflict)):
    if opinion == opinion_up:
        continue
    sl.draw_point(opinion_up, **point_draw_args)
    sl.draw_text_at_point(opinion_up, text=op_tr_str.format(l="X", u="[A;S_" + str(idx) + "]"), **text_draw_args)

point_draw_args = default_point_draw_args.copy()
point_draw_args["color"] = 'tab:green'
text_draw_args = default_text_draw_args.copy()
text_draw_args["va"] = "bottom"
for idx, (opinion, opinion_up) in enumerate(zip(discounted_opinions, discounted_opinions_harmony)):
    if opinion == opinion_up:
        continue
    if idx == cs_shift:
        point_draw_args['marker'] = MarkerStyle("o", fillstyle="right")
        sl.draw_point(opinion_up, **point_draw_args)
        continue
    elif 'marker' in text_draw_args:
        point_draw_args.pop('marker')

    if idx == 0:
        text_draw_args["ha"] = "left"
        text_draw_args["va"] = "top"
        text_draw_args["offset"] = [0.00, 0.00]
    else:
        text_draw_args["ha"] = "right"
        text_draw_args["va"] = "top"
        text_draw_args["offset"] = [-0.02, 0.0]

    sl.draw_point(opinion_up, **point_draw_args)
    sl.draw_text_at_point(opinion_up, text=op_tr_str.format(l="X", u="[A;S_" + str(idx) + "]"), **text_draw_args)

sl.draw_arrow(discounted_opinions[cs_shift], discounted_opinions_conflict[cs_shift], **default_arrow_draw_args)
sl.draw_arrow(discounted_opinions[bc_shift[0]], discounted_opinions_harmony[bc_shift[0]], **default_arrow_draw_args)
sl.draw_arrow(discounted_opinions[bc_shift[1]], discounted_opinions_harmony[bc_shift[1]], **default_arrow_draw_args)

######################################################
# trust opinions

sl.reset_plot()
fig2, ax2 = sl.create_triangle_plot(hypo_texts=('distrust', 'trust'))

point_draw_args = default_point_draw_args.copy()
point_draw_args["color"] = 'tab:blue'
text_draw_args = default_text_draw_args.copy()
text_draw_args["ha"] = "right"
text_draw_args["va"] = "bottom"
text_draw_args["offset"] = [-0.01, 0.02]
for idx, t_op in enumerate(t_vec):
    trust = t_op.trust
    if idx == 2:
        text_draw_args["ha"] = "left"
        text_draw_args["va"] = "bottom"
        text_draw_args["offset"] = [0.01, 0.02]
    sl.draw_point(trust, **point_draw_args)
    sl.draw_text_at_point(trust, text=op_str.format(u="A", l="S_" + str(idx)), **text_draw_args)

point_draw_args = default_point_draw_args.copy()
point_draw_args["color"] = 'tab:orange'
text_draw_args = default_text_draw_args.copy()
text_draw_args["offset"] = [0.01, -0.01]

for idx, (t_op, t_op_up) in enumerate(zip(t_vec, t_vec_updated_conflict)):
    if t_op.trust == t_op_up.trust:
        continue
    trust = t_op_up.trust
    sl.draw_point(trust, **point_draw_args)
    sl.draw_text_at_point(trust, text=op_tr_str.format(u="A", l="S_" + str(idx)), **text_draw_args)

point_draw_args = default_point_draw_args.copy()
point_draw_args["color"] = 'tab:green'
text_draw_args = default_text_draw_args.copy()
text_draw_args["ha"] = "left"
text_draw_args["va"] = "bottom"
text_draw_args["offset"] = [0.01, 0.01]
for idx, (t_op, t_op_up) in enumerate(zip(t_vec, t_vec_updated_harmony)):
    if t_op.trust == t_op_up.trust:
        continue
    trust = t_op_up.trust
    if idx == cs_shift:
        point_draw_args['marker'] = MarkerStyle("o", fillstyle="right")
    elif 'marker' in text_draw_args:
        point_draw_args.pop('marker')
    sl.draw_point(trust, **point_draw_args)
    if idx != cs_shift:
        sl.draw_text_at_point(trust, text=op_tr_str.format(u="A", l="S_" + str(idx)), **text_draw_args)

sl.draw_arrow(t_vec[cs_shift].trust, t_vec_updated_conflict[cs_shift].trust, **default_arrow_draw_args)
sl.draw_arrow(t_vec[bc_shift[0]].trust, t_vec_updated_harmony[bc_shift[0]].trust, **default_arrow_draw_args)
sl.draw_arrow(t_vec[bc_shift[1]].trust, t_vec_updated_harmony[bc_shift[1]].trust, **default_arrow_draw_args)

######################################################
# fusion opinions

sl.reset_plot()
fig3, ax3 = sl.create_triangle_plot(hypo_texts=('disbelief', 'belief'))

point_draw_args = default_point_draw_args.copy()
text_draw_args = default_text_draw_args.copy()
if SHOW_NO_TR_FUSION:
    point_draw_args["color"] = 'tab:gray'
    sl.draw_point(normal_fusion, **point_draw_args)
    text_draw_args["offset"] = [-0.015, 0.03]
    text_draw_args["ha"] = 'right'
    sl.draw_text_at_point(normal_fusion, text=op_str.format(l="X", u="\diamond (\mathbb{S})"), **text_draw_args)

point_draw_args["color"] = 'tab:blue'
sl.draw_point(trusted_fusion, **point_draw_args)
text_draw_args["offset"] = [0., -0.02]
text_draw_args["ha"] = 'center'
text_draw_args["va"] = 'top'
sl.draw_text_at_point(trusted_fusion, text=op_str.format(l="X", u="\diamond ([A;\mathbb{S}])"), **text_draw_args)

point_draw_args = default_point_draw_args.copy()
point_draw_args["color"] = 'tab:orange'
text_draw_args = default_text_draw_args.copy()
text_draw_args["offset"] = [0., 0.01]
text_draw_args["ha"] = 'center'
text_draw_args["va"] = 'bottom'
sl.draw_point(fusion_conflict, **point_draw_args)
sl.draw_text_at_point(fusion_conflict, text=op_tr_str.format(l="X", u="\diamond ([A;\mathbb{S}])"), **text_draw_args)

point_draw_args = default_point_draw_args.copy()
point_draw_args["color"] = 'tab:green'
text_draw_args = default_text_draw_args.copy()
text_draw_args["offset"] = [-0.02, -0.04]
text_draw_args["ha"] = 'center'
text_draw_args["va"] = 'top'
sl.draw_point(fusion_harmony, **point_draw_args)
sl.draw_text_at_point(fusion_harmony, text=op_tr_str.format(l="X", u="\diamond ([A;\mathbb{S}])"), **text_draw_args)

arrow_draw_args = default_arrow_draw_args.copy()
arrow_draw_args["length_cap"] = 0.9
sl.draw_arrow(trusted_fusion, fusion_conflict, **arrow_draw_args)
arrow_draw_args["length_cap"] = 0.8
sl.draw_arrow(trusted_fusion, fusion_harmony, **arrow_draw_args)


x_start = -0.05
x_end = 1.05
y_start = -0.175
y_end = 0.95

x_diff = x_end - x_start
y_diff = y_end - y_start

relative_size = x_diff / y_diff

figure_size_y = 5

for _ax, _fig in zip([ax,ax2,ax3],[fig,fig2, fig3]):
    _ax.set_xlim(x_start, x_end)
    _ax.set_ylim(y_start, y_end)
    _fig.set_size_inches(figure_size_y * relative_size,figure_size_y)

plt.show()

fig.savefig("exp_tr_opinions.pdf", **export_args)
fig2.savefig("exp_tr_trust.pdf", **export_args)
fig3.savefig("exp_tr_fusion.pdf", **export_args)




