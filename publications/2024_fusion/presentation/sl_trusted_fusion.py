#!/usr/bin/python3
import numpy as np
import subjective_logic as sl
import matplotlib
import matplotlib.pyplot as plt
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
    "format": 'png',
    "dpi": 500,
    "transparent": True,
    "bbox_inches": 'tight',
    "pad_inches": 0,
}

# general drawing settings (what exactly should be drawn)
SHOW_ACC = False
# SHOW_AVG = not SHOW_ACC
SHOW_AVG = False
SHOW_NO_TR = False
SHOW_NO_TR_FUSION = False
SHOW_TR_FUSION = False
SHOW_DISCOUNTED = False
SHOW_JOSANG = False

# config settings, sets what should be drawn for a certain slide state
INIT_PICS = False
TRUSTED_FUSION = False
TRUST_REVISION = False
TRUST_REVISION_COMPARISON = True

SHOW_NO_TR = True
SHOW_NO_TR_FUSION = True

if TRUSTED_FUSION or TRUST_REVISION or TRUST_REVISION_COMPARISON:
    SHOW_DISCOUNTED = True
    SHOW_TR_FUSION = True

if TRUST_REVISION or TRUST_REVISION_COMPARISON:
    SHOW_JOSANG = True

if TRUST_REVISION_COMPARISON:
    SHOW_AVG = True




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
opinions = sl.TrustedOpinion2d.extractOpinions(t_vec)
discounted_opinions = sl.TrustedOpinion2d.extractDiscountedOpinions(t_vec)


######################################################
# calc normal and trusted fusion
######################################################
normal_fusion = sl.Fusion.fuse_opinions(sl.Fusion.FusionType.CUMULATIVE, opinions)
trusted_fusion = sl.TrustedFusion.fuse_opinions(sl.Fusion.FusionType.CUMULATIVE, t_vec)

######################################################
# calc trusted fusion using trust revision
######################################################
weighted_types_cs_acc = [
    (sl.TrustRevision.TrustRevisionType.CONFLICT_SHARES, sl.Conflict.ConflictType.ACCUMULATE, 1.),
]
weighted_types_cs_avg = [
    (sl.TrustRevision.TrustRevisionType.CONFLICT_SHARES, sl.Conflict.ConflictType.AVERAGE, 2.),
]
weighted_types_bc = [
    (sl.TrustRevision.TrustRevisionType.REFERENCE_FUSION, sl.Conflict.ConflictType.BELIEF_CUMULATIVE, 2.),
]

fusion_cs_acc, t_vec_updated_cs_acc = sl.TrustedFusion.fuse_opinions_(sl.Fusion.FusionType.CUMULATIVE, weighted_types_cs_acc, t_vec)
fusion_cs_avg, t_vec_updated_cs_avg = sl.TrustedFusion.fuse_opinions_(sl.Fusion.FusionType.CUMULATIVE, weighted_types_cs_avg, t_vec)
fusion_bc, t_vec_updated_bc = sl.TrustedFusion.fuse_opinions_(sl.Fusion.FusionType.CUMULATIVE, weighted_types_bc, t_vec)

discounted_opinions_cs_acc = sl.TrustedOpinion2d.extractDiscountedOpinions(t_vec_updated_cs_acc)
discounted_opinions_cs_avg = sl.TrustedOpinion2d.extractDiscountedOpinions(t_vec_updated_cs_avg)
discounted_opinions_bc = sl.TrustedOpinion2d.extractDiscountedOpinions(t_vec_updated_bc)

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
# source opinions

fig, ax = sl.create_triangle_plot(hypo_texts=('disbelief\n(sunny)', 'belief\n(rainy)'))
point_draw_args = default_point_draw_args.copy()
text_draw_args = default_text_draw_args.copy()
if SHOW_NO_TR:
    for idx, opinion in enumerate(opinions):
        sl.draw_point(opinion, **point_draw_args)
        # op_str=r"$\omega_X^{S_{{idx}}}$"
        if idx == 2 :
            text_draw_args["ha"] = "right"
            text_draw_args["va"] = "top"
            text_draw_args["offset"] = [0.0, -0.02]
            if SHOW_DISCOUNTED:
                text = op_str.format(l="X", u="S_" + str(idx)) + " = "  + op_str.format(l="X", u="[A;S_" + str(idx) + "]")
                sl.draw_text_at_point(opinion, text=text, **text_draw_args)
                continue
        sl.draw_text_at_point(opinion, text=op_str.format(l="X", u="S_" + str(idx)), **text_draw_args)

if SHOW_DISCOUNTED:
    point_draw_args = default_point_draw_args.copy()
    point_draw_args["color"] = 'tab:blue'
    text_draw_args = default_text_draw_args.copy()
    for idx, opinion in enumerate(discounted_opinions):
        sl.draw_point(opinion, **point_draw_args)
        if idx == 2 and SHOW_NO_TR:
            continue
        if idx == 1:
            text_draw_args["ha"] = "right"
            text_draw_args["va"] = "bottom"
        sl.draw_text_at_point(opinion, text=op_str.format(l="X", u="[A;S_" + str(idx) + "]"), **text_draw_args)

point_draw_args = default_point_draw_args.copy()
point_draw_args["color"] = 'tab:green'
text_draw_args = default_text_draw_args.copy()
text_draw_args["ha"] = "right"
text_draw_args["va"] = "bottom"
text_draw_args["offset"] = [-0.01, 0.]
if SHOW_ACC:
    for idx, (opinion, opinion_up) in enumerate(zip(discounted_opinions, discounted_opinions_cs_acc)):
        if opinion == opinion_up:
            continue
        sl.draw_point(opinion_up, **point_draw_args)
        sl.draw_text_at_point(opinion_up, text=op_tr_str.format(l="X", u="[A;S_" + str(idx) + "]"), **text_draw_args)

if SHOW_AVG:
    for idx, (opinion, opinion_up) in enumerate(zip(discounted_opinions, discounted_opinions_cs_avg)):
        if opinion == opinion_up:
            continue
        sl.draw_point(opinion_up, **point_draw_args)
        sl.draw_text_at_point(opinion_up, text=op_tr_str.format(l="X", u="[A;S_" + str(idx) + "]"), **text_draw_args)

if SHOW_AVG or SHOW_ACC:
    sl.draw_arrow(discounted_opinions[cs_shift], discounted_opinions_cs_avg[cs_shift], **default_arrow_draw_args)

if SHOW_JOSANG:
    point_draw_args = default_point_draw_args.copy()
    point_draw_args["color"] = 'tab:orange'
    text_draw_args = default_text_draw_args.copy()
    text_draw_args["va"] = "bottom"
    for idx, (opinion, opinion_up) in enumerate(zip(discounted_opinions, discounted_opinions_bc)):
        if opinion == opinion_up:
            continue
        sl.draw_point(opinion_up, **point_draw_args)
        sl.draw_text_at_point(opinion_up, text=op_tr_str.format(l="X", u="[A;S_" + str(idx) + "]"), **text_draw_args)

    sl.draw_arrow(discounted_opinions[bc_shift], discounted_opinions_bc[bc_shift], **default_arrow_draw_args)


######################################################
# trust opinions

sl.reset_plot()
fig2, ax2 = sl.create_triangle_plot(hypo_texts=('distrust', 'trust'))

if SHOW_TR_FUSION:
    point_draw_args = default_point_draw_args.copy()
    point_draw_args["color"] = 'tab:blue'
    text_draw_args = default_text_draw_args.copy()
    text_draw_args["ha"] = "right"
    text_draw_args["va"] = "bottom"
    text_draw_args["offset"] = [-0.01, 0.02]
    for idx, t_op in enumerate(t_vec):
        trust = t_op.trust()
        if idx == 2:
            text_draw_args["ha"] = "left"
            text_draw_args["va"] = "bottom"
            text_draw_args["offset"] = [0.01, 0.02]
        sl.draw_point(trust, **point_draw_args)
        sl.draw_text_at_point(trust, text=op_str.format(u="A", l="S_" + str(idx)), **text_draw_args)

point_draw_args = default_point_draw_args.copy()
point_draw_args["color"] = 'tab:green'
text_draw_args = default_text_draw_args.copy()
text_draw_args["offset"] = [0.01, -0.01]
if SHOW_ACC:
    for idx, (t_op, t_op_up) in enumerate(zip(t_vec, t_vec_updated_cs_acc)):
        if t_op.trust() == t_op_up.trust():
            continue
        trust = t_op_up.trust()
        sl.draw_point(trust, **point_draw_args)
        sl.draw_text_at_point(trust, text=op_tr_str.format(u="A", l="S_" + str(idx)), **text_draw_args)

if SHOW_AVG:
    for idx, (t_op, t_op_up) in enumerate(zip(t_vec, t_vec_updated_cs_avg)):
        if t_op.trust() == t_op_up.trust():
            continue
        trust = t_op_up.trust()
        sl.draw_point(trust, **point_draw_args)
        sl.draw_text_at_point(trust, text=op_tr_str.format(u="A", l="S_" + str(idx)), **text_draw_args)

if SHOW_AVG or SHOW_ACC:
    sl.draw_arrow(t_vec[cs_shift].trust(), t_vec_updated_cs_avg[cs_shift].trust(), **default_arrow_draw_args)

if SHOW_JOSANG:
    point_draw_args = default_point_draw_args.copy()
    point_draw_args["color"] = 'tab:orange'
    text_draw_args = default_text_draw_args.copy()
    text_draw_args["offset"] = [0.01, -0.01]
    for idx, (t_op, t_op_up) in enumerate(zip(t_vec, t_vec_updated_bc)):
        if t_op.trust() == t_op_up.trust():
            continue
        trust = t_op_up.trust()
        sl.draw_point(trust, **point_draw_args)
        sl.draw_text_at_point(trust, text=op_tr_str.format(u="A", l="S_" + str(idx)), **text_draw_args)

    sl.draw_arrow(t_vec[bc_shift].trust(), t_vec_updated_bc[bc_shift].trust(), **default_arrow_draw_args)

######################################################
# fusion opinions


sl.reset_plot()
fig3, ax3 = sl.create_triangle_plot(hypo_texts=('disbelief\n(sunny)', 'belief\n(rainy)'))

point_draw_args = default_point_draw_args.copy()
text_draw_args = default_text_draw_args.copy()
if SHOW_NO_TR_FUSION:
    point_draw_args["color"] = 'tab:gray'
    sl.draw_point(normal_fusion, **point_draw_args)
    text_draw_args["offset"] = [-0.015, 0.03]
    text_draw_args["ha"] = 'right'
    sl.draw_text_at_point(normal_fusion, text=op_str.format(l="X", u="\diamond (\mathbb{S})"), **text_draw_args)

if SHOW_TR_FUSION:
    point_draw_args["color"] = 'tab:blue'
    sl.draw_point(trusted_fusion, **point_draw_args)
    text_draw_args["offset"] = [0., -0.02]
    text_draw_args["ha"] = 'center'
    text_draw_args["va"] = 'top'
    sl.draw_text_at_point(trusted_fusion, text=op_str.format(l="X", u="\diamond ([A;\mathbb{S}])"), **text_draw_args)


point_draw_args = default_point_draw_args.copy()
point_draw_args["color"] = 'tab:green'
text_draw_args = default_text_draw_args.copy()
text_draw_args["offset"] = [0., 0.01]
text_draw_args["ha"] = 'center'
text_draw_args["va"] = 'bottom'

arrow_draw_args = default_arrow_draw_args.copy()

if SHOW_ACC:
    sl.draw_point(fusion_cs_acc, **point_draw_args)
    sl.draw_text_at_point(fusion_cs_acc, text=op_tr_str.format(l="X", u="\diamond ([A;\mathbb{S}])"), **text_draw_args)

if SHOW_AVG:
    sl.draw_point(fusion_cs_avg, **point_draw_args)
    sl.draw_text_at_point(fusion_cs_avg, text=op_tr_str.format(l="X", u="\diamond ([A;\mathbb{S}])"), **text_draw_args)

if SHOW_AVG or SHOW_ACC:
    arrow_draw_args["length_cap"] = 0.9
    sl.draw_arrow(trusted_fusion, fusion_cs_avg, **arrow_draw_args)

if SHOW_JOSANG:
    point_draw_args = default_point_draw_args.copy()
    point_draw_args["color"] = 'tab:orange'
    text_draw_args = default_text_draw_args.copy()
    text_draw_args["offset"] = [0., 0.01]
    text_draw_args["ha"] = 'left'
    text_draw_args["va"] = 'bottom'
    sl.draw_point(fusion_bc, **point_draw_args)
    sl.draw_text_at_point(fusion_bc, text=op_tr_str.format(l="X", u="\diamond ([A;\mathbb{S}])"), **text_draw_args)

    arrow_draw_args["length_cap"] = 0.8
    sl.draw_arrow(trusted_fusion, fusion_bc, **arrow_draw_args)


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

if INIT_PICS:
    fig.savefig("sl_trusted_fusion_init_opinions.png", **export_args)
    fig2.savefig("sl_trusted_fusion_init_trust.png", **export_args)
    fig3.savefig("sl_trusted_fusion_init_fusion.png", **export_args)
if TRUSTED_FUSION:
    fig.savefig("sl_trusted_fusion_tf_opinions.png", **export_args)
    fig2.savefig("sl_trusted_fusion_tf_trust.png", **export_args)
    fig3.savefig("sl_trusted_fusion_tf_fusion.png", **export_args)
if TRUST_REVISION:
    fig.savefig("sl_trusted_fusion_tr_opinions.png", **export_args)
    fig2.savefig("sl_trusted_fusion_tr_trust.png", **export_args)
    fig3.savefig("sl_trusted_fusion_tr_fusion.png", **export_args)
if TRUST_REVISION_COMPARISON:
    fig.savefig("sl_trusted_fusion_tr_opinions_comp.png", **export_args)
    fig2.savefig("sl_trusted_fusion_tr_trust_comp.png", **export_args)
    fig3.savefig("sl_trusted_fusion_tr_fusion_comp.png", **export_args)




