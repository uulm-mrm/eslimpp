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

define_opinion_tikz_string = "\\defineOpinion[{prior}]{{{name}}}{{{belief}}}{{{disbelief}}}{{{uncertainty}}}"
tikz_filename_trust = "opinion_definitions_trust.tex"
tikz_filename_observ = "opinion_definitions_observ.tex"
tikz_filename_fusion = "opinion_definitions_fusion.tex"

######################################################
# create opinions
######################################################
meas_s1 = sl.Opinion(0.2, 0.7)
trust_s1 = sl.Opinion2d([0.2, 0.3], [1.0, 0.0])
t_s1 = sl.TrustedOpinion(trust_s1, meas_s1)

meas_s2 = sl.Opinion(0.0, 0.75)
trust_s2 = sl.Opinion2d([0.2, 0.2], [1.0, 0.0])
t_s2 = sl.TrustedOpinion(trust_s2, meas_s2)

meas_s3 = sl.Opinion(0.8, 0.05)
trust_s3 = sl.Opinion2d([0.7, 0.0], [1.0, 0.0])
t_s3 = sl.TrustedOpinion(trust_s3, meas_s3)
# meas_s1 = sl.Opinion(0.2, 0.7)
# trust_s1 = sl.Opinion2d([0.3, 0.3], [1.0, 0.0])
# t_s1 = sl.TrustedOpinion(trust_s1, meas_s1)
#
# meas_s2 = sl.Opinion(0.0, 0.7)
# trust_s2 = sl.Opinion2d([0.3, 0.2], [1.0, 0.0])
# t_s2 = sl.TrustedOpinion(trust_s2, meas_s2)
#
# meas_s3 = sl.Opinion(0.8, 0.05)
# trust_s3 = sl.Opinion2d([0.7, 0.0], [1.0, 0.0])
# t_s3 = sl.TrustedOpinion(trust_s3, meas_s3)

t_vec = [t_s1, t_s2, t_s3]
opinions = sl.TrustedOpinion2d.extractOpinions(t_vec)
discounted_opinions = sl.TrustedOpinion2d.extractDiscountedOpinions(t_vec)


######################################################
# calc normal and trusted fusion
######################################################
normal_fusion = sl.Fusion.fuse_opinions(sl.FusionType.CUMULATIVE, opinions)
trusted_fusion = sl.TrustedFusion.fuse_opinions(sl.FusionType.CUMULATIVE, t_vec)

######################################################
# calc trusted fusion using trust revision
######################################################
weighted_types_cs_acc = [
    sl.WeightedTypes(sl.RelationType.CONFLICT, sl.TrustRevisionType.SHARES, sl.ConflictType.ACCUMULATE, 2),
]
weighted_types_cs_avg = [
    sl.WeightedTypes(sl.RelationType.CONFLICT, sl.TrustRevisionType.SHARES, sl.ConflictType.AVERAGE, 2),
]
weighted_types_bc = [
    sl.WeightedTypes(sl.RelationType.CONFLICT, sl.TrustRevisionType.REFERENCE_FUSION, sl.ConflictType.BELIEF_AVERAGE, 2),
]

fusion_cs_acc, t_vec_updated_cs_acc = sl.TrustedFusion.fuse_opinions_(sl.FusionType.CUMULATIVE, weighted_types_cs_acc, t_vec)
fusion_cs_avg, t_vec_updated_cs_avg = sl.TrustedFusion.fuse_opinions_(sl.FusionType.CUMULATIVE, weighted_types_cs_avg, t_vec)
fusion_bc, t_vec_updated_bc = sl.TrustedFusion.fuse_opinions_(sl.FusionType.CUMULATIVE, weighted_types_bc, t_vec)

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
        if idx == 2:
            text_draw_args["ha"] = "right"
            text_draw_args["va"] = "top"
            text_draw_args["offset"] = [0.0, -0.02]
            text = op_str.format(l="X", u="S_" + str(idx)) + " = "  + op_str.format(l="X", u="[A;S_" + str(idx) + "]")
            sl.draw_text_at_point(opinion, text=text, **text_draw_args)
            continue
        sl.draw_text_at_point(opinion, text=op_str.format(l="X", u="S_" + str(idx)), **text_draw_args)

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

point_draw_args = default_point_draw_args.copy()
point_draw_args["color"] = 'tab:orange'
text_draw_args = default_text_draw_args.copy()
text_draw_args["va"] = "bottom"
for idx, (opinion, opinion_up) in enumerate(zip(discounted_opinions, discounted_opinions_bc)):
    if opinion == opinion_up:
        continue
    sl.draw_point(opinion_up, **point_draw_args)
    sl.draw_text_at_point(opinion_up, text=op_tr_str.format(l="X", u="[A;S_" + str(idx) + "]"), **text_draw_args)

sl.draw_arrow(discounted_opinions[cs_shift], discounted_opinions_cs_avg[cs_shift], **default_arrow_draw_args)
sl.draw_arrow(discounted_opinions[bc_shift], discounted_opinions_bc[bc_shift], **default_arrow_draw_args)
sl.draw_arrow(discounted_opinions[bc_shift+1], discounted_opinions_bc[bc_shift+1], **default_arrow_draw_args)

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
        if t_op.trust == t_op_up.trust:
            continue
        trust = t_op_up.trust
        sl.draw_point(trust, **point_draw_args)
        sl.draw_text_at_point(trust, text=op_tr_str.format(u="A", l="S_" + str(idx)), **text_draw_args)

point_draw_args = default_point_draw_args.copy()
point_draw_args["color"] = 'tab:orange'
text_draw_args = default_text_draw_args.copy()
text_draw_args["offset"] = [0.01, -0.01]
for idx, (t_op, t_op_up) in enumerate(zip(t_vec, t_vec_updated_bc)):
    if t_op.trust == t_op_up.trust:
        continue
    trust = t_op_up.trust
    sl.draw_point(trust, **point_draw_args)
    sl.draw_text_at_point(trust, text=op_tr_str.format(u="A", l="S_" + str(idx)), **text_draw_args)

sl.draw_arrow(t_vec[cs_shift].trust, t_vec_updated_cs_avg[cs_shift].trust, **default_arrow_draw_args)
sl.draw_arrow(t_vec[bc_shift].trust, t_vec_updated_bc[bc_shift].trust, **default_arrow_draw_args)
sl.draw_arrow(t_vec[bc_shift+1].trust, t_vec_updated_bc[bc_shift+1].trust, **default_arrow_draw_args)

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
if SHOW_ACC:
    sl.draw_point(fusion_cs_acc, **point_draw_args)
    sl.draw_text_at_point(fusion_cs_acc, text=op_tr_str.format(l="X", u="\diamond ([A;\mathbb{S}])"), **text_draw_args)
if SHOW_AVG:
    sl.draw_point(fusion_cs_avg, **point_draw_args)
    sl.draw_text_at_point(fusion_cs_avg, text=op_tr_str.format(l="X", u="\diamond ([A;\mathbb{S}])"), **text_draw_args)

point_draw_args = default_point_draw_args.copy()
point_draw_args["color"] = 'tab:orange'
text_draw_args = default_text_draw_args.copy()
text_draw_args["offset"] = [0., 0.01]
text_draw_args["ha"] = 'left'
text_draw_args["va"] = 'bottom'
sl.draw_point(fusion_bc, **point_draw_args)
sl.draw_text_at_point(fusion_bc, text=op_tr_str.format(l="X", u="\diamond ([A;\mathbb{S}])"), **text_draw_args)

arrow_draw_args = default_arrow_draw_args.copy()
arrow_draw_args["length_cap"] = 0.9
sl.draw_arrow(trusted_fusion, fusion_cs_avg, **arrow_draw_args)
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

# fig.savefig("exp_tr_opinions.pdf", **export_args)
# fig2.savefig("exp_tr_trust.pdf", **export_args)
# fig3.savefig("exp_tr_fusion.pdf", **export_args)

plt.show()


# trust_file = open(tikz_filename_trust, 'w')
# observ_file = open(tikz_filename_observ, 'w')
# fusion_file = open(tikz_filename_fusion, 'w')
#
# for idx, t_op in enumerate(t_vec):
#     trust = t_op.trust
#     name = f'TAS{idx}'
#     trust_file.write(define_opinion_tikz_string.format(prior=trust.getBinomialPrior(), name=name, belief=trust.belief(), disbelief=trust.disbelief(), uncertainty=trust.uncertainty()))
#     trust_file.write('\n')
#
# for idx, t_op in enumerate(t_vec_updated_bc):
#     trust = t_op.trust
#     name = f'JTAS{idx}'
#     trust_file.write(define_opinion_tikz_string.format(prior=trust.getBinomialPrior(), name=name, belief=trust.belief(), disbelief=trust.disbelief(), uncertainty=trust.uncertainty()))
#     trust_file.write('\n')
#
# for idx, t_op in enumerate(t_vec_updated_cs_avg):
#     trust = t_op.trust
#     name = f'CTAS{idx}'
#     trust_file.write(define_opinion_tikz_string.format(prior=trust.getBinomialPrior(), name=name, belief=trust.belief(), disbelief=trust.disbelief(), uncertainty=trust.uncertainty()))
#     trust_file.write('\n')
#
# for idx, op in enumerate(opinions):
#     name = f'OSX{idx}'
#     observ_file.write(define_opinion_tikz_string.format(prior=op.getBinomialPrior(), name=name, belief=op.belief(), disbelief=op.disbelief(), uncertainty=op.uncertainty()))
#     observ_file.write('\n')
#
# for idx, op in enumerate(discounted_opinions):
#     name = f'DOSX{idx}'
#     observ_file.write(define_opinion_tikz_string.format(prior=op.getBinomialPrior(), name=name, belief=op.belief(), disbelief=op.disbelief(), uncertainty=op.uncertainty()))
#     observ_file.write('\n')
#
# for idx, t_op in enumerate(t_vec_updated_bc):
#     op = t_op.discounted_opinion()
#     name = f'JDOSX{idx}'
#     observ_file.write(define_opinion_tikz_string.format(prior=op.getBinomialPrior(), name=name, belief=op.belief(), disbelief=op.disbelief(), uncertainty=op.uncertainty()))
#     observ_file.write('\n')
#
# for idx, t_op in enumerate(t_vec_updated_cs_avg):
#     op = t_op.discounted_opinion()
#     name = f'CDOSX{idx}'
#     observ_file.write(define_opinion_tikz_string.format(prior=op.getBinomialPrior(), name=name, belief=op.belief(), disbelief=op.disbelief(), uncertainty=op.uncertainty()))
#     observ_file.write('\n')
#
# fusion_ops = [normal_fusion, trusted_fusion, fusion_bc, fusion_cs_avg]
# fusion_names = ['FN', 'TF', 'TRJ', 'TRC']
# for name, op in zip(fusion_names, fusion_ops):
#     fusion_file.write(define_opinion_tikz_string.format(prior=op.getBinomialPrior(), name=name, belief=op.belief(), disbelief=op.disbelief(), uncertainty=op.uncertainty()))
#     fusion_file.write('\n')
#
#
# trust_file.close()
# observ_file.close()
# fusion_file.close()








