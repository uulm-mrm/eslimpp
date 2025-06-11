import subjective_logic as sl
import matplotlib.pyplot as plt
import numpy as np
import bezier
import math

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
    'disbelief_label_position': 0.7,
    'uncertainty_label_position': 0.7,
}


##############################
# slide SL basics
##############################

# opinion = sl.Opinion(0.4, 0.3)
# opinion.prior_belief_masses = [0.2, 0.8]
# variable_postfix = '_X^A'
# sl.draw_full_opinion_triangle(opinion, '_X', ('$\\overline{x}$\n(disbelief)','$x$\n(belief)'), draw_flags_opinion)

##############################
# slide SL fusion cum fusion
##############################

# opinion = sl.Opinion(0.6, 0.1)
# opinion.prior_belief_masses = [0.2, 0.8]
# variable_postfix = '_X^A'
# sl.create_triangle_plot(('$\\overline{x}$\n(disbelief)','$x$\n(belief)'))
# sl.draw_point(opinion, color='tab:blue')
# # sl.draw_text_at_point(opinion, text=fr'$\bm{{\omega{variable_postfix}}}$', ha='left', va='top', fontsize=20, offset=[0.01, -0.01])
#
# opinion3 = sl.Opinion(0.6, 0.1)
# opinion.prior_belief_masses = [0.2, 0.8]
# sl.create_triangle_plot(('$\\overline{x}$\n(disbelief)','$x$\n(belief)'))
# sl.draw_point(opinion3, color='tab:blue')
#
# opinion2 = sl.Opinion(0.1, 0.6)
# opinion2.prior_belief_masses = [0.2, 0.8]
# variable_postfix = '_X^B'
# sl.draw_point(opinion2, color='tab:red')
#
# opinion4 = sl.Opinion(0.1, 0.6)
# opinion4.prior_belief_masses = [0.2, 0.8]
# sl.draw_point(opinion4, color='tab:red')
#
# # sl.draw_text_at_point(opinion2, text=fr'$\bm{{\omega{variable_postfix}}}$', ha='left', va='top', fontsize=20, offset=[0.01, -0.01])
# # sl.reset_plot()
#
# fused = opinion.cum_fuse(opinion2)
# variable_postfix = '_X^{A \\diamond B}'
# sl.create_triangle_plot(('$\\overline{x}$\n(disbelief)','$x$\n(belief)'))
# sl.draw_point(fused, color='tab:green')
# # sl.draw_text_at_point(fused, text=fr'$\bm{{\omega{variable_postfix}}}$', ha='right', va='top', fontsize=20, offset=[-0.01, -0.01])
#
# fused2 = opinion.cum_fuse_(opinion2).cum_fuse_(opinion3).cum_fuse_(opinion4)
# sl.create_triangle_plot(('$\\overline{x}$\n(disbelief)','$x$\n(belief)'))
# sl.draw_point(fused2, alpha=0.5, color='tab:green')


##############################
# slide trust and trust discount
##############################

# trust = sl.Opinion(0.3, 0.4)
# trust.prior_belief_masses = [0.9, 0.1]
# draw_flags_trust = {
# }
# sl.draw_full_opinion_triangle(trust, '_B^A', (r'distrust', r'trust'), draw_flags_trust)
# sl.reset_plot()
#
# opinion = sl.Opinion(0.4, 0.3)
# opinion.prior_belief_masses = [0.2, 0.8]
# draw_flags_opinion = {
# }
# sl.draw_full_opinion_triangle(opinion, '_X^{B}', None, draw_flags_opinion)
# sl.reset_plot()
#
# discount = trust.getBinomialProjection()
# opinion.trust_discount_(discount)
#
# draw_flags_trusted_opinion = {
#     'belief_label_position': 1.5,
#     'disbelief_label_position': 2.0,
#     'uncertainty_label_position': 0.7,
#     'projection_offset': [0.05, -0.02],
# }
# ax, _ = sl.draw_full_opinion_triangle(opinion, '_X^{A;B}', None, draw_flags_trusted_opinion)
#
# lower_left = sl.get_bari_point_between(sl.Opinion(0., 0.), sl.Opinion(0., 1.), position=discount)
# lower_right = sl.get_bari_point_between(sl.Opinion(0., 0.), sl.Opinion(1., 0.), position=discount)
#
# # define nodes for triangle and plot empty triangle
# nodes = np.asfortranarray([
#     [lower_left[0], lower_right[0], 0.5],
#     [lower_left[1], lower_right[1], math.sqrt(3) / 2],
# ])
# triangle = bezier.Triangle(nodes, degree=1)
# triangle.plot(2, ax=ax, alpha=0.3, color="Black")

##############################
# slide conflict
##############################

# opinion = sl.Opinion(0.6, 0.1)
# opinion.prior_belief_masses = [0.2, 0.8]
# opinion_projected = sl.OpinionNoBase(*opinion.getProjection())
# variable_postfix = '_X^B'
# # sl.create_triangle_plot(('$\\overline{x}$\n(disbelief)','$x$\n(belief)'))
# sl.create_triangle_plot()
# sl.draw_point(opinion, color='tab:blue')
# sl.draw_text_at_point(opinion, text=fr'$\bm{{\omega{variable_postfix}}}$', ha='right', va='bottom', fontsize=20, offset=[-0.01, 0.01])
# sl.draw_line(opinion, opinion_projected, '--', dashes=(10, 10), color='gray', linewidth=0.5)
# sl.draw_point(opinion_projected, color='black')
# sl.draw_text_at_point(opinion_projected, text=fr'$P{variable_postfix}(x)$', ha='center', va='top', fontsize=20,
#                    offset=[0.,-0.02])
#
# opinion2 = sl.Opinion(0.1, 0.6)
# opinion2.prior_belief_masses = [0.5, 0.5]
# opinion2_projected = sl.OpinionNoBase(*opinion2.getProjection())
# variable_postfix = '_X^C'
# sl.draw_point(opinion2, color='tab:red')
# sl.draw_text_at_point(opinion2, text=fr'$\bm{{\omega{variable_postfix}}}$', ha='left', va='bottom', fontsize=20, offset=[0.01, 0.01])
# sl.draw_line(opinion2, opinion2_projected, '--', dashes=(10, 10), color='gray', linewidth=0.5)
# sl.draw_point(opinion2_projected, color='black')
# sl.draw_text_at_point(opinion2_projected, text=fr'$P{variable_postfix}(x)$', ha='center', va='top', fontsize=20,
#                       offset=[0.,-0.02])

##############################
# slide trust revision
##############################
#
# trust = sl.Opinion(0.5, 0.2)
# trust.prior_belief_masses = [1., 0.]
# opinion = sl.Opinion(0.7, 0.1)
# opinion.prior_belief_masses = [0.6, 0.4]
# opinion2 = sl.Opinion(0.1, 0.7)
# opinion2.prior_belief_masses = [0.4, 0.6]
#
# t_op1 = sl.TrustedOpinion(trust, opinion)
# t_op2 = sl.TrustedOpinion(trust, opinion2)
# t_revised = t_op1.revise_trust(t_op2)[0]
#
# trust_projected = sl.OpinionNoBase(*t_op1.trust().getProjection())
# revised_projected = sl.OpinionNoBase(*t_revised.trust().getProjection())
#

# sl.create_triangle_plot((r'distrust', r'trust'))
# sl.draw_point(t_op1.trust(), color='black')
# variable_postfix = '_B^A'
# sl.draw_text_at_point(t_op1.trust(), text=fr'$\bm{{\omega{variable_postfix}}}$', ha='left', va='bottom', fontsize=20, offset=[0.02, 0.01])
# sl.draw_line(t_op1.trust(), trust_projected, '--', dashes=(10, 10), color='gray', linewidth=0.5)
# sl.draw_point(trust_projected, color='black')
# sl.draw_text_at_point(trust_projected, text=fr'$P{variable_postfix}(x)$', ha='center', va='top', fontsize=20,
#                    offset=[0.01,-0.02])
#
# sl.draw_point(t_revised.trust(), color='black')
# variable_postfix = '_B^A'
# sl.draw_text_at_point(t_revised.trust(), text=fr'$\bm{{\check{{\omega}}{variable_postfix}}}$', ha='right', va='bottom', fontsize=20, offset=[-0.02, 0.01])
# sl.draw_line(t_revised.trust(), revised_projected, '--', dashes=(10, 10), color='gray', linewidth=0.5)
# sl.draw_point(revised_projected, color='black')
# sl.draw_text_at_point(revised_projected, text=fr'$\check{{P}}{variable_postfix}(x)$', ha='center', va='top', fontsize=20,
#                       offset=[-0.01,-0.02])
#
# sl.draw_arrow(t_op1.trust(), t_revised.trust(), lw=0., fc='black', width = 0.01, length_includes_head=True)

##############################
# slide trust revision result

# variable_postfix = '_X^A'
# sl.create_triangle_plot()
# sl.draw_point(opinion, color='tab:blue')
# # sl.draw_text_at_point(opinion, text=fr'$\bm{{\omega{variable_postfix}}}$', ha='left', va='top', fontsize=20, offset=[0.01, -0.01])
#
# variable_postfix = '_X^B'
# sl.draw_point(opinion2, color='tab:red')
# fused = opinion.cum_fuse(opinion2)
# # variable_postfix = '_X^{A \\diamond B}'
# # sl.create_triangle_plot(('$\\overline{x}$\n(disbelief)','$x$\n(belief)'))
# sl.draw_point(fused, color='tab:green')
# # # sl.draw_text_at_point(fused, text=fr'$\bm{{\omega{variable_postfix}}}$', ha='right', va='top', fontsize=20, offset=[-0.01, -0.01])
#
# trusted_fusion = sl.TrustedFusion.cum_fuse([t_op1, t_op2], (1.,0.))
# # sl.draw_point(trusted_fusion, alpha=0.5, color='tab:green')

##############################
# slide multi agent trust revision
##############################

trust = sl.Opinion(0.5, 0.2)
trust.prior_belief_masses = [1., 0.]
opinion = sl.Opinion(0.7, 0.1)
opinion.prior_belief_masses = [0.6, 0.4]
opinion2 = sl.Opinion(0.1, 0.7)
opinion2.prior_belief_masses = [0.4, 0.6]
opinion3 = sl.Opinion(0.6, 0.1)
opinion3.prior_belief_masses = [0.6, 0.4]
opinion4 = sl.Opinion(0.1, 0.6)
opinion4.prior_belief_masses = [0.4, 0.6]

t_op1 = sl.TrustedOpinion(trust, opinion)
t_op2 = sl.TrustedOpinion(trust, opinion2)
t_op3 = sl.TrustedOpinion(trust, opinion3)
t_op4 = sl.TrustedOpinion(trust, opinion4)

sl.create_triangle_plot()
sl.draw_point(opinion, color='tab:blue')
sl.draw_point(opinion3, color='tab:blue')
# sl.draw_text_at_point(opinion, text=fr'$\bm{{\omega{variable_postfix}}}$', ha='left', va='top', fontsize=20, offset=[0.01, -0.01])

sl.draw_point(opinion2, color='tab:blue')
# sl.draw_point(opinion4, color='tab:blue')
fused = opinion.cum_fuse(opinion2).cum_fuse(opinion3)
# variable_postfix = '_X^{A \\diamond B}'
# sl.create_triangle_plot(('$\\overline{x}$\n(disbelief)','$x$\n(belief)'))
sl.draw_point(fused, color='tab:green')
# # sl.draw_text_at_point(fused, text=fr'$\bm{{\omega{variable_postfix}}}$', ha='right', va='top', fontsize=20, offset=[-0.01, -0.01])

# trusted_fusion = sl.TrustedFusion.cum_fuse([t_op1, t_op2, t_op3], (1.,0.))
# sl.draw_point(trusted_fusion, alpha=0.5, color='tab:green')
# trusted_fusion = sl.TrustedFusion.cum_fuse([t_op1, t_op2, t_op3], (6.,0.))
# sl.draw_point(trusted_fusion, alpha=0.5, color='tab:green')

trusted_fusion = sl.TrustedFusion.cum_fuse([t_op1, t_op2, t_op3], (0.,3.))
sl.draw_point(trusted_fusion, alpha=0.5, color='tab:green')

plt.show()
