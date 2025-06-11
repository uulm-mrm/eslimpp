import numpy as np
import subjective_logic as sl
import matplotlib.pyplot as plt
import math
from typing import Union

triangle_vectors = None
ax = None
fig = None

def get_bari_point(variable):
    global triangle_vectors
    vector = variable
    if (isinstance(variable, sl.Opinion2d) or
        isinstance(variable, sl.OpinionNoBase2d) or
        isinstance(variable, sl.Opinion2f) or
        isinstance(variable, sl.OpinionNoBase2f)):
        vector = np.array([variable.disbelief(), variable.belief(), variable.uncertainty()])

    if triangle_vectors is None:
        return np.array([0,0])

    return (triangle_vectors @ vector)

def init_plot_config():
    plt.rcParams['text.usetex'] = True
    plt.rc('text.latex', preamble=r'\usepackage{bm}\usepackage{amsfonts}')


def reset_plot():
    global ax
    global fig
    ax = None
    fig = None


def create_triangle_plot(hypo_texts=None):
    global ax
    global fig
    global triangle_vectors

    init_plot_config()

    if ax is None or fig is None:
        fig, ax = plt.subplots()

    ax.axis('off')
    ax.set_xlim([-0.1, 1.2])
    # ax.set_ylim([0.,1.0])

    if hypo_texts is None:
        # hypo_texts = ('disbelief\n(free)', 'belief\n(occupied)')
        hypo_texts = ('disbelief', 'belief')

    ax.text(-0.01, -0.01, hypo_texts[0], fontsize=20, ha='left', va='top', ma='center')
    ax.text(1.01, -0.01, hypo_texts[1], fontsize=20, ha='right', va='top', ma='center')
    ax.text(0.5, math.sqrt(0.75), 'uncertainty', fontsize=20, ha='center', va='bottom')

    if triangle_vectors is None:
        triangle_vectors = np.array([
            [0.0, 1.0, 0.5],
            [0.0, 0.0, math.sqrt(3)/2]
        ])

    draw_line(sl.OpinionNoBase(0.0,1.0), sl.OpinionNoBase(1.0,0.0), color='black',linewidth=1)
    draw_line(sl.OpinionNoBase(0.0,1.0), sl.OpinionNoBase(0.0,0.0), color='black',linewidth=1)
    draw_line(sl.OpinionNoBase(1.0,0.0), sl.OpinionNoBase(0.0,0.0), color='black',linewidth=1)

    return fig, ax




def get_bari_point_between(opinion1, opinion2, position=0.5):
    global triangle

    point_start = get_bari_point(opinion1)
    point_end = get_bari_point(opinion2)
    return (point_start * (1 - position) + point_end * position)


def draw_point(opinion, *args, **kwargs):
    global triangle
    global ax

    draw_args = {
        "edgecolor": 'black',
        "linewidth": 0.5,
        "zorder": 99,
    }
    draw_args.update(**kwargs)

    point = get_bari_point(opinion)
    ax.scatter(point[0], point[1], *args, **draw_args)


def draw_text_at_point(opinion, text, *args, **kwargs):
    global triangle
    global ax
    point = get_bari_point(opinion)

    draw_args = {
        "offset": [0, 0]
    }
    draw_args.update(**kwargs)
    offset = draw_args.pop('offset')

    ax.text(point[0] + offset[0], point[1] + offset[1], text, *args, **draw_args)


def draw_line(opinion_start, opinion_end, *args, **kwargs):
    global triangle
    global ax
    point_start = get_bari_point(opinion_start)
    point_end = get_bari_point(opinion_end)

    points = np.column_stack((point_start, point_end))
    ax.plot(points[0, :], points[1, :], *args, **kwargs)


def draw_arrow(opinion_start, opinion_end, *args, **kwargs):
    global triangle
    global ax

    length_cap = 1.0
    start_offset = 0.0
    end_offset = 0.0
    # x is in direction of diff
    # y is in orthogonal to the direction of diff
    x_offset = 0.0
    y_offset = 0.0
    if "length_cap" in kwargs:
        length_cap = kwargs.pop("length_cap")
    if "start_offset" in kwargs:
        start_offset = kwargs.pop("start_offset")
    if "end_offset" in kwargs:
        start_offset = kwargs.pop("end_offset")
    if "x_offset" in kwargs:
        x_offset = kwargs.pop("x_offset")
    if "y_offset" in kwargs:
        y_offset = kwargs.pop("y_offset")


    # get points in triangle
    point_start = get_bari_point(opinion_start)
    point_end = get_bari_point(opinion_end)
    # correct dimensions
    point_start = point_start
    point_end = point_end
    # get raw diff
    diff = point_end - point_start


    # shift start/end, and correct diff afterwards incl. length cap
    point_start += start_offset * diff
    point_end -= end_offset * diff
    diff = diff * (1 - start_offset - end_offset) * length_cap

    diff_norm = diff / np.linalg.norm(diff)
    print(diff_norm)
    offset = x_offset * diff_norm + y_offset * np.array([-diff_norm[1], diff_norm[0]])

    ax.arrow(point_start[0] + offset[0], point_start[1] + offset[1], diff[0], diff[1], *args, **kwargs)


def draw_text_between(text, opinion1, opinion2, position=0.5, *args, **kwargs):
    global triangle
    global ax

    point_text = get_bari_point_between(opinion1, opinion2, position)
    ax.text(point_text[0], point_text[1], text, *args, **kwargs)


def draw_full_opinion_triangle(opinion, variable_postfix, hypo_texts=None, draw_flags_in=None):
    global ax
    global fig

    draw_flags = {
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
        'prior_offset': [0.01, -0.01],
        'projection_offset': [0.01, -0.02],
    }
    if draw_flags_in is not None:
        draw_flags.update(draw_flags_in)

    if not draw_flags['draw_hypo_texts']:
        hypo_texts = ('', '')

    create_triangle_plot(hypo_texts)

    opinion_projected = sl.OpinionNoBase(*opinion.getProjection())

    # hack since trust_discount in this direction can end up in negative uncertainty (e.g. -1e-17)
    uncert = opinion.uncertainty()
    zero_uncert = sl.OpinionNoBase(opinion.belief() + 0.5 * uncert, opinion.disbelief() + 0.5 * uncert)

    belief = opinion.belief()
    zero_belief = sl.OpinionNoBase(0., opinion.disbelief() + 0.5 * belief)

    disbelief = opinion.disbelief()
    zero_disbelief = sl.OpinionNoBase(opinion.belief() + 0.5 * disbelief, 0.)

    vacuous = sl.OpinionNoBase(0., 0.)
    vacuous_projected = sl.OpinionNoBase(*vacuous.getProjection(opinion.prior_belief_masses))

    if draw_flags['draw_prior']:
        draw_line(vacuous, vacuous_projected, '--', dashes=(10, 10), color='gray', linewidth=0.5)
        draw_point(vacuous_projected, color='black')
        if draw_flags['draw_prior_label']:
            draw_text_at_point(vacuous_projected, text=fr'$a{variable_postfix}$', ha='center', va='top', fontsize=20,
                               offset=draw_flags['prior_offset'])

    if draw_flags['draw_projection']:
        draw_line(opinion, opinion_projected, '--', dashes=(10, 10), color='gray', linewidth=0.5)
        draw_point(opinion_projected, color='black')
        if draw_flags['draw_prior_label']:
            draw_text_at_point(opinion_projected, text=fr'$P{variable_postfix}$', ha='center', va='top', fontsize=20,
                               offset=draw_flags['projection_offset'])

    if draw_flags['draw_axis']:
        draw_line(opinion, zero_uncert, color='black', linewidth=0.5)
        draw_line(opinion, zero_belief, color='black', linewidth=0.5)
        draw_line(opinion, zero_disbelief, color='black', linewidth=0.5)
        if draw_flags['draw_axis_label']:
            uncert_position = draw_flags['uncertainty_label_position']
            belief_position = draw_flags['belief_label_position']
            disbelief_position = draw_flags['disbelief_label_position']
            draw_text_between(fr'$u{variable_postfix}$', opinion, zero_uncert, position=uncert_position, fontsize=20,
                              ha='center', va='center',
                              bbox=dict(facecolor='white', alpha=0.8, edgecolor='white', pad=0))
            draw_text_between(fr'$b{variable_postfix}$', opinion, zero_belief, position=belief_position, fontsize=20,
                              ha='center', va='center',
                              bbox=dict(facecolor='white', alpha=0.8, edgecolor='white', pad=0))
            draw_text_between(fr'$d{variable_postfix}$', opinion, zero_disbelief, position=disbelief_position,
                              fontsize=20, ha='center', va='center',
                              bbox=dict(facecolor='white', alpha=0.8, edgecolor='white', pad=0))

    if draw_flags['draw_opinion']:
        draw_point(opinion, color='black')
        if draw_flags['draw_opinion_label']:
            draw_text_at_point(opinion, text=fr'$\omega{variable_postfix}$', ha='left', va='top', fontsize=20,
                               offset=[0.01, -0.01])

    return fig, ax


def draw_trust_discount():
    global ax

    opinion = sl.OpinionNoBase(0.4, 0.2)
    prior = 0.2
    opinion_projected = sl.OpinionNoBase(*opinion.getProjection([prior, 1.0 - prior]))

    # hack since trust_discount in this direction can end up in negative uncertainty (e.g. -1e-17)
    uncert = opinion.uncertainty()
    zero_uncert = sl.OpinionNoBase(opinion.belief() + 0.5 * uncert, opinion.disbelief() + 0.5 * uncert)

    belief = opinion.belief()
    zero_belief = sl.OpinionNoBase(0., opinion.disbelief() + 0.5 * belief)

    disbelief = opinion.disbelief()
    zero_disbelief = sl.OpinionNoBase(opinion.belief() + 0.5 * disbelief, 0.)

    vacuous = sl.OpinionNoBase(0., 0.)
    vacuous_projected = sl.OpinionNoBase(*vacuous.getProjection([prior, 1.0 - prior]))


if __name__ == '__main__':
    # draw_full_opinion_triangle('_B^A', (r'distrust', r'trust'))

    trust = sl.Opinion(0.3, 0.4)
    trust.prior_belief_masses = [0.9, 0.1]
    draw_flags_trust = {
    }
    draw_full_opinion_triangle(trust, '_B^A', (r'distrust', r'trust'), draw_flags_trust)
    reset_plot()

    opinion = sl.Opinion(0.4, 0.3)
    opinion.prior_belief_masses = [0.2, 0.8]
    draw_flags_opinion = {
    }
    draw_full_opinion_triangle(opinion, '_X^{B}', None, draw_flags_opinion)
    reset_plot()

    discount = trust.getBinomialProjection()
    opinion.trust_discount_(discount)

    draw_flags_trusted_opinion = {
        'belief_label_position': 1.5,
        'disbelief_label_position': 2.0,
        'uncertainty_label_position': 0.7,
    }
    draw_full_opinion_triangle(opinion, '_X^{A;B}', None, draw_flags_trusted_opinion)

    lower_left = get_bari_point_between(sl.Opinion(0., 0.), sl.Opinion(0., 1.), position=discount)
    lower_right = get_bari_point_between(sl.Opinion(0., 0.), sl.Opinion(1., 0.), position=discount)

    # define nodes for triangle and plot empty triangle
    nodes = np.asfortranarray([
        [lower_left[0], lower_right[0], 0.5],
        [lower_left[1], lower_right[1], math.sqrt(3) / 2],
    ])
    triangle = bezier.Triangle(nodes, degree=1)
    triangle.plot(2, ax=ax, alpha=0.3, color="Black")

    # lower_left = get_bari_point_between(sl.Opinion(0.,0.), sl.Opinion(0.,1.))
    # lower_right = get_bari_point_between(sl.Opinion(0.,0.), sl.Opinion(1.,0.))
    #
    # # define nodes for triangle and plot empty triangle
    # nodes = np.asfortranarray([
    #     [lower_left[0], lower_right[0], 0.5],
    #     [lower_left[1], lower_right[1], math.sqrt(3) / 2],
    # ])
    # triangle = bezier.Triangle(nodes, degree=1)
    #
    # draw_flags_trusted_opinion = {
    #     'draw_hypo_texts': False,
    #     'draw_axis': True,
    #     'draw_axis_label': False,
    #     'draw_opinion': True,
    #     'draw_prior': False,
    #     'draw_projection': False,
    # }
    # draw_full_opinion_triangle(opinion, '_X^{A;B}',None, draw_flags_trusted_opinion)

    # draw_full_opinion_triangle('_X^{A;B}',None, {'draw_axis_label': False})
    plt.show()
