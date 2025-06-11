import subjective_logic as sl
import numpy as np

dimension = 4
print(f'running the experiment of gathering information with a dimension of {dimension}')

prior = np.array(dimension * [1/dimension])
fair_belief = np.array(dimension * [1/dimension])

fair_op = sl.Opinion(fair_belief, prior)
print(f'the fair dice reference opinion is given by {fair_op}')

observations = [1,1,2,4]
evidence_vector = np.zeros((dimension,))

for obs in observations:
    evidence_vector[obs - 1] += 1

print(f'after making the observations: {observations},\nthe evidence vector is: {evidence_vector}')

obs_dirichlet = sl.DirichletDistribution(evidence_vector, prior)
obs_opinions = obs_dirichlet.as_opinion()

print(f'the resulting observation opinion is given by {obs_opinions}')

doc = obs_opinions.degree_of_conflict(fair_op)

print(f'which results in a degree of conflict: {doc}')

observations = [1,4,2,1,4,1,1,2,3,1,2,1]
print(f'adding the observations: {observations}')

for obs in observations:
    evidence_vector[obs - 1] += 1

obs_dirichlet = sl.DirichletDistribution(evidence_vector, prior)
obs_opinions = obs_dirichlet.as_opinion()
print(f'the remaining uncertainty of the observation opinion: {obs_opinions.uncertainty()}')

print(f'results in the new observation opinion, given by {obs_opinions}')

doc = obs_opinions.degree_of_conflict(fair_op)
print(f'and a degree of conflict: {doc}')
