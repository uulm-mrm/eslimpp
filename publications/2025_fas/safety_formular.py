#!/usr/bin/python3
from subjective_logic import Opinion2d as Opinion

from system_state import SystemState, Mode

state = SystemState()

state.sensor_1 = Opinion([0.9,0.0],[0.9,0.1])
state.processing_1 = Opinion([0.9,0.0],[0.9,0.1])
state.concurrent_sa = Opinion([0.9,0.0],[0.9,0.1])

state.sensor_2 = Opinion([0.9,0.0],[0.9,0.1])
state.processing_2 = Opinion([0.9,0.0],[0.9,0.1])

state.sensor_3 = Opinion([0.9,0.0],[0.9,0.1])
state.processing_3 = Opinion([0.9,0.0],[0.9,0.1])


state.fusion = Opinion([0.9,0.0],[0.9,0.1])
state.planning = Opinion([0.9,0.0],[0.9,0.1])


state.fusion = Opinion([0.0,0.9],[0.9,0.1])
state.mode = Mode.MINIMAL_FEASIBLE
print('minimal:')
print('overall proj: ', state.getOverall().getBinomialProjection())

state.mode = Mode.STATE_OF_HEALTH
print('soh:')
print('overall proj: ', state.getOverall().getBinomialProjection())
print('\n\n')
state.fusion = Opinion([0.9,0.0],[0.9,0.1])

# print('overall:      ', state.getOverall())
state.mode = Mode.MINIMAL_FEASIBLE
print('minimal:')
print('overall proj: ', state.getOverall().getBinomialProjection())

state.mode = Mode.STATE_OF_HEALTH
print('soh:')
print('overall proj: ', state.getOverall().getBinomialProjection())
print('\n\n')


state.sensor_1 = Opinion([0.0,0.9],[0.9,0.1])
state.concurrent_sa = Opinion([0.0,0.9],[0.9,0.1])

print('one sensor bad')
state.mode = Mode.MINIMAL_FEASIBLE
print('minimal:')
print('overall proj: ', state.getOverall().getBinomialProjection())

state.mode = Mode.STATE_OF_HEALTH
print('soh:')
print('overall proj: ', state.getOverall().getBinomialProjection())
print('\n\n')


state.sensor_2 = Opinion([0.0,0.9],[0.9,0.1])

print('two sensors bad')
state.mode = Mode.MINIMAL_FEASIBLE
print('minimal:')
print('overall proj: ', state.getOverall().getBinomialProjection())

state.mode = Mode.STATE_OF_HEALTH
print('soh:')
print('overall proj: ', state.getOverall().getBinomialProjection())
print('\n\n')

state.sensor_3 = Opinion([0.0,0.9],[0.9,0.1])

print('three sensors bad')
state.mode = Mode.MINIMAL_FEASIBLE
print('minimal:')
print('overall proj: ', state.getOverall().getBinomialProjection())

state.mode = Mode.STATE_OF_HEALTH
print('soh:')
print('overall proj: ', state.getOverall().getBinomialProjection())
print('\n\n')
