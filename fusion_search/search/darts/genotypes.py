from collections import namedtuple

Genotype = namedtuple('Genotype', 'edges steps concat')
StepGenotype = namedtuple('StepGenotype', 'inner_edges inner_steps inner_concat')

PRIMITIVES = [
    'none',
    'skip'
]

STEP_EDGE_PRIMITIVES = [
    'none',
    'skip'
]


# We must keep this order consistent with the order in /darts/models/search/darts/utils.py

STEP_STEP_PRIMITIVES = [
    'Sum',
    'ScaleDotAttn',
    'LinearGLU',
    'ConcatFC',
    'SE1',
    'CatConvMish'

]

