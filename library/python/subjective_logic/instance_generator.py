import numpy as np
import subjective_logic as sl
import inspect

def gen_instance(name: str, *args):
    if len(args) == 1:
        updated_args = []
        dimension = args[0]
        if isinstance(args[0], np.ndarray) or isinstance(args[0], list) or 'Array' in type(args[0]).__name__:
            dimension = len(args[0])
            updated_args.append(args[0])
        class_name = name + str(dimension) + "d"
        return getattr(sl, class_name)(*updated_args)
    elif len(args) == 2 and ( \
               isinstance(args[0], list) and isinstance(args[1], list) or \
               isinstance(args[0], np.ndarray) and isinstance(args[1], np.ndarray)) \
            and len(args[0]) == len(args[1]) :
        dimension = len(args[0])
        class_name = name + str(dimension) + "d"
        return getattr(sl, class_name)(*args)

    class_name = name + str(len(args)) + "d"
    return getattr(sl, class_name)(*args)


def DirichletDistribution(*args):
    return gen_instance(inspect.stack()[0][3], *args)


def Opinion(*args):
    return gen_instance(inspect.stack()[0][3], *args)


def OpinionNoBase(*args):
    return gen_instance(inspect.stack()[0][3], *args)
