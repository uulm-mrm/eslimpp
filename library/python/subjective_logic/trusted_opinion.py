import inspect
import subjective_logic._subjective_logic_lib_python_api as sl

def TrustedOpinion(n: int):
    class_name = inspect.stack()[0][3] + str(n) + "d"
    return getattr(sl, class_name)()

def TrustedOpinion(*args):
    if len(args) == 1 :
       class_name = inspect.stack()[0][3] + str(args[0]) + "d"
       return getattr(sl, class_name)()

    size = str(args[1].dimension)

    appendix = ""
    if isinstance(args[1], getattr(sl,"OpinionNoBase" + size + "d")):
        appendix = "NoBase" + size + "d"
    elif isinstance(args[1], getattr(sl,"OpinionNoBase" + size + "f")):
        appendix = "NoBase" + size + "f"
    elif isinstance(args[1], getattr(sl,"Opinion" + size + "f")):
        appendix = size + "f"
    else:
        appendix = size + "d"

    class_name = inspect.stack()[0][3] + appendix
    return getattr(sl, class_name)(*args)
