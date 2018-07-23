from functools import partial

class repeatable_partial(partial):
    """
    In python < 3.5, stacked partials on the same function do not add more args
    and kwargs to the function being partialed, but rather partial the
    partial.

    This is unlogical behavior, which has been corrected in py35. This objects
    rectifies this behavior in earlier python versions as well.

    This is essentially just a copy-paste of python 3.5's __new__ method.
    """
    def __new__(*args, **keywords):
        if not args:
            raise TypeError("descriptor '__new__' of partial needs an argument")
        if len(args) < 2:
            raise TypeError("type 'partial' takes at least one argument")
        cls = args[0]
        func = args[1]
        args = args[2:]
        if not callable(func):
            raise TypeError("the first argument must be callable")
        args = tuple(args)

        if hasattr(func, "func"):
            args = func.args + args
            tmpkw = func.keywords.copy()
            tmpkw.update(keywords)
            keywords = tmpkw
            del tmpkw
            func = func.func

        return super(repeatable_partial, cls).__new__(cls, func, *args, **keywords)