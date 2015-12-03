from sympy.core.symbol import Symbol
import os
import sys
import inspect

class Argument(Symbol):
    def __new__(cls, name=None, **assumptions):
        # Super dirty way? to determine the variable name from the calling line.
        if not name or type(name) != str:
            frame, filename, line_number, function_name, lines, index = inspect.stack()[1]

            # Check if the script is running in an exe
            # If this is the case, symfit cannot find the vaiable name from the calling line and it will search
            # in the exe directory to find the source. Make sure to copy the required source files for this to work.
            if getattr(sys, 'frozen', False):
                basedir = sys._MEIPASS #  Find the exe base dir
                fname = os.path.basename(filename)
                # Find the source file in the pyinstaller directory
                source_file = None
                for root, dirs, files in os.walk(basedir):
                    if fname in files:
                        source_file = os.path.join(root, fname)
                        break

                if source_file is None:
                    raise IOError("Source code file not found")

                # Get the correct line from the source code
                l = 0
                with open(source_file, 'r') as fid:
                    for line in fid:
                        l+=1
                        if l == line_number:
                            lines = line
            caller = lines[0].strip()
            if '==' in caller:
                pass
            else:
                try:
                    terms = caller.split('=')
                except ValueError:
                    generated_name = name
                else:
                    generated_name = terms[0].strip()  # lhs
                return super(Argument, cls).__new__(cls, generated_name, **assumptions)
        return super(Argument,cls).__new__(cls, name, **assumptions)

    def __init__(self, name=None, *args, **kwargs):
        if name is not None:
            self.name = name
        super(Argument, self).__init__(*args, **kwargs)


class Parameter(Argument):
    """ Parameter objects are used to facilitate bounds on function parameters,
    as well as to allow AbstractFunction instances to share parameters between
    them.
    """
    def __init__(self, value=1.0, min=None, max=None, fixed=False, name=None, *args, **kwargs):
        super(Parameter, self).__init__(name, *args, **kwargs)
        self.value = value
        self.fixed = fixed
        if not self.fixed:
            self.min = min
            self.max = max


class Variable(Argument):
    pass