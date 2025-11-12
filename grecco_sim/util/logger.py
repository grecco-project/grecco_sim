import builtins
import sys
import os
import datetime
from contextlib import contextmanager
from typing import Any

# from wurlitzer import pipes
import io


import pandas as pd


@contextmanager
def show_output():
    """To be able to interchange the suppressor"""
    yield
    return


"""@contextmanager
def suppress_stdout():
    buf = io.StringIO()
    with open(os.devnull, "w") as devnull:
        with pipes(stdout=buf, stderr=buf):
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout
"""


@contextmanager
def suppress_output(to=os.devnull):
    """
    usage:

    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    """
    if "pytest" in sys.modules or sys.platform not in ["linux", "linux2"]:
        # Prevent output suppression in a test session as the filesystem operations
        # crash in github pipelines
        yield
        return

    fd = sys.stdout.fileno()

    #### assert that Python and C stdio write using the same file descriptor
    #### assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(to, "w") as file:
            _redirect_stdout(to=file)
            file.close()
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different


def _format_print_str(
    *msg: Any,
    print_time: bool = False,
    print_code_pos: bool = True,
    basename: bool = False,
) -> str:

    # COnvert from tuple to single item if only one printable is given
    if len(msg) == 1:
        msg = msg[0]

    _msg_str = str(msg)

    if _msg_str == "\n":
        return ""

    if print_time:
        now = datetime.datetime.now()
        time_string = now.strftime("%x-%X %z: ")
    else:
        time_string = ""

    frame = sys._getframe()
    frame = frame.f_back.f_back

    if print_code_pos:
        if frame is not None:
            code_position_string = frame.f_code.co_filename
            if basename:
                code_position_string = os.path.basename(code_position_string)
            code_position_string += ":" + str(frame.f_lineno) + ": "
        else:
            code_position_string = "Unknown output location: "

    if len(_msg_str) > 0 and _msg_str[-1] == "\n":
        _msg_str = _msg_str[:-1]

    if len(_msg_str) > 0 and _msg_str[0].isspace():
        linebreak = "\n"
    else:
        linebreak = ""

    return time_string + code_position_string + linebreak + _msg_str


_old_print_func = builtins.print


def _myprint(*msg, **kwargs):
    _old_print_func(_format_print_str(*msg), **kwargs)


def set_logger(basename=False, print_time=False):
    """Override built-in print function with custom one showing code position of the print statement

    Args:
        basename (bool, optional): print complete filepath or just the filename. Defaults to False.
        print_time (bool, optional): print the time of the print call. Defaults to False.
    """
    builtins.print = _myprint


def default_printing(func):
    def wrapper_suppress_output(*args, **kwargs):
        tmp_print_func = builtins.print
        builtins.print = _old_print_func
        res = func(*args, **kwargs)
        builtins.print = tmp_print_func
        return res

    return wrapper_suppress_output


def set_pandas_print():
    pd.set_option("display.max_columns", 22)
    pd.set_option("display.width", 300)


if __name__ == "__main__":
    print("stff")
    set_logger()
    print("stuffother stuff", "lalal")
    with suppress_output():
        print("stuffother stllalalaluff")

    print("mystuff")
