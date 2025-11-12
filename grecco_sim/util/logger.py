import builtins
import sys
import os
import datetime
from contextlib import contextmanager
from typing import Any
import pandas as pd


@contextmanager
def show_output():
    """To be able to interchange the suppressor"""
    yield
    return


def _redirect_stdout(to):
    old_fd = os.dup(sys.stdout.fileno())  # copy current stdout FD
    os.dup2(to.fileno(), sys.stdout.fileno())  # point stdout to file
    return old_fd


@contextmanager
def suppress_output(to=os.devnull):
    if "pytest" in sys.modules or sys.platform not in ("linux", "linux2"):
        yield
        return

    # Ziel öffnen (falls Pfad)
    opened_here = False
    if isinstance(to, (str, os.PathLike)):
        f = open(to, "w")
        opened_here = True
    else:
        f = to

    saved_out = _redirect_stdout(f)
    try:
        yield
    finally:
        # Restore original stdout
        stdout_fd = sys.__stdout__.fileno()
        os.dup2(saved_out, stdout_fd)
        os.close(saved_out)

        if opened_here:
            f.close()


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


@contextmanager
def auto_output_redirect(output_pattern):
    with open(output_pattern, "w", buffering=1, encoding="utf-8") as f:
        old_fd = _redirect_stdout(f)
        try:
            yield
        finally:
            # restore stdout to the duplicated original
            os.dup2(old_fd, sys.stdout.fileno())
            os.close(old_fd)

    with open(output_pattern, "w") as f:
        saved_out = _redirect_stdout(f)
        try:
            yield
        finally:
            # Restore original stdout
            stdout_fd = sys.__stdout__.fileno()
            os.dup2(saved_out, stdout_fd)
            os.close(saved_out)


if __name__ == "__main__":
    print("stff")
    set_logger()
    print("stuffother stuff", "lalal")
    with suppress_output():
        print("stuffother stllalalaluff")

    print("mystuff")
