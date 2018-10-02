import os


def check_if_plottable():
    return os.environ.get('DISPLAY', '') != ''
