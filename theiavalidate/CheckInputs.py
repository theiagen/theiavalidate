import os
import argparse
import pandas as pd


def to_numeric_safe(value):
    """Safely converts a value to a number without raising errors
    """
    try:
        return pd.to_numeric(value, errors='raise')
    except ValueError:
        return value


def is_file_readable(filename):
    """This function checks if the file is accessible

    Args:
        filename (String): the path to a file

    Raises:
        argparse.ArgumentTypeError: an error saying that the file could not be accessed

    Returns:
        String: the path of the file
    """
    if not os.path.exists(filename) and filename != "-":
        raise argparse.ArgumentTypeError(
            "{0} cannot be accessed".format(filename))
    return filename


def is_comma_delimited_list(string):
    """This function checks if a list is comma-delimited

    Args:
        string (String): A list, hopefully comma-delimited

    Returns:
        String: The split String or None
    """
    if string is not None:
        return string.split(",")
    else:
        return None
