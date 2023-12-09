#!/usr/bin/env python3

import argparse
from . import CheckInputs
from .__init__ import __VERSION__
from .Validator import Validator

DEFAULT_NA_VALUES = [
  '-1.#IND', '1.#QNAN', '1.#IND', '-1.#QNAN', '#N/A N/A', '#N/A', 'N/A', 'n/a',
  '', '#NA', 'NULL', 'null', 'NaN','-NaN', 'nan', '-nan', 'None'
]

def main():
  parser = argparse.ArgumentParser(
    description = "This tool compares two tab-delimited files and outputs a report of the differences between the two files.",
    usage = "python3 theiavalidate/theiavalidate.py table1 table2 [options]",
    formatter_class = lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=10))
  parser.add_argument("table1", 
                      help="the first table to compare", type=CheckInputs.is_table_valid)
  parser.add_argument("table2", 
                      help="the second table to compare", type=CheckInputs.is_table_valid)
  parser.add_argument("-v", "--version", 
                      action='version', version=str(__VERSION__))
  parser.add_argument("-c", "--columns_to_compare",
                      help="a comma-separated list of columns to compare\nrequired for a successful run", default=None, metavar="\b", type=CheckInputs.is_comma_delimited_list)
  parser.add_argument("-m", "--validation_criteria", 
                      help="a tab-delimited file containing the validation criteria to check", default=None, metavar="\b", type=CheckInputs.is_table_valid)
  parser.add_argument("-l", "--column_translation", 
                      help="a tab-delimited file that links column names between the two tables", default=None, metavar="\b", type=CheckInputs.is_table_valid)
  parser.add_argument("-o", "--output_prefix", 
                      help="the output file name prefix\ndo not include any spaces", default="theiavalidate", metavar="\b")
  parser.add_argument("-n", "--na_values", 
                      help=f"the values that should be considered NA\ndefault values = {DEFAULT_NA_VALUES}", 
                      default=DEFAULT_NA_VALUES, metavar="\b", type=int)
  parser.add_argument("--verbose", 
                      help="increase stdout verbosity", action="store_true", default=False)
  parser.add_argument("--debug", 
                      help="increase stdout verbosity to debug; overwrites --verbose", action="store_true", default=False)

  options = parser.parse_args()
  
  validate = Validator(options)
  validate.compare()
  
  
if __name__ == "__main__":
  main()