from theiavalidate.Validator import Validator
from theiavalidate.theiavalidate import DEFAULT_NA_VALUES

import numpy as np
import pandas as pd
import unittest

class MockOptions:
  """
  Mock the "options" object that is created in theiavalidate.py. In
  theiavalidate.py, this object is created from command-line arguments using
  the argparse package, but here we will simulate this object with a
  different class to more easily create Validator objects.
  """
  def __init__(self, options_dict=None):
    # defaults
    self.table1 = None
    self.table2 = None
    self.version = None
    self.columns_to_compare = []
    self.validation_criteria = None
    self.column_translation = None
    self.output_prefix = None
    self.na_values = DEFAULT_NA_VALUES
    self.verbose = False
    self.debug = False

    # overwrite defaults with options_dict
    if options_dict is not None:
      for key, value in options_dict.items():
        setattr(self, key, value)


class TestDetermineFileColumns(unittest.TestCase):
    def setUp(self):
      self.validator = Validator(MockOptions())

    def run_determine_file_columns(self, data1, data2):
      self.validator.table1 = pd.DataFrame(data1)
      self.validator.table2 = pd.DataFrame(data2)
      self.validator.determine_file_columns()

    def test_no_file_columns(self):
      data = {
        "col1": [1, 2, 3],
        "col2": ["foo", "bar", "baz"]
      }
      self.run_determine_file_columns(data, data)
      self.assertEqual(len(self.validator.file_columns), 0)

    def test_some_file_columns(self):
      data1 = {
        "col1": [1, 2, 3],
        "col2": ["gs://foo", "gs://bar", "gs://baz"]
      }
      data2 = {
        "col1": [1, 2, 3],
        "col2": ["gs://eggs", "gs://spam", "gs://monty"]
      }
      self.run_determine_file_columns(data1, data2)
      self.assertEqual(self.validator.file_columns, {"col2"})

    def test_missing_uri(self):
      data1 = {
        "col1": [1, 2, 3],
        "col2": ["gs://foo", np.nan, "gs://baz"]
      }
      data2 = {
        "col1": [1, 2, 3],
        "col2": ["gs://eggs", "gs://spam", "gs://monty"]
      }
      self.run_determine_file_columns(data1, data2)
      self.assertEqual(self.validator.file_columns, {"col2"})

    def test_both_columns_null(self):
      data1 = {
        "col1": ["gs://foo", "gs://bar", "gs://baz"],
        "col2": [np.nan, np.nan, np.nan]
      }
      data2 = {
        "col1": ["gs://eggs", "gs://spam", "gs://monty"],
        "col2": [np.nan, np.nan, np.nan]
      }
      self.run_determine_file_columns(data1, data2)
      self.assertEqual(self.validator.file_columns, {"col1"})

    def test_one_column_null(self):
      data1 = {
        "col1": ["gs://foo", "gs://bar", "gs://baz"],
        "col2": ["gs://x", "gs://y", "gs://z"]
      }
      data2 = {
        "col1": ["gs://eggs", "gs://spam", "gs://monty"],
        "col2": [np.nan, np.nan, np.nan]
      }
      self.run_determine_file_columns(data1, data2)
      self.assertEqual(self.validator.file_columns, {"col1", "col2"})

    def test_one_column_not_null(self):
      data1 = {
        "col1": ["gs://foo", "gs://bar", "gs://baz"],
        "col2": ["gs://x", "gs://y", "gs://z"]
      }
      data2 = {
        "col1": ["gs://eggs", "gs://spam", "gs://monty"],
        "col2": [1, 2, 3]
      }
      self.run_determine_file_columns(data1, data2)
      self.assertEqual(self.validator.file_columns, {"col1"})


class TestCompareFiles(unittest.TestCase):
  def setUp(self):
    self.validator = Validator(MockOptions())
    self.file_comparison_dir = "tests/file1_files"
    self.file_comparison_dir = "tests/file2_files"

  def test_matching_files(self):
    df1 = pd.DataFrame({
      "col1": ["gs://match1-1.txt", "gs://match1-2.txt", "gs://match1-2.txt"],
      "col2": ["gs://match2-1.txt", "gs://match2-2.txt", "gs://match2-3.txt"]
    })
    df2 = pd.DataFrame({
      "col1": ["gs://match1-1.txt", "gs://match1-2.txt", "gs://match1-2.txt"],
      "col2": ["gs://match2-1.txt", "gs://match2-2.txt", "gs://match2-3.txt"]
    })
    observed = self.validator.compare_files(df1, df2)
    expected = pd.DataFrame({
      "Number of differences (exact match)": [0, 0]
    })
    expected.index = ["col1, col2"]
    pd.testing.assert_frame_equal(observed, expected)

  def test_mismatching_files(self):
    pass

  def test_mix_matching_files(self):
    pass

  def test_null_file(self):
    pass

