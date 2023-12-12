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

    def test_mixed_nulls(self):
      data1 = {
        "col1": ["gs://foo", "gs://foo", np.nan],
        "col2": ["gs://x", "gs://y", np.nan]
      }
      data2 = {
        "col1": ["gs://eggs", np.nan, np.nan],
        "col2": [np.nan, "gs://b", np.nan]
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
  SAMPLES_INDEX = ["sample1", "sample2", "sample3"]
  COLUMNS_INDEX = ["col1", "col2"]

  def setUp(self):
    self.validator = Validator(MockOptions())
    self.validator.table1_name = "table1"
    self.validator.table2_name = "table2"
    self.validator.table1_files_dir = "tests/table1_files"
    self.validator.table2_files_dir = "tests/table2_files"
    self.diff_dir = "/dev/null"

  def create_matching_files_tables(self):
    df1 = pd.DataFrame({
      "col1": ["gs://match1-1.txt", "gs://match1-2.txt", "gs://match1-3.txt"],
      "col2": ["gs://match2-1.txt", "gs://match2-2.txt", "gs://match2-3.txt"]
    })
    df2 = pd.DataFrame({
      "col1": ["gs://match1-1.txt", "gs://match1-2.txt", "gs://match1-3.txt"],
      "col2": ["gs://match2-1.txt", "gs://match2-2.txt", "gs://match2-3.txt"]
    })
    for df in [df1, df2]:
      df.index = self.SAMPLES_INDEX
    return df1, df2

  def create_mismatching_files_tables(self):
    df1 = pd.DataFrame({
      "col1": ["gs://mismatch1-1.txt", "gs://mismatch1-2.txt", "gs://mismatch1-3.txt"],
      "col2": ["gs://mismatch2-1.txt", "gs://mismatch2-2.txt", "gs://mismatch2-3.txt"]
    })
    df2 = pd.DataFrame({
      "col1": ["gs://mismatch1-1.txt", "gs://mismatch1-2.txt", "gs://mismatch1-3.txt"],
      "col2": ["gs://mismatch2-1.txt", "gs://mismatch2-2.txt", "gs://mismatch2-3.txt"]
    })
    for df in [df1, df2]:
      df.index = self.SAMPLES_INDEX
    return df1, df2
  
  def create_mix_matching_files_tables(self):
    df1 = pd.DataFrame({
      "col1": ["gs://match1-1.txt", "gs://match1-2.txt", "gs://match1-3.txt"],
      "col2": ["gs://mismatch2-1.txt", "gs://match2-2.txt", "gs://mismatch2-3.txt"]
    })
    df2 = pd.DataFrame({
      "col1": ["gs://match1-1.txt", "gs://match1-2.txt", "gs://match1-3.txt"],
      "col2": ["gs://mismatch2-1.txt", "gs://match2-2.txt", "gs://mismatch2-3.txt"]
    })
    for df in [df1, df2]:
      df.index = self.SAMPLES_INDEX
    return df1, df2

  def create_null_files_tables(self):
    df1 = pd.DataFrame({
      "col1": [np.nan, "gs://match1-2.txt", "gs://match1-3.txt"],
      "col2": [np.nan, "gs://match2-2.txt", "gs://mismatch2-3.txt"]
    })
    df2 = pd.DataFrame({
      "col1": [np.nan, "gs://match1-2.txt", "gs://match1-3.txt"],
      "col2": ["gs://match2-1.txt", np.nan, np.nan]
    })
    for df in [df1, df2]:
      df.index = self.SAMPLES_INDEX
    return df1, df2
  
  def test_matching_files_exact_matches(self):
    df1, df2 = self.create_matching_files_tables()
    self.validator.compare_files(df1, df2)
    expected = pd.DataFrame({
      "col1": [True, True, True],
      "col2": [True, True, True]
    })
    expected.index = self.SAMPLES_INDEX
    pd.testing.assert_frame_equal(self.validator.file_exact_matches, expected)

  def test_mismatching_files_exact_matches(self):
    df1, df2 = self.create_mismatching_files_tables()
    self.validator.compare_files(df1, df2)
    expected = pd.DataFrame({
      "col1": [False, False, False],
      "col2": [False, False, False]
    })
    expected.index = self.SAMPLES_INDEX
    pd.testing.assert_frame_equal(self.validator.file_exact_matches, expected)

  def test_mix_matching_files_exact_matches(self):
    df1, df2 = self.create_mix_matching_files_tables()
    self.validator.compare_files(df1, df2)
    expected = pd.DataFrame({
      "col1": [True, True, True],
      "col2": [False, True, False]
    })
    expected.index = self.SAMPLES_INDEX
    pd.testing.assert_frame_equal(self.validator.file_exact_matches, expected)

  def test_null_files_exact_matches(self):
    df1, df2 = self.create_null_files_tables()
    self.validator.compare_files(df1, df2)
    expected = pd.DataFrame({
      "col1": [True, True, True],
      "col2": [False, False, False]
    })
    expected.index = self.SAMPLES_INDEX
    pd.testing.assert_frame_equal(self.validator.file_exact_matches, expected)

  def test_null_files_number_of_differences(self):
    df1, df2 = self.create_null_files_tables()
    self.validator.compare_files(df1, df2)
    self.validator.set_file_number_of_differences()
    expected = pd.DataFrame({
      self.validator.NUM_DIFFERENCES_COL: [0, 3]
    })
    expected.index = self.COLUMNS_INDEX
    pd.testing.assert_frame_equal(self.validator.file_number_of_differences, expected)

  def test_mismatching_files_number_of_differences(self):
    df1, df2 = self.create_mismatching_files_tables()
    self.validator.compare_files(df1, df2)
    self.validator.set_file_number_of_differences()
    expected = pd.DataFrame({
      self.validator.NUM_DIFFERENCES_COL: [3, 3]
    })
    expected.index = self.COLUMNS_INDEX
    pd.testing.assert_frame_equal(self.validator.file_number_of_differences, expected)

  def test_mix_matching_files_number_of_differences(self):
    df1, df2 = self.create_mix_matching_files_tables()
    self.validator.compare_files(df1, df2)
    self.validator.set_file_number_of_differences()
    expected = pd.DataFrame({
      self.validator.NUM_DIFFERENCES_COL: [0, 2]
    })
    expected.index = self.COLUMNS_INDEX
    pd.testing.assert_frame_equal(self.validator.file_number_of_differences, expected) 

  def test_null_files_number_of_differences(self):
    df1, df2 = self.create_null_files_tables()
    self.validator.compare_files(df1, df2)
    self.validator.set_file_number_of_differences()
    expected = pd.DataFrame({
      self.validator.NUM_DIFFERENCES_COL: [0, 3]
    })
    expected.index = self.COLUMNS_INDEX
    pd.testing.assert_frame_equal(self.validator.file_number_of_differences, expected)

  def test_matching_files_exact_differences(self):
    df1, df2 = self.create_matching_files_tables()
    self.validator.compare_files(df1, df2)
    expected = pd.DataFrame({
      ("col1", "table1"): [np.nan, np.nan, np.nan],
      ("col1", "table2"): [np.nan, np.nan, np.nan],
      ("col2", "table1"): [np.nan, np.nan, np.nan],
      ("col2", "table2"): [np.nan, np.nan, np.nan]
    }).astype(object)
    expected.index = self.SAMPLES_INDEX
    pd.testing.assert_frame_equal(self.validator.file_exact_differences, expected)

  def test_mismatching_files_exact_differences(self):
    df1, df2 = self.create_mismatching_files_tables()
    self.validator.compare_files(df1, df2)
    expected = pd.DataFrame({
      ("col1", "table1"): ["gs://mismatch1-1.txt", "gs://mismatch1-2.txt", "gs://mismatch1-3.txt"],
      ("col1", "table2"): ["gs://mismatch1-1.txt", "gs://mismatch1-2.txt", "gs://mismatch1-3.txt"],
      ("col2", "table1"): ["gs://mismatch2-1.txt", "gs://mismatch2-2.txt", "gs://mismatch2-3.txt"],
      ("col2", "table2"): ["gs://mismatch2-1.txt", "gs://mismatch2-2.txt", "gs://mismatch2-3.txt"]
    }).astype(object)
    expected.index = self.SAMPLES_INDEX
    pd.testing.assert_frame_equal(self.validator.file_exact_differences, expected)

  def test_mix_matching_files_exact_differences(self):
    df1, df2 = self.create_mix_matching_files_tables()
    self.validator.compare_files(df1, df2)
    expected = pd.DataFrame({
      ("col1", "table1"): [np.nan, np.nan, np.nan],
      ("col1", "table2"): [np.nan, np.nan, np.nan],
      ("col2", "table1"): ["gs://mismatch2-1.txt", np.nan, "gs://mismatch2-3.txt"],
      ("col2", "table2"): ["gs://mismatch2-1.txt", np.nan, "gs://mismatch2-3.txt"]
    }).astype(object)
    expected.index = self.SAMPLES_INDEX
    pd.testing.assert_frame_equal(self.validator.file_exact_differences, expected)

  def test_null_files_exact_differences(self):
    df1, df2 = self.create_mix_matching_files_tables()
    self.validator.compare_files(df2, df2)
    expected = pd.DataFrame({
      ("col1", "table1"): [np.nan, np.nan, np.nan],
      ("col1", "table2"): [np.nan, np.nan, np.nan],
      ("col2", "table1"): [np.nan, "gs://match2-2.txt", "gs://mismatch2-3.txt"],
      ("col2", "table2"): ["gs://match2-1.txt", np.nan, np.nan]
    }).astype(object)
    expected.index = self.SAMPLES_INDEX
    pd.testing.assert_frame_equal(self.validator.file_exact_differences, expected)

