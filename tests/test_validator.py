# To run these unit tests, run "python3 -m unittest" from the root of the
# project directory.

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
  def __init__(self):
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



class TestDetermineFileColumns(unittest.TestCase):
  """
  Test detecting which columns in the tables correspond to files. If there is at
  least one URI and no other values except np.nan in both tables, we should
  treat the column as a "file_column".
  """
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
  """
  Test comparing files (exact match). Identical files or two np.nans
  should count as an exact match, anything else should count as a mismatch.
  """
  SAMPLES_INDEX = ["sample1", "sample2", "sample3"]
  COLUMNS_INDEX = ["col1", "col2"]

  def setUp(self):
    self.validator = Validator(MockOptions())
    self.validator.table1_name = "table1"
    self.validator.table2_name = "table2"
    self.validator.table1_files_dir = "tests/table1_files"
    self.validator.table2_files_dir = "tests/table2_files"
    self.diff_dir = "/dev/null"  # discard diff files

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
    df1, df2 = self.create_null_files_tables()
    self.validator.compare_files(df1, df2)
    expected = pd.DataFrame({
      ("col1", "table1"): [np.nan, np.nan, np.nan],
      ("col1", "table2"): [np.nan, np.nan, np.nan],
      ("col2", "table1"): [np.nan, "gs://match2-2.txt", "gs://mismatch2-3.txt"],
      ("col2", "table2"): ["gs://match2-1.txt", np.nan, np.nan]
    }).astype(object)
    expected.index = self.SAMPLES_INDEX
    pd.testing.assert_frame_equal(self.validator.file_exact_differences, expected)

class TestValidateFiles(unittest.TestCase):
  """
  Test comparing files using the validation criteria. EXACT follows the same
  logic as compare_files(), SET should treat files as matching if after
  sorting they are identical, IGNORE should "skip" the files. Other criteria
  should result in an Exception.
  """
  SAMPLES_INDEX = ["sample1", "sample2", "sample3", "sample4", "sample5"]
  COLUMNS_INDEX = ["exact_col", "set_col", "ignore_col", "float_col"]
  TABLE1_FILE_URIS = ["gs://match1-1.txt", "gs://mismatch1-2.txt", "gs://match1-3", "gs://sortmatch1-1.txt", np.nan]
  TABLE2_FILE_URIS = ["gs://match1-1.txt", "gs://mismatch1-2.txt", np.nan, "gs://sortmatch1-1.txt", np.nan]
  EXACT_MATCHES_MASK = [True, False, False, False, True]

  def setUp(self):
    self.validator = Validator(MockOptions())
    self.validator.validation_criteria = pd.DataFrame({
      "exact_col": "EXACT",
      "set_col": "SET",
      "ignore_col": "IGNORE",
      "float_col": 0.1,
    }, index=["column", "criteria"]
    )

    # This numeric convertion is done in Validator init method
    self.validator.validation_criteria = (self.validator.validation_criteria
      .apply(pd.to_numeric, errors="ignore").convert_dtypes()
    )

    # assign the same URIs to each column, will test that the validation
    # results vary depending on the the validation criterion
    self.validator.table1 = pd.DataFrame({
      "samples": self.SAMPLES_INDEX,
      "exact_col": self.TABLE1_FILE_URIS,
      "set_col": self.TABLE1_FILE_URIS,
      "ignore_col": self.TABLE1_FILE_URIS,
      "float_col": self.TABLE1_FILE_URIS  # uh-oh
    })
    
    self.validator.table2 = pd.DataFrame({
      "samples": self.SAMPLES_INDEX,
      "exact_col": self.TABLE2_FILE_URIS,
      "set_col": self.TABLE2_FILE_URIS,
      "ignore_col": self.TABLE2_FILE_URIS,
      "float_col": self.TABLE2_FILE_URIS  # uh-oh
    })

    # the exact matches will be identical regardless of validation criteria
    self.validator.file_exact_matches = pd.DataFrame({
      "exact_col": self.EXACT_MATCHES_MASK,
      "set_col": self.EXACT_MATCHES_MASK,
      "ignore_col": self.EXACT_MATCHES_MASK,
      "float_col": self.EXACT_MATCHES_MASK
    })
    self.validator.file_exact_matches.index = self.SAMPLES_INDEX

    self.validator.file_number_of_differences = pd.DataFrame({
      self.validator.NUM_DIFFERENCES_COL: [3, 3, 3, 3]
    })
    self.validator.file_number_of_differences.index = self.COLUMNS_INDEX

    self.validator.table1_name = "table1"
    self.validator.table2_name = "table2"
    self.validator.table1_files_dir = "tests/table1_files"
    self.validator.table2_files_dir = "tests/table2_files"

    self.validator.validation_table = pd.DataFrame()

  def test_validate_exact(self):
    column = self.validator.validation_criteria["exact_col"]
    observed = self.validator.validate_files(column)
    expected = ("EXACT", 3)
    self.assertEqual(observed, expected)

  def test_validate_ignore(self):
    column = self.validator.validation_criteria["ignore_col"]
    observed = self.validator.validate_files(column)
    expected = ("IGNORE", 0)
    self.assertEqual(observed, expected)

  def test_validate_set(self):
    column = self.validator.validation_criteria["set_col"]
    observed = self.validator.validate_files(column)
    expected = ("SET", 2)  # sorted file should not count as different
    self.assertEqual(observed, expected)

  def test_validate_float(self):
    # have not implemented % difference for files
    column = self.validator.validation_criteria["set_col"]
    self.assertRaises(Exception, self.validator.validate_files(column))

  def test_validation_table(self):
    for column in ["exact_col", "set_col", "ignore_col"]:
      column = self.validator.validation_criteria[column]
      self.validator.validate_files(column)
    
    # these steps are done in run_validation_checks
    self.validator.validation_table.set_index(self.validator.table1["samples"], inplace=True)
    self.validator.validation_table.rename_axis(None, axis="index", inplace=True)
    self.validator.validation_table.columns = pd.MultiIndex.from_tuples(
      self.validator.validation_table.columns, names=["Column", "Table"]
    )

    # exact_col should count sortmatch file as a mismatch, while set_col should
    # count it as a match.
    # no column should be generated for ignore_col.
    expected = pd.DataFrame({
      ("exact_col", "table1"): [np.nan, "gs://mismatch1-2.txt", "gs://match1-3", "gs://sortmatch1-1.txt", np.nan],
      ("exact_col", "table2"): [np.nan, "gs://mismatch1-2.txt", np.nan, "gs://sortmatch1-1.txt", np.nan],
      ("set_col", "table1"): [np.nan, "gs://mismatch1-2.txt", "gs://match1-3", np.nan, np.nan],
      ("set_col", "table2"): [np.nan, "gs://mismatch1-2.txt", np.nan, np.nan, np.nan],
    })
    expected.set_index(self.validator.table1["samples"], inplace=True)
    expected.rename_axis(None, axis="index", inplace=True)
    expected.columns = pd.MultiIndex.from_tuples(expected.columns, names=["Column", "Table"])
    pd.testing.assert_frame_equal(self.validator.validation_table, expected)
