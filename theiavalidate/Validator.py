from CheckInputs import to_numeric_safe
from datetime import date
from pretty_html_table import build_table

import difflib
import filecmp
import logging
import numpy as np
import os
import pandas as pd
import pdfkit as pdf
import re
import subprocess
import sys


class Validator:
    """This class controls the comparison process between the two input tables
    """
    NUM_DIFFERENCES_COL = "Number of differences (exact match)"

    def __init__(self, options):
        logging.basicConfig(
            encoding='utf-8', level=logging.ERROR, stream=sys.stderr)
        self.logger = logging.getLogger(__name__)
        self.verbose = options.verbose
        self.debug = options.debug

        if self.verbose:
            self.logger.setLevel(logging.INFO)
            self.logger.info("Verbose mode enabled")
        else:
            self.logger.setLevel(logging.ERROR)

        if self.debug:
            self.logger.setLevel(logging.DEBUG)
            self.logger.debug("Debug mode enabled")

        self.logger.info("Organizing the arguments")

        self.logger.debug(
            "The following options were passed to the Validator class: {}".format(options))

        self.table1_name = options.table1
        self.table2_name = options.table2

        self.column_translation = options.column_translation
        self.validation_criteria = options.validation_criteria
        self.columns_to_compare = options.columns_to_compare
        self.columns_to_compare.append("samples")

        self.file_columns = set()  # columns that contain GCP URIs to files
        self.table1_files_dir = "table1_files"
        self.table2_files_dir = "table2_files"
        self.diff_dir = "file_diffs"

        # DataFrames for holding file comparison results
        self.file_exact_matches = None
        self.file_exact_differences = None
        self.file_number_of_differences = None
        self.file_validations = None

        self.output_prefix = options.output_prefix
        self.na_values = options.na_values

        if (self.column_translation is not None):
            self.logger.debug("Creating a dictionary from the column translation file named {}".format(self.column_translation))
            self.column_translation = pd.read_csv(self.column_translation, sep="\t", index_col=0).squeeze().to_dict()
            self.logger.debug("Column translation: {}".format(self.column_translation))
        if (self.validation_criteria is not None):
            self.logger.debug("Creating a dataframe from the validation criteria file named {}".format(self.validation_criteria))
            self.validation_criteria = pd.read_csv(self.validation_criteria, sep="\t", index_col=0).transpose()
            self.validation_criteria = self.validation_criteria.apply(to_numeric_safe).convert_dtypes()
            self.logger.debug("Validation criteria: {}".format(self.validation_criteria))

    def convert_table_to_dataframe(self, table):
        """This function converts a TSV file into a pandas dataframe, renames the headers if a dictionary is provided, and replaces "None" string cells with NaNs
        """
        self.logger.debug("Converting {} to a Pandas Dataframe".format(table))
        df = pd.read_csv(table, sep="\t", keep_default_na=False,
                         header=0, index_col=False, na_values=self.na_values)

        # change first column to samples
        self.logger.debug("Renaming the first column to 'samples'")
        df.columns.values[0] = "samples"

        # rename any columns to match the second row in the column translation file
        if (self.column_translation is not None):
            self.logger.debug(
                "Renaming columns with the following dictionary: %s" % self.column_translation)
            df.rename(columns=self.column_translation, inplace=True)

        # replace "None" string cells with NaNs
        self.logger.debug("Replacing 'None' strings with `np.nan`s")
        df.replace("None", np.nan, inplace=True)

        self.logger.debug("Done converting table to dataframe!")
        return df

    def create_filtered_table(self):
        """This function filters out the columns that are not in the list of columns to compare  and writes out the resulting table
        """
        self.logger.debug(
            "Removing columns that are not in the list of columns to compare")

        drop_list1 = []
        drop_list2 = []

        self.logger.debug(
            "Removing excess columns in {}".format(self.table1_name))
        for item in self.table1.columns:
            if item not in self.columns_to_compare:
                drop_list1.append(item)
        self.table1.drop(drop_list1, axis=1, inplace=True)

        self.logger.debug(
            "Removing excess columns in {}".format(self.table2_name))
        for item in self.table2.columns:
            if item not in self.columns_to_compare:
                drop_list2.append(item)
        self.table2.drop(drop_list2, axis=1, inplace=True)

        # identify any columns that we want to compare do not appear in the data table
        self.logger.debug(
            "Adding any columns that are missing from each table that we want to compare")
        for column in self.columns_to_compare:
            if column not in self.table1.columns:
                # add the column to the data table
                self.table1[column] = np.nan
            if column not in self.table2.columns:
                # add the column to the data table
                self.table2[column] = np.nan

        # reorder the columns to have matching order
        self.logger.debug(
            "Reordering the tables to have matching column order")
        self.table2 = self.table2[self.table1.columns]

        # create filtered tables for output
        self.logger.debug("Writing out the filtered tables to tsv files")
        self.table1.to_csv("filtered_" + self.table1_name,
                           sep="\t", index=False)
        self.table2.to_csv("filtered_" + self.table2_name,
                           sep="\t", index=False)

    def count_populated_cells(self):
        """
        This function counts the number of populated cells in each table and adds those counts to a summary table
        """
        self.logger.debug(
            "Counting the number of populated cells in each table")
        table1_populated_rows = pd.DataFrame(self.table1.count(), columns=[
                                             "Number of samples populated in {}".format(self.table1_name)])
        table2_populated_rows = pd.DataFrame(self.table2.count(), columns=[
                                             "Number of samples populated in {}".format(self.table2_name)])

        # remove the sample name rows from the summary_output table (should be identical, no point checking here)
        self.logger.debug(
            "Removing the sample name row from the tables so we don't compare them")
        table1_populated_rows.drop("samples", axis=0, inplace=True)
        table2_populated_rows.drop("samples", axis=0, inplace=True)

        # create a summary table
        self.logger.debug(
            "Creating the summary table with the number of populated cells")
        self.summary_output = pd.concat(
            [table1_populated_rows, table2_populated_rows], join="outer", axis=1)

    def determine_file_columns(self):
        """
        Determine the columns with GCP URIs so that they are excluded from regular
        comparisons and instead file comparisons are performed.
        """
        for df in [self.table1, self.table2]:
            # select columns with at least one GCP URI among nulls
            file_columns = df.columns[(df.apply(lambda x: x.astype(str).str.startswith("gs://")
                                                | x.isnull()).all())
                                      & (~df.isnull().all())]

            file_columns = file_columns.tolist()
            self.file_columns.update(file_columns)

        # Ensure file_columns set only has GCP URIs and nulls
        for df in [self.table1, self.table2]:
            remove_columns = df.columns[~(df.apply(lambda x: x.astype(str).str.startswith('gs://')
                                                   | x.isnull()).all())]

            # Convert the Index object to a set
            remove_columns = set(remove_columns.tolist())
            self.file_columns = self.file_columns - remove_columns

    def perform_exact_match(self):
        """
        This function performs an exact match and creates an Excel file 
        that contains the exact match differences
        """
        self.logger.debug("Performing an exact match and removing the sample name column")

        if self.file_columns:
            # exclude file_columns for string comparison
            table1 = self.table1.drop(list(self.file_columns), axis=1)
            table2 = self.table2.drop(list(self.file_columns), axis=1)

            # handle file comparisons separately from strings
            files_df1 = self.table1.set_index("samples")
            files_df2 = self.table2.set_index("samples")
            files_df1 = files_df1[list(self.file_columns)]
            files_df2 = files_df2[list(self.file_columns)]
            self.compare_files(files_df1, files_df2)
            self.set_file_number_of_differences()
        else:
            table1 = self.table1
            table2 = self.table2

        # count the number of differences using exact string matches
        # temporarily make NaNs null since NaN != NaN for the pd.DataFrame.eq() function
        number_of_differences = pd.DataFrame((~table1.fillna("NULL").astype(str).eq(
            table2.fillna("NULL").astype(str))).sum(), columns=[self.NUM_DIFFERENCES_COL])
        
        # remove the samplename row
        number_of_differences.drop("samples", axis=0, inplace=True)

        # add the number of differences to the summary output table
        self.logger.debug("Adding the number of exact match differences to the summary table")
        self.summary_output = pd.concat(
            [self.summary_output, number_of_differences], join="outer", axis=1)

        # include any file differences
        if self.file_number_of_differences is not None:
            self.summary_output = self.summary_output.combine_first(self.file_number_of_differences)
        self.summary_output[self.NUM_DIFFERENCES_COL] = self.summary_output[self.NUM_DIFFERENCES_COL].astype(int)

        # ensure number of differences column is the last column (for PDF report appearance)
        self.summary_output[self.NUM_DIFFERENCES_COL] = self.summary_output.pop(self.NUM_DIFFERENCES_COL)

        # get a table of self-other differences
        # also: temporarily drop the sample name column for comparison and then set it as the index for the output data frame
        self.logger.debug("Creating a table of self-other differences")
        exact_differences_table = table1.drop("samples", axis=1).compare(
            table2.drop("samples", axis=1), keep_shape=True).set_index(table1["samples"])
        # rename the self and other with the table names
        self.logger.debug("Renaming the self and other to be the table names")
        exact_differences_table.rename(
            columns={"self": self.table1_name, "other": self.table2_name}, 
            level=-1, inplace=True)

        # add file exact differences
        exact_differences_table = pd.concat(
            [exact_differences_table, self.file_exact_differences], axis=1)

        # replace matching values (NAs) with blanks
        self.logger.debug("Replacing all NA values with blanks")
        exact_differences_table.replace(np.nan, "", inplace=True)

        self.logger.debug("Writing the self-other differences table to a TSV file")
        exact_differences_table.to_csv(
            self.output_prefix + "_exact_differences.tsv", sep="\t", index=True)

    def compare_files(self, file_df1, file_df2):
        """
        Determine which pairs of files referenced in the DataFrames are identical
        """
        self.file_exact_matches = pd.DataFrame(index=file_df1.index,
                                               columns=file_df1.columns)

        # create similar table to one generated by df1.compare(df2)
        # for adding to the exact differences TSV
        self.file_exact_differences = pd.DataFrame(
            index=file_df1.index,
            columns=pd.MultiIndex.from_product(
                [file_df1.columns, [self.table1_name, self.table2_name]])
        )

        for col in file_df1.columns:
            for row in file_df1.index:
                uri1 = file_df1.loc[row, col]
                uri2 = file_df2.loc[row, col]
                if pd.isnull(uri1) and pd.isnull(uri2):
                    # count two nulls as matching
                    self.file_exact_matches.loc[row, col] = True
                elif (not pd.isnull(uri1) and not pd.isnull(uri2)):
                    file1 = os.path.join(
                        self.table1_files_dir, uri1.removeprefix("gs://"))
                    file2 = os.path.join(
                        self.table2_files_dir, uri2.removeprefix("gs://"))
                    is_match = filecmp.cmp(file1, file2, shallow=False)
                    self.file_exact_matches.loc[row, col] = is_match
                    if is_match:
                        # don't add URIs to exact differences table if files match
                        self.file_exact_differences.loc[row,
                                                        (col, self.table1_name)] = np.nan
                        self.file_exact_differences.loc[row,
                                                        (col, self.table2_name)] = np.nan
                        continue
                    else:
                        output_filename = f"{row}_{col}_diff.txt"
                        output_path = os.path.join(
                            self.diff_dir, output_filename)
                        self.create_diff(file1, file2, output_path)
                else:
                    # count as not matching if pair is missing
                    self.file_exact_matches.loc[row, col] = False

                self.file_exact_differences.loc[row,
                                                (col, self.table1_name)] = uri1
                self.file_exact_differences.loc[row,
                                                (col, self.table2_name)] = uri2

        self.file_exact_matches = self.file_exact_matches.astype(bool)

    def set_file_number_of_differences(self):
        self.file_number_of_differences = pd.DataFrame(
            columns=[self.NUM_DIFFERENCES_COL])
        for col in self.file_exact_matches.columns:
            count = self.file_exact_matches[col].dropna().ne(True).sum()
            self.file_number_of_differences.loc[col] = count

    def create_diff(self, file1, file2, output_path):
        # create unified diff
        with open(file1, "r") as f1, open(file2, "r") as f2:
            diff = difflib.unified_diff(
                f1.readlines(),
                f2.readlines(),
                fromfile=file1,
                tofile=file2,
                lineterm='',
            )
            diff = "".join(diff)

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as out:
                out.write(diff)

    def calculate_percent_difference(self, x1, x2):
        """Does the math part of percent difference

        Args:
            x1 (pd.Series): the series of numbers from a cell from table1
            x2 (pd.Series): the series of numbers from a cell from table2
        
        Returns:
            Array: an array with the percent difference calculations for the two cells
        """
        diffs = []
        if len(x1) == len(x2):
            for x, y in zip(x1, x2):
                if x != 0 and y != 0:
                    diffs.append(np.abs(x - y) / np.mean([x, y]))
                else:
                    diffs.append(0)
            return diffs
        else:
            return [np.nan]
        
    def percent_difference(self, value1, value2, delimiter, threshold):
        """This function calculates the percent difference between two values
        
        Args:
            value1 (pd.Series): the value from the first table
            value2 (pd.Series): the value from the second table
            delimiter (String): delimiters to split the values on if necessary
            threshold (Float): the threshold the percent differences must be under 
            
        Returns:
            pd.Series: Boolean array indicate if the percent differences are under the threshold
        """
        self.logger.debug("PERCENT_DIFFERENCE:In the case where value1 and value2 can be pd.Series of lists, we will split those on the delimiter ({})".format(delimiter))
        
        # thank you my lord and savior chatgpt
        self.logger.debug("PERCENT_DIFFERENCE:We will now convert the content of those lists into numeric values safely because math and strings don't mix")
        value1_numbers = value1.apply(lambda x: [to_numeric_safe(num) for num in re.split(delimiter, str(x)) if to_numeric_safe(num) != num])
        value2_numbers = value2.apply(lambda x: [to_numeric_safe(num) for num in re.split(delimiter, str(x)) if to_numeric_safe(num) != num])

        self.logger.debug("PERCENT_DIFFERENCE:The content of value1_numbers is as follows:\n{}".format(value1_numbers))
        self.logger.debug("PERCENT_DIFFERENCE:The content of value2_numbers is as follows:\n{}".format(value2_numbers))

        self.logger.debug("PERCENT_DIFFERENCE:The number of items in each list within value1 and value2 could differ")
        self.logger.debug("PERCENT_DIFFERENCE:This is undesired behavior and so we will calculate a Boolean pd.Series to confirm")
        equal_lengths = value1_numbers.apply(len) == value2_numbers.apply(len)
        self.logger.debug("PERCENT_DIFFERENCE:This array indicates if the length of the items between each row in the tables is equal:\n{}".format(equal_lengths))

        self.logger.debug("PERCENT_DIFFERENCE:Now calculating the percent differences between {} in table1 and table2".format(value1.name))
       
        math_differences = value1_numbers.combine(value2_numbers, lambda x1, x2: (self.calculate_percent_difference(x1, x2)))
        self.logger.debug("PERCENT_DIFFERENCE:The following is a pd.Series with the results of the mathmatical comparison [ |x-y|/((x+y)/2) ]:\n{}".format(math_differences))
       
        acceptable_differences = math_differences.apply(lambda x: all(diff <= threshold if not np.isnan(diff) else False for diff in x ))
        self.logger.debug("PERCENT_DIFFERENCE:The following is a Boolean pd.Series of when these are within the threshold ({}) OR are nan:\n{}".format(threshold, acceptable_differences))

        self.logger.debug("PERCENT_DIFFERENCE:This is the sum of the equal_lengths and acceptable_differences pd.Series:\n{}".format(equal_lengths & acceptable_differences))
        return equal_lengths & acceptable_differences

    def range_difference(self, value1, value2, delimiter, threshold):
        """This function calculates the numerical difference between two values

        Args:
            value1 (pd.Series): The value from the first table
            value2 (pd.Series): The value from the second table
            delimiter (String): delimiters to split the values on if necessary
            threshold (Float): the threshold the absolute difference must be under 
        
        Returns:
            pd.Series: Boolean array indicate if the absolute differences are under the threshold
        """
        
        self.logger.debug("RANGE_DIFFERENCE:In the case where value1 and value2 can be pd.Series of lists, we will split those on the delimiter ({})".format(delimiter))
        
        self.logger.debug("RANGE_DIFFERENCE:We will now convert the content of those lists into numeric values safely because math and strings don't mix")
        value1_numbers = value1.apply(lambda x: [to_numeric_safe(num) for num in re.split(delimiter, str(x)) if to_numeric_safe(num) != num and ~np.isnan(to_numeric_safe(num))])
        value2_numbers = value2.apply(lambda x: [to_numeric_safe(num) for num in re.split(delimiter, str(x)) if to_numeric_safe(num) != num and ~np.isnan(to_numeric_safe(num))])
    
        self.logger.debug("RANGE_DIFFERENCE:The content of value1_numbers is as follows:\n{}".format(value1_numbers))
        self.logger.debug("RANGE_DIFFERENCE:The content of value2_numbers is as follows:\n{}".format(value2_numbers))

        self.logger.debug("RANGE_DIFFERENCE:The number of items in each list within value1 and value2 could differ")
        self.logger.debug("RANGE_DIFFERENCE:This is undesired behavior and so we will calculate a Boolean pd.Series to confirm")
        equal_lengths = value1_numbers.apply(len) == value2_numbers.apply(len)
        self.logger.debug("RANGE_DIFFERENCE:This array indicates if the length of the items between each row in the tables is equal:\n{}".format(equal_lengths))

        self.logger.debug("RANGE_DIFFERENCE:Now calculating the percent differences between {} in table1 and table2".format(value1.name))
        
        math_differences = value1_numbers.combine(value2_numbers, lambda x1, x2: np.absolute(np.array(x2) - np.array(x1) if len(x1) == len(x2) else [np.nan])).dropna()
        self.logger.debug("PERCENT_DIFFERENCE:The following is a pd.Series with the results of the mathmatical comparison [ |x-y| ]:\n{}".format(math_differences))      

        acceptable_differences = math_differences.apply(lambda x: all(diff <= threshold if not np.isnan(diff) else False for diff in x ))
        self.logger.debug("PERCENT_DIFFERENCE:The following is a Boolean pd.Series of when these are within the threshold ({}):\n{}".format(threshold, acceptable_differences))

        self.logger.debug("PERCENT_DIFFERENCE:This is the sum of the equal_lengths and acceptable_differences pd.Series:\n{}".format(equal_lengths & acceptable_differences))
        return equal_lengths & acceptable_differences

    def convert_dtype(self, val):
        """convert variable dtypes appropriately

        Args:
            val (unknown dtype): the value to appropriately type

        Returns:
            unknown dtype: the appropriately typed value
        """
        if isinstance(val, str) and val.replace(".", "", 1).isdigit():
            return int(val) if "." not in val else float(val)
        return val

    def validate(self, column, top_level=True):
        """This function checks column content to see if it meets user-defined validation criteria
        """
        self.logger.debug("VALIDATE:We are **not** recursing:\n\t{}".format(top_level))
        try:
            if pd.isnull(column.iloc[1]):
                delimiter = "[,]" # formatted for re.split()
                self.logger.debug("VALIDATE:No delimiter was provided; defaulting the delimiter to '{}'".format(delimiter))
            else:
                delimiter = "[{}]".format(re.escape(column.iloc[1]))
                self.logger.debug("VALIDATE:A delimiter was provided; setting the delimiter to '{}'".format(delimiter))

        except:
            delimiter = "[,]" # formatted for re.split()
            self.logger.debug("VALIDATE:No delimiter column was provided; defaulting the delimiter to '{}'".format(delimiter))
            
        validation_criteria = column.iloc[0]
        self.logger.debug("VALIDATE:The validation criteria is {}; Making sure that it is the appropriate dtype".format(validation_criteria))
        
        self.logger.debug("VALIDATE:Before conversion, the criteria type is: {}".format(type(validation_criteria)))
        validation_criteria = self.convert_dtype(validation_criteria)
        self.logger.debug("VALIDATE:After conversion, the criteria type is: {}".format(type(validation_criteria)))

        if column.name in self.table1.columns:
            self.logger.debug("VALIDATE:The name of the column ({}) is found in table1".format(column.name))
            # check the data type of the validation criteria; based on its type, we can assume the comparison to perform
            if column.name in self.file_columns:
                self.logger.debug("VALIDATE:The column contains files; performing file validation")
                
                # handle file validation separately from strings, floats
                formatted_validation_criteria, number_of_differences = self.validate_files(column)
                
                self.logger.debug("VALIDATE:File comparison performed: {}, {}".format(formatted_validation_criteria, number_of_differences))
                return (formatted_validation_criteria, number_of_differences)
            elif isinstance(validation_criteria, str):  # if a string value is in column[0]
                self.logger.debug("VALIDATE:The criteria ({}) was identified as a String; now determining what content it holds".format(validation_criteria))
                
                if validation_criteria == "EXACT":  # count the number of exact match failures/differences
                    self.logger.debug("VALIDATE:The criteria was identified as 'EXACT'")
                    self.logger.info("Performing an EXACT match on column {} and counting the number of differences".format(column.name))
                    
                    self.logger.debug("VALIDATE:Extracting column of interest and filling NA values with NULL since NA != NA")
                    table1_na_are_null = self.table1[column.name].fillna("NULL")
                    table2_na_are_null = self.table2[column.name].fillna("NULL")
                    self.logger.debug("VALIDATE:{} with NULLs instead of NA in table1:\n{}".format(column.name, table1_na_are_null))
                    self.logger.debug("VALIDATE:{} with NULLs instead of NA in table2:\n{}".format(column.name, table2_na_are_null))
                    
                    self.logger.debug("VALIDATE:Creating Boolean pd.Series that has True when values between table1 and table2 are equal")
                    exact_matches = table1_na_are_null.eq(table2_na_are_null)
                    self.logger.debug("VALIDATE:This is the boolean array of exact matches between these two tables:\n{}".format(exact_matches))

                    self.logger.debug("VALIDATE:Now counting the number of 'FALSE' items in the array")
                    number_of_differences = (~exact_matches).values.sum()
                    self.logger.debug("VALIDATE:There are {} 'FALSE' values in the exact_matches pd.Series".format(number_of_differences))
                    
                     # only write out columns that fail validation criteria
                    if number_of_differences > 0:
                        self.logger.debug("VALIDATE:Since differences were identified, we will write out these columns to the output table")
                        self.validation_table[(column.name, self.table1_name)] = self.table1[column.name].where(~exact_matches)
                        self.validation_table[(column.name, self.table2_name)] = self.table2[column.name].where(~exact_matches)

                    self.logger.debug("VALIDATE:Returning 'EXACT' and the number of differences ({})".format(number_of_differences))
                    return ("EXACT", number_of_differences)
                # do not check; there are no failures (0)
                elif validation_criteria == "IGNORE":
                    self.logger.debug("VALIDATE:The criteria was identified as 'IGNORE'")
                    self.logger.info("Ignoring column {} and indicating 0 failures".format(column.name))
                    return ("IGNORE", 0)
                elif validation_criteria == "SET":  # check list items for identical content
                    self.logger.debug("VALIDATE:The criteria was identified as 'SET'")
                    self.logger.info("Performing a set comparison on column {} and counting the number of differences".format(column.name))
                    
                    self.logger.debug("VALIDATE:Turning the columns into sets and filling NA values with NULL")
                    table1_set = self.table1[column.name].fillna("NULL").apply(lambda x: set(re.split(delimiter, x)))
                    table2_set = self.table2[column.name].fillna("NULL").apply(lambda x: set(re.split(delimiter, x)))
                    
                    self.logger.debug("VALIDATE:{} as a set for table1:\n{}".format(column.name, table1_set))
                    self.logger.debug("VALIDATE:{} as a set for table2:\n{}".format(column.name, table2_set))
                    
                    self.logger.debug("VALIDATE:Creating Boolean pd.Series that has True when the sets between table1 and table2 are equal")
                    set_matches = (table1_set.eq(table2_set))
                    self.logger.debug("VALIDATE:This is the boolean array of set matches between these two tables:\n{}".format(set_matches))

                    self.logger.debug("VALIDATE:Now counting the number of 'FALSE' items in the array")
                    number_of_differences = (~set_matches).values.sum()
                    self.logger.debug("VALIDATE:There are {} 'FALSE' values in the exact_matches pd.Series".format(number_of_differences))
                    
                     # only write out columns that fail validation criteria
                    if number_of_differences > 0:
                        self.logger.debug("VALIDATE:Since differences were identified, we will write out these columns to the output table")
                        self.logger.debug("VALIDATE:For sets, we will only output what is different between the sets")
                        set_differences = table1_set.combine(table2_set, lambda x, y: (x - y, y - x))
                                            
                        self.validation_table[(column.name, self.table1_name)] = set_differences.apply(lambda x: ', '.join(sorted(x[0])) if x else '')
                        self.validation_table[(column.name, self.table2_name)] = set_differences.apply(lambda x: ', '.join(sorted(x[1])) if x else '')

                    self.logger.debug("VALIDATE:Returning 'SET' and the number of differences ({})".format(number_of_differences))
                    return ("SET", number_of_differences)
                elif "," in validation_criteria:
                    self.logger.debug("VALIDATE:The criteria was identified as COMPOUND: '{}'".format(validation_criteria))
                    self.logger.info("A comma was identified within the criteria ({}); the criteria will be checked left to right for {}".format(validation_criteria, column.name))
                    
                    self.logger.debug("VALIDATE:Splitting the criteria on the comma")
                    criteria_options = validation_criteria.split(",")
                    self.logger.debug("VALIDATE:The split criteria is as follows: {}".format(criteria_options))
                    
                    # a temporary "differences" series here that we'll update as we recurse
                    # defauling all to "true" (i.e., all rows pass validation)
                    self.logger.debug("VALIDATE:Creating temporary 'matches' series here that will be updated as recursion happens")
                    failing_rows = pd.Series(True, index=self.table1.index)
                    self.logger.debug("VALIDATE:This is the INITIAL failing_rows pd.Series (all fail):\n{}".format(failing_rows))
                    
                    self.logger.debug("VALIDATE:Now iterating through the {} items in criteria_options".format(len(criteria_options)))
                    for criteria in criteria_options:
                        self.logger.debug("VALIDATE:The criteria is: {}".format(criteria))
                        
                        # duplicate the column value and rewrite the criteria to pass recurisvely in
                        self.logger.debug("VALIDATE:Duplicating the `column` input variable that contains any delimiters and the column.name")
                        temp_column = column.copy()
                        self.logger.debug("VALIDATE:Replacing the full criteria with a singular item in the `criteria_options` list")
                        temp_column.iloc[0] = criteria
                        self.logger.debug("VALIDATE:Here is the temporary column that will be used to recurse:\n{}".format(temp_column))
                        
                        self.logger.debug("Passing in the temporary column and `top_level=FALSE` to validate() since it is a recursive entry")
                        self.logger.debug("VALIDATE:\t***ENTERING RECURSION***")
                        validation_text, num_failures = self.validate(temp_column, False)
                        self.logger.debug("VALIDATE:\t***EXITING RECURSION***")
                       
                        self.logger.debug("VALIDATE:The output from the recursion is as follows:\n{}\n{}".format(validation_text, num_failures))
                        
                        if (num_failures) == 0:
                            self.logger.debug("VALIDATE:The number of failures is 0! All elements passed validation! No need to continue this loop.")
                            
                            self.logger.debug("VALIDATE:This is the current content of failing_rows:\n{}".format(failing_rows))
                            self.logger.debug("VALIDATE:Setting the failing_rows pd.Series to be all True since all passed")
                            failing_rows = pd.Series(True, index=self.table1.index)
                            self.logger.debug("VALIDATE:The current value of failing_rows is now:\n{}".format(failing_rows))

                            break
                        else:
                            self.logger.debug("VALIDATE:Some ({}) elements failed this round of validation".format(num_failures))
                            self.logger.debug("VALIDATE:Now identifying which rows did not pass this round of validation.")

                            self.logger.debug("VALIDATE:Rows that passed this round of validation will be blank AT LEAST ONE of the columns the output table.")                            
                            self.logger.debug("VALIDATE:Here is the current state of failing_rows before updating:\n{}".format(failing_rows))
                           
                            failing_rows &= (self.validation_table[(column.name, self.table1_name)].replace(r'^\s*$', np.nan, regex=True).notna() 
                                            | self.validation_table[(column.name, self.table2_name)].replace(r'^\s*$', np.nan, regex=True).notna())
                            self.logger.debug("VALIDATE:Here is the current state of failing_rows AFTER updating:\n{}".format(failing_rows))                                                      
                            
                    number_of_differences = (failing_rows).values.sum()
                    self.logger.debug("VALIDATE:There are {} differences".format(number_of_differences))

                    # correct the output table with final results
                    # only write out columns that fail validation criteria
                    if number_of_differences > 0 and top_level == True:
                        self.logger.debug("VALIDATE: Some rows failed both rounds of validation, so we will write them to the table")
                        self.validation_table[(column.name, self.table1_name)] = self.table1[column.name].where(failing_rows)
                        self.validation_table[(column.name, self.table2_name)] = self.table2[column.name].where(failing_rows)

                    self.logger.debug("VALIDATE:Returning '{}' and the number of differences ({})".format(validation_criteria, number_of_differences))
                    return (validation_criteria, number_of_differences)
                else:
                    self.logger.info("String value ({}) not yet recognized".format(validation_criteria))
                    return ("String value not yet recognized", np.nan)
            elif isinstance(validation_criteria, float) is True:  # if a float
                self.logger.debug("VALIDATE:The criteria ({}) was identified as a Float; now performing a PERCENT_DIFF".format(validation_criteria))
                self.logger.info("Performing a percent difference comparison on column {} and counting the number of differences".format(column.name))
                
                self.logger.debug("VALIDATE:Passing the columns of interest, any indicated delimiters, and the validation_criteria to the percent_difference function.")
                acceptable_differences = self.percent_difference(self.table1[column.name], self.table2[column.name], delimiter, validation_criteria)
                self.logger.debug("VALIDATE:Here the result of that function:\n{}".format(acceptable_differences))
                
                self.logger.debug("VALIDATE:Now counting the number of 'FALSE' items in the acceptable_differences array")
                number_of_differences = (~acceptable_differences).values.sum()
                self.logger.debug("VALIDATE:There are {} 'FALSE' values in the array".format(number_of_differences))
                
                 # only write out columns that fail validation criteria
                if number_of_differences > 0:
                    self.logger.debug("VALIDATE:Since differences were identified, we will write out these columns to the output table")
                    self.validation_table[(column.name, self.table1_name)] = self.table1[column.name].where(~acceptable_differences)
                    self.validation_table[(column.name, self.table2_name)] = self.table2[column.name].where(~acceptable_differences)

                self.logger.debug("VALIDATE:Returning 'PERCENT_DIFF: {}' and the number of differences ({})".format(format(validation_criteria, ".2%"), number_of_differences))
                return ("PERCENT_DIFF: " + format(validation_criteria, ".2%"), number_of_differences)

            elif isinstance(validation_criteria, int):
                self.logger.debug("VALIDATE:The criteria ({}) was identified as a Int; now performing a RANGE".format(validation_criteria))
                self.logger.info("Performing a range functionality on column {} and indicating the number of differences".format(column.name))

                acceptable_differences = self.range_difference(self.table1[column.name], self.table2[column.name], delimiter, validation_criteria)
                self.logger.debug("VALIDATE:Here the result of that function:\n{}".format(acceptable_differences))
                
                self.logger.debug("VALIDATE:Now counting the number of 'FALSE' items in the acceptable_differences array")
                number_of_differences = (~acceptable_differences).values.sum()
                self.logger.debug("VALIDATE:There are {} 'FALSE' values in the array".format(number_of_differences))
                
                 # only write out columns that fail validation criteria
                if number_of_differences > 0:
                    self.logger.debug("VALIDATE:Since differences were identified, we will write out these columns to the output table")
                    self.validation_table[(column.name, self.table1_name)] = self.table1[column.name].where(~acceptable_differences)
                    self.validation_table[(column.name, self.table2_name)] = self.table2[column.name].where(~acceptable_differences)

                self.logger.debug("VALIDATE:Returning 'RANGE: {}' and the number of differences ({})".format(validation_criteria, number_of_differences))
                return ("RANGE: " + format(validation_criteria), number_of_differences)
            else:  # it's an object type, do not check
                self.logger.debug("VALIDATE:The criteria ({}) was identified as a Object; skipping this and returning np.nan".format(validation_criteria))
                self.logger.info("Ignoring column {} and indicating np.nan failures".format(column.name))
                return ("OBJECT TYPE VALUE; IGNORED FOR NOW", np.nan)
        else:
            self.logger.debug("VALIDATE:The column was not found in table1!")
            self.logger.info("Column {} was not found; indicating np.nan failures".format(column.name))
            return ("COLUMN " + column.name + " NOT FOUND", np.nan)

    def validate_files(self, column):
        """
        Perform validation of matching file contents based on which of EXACT,
        IGNORE, or SET is assigned as the column's validation criterion. For SET,
        sort lines in file before comparing.
        """
        validation_criteria = column.iloc[0]
        if validation_criteria == "EXACT":
            # we already know where the exact matches are from compare_files()
            self.validation_table[(column.name, self.table1_name)] = (self.table1
                                                                      .set_index("samples")[column.name]
                                                                      .where(~self.file_exact_matches[column.name])
                                                                      .reset_index()[column.name]
                                                                      )
            self.validation_table[(column.name, self.table2_name)] = (self.table2
                                                                      .set_index("samples")[column.name]
                                                                      .where(~self.file_exact_matches[column.name])
                                                                      .reset_index()[column.name]
                                                                      )
            number_of_differences = self.file_number_of_differences.loc[
                column.name, self.NUM_DIFFERENCES_COL]
        elif validation_criteria == "IGNORE":
            number_of_differences = 0
        elif validation_criteria == "SET":
            # for SET, sort lines in files then compare
            concat_columns = pd.concat([self.table1[column.name], self.table2[column.name]], axis=1)
            concat_columns = concat_columns.map(lambda x: x.removeprefix("gs://") if pd.notnull(x) else x)
            sorted_file_matches = concat_columns.apply(self.compare_sorted_files, axis=1)
            self.validation_table[(column.name, self.table1_name)] = (self.table1[column.name].where(~sorted_file_matches))
            self.validation_table[(column.name, self.table2_name)] = (self.table2[column.name].where(~sorted_file_matches))
            number_of_differences = len(sorted_file_matches) - sorted_file_matches.sum()
        else:
            raise Exception("Only EXACT, IGNORE, and SET validation criteria are implemented for file columns")
        return (validation_criteria, number_of_differences)

    def compare_sorted_files(self, row):
        """
        Compare two files sorted alphabetically by line for a pair of file URIs.
        """
        file1 = row.iloc[0]
        file2 = row.iloc[1]
        if pd.isnull(file1) and pd.isnull(file2):
            # count two nulls as matching
            return True
        if pd.notnull(file1) and pd.notnull(file2):
            file1 = os.path.join(self.table1_files_dir, file1)
            file2 = os.path.join(self.table2_files_dir, file2)
            with open(file1, "r") as f1, open(file2, "r") as f2:
                lines1 = f1.readlines()
                lines2 = f2.readlines()
            lines1.sort()
            lines2.sort()
            return lines1 == lines2
        # count null + not-null as mismatching
        return False

    def run_validation_checks(self):
        """
        This function creates, formats, and runs the validation criteria checks
        """
        self.validation_table = pd.DataFrame()

        self.logger.debug("Performing the validation checks")
        self.summary_output[["Validation Criteria", "Number of samples failing the validation criteria"]] = pd.DataFrame(
            self.validation_criteria.apply(lambda x: self.validate(x.astype(object)), result_type="expand")).transpose()
        # format the validation criteria differences table
        self.logger.debug("Formatting the validation criteria differences table")
        self.validation_table.set_index(self.table1["samples"], inplace=True)
        self.validation_table.rename_axis(None, axis="index", inplace=True)

        self.validation_table.columns = pd.MultiIndex.from_tuples(
            self.validation_table.columns, names=["Column", "Table"])

        self.logger.debug("Writing the validation criteria differences table out to a TSV file")
        self.validation_table.to_csv(
            self.output_prefix + "_validation_criteria_differences.tsv", sep="\t")

    def make_pdf_report(self):
        """This function turns the summary DataFrame into a pdf report
        """
        self.logger.debug("Turning the summary dataframe into a HTML report")
        pd.set_option('display.max_colwidth', None)

        # make pretty html table
        html_table_light_grey = build_table(self.summary_output,
                                            'grey_light',
                                            index=True,
                                            text_align='center',
                                            # conditions={
                                            #   'Number of differences (exact match)': {
                                            #     'min': 1,
                                            #     'max': 0,
                                            #     'min_color': 'black',
                                            #     'max_color': 'red'
                                            #   }
                                            # }
                                            )

        # save to html file
        with open(self.output_prefix + "_summary.html", 'w') as outfile:
            outfile.write("<p>Validation analysis performed on " +
                          str(date.today().isoformat()) + ".</p>")
            outfile.write(html_table_light_grey)

            if (self.validation_criteria is not None):
                outfile.write("<p>Validation Criteria:</p>")
                outfile.write("<ul>")
                outfile.write(" <dt>EXACT</dt>")
                outfile.write(" <dd>Performs an exact string match</dd>")
                outfile.write(" <dt>IGNORE</dt>")
                outfile.write(
                    " <dd>Ignores the values; indicates 0 failures</dd>")
                outfile.write(" <dt>SET</dt>")
                outfile.write(
                    " <dd>Compares items in a list without regard to order</dd>")
                outfile.write(" <dt>PERCENT_DIFF</dt>")
                outfile.write(
                    " <dd>Tests if two values are more than the indicated percent difference (must be in decimal format)</dd>")
                outfile.write(" <dt>RANGE</dt>")
                outfile.write(
                    " <dd>Tests if two values are more than +- the indicated number (must be in integer format)</dd>")
                outfile.write(" <dt>COMBINED</dt>")
                outfile.write(
                    " <dd>Tests both options and returns a pass for that row if at least one criteria is met</dd>")      
                outfile.write("</ul>")

        # convert to pdf
        options = {
            'page-size': 'Letter',
            'title': self.table1_name + ' vs. ' + self.table2_name,
            'margin-top': '0.25in',
            'margin-right': '0.25in',
            'margin-bottom': '0.25in',
            'margin-left': '0.25in'
        }

        self.logger.debug("Converting the summary html file to a PDF")
        pdf.from_file(self.output_prefix + "_summary.html",
                      self.output_prefix + "_summary.pdf", options=options)

    def compare(self):
        """This function orchestrates all others
        """
        self.logger.info("Comparing %s and %s" %
                         (self.table1_name, self.table2_name))

        self.logger.info("Converting the tables into Pandas dataframes")
        self.table1 = self.convert_table_to_dataframe(self.table1_name)
        self.table2 = self.convert_table_to_dataframe(self.table2_name)

        self.logger.debug("Grabbing the basename of the input files")
        self.table1_name = os.path.basename(self.table1_name)
        self.table2_name = os.path.basename(self.table2_name)

        self.logger.info("Filtering the tables to only include the columns to compare")
        self.create_filtered_table()

        self.logger.info("Counting how many cells have values")
        self.count_populated_cells()

        self.logger.info("Determining if any columns need file comparisons")
        self.determine_file_columns()

        if self.file_columns:
            dir1 = f"{self.table1_files_dir}/"
            dir2 = f"{self.table2_files_dir}/"
            os.mkdir(dir1)
            os.mkdir(dir2)

            self.logger.info("Localizing files to compare...")
            self.table1[list(self.file_columns)].apply(
                localize_files, directory=dir1)
            self.table2[list(self.file_columns)].apply(
                localize_files, directory=dir2)

        self.logger.info("Performing an exact string match")
        self.perform_exact_match()

        if (self.validation_criteria is not None):
            self.logger.info("Performing validation criteria checks")
            self.run_validation_checks()

        self.logger.info("Creating a PDF report")
        self.make_pdf_report()

        self.logger.info("Done!")


def localize_files(row, directory):
    """
    Download files to compare from GCP.
    """
    for value in row:
        if isinstance(value, str) and value.startswith("gs://"):
            # it would be much faster to copy files all at once, but any files with
            # the same name would be clobbered, so create local directories matching
            # gsutil path and loop to copy
            remote_path = os.path.dirname(value.removeprefix("gs://"))
            destination_path = os.path.join(directory, remote_path)
            os.makedirs(destination_path, exist_ok=True)
            subprocess.run(["gsutil", "-m", "cp", value, destination_path])
        return value
