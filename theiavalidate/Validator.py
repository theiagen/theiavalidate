from datetime import date
from pretty_html_table import build_table
import logging
import numpy as np
import os
import pandas as pd
import pdfkit as pdf
import subprocess
import sys

class Validator:
  """
  This class runs the parsing module for theiavalidate
  """
  def __init__(self, options):
    logging.basicConfig(encoding='utf-8', level=logging.ERROR, stream=sys.stderr)
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
    
    self.logger.debug("The following options were passed to the Validator class: {}".format(options))
    
    self.table1_name = options.table1
    self.table2_name = options.table2
      
    self.column_translation = options.column_translation
    self.validation_criteria = options.validation_criteria
    self.columns_to_compare = options.columns_to_compare
    self.columns_to_compare.append("samples")
    
    self.output_prefix = options.output_prefix
    self.na_values = options.na_values
      
    if (self.column_translation is not None):
      self.logger.debug("Creating a dictionary from the column translation file named {}".format(self.column_translation))
      self.column_translation = pd.read_csv(self.column_translation, sep="\t", header=None, index_col=0).squeeze().to_dict()
      self.logger.debug("Column translation: {}".format(self.column_translation))
    if (self.validation_criteria is not None):
      self.logger.debug("Creating a dictionary from the validation criteria file named {}".format(self.validation_criteria))
      self.validation_criteria = pd.read_csv(self.validation_criteria, sep="\t", index_col=0).transpose()
      self.validation_criteria = self.validation_criteria.apply(pd.to_numeric, errors="ignore").convert_dtypes()
      self.logger.debug("Validation criteria: {}".format(self.validation_criteria))
    
  """
  This function converts a TSV file into a pandas dataframe, renames the headers if a dictionary 
  is provided, and replaces "None" string cells with NaNs
  """
  def convert_table_to_dataframe(self, table):
    self.logger.debug("Converting {} to a Pandas Dataframe".format(table))
    df = pd.read_csv(table, sep="\t", keep_default_na=False, header=0, index_col=False, na_values=self.na_values)
    
    # change first column to samples
    self.logger.debug("Renaming the first column to 'samples'")
    df.columns.values[0] = "samples"
    
    # rename any columns to match the second row in the column translation file
    if (self.column_translation is not None):
      self.logger.debug("Renaming columns with the following dictionary: %s" % self.column_translation)
      df.rename(columns=self.column_translation, inplace=True)
    
    # replace "None" string cells with NaNs
    self.logger.debug("Replacing 'None' strings with `np.nan`s")
    df.replace("None", np.nan, inplace=True)
    
    self.logger.debug("Done converting table to dataframe!")
    return df
  
  """ 
  This function filters out the columns that are not in the list of columns to compare 
  and writes out the resulting table
  """
  def create_filtered_table(self):
    self.logger.debug("Removing columns that are not in the list of columns to compare")
    
    drop_list1 = []
    drop_list2 = []
    
    self.logger.debug("Removing excess columns in {}".format(self.table1_name))
    for item in self.table1.columns:
      if item not in self.columns_to_compare:
        drop_list1.append(item)
    self.table1.drop(drop_list1, axis=1, inplace=True)
    
    self.logger.debug("Removing excess columns in {}".format(self.table2_name))
    for item in self.table2.columns:
      if item not in self.columns_to_compare:
        drop_list2.append(item)
    self.table2.drop(drop_list2, axis=1, inplace=True)
    
    # identify any columns that we want to compare do not appear in the data table
    self.logger.debug("Adding any columns that are missing from each table that we want to compare")
    for column in self.columns_to_compare:
      if column not in self.table1.columns:
        # add the column to the data table
        self.table1[column] = np.nan
      if column not in self.table2.columns:
        # add the column to the data table
        self.table2[column] = np.nan
  
    # reorder the columns to have matching order
    self.logger.debug("Reordering the tables to have matching column order")
    self.table2 = self.table2[self.table1.columns]

    # create filtered tables for output
    self.logger.debug("Writing out the filtered tables to tsv files")
    self.table1.to_csv("filtered_" + self.table1_name, sep="\t", index=False)
    self.table2.to_csv("filtered_" + self.table2_name, sep="\t", index=False)    
     
  """
  This function counts the number of populated cells in each table and adds those counts to a summary table
  """  
  def count_populated_cells(self):
    self.logger.debug("Counting the number of populated cells in each table")
    table1_populated_rows = pd.DataFrame(self.table1.count(), columns=["Number of samples populated in {}".format(self.table1_name)])
    table2_populated_rows = pd.DataFrame(self.table2.count(), columns=["Number of samples populated in {}".format(self.table2_name)])
        
    # remove the sample name rows from the summary_output table (should be identical, no point checking here)
    self.logger.debug("Removing the sample name row from the tables so we don't compare them")
    table1_populated_rows.drop("samples", axis=0, inplace=True)
    table2_populated_rows.drop("samples", axis=0, inplace=True)

    # create a summary table
    self.logger.debug("Creating the summary table with the number of populated cells")
    self.summary_output = pd.concat([table1_populated_rows, table2_populated_rows], join="outer", axis=1)
  
  """
  This function performs an exact match and creates and Excel file that contains the exact match differences
  """
  def perform_exact_match(self):
    self.logger.debug("Performing an exact match and removing the sample name column")
    # count the number of differences using exact string matches
    # temporarily make NaNs null since NaN != NaN for the pd.DataFrame.eq() function
    # also: remove the samplename row
    number_of_differences = pd.DataFrame((~self.table1.fillna("NULL").astype(str).eq(self.table2.fillna("NULL").astype(str))).sum(), columns = ["Number of differences (exact match)"])
    number_of_differences.drop("samples", axis=0, inplace=True)
    
    # add the number of differences to the summary output table
    self.logger.debug("Adding the number of exact match differences to the summary table")
    self.summary_output = pd.concat([self.summary_output, number_of_differences], join="outer", axis=1)
    
    # get a table of self-other differences
    # also: temporarily drop the sample name column for comparison and then set it as the index for the output data frame
    self.logger.debug("Creating a table of self-other differences")
    exact_differences_table = self.table1.drop("samples", axis=1).compare(self.table2.drop("samples", axis=1), keep_shape=True).set_index(self.table1["samples"])
    # rename the self and other with the table names
    self.logger.debug("Renaming the self and other to be the table names")
    exact_differences_table.rename(columns={"self": self.table1_name, "other": self.table2_name}, level=-1, inplace=True)
    # replace matching values (NAs) with blanks
    self.logger.debug("Replacing all NA values with blanks")
    exact_differences_table.replace(np.nan, "", inplace=True)
    
    self.logger.debug("Writing the self-other differences table to a TSV file")
    exact_differences_table.to_csv(self.output_prefix + "_exact_differences.tsv", sep="\t", index=True)

  """
  This function calculates the percent difference between two values
  """
  def percent_difference(self, value1, value2):
    value1 = value1.apply(pd.to_numeric)
    value2 = value2.apply(pd.to_numeric)
    self.logger.debug("Calculating the percent difference between {}s".format(value1.name))
    # |x-y|/((x+y)/2)
    difference = np.absolute(value2.sub(value1)).div(value2.add(value1)/2)
    return difference

  """ 
  This function checks column content to see if it meets user-defined validation criteria
  """
  def validate(self, column):
    if column.name in self.table1.columns:
      # check the data type of the validation criteria; based on its type, we can assume the comparison to perform
      if pd.api.types.is_string_dtype(column) == True: # if a string
        if column[0] == "EXACT": # count the number of exact match failures/differences
          self.logger.debug("Performing an exact match on column {} and counting the number of differences".format(column.name))
          exact_matches = ~self.table1[column.name].fillna("NULL").eq(self.table2[column.name].fillna("NULL"))

          self.validation_table[(column.name, self.table1_name)] = self.table1[column.name].where(exact_matches)
          self.validation_table[(column.name, self.table2_name)] = self.table2[column.name].where(exact_matches)

          number_of_differences = exact_matches.sum()
          return ("EXACT", number_of_differences)
        elif column[0] == "IGNORE": # do not check; there are no failures (0)
          self.logger.debug("Ignoring column {} and indicating 0 failures".format(column.name))
          return ("IGNORE", 0)
        elif column[0] == "SET": # check list items for identical content
          self.logger.debug("Performing a set comparison on column {} and counting the number of differences".format(column.name))
          differences = (~self.table1[column.name].fillna("NULL").apply(lambda x: set(x.split(","))).eq(self.table2[column.name].fillna("NULL").apply(lambda x: set(x.split(",")))))
          
          self.validation_table[(column.name, self.table1_name)] = self.table1[column.name].where(differences)
          self.validation_table[(column.name, self.table2_name)] = self.table2[column.name].where(differences)
          
          number_of_differences = (differences).sum() 
          return ("SET", number_of_differences)
        else:
          self.logger.debug("String value ({}) not recognized".format(column[0]))
          return("String value not recognized", np.nan)
      elif pd.api.types.is_float_dtype(column) == True: # if a float
        self.logger.debug("Performing a percent difference comparison on column {} and counting the number of differences".format(column.name))
        differences = self.percent_difference(self.table1[column.name], self.table2[column.name]).gt(column[0])
              
        self.validation_table[(column.name, self.table1_name)] = self.table1[column.name].where(differences)
        self.validation_table[(column.name, self.table2_name)] = self.table2[column.name].where(differences)
      
        number_of_differences = differences.sum()
        return ("PERCENT_DIFF: " + format(column[0], ".2%"), number_of_differences)
      elif pd.api.types.is_integer_dtype(column) == True: # if an integer
        self.logger.debug("Ignoring column {} and indicating np.nan failures".format(column.name))
        return ("INTEGER; IGNORED FOR NOW", np.nan)
      else: # it's an object type, do not check
        self.logger.debug("Ignoring column {} and indicating np.nan failures".format(column.name))
        return ("OBJECT TYPE VALUE; IGNORED FOR NOW", np.nan)
    else:
      self.logger.debug("Column {} was not found; indicating np.nan failures".format(column.name))
      return ("COLUMN " + column.name + " NOT FOUND", np.nan)
  
  """ 
  This function creates, formats, and runs the validation criteria checks
  """                                                                
  def run_validation_checks(self):
      self.validation_table = pd.DataFrame()
      
      self.logger.debug("Performing the validation checks")
      self.summary_output[["Validation Criteria", "Number of samples failing the validation criteria"]] = pd.DataFrame(self.validation_criteria.apply(lambda x: self.validate(x), result_type="expand")).transpose()
      
      # format the validation criteria differences table
      self.logger.debug("Formatting the validation criteria differences table")
      self.validation_table.set_index(self.table1["samples"], inplace=True)
      self.validation_table.rename_axis(None, axis="index", inplace=True)
      self.validation_table.transpose()
      
      self.validation_table.columns = pd.MultiIndex.from_tuples(self.validation_table.columns, names=["Column", "Table"])

      self.logger.debug("Writing the validation criteria differences table out to a TSV file")
      self.validation_table.to_csv(self.output_prefix + "_validation_criteria_differences.tsv", sep="\t")

  """ 
  This function turns the summary DataFrame into a pdf report
  """
  def make_pdf_report(self):
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
      outfile.write("<p>Validation analysis performed on " + str(date.today().isoformat()) + ".</p>")
      outfile.write(html_table_light_grey)
      
      if (self.validation_criteria is not None):
        outfile.write("<p>Validation Criteria:</p>")
        outfile.write("<ul>")
        outfile.write(" <dt>EXACT</dt>")
        outfile.write(" <dd>Performs an exact string match</dd>")
        outfile.write(" <dt>IGNORE</dt>")
        outfile.write(" <dd>Ignores the values; indicates 0 failures</dd>")
        outfile.write(" <dt>SET</dt>")
        outfile.write(" <dd>Compares items in a list without regard to order</dd>")
        outfile.write(" <dt>PERCENT_DIFF</dt>")
        outfile.write(" <dd>Tests if two values are more than the indicated percent difference (must be in decimal format)</dd>")
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
    pdf.from_file(self.output_prefix + "_summary.html", self.output_prefix + "_summary.pdf", options=options)

  """ 
  This function orchestrates all others
  """
  def compare(self):
    self.logger.info("Comparing %s and %s" % (self.table1_name, self.table2_name))
    
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

    dir1 = f"table1_files/"
    dir2 = f"table2_files/"
    os.mkdir(dir1)
    os.mkdir(dir2)

    # localize files to compare
    self.table1.apply(localize_files, directory=dir1, axis=1)
    self.table2.apply(localize_files, directory=dir2, axis=1)

    subprocess.run(["ls", "-R", dir1])
    subprocess.run(["ls", "-R", dir2])
    
    self.logger.info("Performing an exact string match")
    self.perform_exact_match()

    if (self.validation_criteria is not None):
      self.logger.info("Performing validation criteria checks")
      self.run_validation_checks()
      
    self.logger.info("Creating a PDF report")
    self.make_pdf_report()
    
    self.logger.info("Done!")

def localize_files(row, directory):
  for value in row:
    if isinstance(value, str) and value.startswith("gs://"):
      # copy files to to compare_files/ directory
      # it would be much faster to copy them all at once, but any files with
      # the same name would be clobbered, so create local directories matching
      # gsutil path and loop to copy
      remote_path = os.path.dirname(value[5:])  # exclude 'gs://' prefix
      destination_path = os.path.join(directory, remote_path)
      os.makedirs(destination_path)
      subprocess.run(["gsutil", "-m", "cp", value, destination_path])
