#!/bin/bash

# to generate the files in the example-outputs, please run this command:

theiavalidate \
  example-table1.tsv \
  example-table2.tsv \
  -m example-validation_criteria.tsv \
  -c "assembly_length,gambit_predicted_taxon,amrfinderplus_amr_core_genes,extra_column,busco_results" \
  -l example-column_translation.tsv \
  --debug
