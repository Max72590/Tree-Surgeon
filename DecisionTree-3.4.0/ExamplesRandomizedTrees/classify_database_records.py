#!/usr/bin/env python

###   classify_database_records.py

###   This script demonstrates how you can carry out an evaluation of the
###   predictive power of the set of decision trees constructed by
###   RandomizedTreesForBigData.

###   Note that RandomizedTreesForBigData constructs decision trees using
###   training samples drawn randomly from the training database.  For
###   evaluation, the script shown here draws yet another random set of
###   samples from the training database and checks whether the majority
###   vote classification returned by all the decision trees agrees with
###   the true labels for the data samples used for evaluation.

###   The script shown below puts out the following outputs:
###
###   ---  It shows for each test sample the class label as calculated
###        by RandomizedTreesForBigData and the class label as present
###        in the training database.
###
###   ---  It presents the overall classification error.
###
###   ---  It presents the confusion matrix that is obtaining by
###        aggregating the calculated-class-labels versus the
###        true-class-labels

import RandomizedTreesForBigData
import sys
import string
import re
import operator
import random


interaction_needed = False
if (len(sys.argv) > 1) and (sys.argv[1] == 'with_interaction'):
    interaction_needed = True    

# Number of records extracted randomly from the training set to classify:
number_of_records_to_classify = 1000


### IMPORTANT:  The database file mentioned below is proprietary and is NOT
###             included in the module package:
training_datafile = "/home/kak/DecisionTree_data/AtRisk/AtRiskModel_File_modified.csv"

csv_class_column_index = 48

csv_columns_for_features = [41,49,50]

how_many_trees = 7


####################################### support functions #################################
def total_num_training_samples_in_file(filename):
    with open(filename) as f:
        for i, line in enumerate(f):
            pass
        f.close()
    return i    # Note that i is less by 1 relative to total number of records. But that's ok because of header.

def cleanup_csv(line):
    '''
    Introduced in Version 3.2.4, I wrote this function in response to a need to
    create a decision tree for a very large national econometric database.  The
    fields in the CSV file for this database are allowed to be double quoted and such
    fields may contain commas inside them.  This function also replaces empty fields
    with the generic string 'NA' as a shorthand for "Not Available".  IMPORTANT: This
    function skips over the first field in each record.  It is assumed that the first
    field in each record is an ID number for the record.
    '''
    line = line.translate(bytes.maketrans(b"()[]{}'", b"       ")) \
           if sys.version_info[0] == 3 else line.translate(string.maketrans("()[]{}'", "       "))
    double_quoted = re.findall(r'"[^\"]+"', line[line.find(',') : ])
    for item in double_quoted:
        clean = re.sub(r',', r'', item[1:-1].strip())
        parts = re.split(r'\s+', clean.strip())
        line = str.replace(line, item, '_'.join(parts))
    white_spaced = re.findall(r',\s*[^,]+\s+[^,]+\s*,', line)
    for item in white_spaced:
        if re.match(r',\s+,', item) : continue
        replacement = '_'.join(re.split(r'\s+', item[:-1].strip())) + ','
        line = str.replace(line, item, replacement)
    fields = re.split(r',', line)
    newfields = []
    for field in fields:
        newfield = field.strip()
        if newfield == '':
            newfields.append('NA')
        else:
            newfields.append(newfield)
    line = ','.join(newfields)
    return line

################## construct list of record indexes to classify  #########################

how_many_total_training_samples = total_num_training_samples_in_file(training_datafile)
records_to_classify = random.sample(range(1,how_many_total_training_samples+1), number_of_records_to_classify)

################################## construct randomized trees ############################

rt = RandomizedTreesForBigData.RandomizedTreesForBigData(
                                training_datafile = training_datafile,
                                csv_class_column_index = csv_class_column_index,
                                csv_columns_for_features = csv_columns_for_features,
                                entropy_threshold = 0.01,
                                max_depth_desired = 8,
                                symbolic_to_numeric_cardinality_threshold = 10,
                                how_many_trees = how_many_trees,
                                looking_for_needles_in_haystack = 1,
                              )
print("Reading the training data ...")
rt.get_training_data_for_N_trees()

##   UNCOMMENT the following statement if you want to see the training data used for each tree::
#rt.show_training_data_for_all_trees()

print("\nCalculating first order probabilities...")
rt.calculate_first_order_probabilities()

print("\nCalculating class priors...")
rt.calculate_class_priors()

print("\nConstructing ALL decision trees ....\n")
rt.construct_all_decision_trees()

##   UNCOMMENT the following statement if you want to see all decision trees individually:
rt.display_all_decision_trees()

####################  Extract test samples from the database file  ######################

records_to_classify_local = records_to_classify[:]
record_ids_with_class_labels = {}
record_ids_with_features_and_vals = {}
all_fields = []
with open(training_datafile) as f:
    for i,line in enumerate(f):
        if i == 0:
            all_fields = cleanup_csv(line).split(r',')
            continue 
        if len(records_to_classify_local) == 0:
            break
        record = cleanup_csv(line).split(r',')
        if int(record[0]) in records_to_classify_local:
            records_to_classify_local.remove(int(record[0]))
            record_ids_with_class_labels[int(record[0])] = record[csv_class_column_index]
            features_and_vals = list(map(operator.add, [all_fields[i] for i in csv_columns_for_features],
                                list(map(operator.add, '=' * len(csv_columns_for_features),  [record[i]
                                                       for i in csv_columns_for_features]))))
            record_ids_with_features_and_vals[int(record[0])] = features_and_vals
    f.close()
# Now classify all the records extracted from the database file:    
original_classifications = {}
calculated_classifications = {}
for record_index in record_ids_with_features_and_vals:
    test_sample = record_ids_with_features_and_vals[record_index]
    # Let's now get rid of those feature=value combos when value is 'NA'
    unknown_value_for_a_feature_flag = None
    for feature_and_val in test_sample:
        if feature_and_val[feature_and_val.find('=')+1:] == 'NA':
            unknown_value_for_a_feature_flag = True
            break
    if unknown_value_for_a_feature_flag:
        continue
    rt.classify_with_all_trees( test_sample )
    classification = rt.get_majority_vote_classification()
    print("\nclassification for %5d: %10s       original classification: %s" %
                (record_index, classification, record_ids_with_class_labels[record_index]))
    original_classifications[record_index] = record_ids_with_class_labels[record_index]
    calculated_classifications[record_index] = classification[classification.find('=')+1:]
total_errors = 0
confusion_matrix_row1 = [0,0]
confusion_matrix_row2 = [0,0]
for record_index in calculated_classifications:
    if original_classifications[record_index] != calculated_classifications[record_index]:
        total_errors += 1
    if original_classifications[record_index] == 'NO':
        if calculated_classifications[record_index] == 'NO':
            confusion_matrix_row1[0] += 1
        else:
            confusion_matrix_row1[1] += 1
    if original_classifications[record_index] == 'YES':
        if calculated_classifications[record_index] == 'NO':
            confusion_matrix_row2[0] += 1
        else:
            confusion_matrix_row2[1] += 1
    
percentage_errors =  (total_errors * 100.0) / len(calculated_classifications)

print("\n\nClassification error rate: %s\n" % str(percentage_errors))

print("\nConfusion Matrix:\n\n")
print("%50s          %25s\n" % ("classified as NOT at risk", "classified as at risk"))
print("Known to be NOT at risk: %10d  %35d\n\n" % tuple(confusion_matrix_row1))
print("Known to be at risk:%15d  %35d\n\n" % tuple(confusion_matrix_row2))

#============== Now interact with the user for classifying additional records  ==========
if interaction_needed:
    while 1:
        input = raw_input("\nWould you like to see classification for a particular record: ")
        if input == 'n':
            sys.exit()
        elif input == 'y':
            input = raw_input("\nEnter record numbers whose classifications you want to see: ")
            records_to_classify = list(map(int, input.split()))
            records_to_classify_local = records_to_classify[:]
            record_ids_with_class_labels = {}
            record_ids_with_features_and_vals = {}
            all_fields = []
            with open(training_datafile) as f:
                for i,line in enumerate(f):
                    if i == 0:
                        all_fields = cleanup_csv(line).split(r',')
                        continue 
                    if len(records_to_classify_local) == 0:
                        break
                    record = cleanup_csv(line).split(r',')
                    if int(record[0]) in records_to_classify_local:
                        records_to_classify_local.remove(int(record[0]))
                        record_ids_with_class_labels[int(record[0])] = record[csv_class_column_index]
                        features_and_vals = list(map(operator.add,
                                            [all_fields[i] for i in csv_columns_for_features],
                                            list(map(operator.add, '=' * len(csv_columns_for_features),
                                            [record[i] for i in csv_columns_for_features]))))
                        record_ids_with_features_and_vals[int(record[0])] = features_and_vals
                f.close()
            # Now classify all the records extracted from the database file:    
            for record_index in record_ids_with_features_and_vals:
                test_sample = record_ids_with_features_and_vals[record_index]
                rt.classify_with_all_trees( test_sample )
                classification = rt.get_majority_vote_classification()
                print("\nclassification for %5d: %10s       original classification: %s" %
                            (record_index, classification, record_ids_with_class_labels[record_index]))
        else:
            print("\nYou are allowed to enter only 'y' or 'n'.  Try again.")
