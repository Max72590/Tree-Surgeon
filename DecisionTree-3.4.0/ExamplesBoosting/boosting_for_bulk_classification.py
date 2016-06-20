#!/usr/bin/env python

### boosting_for_bulk_classificaiton.py

##  Call syntax example:

##         boosting_for_bulk_classification.py  training6.csv  test6.csv  out6.csv

##  This script demonstrates how a boosted decision-tree classifier can be used
##  to carry out bulk classification of all your test samples in a file.

import BoostedDecisionTree
import re
import sys
import operator

debug = 0

if len(sys.argv) != 4:
    sys.exit('''This script must be called with exactly three command-line '''
             '''arguments:\n''' 
             '''   1st arg: name of the training datafile\n'''
             '''   2nd arg: name of the test data file\n'''
             '''   3rd arg: the name of the output file to which class '''
             '''labels will be written\n''') 

training_datafile,test_datafile,outputfile = sys.argv[1:4]

training_file_class_name_in_column       = 1
training_file_columns_for_feature_values = [2,3]
how_many_stages                          = 4


####################################  Utility Routines #################################

def convert(value):
  try:
        answer = float(value)
        return answer
  except:
        return value

def sample_index(sample_name):
    m = re.search('_(.+)$', sample_name)
    return int(m.group(1))

def get_test_data_from_csv():
    global all_class_names, feature_values_for_samples_dict
    if not test_datafile.endswith('.csv'): 
        sys.exit("Aborted. get_test_data_from_csv() is only for CSV files")
    class_name_in_column = training_file_class_name_in_column - 1 
    all_data = [line.rstrip().split(',') for line in open(test_datafile,"rU")]
    data_dict = {line[0] : line[1:] for line in all_data}
    if '""' not in data_dict:
        sys.exit('''Aborted. The first row of CSV file must begin '''
                 '''with "" and then list the feature names and the class names''')
    feature_names = [item.strip('"') for item in data_dict['""']]
    class_column_heading = feature_names[class_name_in_column]
    feature_names = [feature_names[i-1] for i in training_file_columns_for_feature_values]
    class_for_sample_dict = { "sample_" + key.strip('"') : class_column_heading + "=" +
                              data_dict[key][class_name_in_column] for key in data_dict if key != '""'}
    feature_values_for_samples_dict = {"sample_" + key.strip('"') : 
          list(map(operator.add, list(map(operator.add, feature_names, "=" * len(feature_names))),
                [str(convert(data_dict[key][i-1].strip('"'))) 
                    for i in training_file_columns_for_feature_values])) for key in data_dict if key != '""'}
    features_and_values_dict = {data_dict['""'][i-1].strip('"') : 
            [convert(data_dict[key][i-1].strip('"')) for key in data_dict if key != '""'] 
                                                         for i in training_file_columns_for_feature_values} 
    all_class_names = sorted(list(set(class_for_sample_dict.values())))
    numeric_features_valuerange_dict = {}
    feature_values_how_many_uniques_dict = {}
    features_and_unique_values_dict = {}
    for feature in features_and_values_dict:
        unique_values_for_feature = list(set(features_and_values_dict[feature]))
        unique_values_for_feature = sorted(list(filter(lambda x: x != 'NA', unique_values_for_feature)))
        feature_values_how_many_uniques_dict[feature] = len(unique_values_for_feature)
        if all(isinstance(x,float) for x in unique_values_for_feature):
            numeric_features_valuerange_dict[feature] = [min(unique_values_for_feature), max(unique_values_for_feature)]
            unique_values_for_feature.sort(key=float)
        features_and_unique_values_dict[feature] = sorted(unique_values_for_feature)
    if debug:
        print("\nAll class names: " + str(all_class_names))
        print("\nEach sample data record:")
        for item in sorted(feature_values_for_samples_dict.items(), key = lambda x: sample_index(x[0]) ):
            print(item[0]  + "  =>  "  + str(item[1]))
        print("\nclass label for each data sample:")
        for item in sorted(class_for_sample_dict.items(), key=lambda x: sample_index(x[0])):
            print(item[0]  + "  =>  "  + str(item[1]))
        print("\nfeatures and the values taken by them:")
        for item in sorted(features_and_values_dict.items()):
            print(item[0]  + "  =>  "  + str(item[1]))
        print("\nnumeric features and their ranges:")
        for item in sorted(numeric_features_valuerange_dict.items()):
            print(item[0]  + "  =>  "  + str(item[1]))
        print("\nnumber of unique values in each feature:")
        for item in sorted(feature_values_how_many_uniques_dict.items()):
            print(item[0]  + "  =>  "  + str(item[1]))


########################################################################################
###############  Read the Input Test Datafile and Classify Each Record  ################

### First construct an instance of the DecisionTree class:

boosted = BoostedDecisionTree.BoostedDecisionTree(
                                training_datafile = training_datafile,
                                csv_class_column_index = training_file_class_name_in_column,
                                csv_columns_for_features = training_file_columns_for_feature_values,
                                entropy_threshold = 0.01,
                                max_depth_desired = 5,
                                symbolic_to_numeric_cardinality_threshold = 10,
                                how_many_stages = how_many_stages,
                              )
print("Reading and processing training data...")
boosted.get_training_data_for_base_tree()

##   UNCOMMENT THE FOLLOWING STATEMENT if you want to see the training data used for
##   just the base tree:
#boosted.show_training_data_for_base_tree()

# This is a required call:
print("Calculating first-order probabilities...")
boosted.calculate_first_order_probabilities_and_class_priors()

# This is a required call:
print("Constructing base decision tree...")
boosted.construct_base_decision_tree()

#   UNCOMMENT THE FOLLOWING TWO STATEMENTS if you would like to see just the base
#   decision tree displayed in your terminal window:
#print("\n\nThe Decision Tree:\n")
#boosted.display_base_decision_tree()

# This is a required call:
print("Constructing the rest of the decision trees....")
boosted.construct_cascade_of_trees()

##  UNCOMMENT the next statement if you want to see the decision trees constructed
##  for each stage of the cascade:
boosted.display_decision_trees_for_different_stages()

### NOW YOU ARE READY TO CLASSIFY THE FILE-BASED TEST DATA:
get_test_data_from_csv()

FILEOUT   = open(outputfile, 'w')

class_names = ",".join(sorted(boosted.get_all_class_names()))
output_string = "sample_index," + class_names + "\n"
FILEOUT.write(output_string)
for item in sorted(feature_values_for_samples_dict.items(), key = lambda x: sample_index(x[0]) ):
    test_sample =  feature_values_for_samples_dict[item[0]]
    boosted.classify_with_boosting(test_sample)
    classification = boosted.trust_weighted_majority_vote_classifier()
    output_string = str(sample_index(item[0]))
    output_string +=  "," + classification[11:]
    FILEOUT.write(output_string + "\n")
FILEOUT.close()

print("Trust weighted majority vote classifications from the boosted decision-tree classifier was written out to %s\n" % outputfile)


