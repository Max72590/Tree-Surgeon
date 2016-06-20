#!/usr/bin/env python

##  randomized_trees_for_classifying_one_test_sample_1.py

##  This script demonstrates using the RandomizedTreesForBigData class for
##  for solving a data classification problem when there is a significant
##  disparity between the populations of the training samples for the
##  the different classes.  You need to set the following two parameters
##  in the call to the constructor for the 'needle-in-a-haystack' logic
##  to work:
##
##              looking_for_needles_in_haystac
##              how_many_trees



import RandomizedTreesForBigData

##  NOTE: The databaes file mentioned below is proprietary and is NOT
##        included in the module package.

training_datafile = "/home/kak/DecisionTree_data/AtRisk/AtRiskModel_File_modified.csv"

rt = RandomizedTreesForBigData.RandomizedTreesForBigData(
                                training_datafile = training_datafile,
                                csv_class_column_index = 48,
                                csv_columns_for_features = [24,32,33,34,41],
                                entropy_threshold = 0.01,
                                max_depth_desired = 8,
                                symbolic_to_numeric_cardinality_threshold = 10,
                                how_many_trees = 5,
                                looking_for_needles_in_haystack = 1,
                              )
print("Reading the training data ...")
rt.get_training_data_for_N_trees()

##   UNCOMMENT the following statement if you want to see the training data used for each tree::
rt.show_training_data_for_all_trees()

print("Calculating first order probabilities...")
rt.calculate_first_order_probabilities()

print("Calculating class priors...")
rt.calculate_class_priors()

print("Constructing all decision trees ....")
rt.construct_all_decision_trees()

##   UNCOMMENT the following statement if you want to see all decision trees individually:
rt.display_all_decision_trees()

print("Reading the test sample....")

test_sample  = ['SATV = 110',
                 'SATM = 130',
                 'SATW = 180',
                 'HSGPA = 1.5']

print("Classify the test sample with each decision tree....")
rt.classify_with_all_trees( test_sample )

##   COMMENT OUT the following statement if you do NOT want to see the classification results
##   produced by each tree separately:
rt.display_classification_results_for_all_trees()

print("\n\nWill now calculate the majority decision from all trees:\n")
decision = rt.get_majority_vote_classification()
print("\nMajority vote decision: %s\n" % decision)
