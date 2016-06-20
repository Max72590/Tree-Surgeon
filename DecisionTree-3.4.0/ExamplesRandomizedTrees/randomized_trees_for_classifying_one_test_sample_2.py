#!/usr/bin/env python

##   randomized_trees_for_classifying_one_test_sample_2.py

##  This script demonstrates how you can use the RandomizedTreesForBigData
##  class for data classification in the big data context.  Assuming you
##  have access to a very large training database, you can draw multiple
##  random datasets from the database and use each for constructing a
##  different decision tree.  Subsequently, the final classification for a
##  new data sample can be based on majority voting by all the decision
##  trees thus constructed.  In order to use this functionality, you need
##  to set the following two constructor parameters of this class:
##
##           how_many_training_samples_per_tree
##
##           how_many_trees


import RandomizedTreesForBigData

training_datafile = "stage3cancer.csv"

rt = RandomizedTreesForBigData.RandomizedTreesForBigData(
                                training_datafile = training_datafile,
                                csv_class_column_index = 2,
                                csv_columns_for_features = [3,4,5,6,7,8],
                                entropy_threshold = 0.01,
                                max_depth_desired = 8,
                                symbolic_to_numeric_cardinality_threshold = 10,
                                how_many_trees = 3,
                                how_many_training_samples_per_tree = 50,
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

test_sample  = ['g2 = 4.2',
                'grade = 2.3',
                'gleason = 4',
                'eet = 1.7',
                'age = 55.0',
                'ploidy = diploid']

print("Classify the test sample with each decision tree....")
rt.classify_with_all_trees( test_sample )

##   COMMENT OUT the following statement if you do NOT want to see the classification results
##   produced by each tree separately:
rt.display_classification_results_for_all_trees()

print("\n\nWill now calculate the majority decision from all trees:\n")
decision = rt.get_majority_vote_classification()
print("\nMajority vote decision: %s\n" % decision)
