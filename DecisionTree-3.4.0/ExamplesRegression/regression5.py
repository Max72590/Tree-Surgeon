#!/usr/bin/env python

##  regression5.py

##  This example script is for demonstrating the power of the regression tree
##  for the case when you have just one predictor variable and one dependent
##  variable.

##  The only difference between this script and regression4.py is that the
##  data file used here is more complex in terms of how the dependent 
##  variable depends on the predictor variable.

##  Remember, the column indexing in the csv file is zero-based.  That is,
##  the first column is indexed 0.

import RegressionTree
import sys

training_datafile = "gendata5.csv"

rt = RegressionTree.RegressionTree( training_datafile = training_datafile,
                                    dependent_variable_column = 2,
                                    predictor_columns = [1],
                                    mse_threshold = 0.01,
                                    max_depth_desired = 2,
                                  )
rt.get_training_data_for_regression()
root_node = rt.construct_regression_tree()

print("\n\nThe Regression Tree:\n")
root_node.display_regression_tree("     ")

test_sample = ['xcoord = 128']

answer = rt.prediction_for_single_data_point(root_node, test_sample)
print("Answer returned: %s" % str(answer))
print("Prediction for test sample: %.9f" % answer['prediction'][0])

rt.predictions_for_all_data_used_for_regression_estimation(root_node)

rt.display_all_plots()

rt.mse_for_tree_regression_for_all_training_samples(root_node)
