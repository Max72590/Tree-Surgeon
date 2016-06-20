#!/usr/bin/env python

##  regression6.py

##  This script shows the power of tree regression on 3D data.  We now have
##  TWO predictor variables and one dependent variable.

##  Remember, the column indexing in the csv file is zero-based.  That is,
##  the first column is indexed 0.

import RegressionTree
import sys

training_datafile = "gen3Ddata1.csv"

rt = RegressionTree.RegressionTree( training_datafile = training_datafile,
                                    dependent_variable_column = 3,
                                    predictor_columns = [1,2],
                                    mse_threshold = 0.01,
                                    max_depth_desired = 2,
                                  )
rt.get_training_data_for_regression()
root_node = rt.construct_regression_tree()

print("\n\nThe Regression Tree:\n")
root_node.display_regression_tree("     ")

test_sample = ['x1_coord = 50', 'x2_coord = 50']
answer = rt.prediction_for_single_data_point(root_node, test_sample)
print("Answer returned: %s" % str(answer))
print("Prediction for test sample: %.9f" % answer['prediction'][0])

rt.predictions_for_all_data_used_for_regression_estimation(root_node)

rt.display_all_plots()

rt.mse_for_tree_regression_for_all_training_samples(root_node)

