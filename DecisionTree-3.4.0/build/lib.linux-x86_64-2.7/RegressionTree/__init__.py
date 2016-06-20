#!/usr/bin/env python

import sys

if sys.version_info[0] == 3:
    from DecisionTree.DecisionTree import __version__
    from DecisionTree.DecisionTree import __author__
    from DecisionTree.DecisionTree import __date__
    from DecisionTree.DecisionTree import __url__
    from DecisionTree.DecisionTree import __copyright__
    from RegressionTree.RegressionTree import RegressionTree
else:
    from DecisionTree import __version__
    from DecisionTree import __author__
    from DecisionTree import __date__
    from DecisionTree import __url__
    from DecisionTree import __copyright__
    from RegressionTree import RegressionTree




