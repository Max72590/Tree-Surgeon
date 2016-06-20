import DecisionTree
import unittest

training_datafile = "training_symbolic.csv"


class TestDecisionTreeClassification(unittest.TestCase):

    def setUp(self):
        print("Testing decision-tree classification on sample training file")
        self.dt = DecisionTree.DecisionTree(training_datafile = training_datafile,
                                            csv_class_column_index = 1,
                                            csv_columns_for_features = [2,3,4,5])
        self.dt.get_training_data()
        self.dt.calculate_first_order_probabilities()
        self.dt.calculate_class_priors()
        self.root_node = self.dt.construct_decision_tree_classifier()

    def test_decision_tree_classification(self):
        test_sample = ['exercising=never', 
                       'smoking=heavy', 
                       'fatIntake=heavy',
                       'videoAddiction=heavy']
        classification = self.dt.classify(self.root_node, test_sample)
        self.assertTrue( abs(float(classification['class=benign']) - 0.005) < 0.01 )
        self.assertTrue( abs(float(classification['class=malignant']) - 0.995) < 0.01 )

def getTestSuites(type):
    return unittest.TestSuite([
            unittest.makeSuite(TestDecisionTreeClassification, type)
                             ])                    
if __name__ == '__main__':
    unittest.main()

