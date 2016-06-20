import DecisionTree
import unittest

training_datafile = "training_symbolic.csv"


class TestEntropyCalculation(unittest.TestCase):

    def setUp(self):
        print("Testing entropy calculation on sample training file")
        self.dt = DecisionTree.DecisionTree(training_datafile = training_datafile,
                                            csv_class_column_index = 1,
                                            csv_columns_for_features = [2,3,4,5])
        self.dt.get_training_data()
        self.dt.calculate_first_order_probabilities()
        self.dt.calculate_class_priors()

    def test_class_entropy_on_priors(self):
        ent = self.dt.class_entropy_on_priors()
        self.assertTrue( abs(ent - 0.958) < 0.01)

    def test_class_entropy_for_given_sequence_of_feature_values(self):
        ent = self.dt.class_entropy_for_a_given_sequence_of_features_and_values_or_thresholds(['smoking=heavy', 'exercising=never', 'fatIntake=low', \
                                             'videoAddiction=none']) 
        self.assertTrue( abs(ent - 0.862) < 0.01)

def getTestSuites(type):
    return unittest.TestSuite([
            unittest.makeSuite(TestEntropyCalculation, type)
                             ])                    
if __name__ == '__main__':
    unittest.main()

