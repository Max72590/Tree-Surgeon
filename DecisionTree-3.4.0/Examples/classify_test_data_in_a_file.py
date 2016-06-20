#!/usr/bin/env python

### classify_test_data_in_a_file_numeric.py

##  This script demonstrates how you can carry out bulk classification of
##  data records in an input csv file.  You can name either a csv file or a
##  txt file for your output where the classification results will be
##  deposited.  If you name a csv file for the output, the output csv will
##  show for each data record all the classes along with their
##  probabilities.  If you name a txt file for your output, what you'll see
##  in that file will depend on the value of the binary variable
##
##       show_hard_classifications
##
##  If this variable is set to 1, only the most probable class will be
##  shown for each data record in the input test datafile.  On the other
##  hand, when this variable is set to 0, for each data record in the input
##  csv file you will see in output file all the classes along with their
##  probabilities.

import DecisionTree
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

### When the following variable is set to 1, only the most probable class for each
### data record is written out to the output file.  This works only for the case
### when the output is sent to a `.txt' file.  If the output is sent to a `.csv' 
### file, you'll see all the class names and their probabilities for each data sample
### in your test datafile.
show_hard_classifications = 0


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
    class_name_in_column = dt._csv_class_column_index - 1 
    all_data = [line.rstrip().split(',') for line in open(test_datafile,"rU")]
    data_dict = {line[0] : line[1:] for line in all_data}
    if '""' not in data_dict:
        sys.exit('''Aborted. The first row of CSV file must begin '''
                 '''with "" and then list the feature names and the class names''')
    feature_names = [item.strip('"') for item in data_dict['""']]
    class_column_heading = feature_names[class_name_in_column]
    feature_names = [feature_names[i-1] for i in dt._csv_columns_for_features]
    class_for_sample_dict = { "sample_" + key.strip('"') : \
                 class_column_heading + "=" + \
                 data_dict[key][class_name_in_column] \
                 for key in data_dict if key != '""'}
    feature_values_for_samples_dict = {"sample_" + key.strip('"') : \
          list(map(operator.add, list(map(operator.add, \
                       feature_names, "=" * len(feature_names))),   \
                [str(convert(data_dict[key][i-1].strip('"'))) \
                           for i in dt._csv_columns_for_features])) \
                                 for key in data_dict if key != '""'}
    features_and_values_dict = {data_dict['""'][i-1].strip('"') : \
            [convert(data_dict[key][i-1].strip('"')) \
                for key in data_dict if key != '""']  \
                        for i in dt._csv_columns_for_features} 
    all_class_names = sorted(list(set(class_for_sample_dict.values())))
    numeric_features_valuerange_dict = {}
    feature_values_how_many_uniques_dict = {}
    features_and_unique_values_dict = {}
    for feature in features_and_values_dict:
        unique_values_for_feature = \
                           list(set(features_and_values_dict[feature]))
        unique_values_for_feature = \
          sorted(list(filter(lambda x: x != 'NA', unique_values_for_feature)))
        feature_values_how_many_uniques_dict[feature] = len(unique_values_for_feature)
        if all(isinstance(x,float) for x in unique_values_for_feature):
            numeric_features_valuerange_dict[feature] = \
                          [min(unique_values_for_feature), max(unique_values_for_feature)]
            unique_values_for_feature.sort(key=float)
        features_and_unique_values_dict[feature] = sorted(unique_values_for_feature)
    if debug:
        print("\nAll class names: " + str(all_class_names))
        print("\nEach sample data record:")
        for item in sorted(feature_values_for_samples_dict.items(), \
                                               key = lambda x: sample_index(x[0]) ):
            print(item[0]  + "  =>  "  + str(item[1]))
        print("\nclass label for each data sample:")
        for item in sorted(class_for_sample_dict.items(), \
                                                  key=lambda x: sample_index(x[0])):
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
dt = DecisionTree.DecisionTree( 
                 training_datafile = training_datafile,
                 csv_class_column_index = 1,    
                 csv_columns_for_features = [2,3],
                 entropy_threshold = 0.01,
                 max_depth_desired = 3,
                 symbolic_to_numeric_cardinality_threshold = 10,
        )

### Initialize probability cache:
dt.get_training_data()
dt.calculate_first_order_probabilities()
dt.calculate_class_priors()

### UNCOMMENT THE NEXT STATEMENT if you would like to see
### the training data that was read from the disk file:
#dt.show_training_data();

### Construct the decision tree for classification:
root_node = dt.construct_decision_tree_classifier()

### UNCOMMENT THE NEXT STATEMENT if you would like to see
### the decision tree displayed in your terminal window:
#root_node.display_decision_tree("   ")

### NOW YOU ARE READY TO CLASSIFY THE FILE-BASED TEST DATA:
get_test_data_from_csv()

FILEOUT   = open(outputfile, 'w')

if show_hard_classifications and not outputfile.endswith('.csv'):
    FILEOUT.write("\nOnly the most probable class shown "  +\
                      "for each test sample\n\n")
elif not show_hard_classifications and not outputfile.endswith('.csv'):
    FILEOUT.write("\nThe classification result for each "  +\
                      "sample ordered in decreasing order of " +\
                      "probability\n\n")
if outputfile.endswith('.csv'):
    class_names_csv = ",".join(sorted(dt._class_names))
    output_string = "sample_index," + class_names_csv + "\n"
    FILEOUT.write(output_string)
    for item in sorted(feature_values_for_samples_dict.items(), \
                                   key = lambda x: sample_index(x[0]) ):
        test_sample =  feature_values_for_samples_dict[item[0]]
        classification = dt.classify(root_node, test_sample)
        solution_path = classification['solution_path']
        del classification['solution_path']
        which_classes = sorted(list(classification.keys()))
        output_string = str(sample_index(item[0]))
        for which_class in which_classes:
            if which_class is not 'solution_path':
                valuestring = classification[which_class]
                output_string +=  "," + valuestring
        FILEOUT.write(output_string + "\n")
else:
    for item in sorted(feature_values_for_samples_dict.items(), \
                                  key = lambda x: sample_index(x[0]) ):
        test_sample =  feature_values_for_samples_dict[item[0]]
        classification = dt.classify(root_node, test_sample)
        solution_path = classification['solution_path']
        del classification['solution_path']
        which_classes = list( classification.keys() )
        which_classes = sorted(which_classes, key=lambda x: classification[x], \
                                                                  reverse=True)
        result_string = item[0] + ":   "
        if show_hard_classifications:
            which_class = which_classes[0]
            m = re.search(r'(.+)=(.+)', which_class)
            class_name = m.group(2)
            valuestring = "%-20s" % (classification[which_class])
            result_string += class_name + " => " + valuestring    + "    "
            FILEOUT.write(result_string + "\n")
        else:
            for which_class in which_classes:
                m = re.search(r'(.+)=(.+)', which_class)
                class_name = m.group(2)
                valuestring = "%-20s" % (classification[which_class]) 
                result_string += class_name + " => " + valuestring    + "    "
            FILEOUT.write(result_string + "\n")
FILEOUT.close()
