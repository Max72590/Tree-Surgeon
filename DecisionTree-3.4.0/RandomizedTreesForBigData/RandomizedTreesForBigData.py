from DecisionTree import DecisionTree

import re
import random
import operator
import string
import sys
from functools import reduce


def convert(value):
    try:
        answer = float(value)
        return answer
    except:
        return value

def sample_index(sample_name):
    '''
    When the training data is read from a CSV file, we assume that the first column
    of each data record contains a unique integer identifier for the record in that
    row. This training data is stored in a dictionary whose keys are the prefix
    'sample_' followed by the identifying integers.  The purpose of this function is
    to return the identifying integer associated with a data record.
    '''
    m = re.search('_(.+)$', sample_name)
    return int(m.group(1))

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

class RandomizedTreesForBigData(object):

    def __init__(self, *args, **kwargs ):
        if kwargs and args:
            raise SyntaxError(  
                   '''RandomizedTreesForBigData constructor can only be called with keyword arguments for
                      the following keywords: training_datafile, entropy_threshold,
                      max_depth_desired, csv_class_column_index, symbolic_to_numeric_cardinality_threshold,
                      number_of_histogram_bins, csv_columns_for_features, number_of_histogram_bins, 
                      how_many_trees, how_many_training_samples_per_tree,
                      looking_for_needles_in_haystack, debug1''') 
        allowed_keys = 'training_datafile','entropy_threshold','max_depth_desired','csv_class_column_index',\
                       'symbolic_to_numeric_cardinality_threshold','csv_columns_for_features',\
                       'number_of_histogram_bins','how_many_trees',\
                       'how_many_training_samples_per_tree','looking_for_needles_in_haystack','debug1'
        keywords_used = kwargs.keys()
        for keyword in keywords_used:
            if keyword not in allowed_keys:
                raise SyntaxError(keyword + ":  Wrong keyword used --- check spelling") 
        training_datafile=entropy_threshold=max_depth_desired=csv_class_column_index=number_of_histogram_bins=None
        symbolic_to_numeric_cardinality_threshold=csv_columns_for_features=how_many_trees=debug1=None
        looking_for_needles_in_haystack=how_many_training_samples_per_tree=None
        if kwargs:
            if 'training_datafile' in kwargs : training_datafile = kwargs['training_datafile']
            else: raise Exception('''You must specify a training datafile''')
            if 'csv_class_column_index' in kwargs: csv_class_column_index = kwargs.pop('csv_class_column_index')
            else: raise Exception('''You must provide a zero-based column index for the class label in each record''')
            if 'csv_columns_for_features' in kwargs: \
                                  csv_columns_for_features = kwargs.pop('csv_columns_for_features')
            else: raise Exception('''You must provide zero-based column index values for the features''')
            if 'how_many_trees' in kwargs : how_many_trees = kwargs.pop('how_many_trees')
            if 'how_many_training_samples_per_tree' in kwargs : \
                         how_many_training_samples_per_tree = kwargs.pop('how_many_training_samples_per_tree')
            if 'looking_for_needles_in_haystack' in kwargs : \
                         looking_for_needles_in_haystack = kwargs.pop('looking_for_needles_in_haystack')
            if 'debug1' in kwargs  :  debug1 = kwargs.pop('debug1')
        if training_datafile:
            self._training_datafile = training_datafile
        elif not training_datafile:
            raise Exception('''You must specify a training datafile''')
        else:
            if args[0] != 'evalmode':
                raise Exception("""When supplying non-keyword arg, it can only be 'evalmode'""")
        if csv_class_column_index:
            self._csv_class_column_index                 =      csv_class_column_index
        else:
            self._csv_class_column_index                 =      None
        if csv_columns_for_features:
            self._csv_columns_for_features               =      csv_columns_for_features
        else: 
            self._csv_columns_for_features               =      None            
        if looking_for_needles_in_haystack:
            if how_many_training_samples_per_tree:
                raise Exception("""\n\nWhen using 'looking_for_needles_in_haystack' option, you are NOT allowed """
                                """to also use the 'how_many_training_samples_per_tree' option.""")
            else:
                self._looking_for_needles_in_haystack    =      looking_for_needles_in_haystack
        else:
            self._looking_for_needles_in_haystack        =      0
        if how_many_training_samples_per_tree:
            if looking_for_needles_in_haystack:
                raise Exception("""\n\nWhen using 'how_many_training_samples_per_tree' option, you are NOT allowed """
                                """to also use the 'looking_for_needles_in_haystack' option.""")
            else:
                self._how_many_training_samples_per_tree =      how_many_training_samples_per_tree
        else:
            self._how_many_training_samples_per_tree =      None
        self._how_many_trees                         =      how_many_trees
        self._training_data_for_trees                =      {}
        self._all_trees                              =      {i:DecisionTree(**kwargs) for i in range(how_many_trees)}
        self._root_nodes                             =      []
        self._classifications                        =      None
        self._all_record_ids                         =      []
        self._training_data_record_indexes           =      {}
        if debug1:
            self._debug1                             =      debug1
        else:
            self._debug1                             =      0

    def get_training_data_for_N_trees(self):
        if not self._training_datafile.endswith('.csv'): 
            TypeError("Aborted. get_training_data_from_csv() is only for CSV files")
        self._training_data_for_trees = {t : [] for t in range(self._how_many_trees)}
        def total_num_training_samples_in_file(filename): 
            with open(filename) as f:
                for i, line in enumerate(f):
                    record = cleanup_csv(line)
                    self._all_record_ids.append(record[0:record.find(',')])
                f.close()
            return i    # Note that i is less by 1 relative to total number of records. But that's ok because of header.
        self._how_many_total_training_samples = total_num_training_samples_in_file(self._training_datafile)
        if self._debug1:
            print("\n\nTotal number of training samples: %d\n" % self._how_many_total_training_samples)
        if self._looking_for_needles_in_haystack:
            self.get_training_data_for_N_trees_balanced()
        else:
            self.get_training_data_for_N_trees_regular()

    def get_training_data_for_N_trees_balanced(self):            
        if self._how_many_training_samples_per_tree:
            raise Exception('''You cannot use the contructor option 'how_many_training_samples_per_tree' if you '''
                            '''have set the option 'looking_for_needles_in_haystack' ''')
        class_names = []
        import sys
        all_record_ids_with_class_labels = {}
        with open(self._training_datafile) as f:
            for i,line in enumerate(f):
                if i == 0: continue
                record = cleanup_csv(line)
                parts = record.split(r',')
                class_names.append(parts[self._csv_class_column_index])
                all_record_ids_with_class_labels[parts[0]] = parts[self._csv_class_column_index]
                if i%10000 == 0:
                    print('.'),
                    sys.stdout.flush()
                sys.stdout = sys.__stdout__
            f.close() 
        unique_class_names = list(set(class_names[1:]))
        if len(unique_class_names) > 2:
            raise Exception("""'looking_for_needles_in_haystack' option has only been tested for the case of """
                            """two data classes.  You have %d data classes. Aborting.""" % len(unique_class_names))
        if self._debug1:
            print("\n\nunique class names: %s" % str(unique_class_names))
        hist = {x : 0 for x in unique_class_names}
        for item in class_names[1:]:
            for unique_val in unique_class_names:
                if item == unique_val:
                    hist[unique_val] += 1
                    break
        if self._debug1:
            print("\nhistogram of the values for the field : "),
            for key in sorted(hist):
                print("   %s => %s" % str(key), str(hist[key])),
        max_number_of_trees_possible = max(list(hist.values())) // min(list(hist.values()))
        if self._debug1:      
            print("\nmaximum number of trees possible: %s" % str(max_number_of_trees_possible))
        if self._how_many_trees > max_number_of_trees_possible:
            raise Exception('''\n\nYou have asked for more trees than can be supported by the training data. '''
                            '''Maxinum number of trees that can be constructed from the training file is: %d\n''' %
                            max_number_of_trees_possible)
        class1 = {item[0] : item[1] for item in all_record_ids_with_class_labels.items()
                                                                       if item[1] == unique_class_names[0]}
        class2 = {item[0] : item[1] for item in all_record_ids_with_class_labels.items()
                                                                       if item[1] == unique_class_names[1]}
        minority_class = class2 if len(class1) >= len(class2) else class1
        majority_class = class1 if len(class1) >= len(class2) else class2
        minority_records = sorted(minority_class.keys())
        majority_records = sorted(majority_class.keys())
        if self._debug1:
            print("minority records: %s" % str(minority_records))
        self._how_many_training_samples_per_tree = 2 * len(minority_records)
        self._training_data_record_indexes  = {t : random.sample(majority_records,
                                len(minority_records)) + minority_records for t in range(self._how_many_trees)}  
        if self._debug1:      
            for t in self._training_data_record_indexes:
                print("\n\n%d   =>   %s\n" % (t, str(self._training_data_record_indexes[t])))
        self._digest_training_data_all_trees()

    def get_training_data_for_N_trees_regular(self):
        self._training_data_record_indexes  = {t : random.sample(self._all_record_ids,
                             self._how_many_training_samples_per_tree) for t in range(self._how_many_trees)}
        self._digest_training_data_all_trees()

    def _digest_training_data_all_trees(self):
        class_name_in_column = self._csv_class_column_index - 1  # subtract 1 because first col has record labels    
        firstline = None
        with open(self._training_datafile) as f:
            for i,line in enumerate(f):
                record = cleanup_csv(line)
                if i == 0:
                    firstline = record
                    continue
                for t in self._training_data_record_indexes:
                    if record[0:record.find(',')] in self._training_data_record_indexes[t]:
                        self._training_data_for_trees[t].append(record)                        
            f.close()
        splitup_data_for_trees = {t : [record.rstrip().split(',') for record in self._training_data_for_trees[t]]
                                             for t in range(self._how_many_trees)}
        data_dict_for_all_trees = {t : {record[0] : record[1:] for record in splitup_data_for_trees[t]} 
                                                               for t in range(self._how_many_trees)}
        all_feature_names = firstline.rstrip().split(',')[1:]
        class_column_heading = all_feature_names[class_name_in_column]        
        feature_names = [all_feature_names[i-1] for i in self._csv_columns_for_features]
        class_for_sample_all_trees = {t : { "sample_" + key.strip('"') : 
               class_column_heading + "=" + data_dict_for_all_trees[t][key][class_name_in_column] 
                           for key in data_dict_for_all_trees[t]} for t in range(self._how_many_trees)}
        sample_names_in_all_trees = {t : ["sample_" + key for key in data_dict_for_all_trees[t]]
                                                               for t in range(self._how_many_trees)}
        feature_values_for_samples_all_trees = {t : {"sample_" + key.strip('"') :         
                  list(map(operator.add, list(map(operator.add, feature_names, "=" * len(feature_names))), 
           [str(convert(data_dict_for_all_trees[t][key][i-1].strip('"'))) for i in self._csv_columns_for_features])) 
                           for key in data_dict_for_all_trees[t]} for t in range(self._how_many_trees)}
        features_and_values_all_trees = {t : {all_feature_names[i-1] :
                 [convert(data_dict_for_all_trees[t][key][i-1].strip('"')) for key in data_dict_for_all_trees[t]]    
                     for i in self._csv_columns_for_features} for t in range(self._how_many_trees)}
        all_class_names_all_trees = {t : sorted(list(set(class_for_sample_all_trees[t].values())))
                                                     for t in range(self._how_many_trees)}
        numeric_features_valuerange_all_trees = {t : {} for t in range(self._how_many_trees)}        
        feature_values_how_many_uniques_all_trees = {t : {} for t in range(self._how_many_trees)}
        features_and_unique_values_all_trees = {t : {} for t in range(self._how_many_trees)}
        for t in range(self._how_many_trees):
            for feature in features_and_values_all_trees[t]:
                unique_values_for_feature = list(set(features_and_values_all_trees[t][feature]))
                unique_values_for_feature = sorted(list(filter(lambda x: x != 'NA', unique_values_for_feature)))
                feature_values_how_many_uniques_all_trees[t][feature] = len(unique_values_for_feature)
                if all(isinstance(x,float) for x in unique_values_for_feature):
                    numeric_features_valuerange_all_trees[t][feature] = \
                                  [min(unique_values_for_feature), max(unique_values_for_feature)]
                    unique_values_for_feature.sort(key=float)
                features_and_unique_values_all_trees[t][feature] = sorted(unique_values_for_feature)
        for t in range(self._how_many_trees):                
            self._all_trees[t]._class_names = all_class_names_all_trees[t]
            self._all_trees[t]._feature_names = feature_names
            self._all_trees[t]._samples_class_label_dict = class_for_sample_all_trees[t]
            self._all_trees[t]._training_data_dict  =  feature_values_for_samples_all_trees[t]
            self._all_trees[t]._features_and_values_dict    =  features_and_values_all_trees[t]
            self._all_trees[t]._features_and_unique_values_dict    =  features_and_unique_values_all_trees[t]
            self._all_trees[t]._numeric_features_valuerange_dict = numeric_features_valuerange_all_trees[t]
            self._all_trees[t]._feature_values_how_many_uniques_dict = feature_values_how_many_uniques_all_trees[t]
        if self._debug1:
            for t in range(self._how_many_trees):            
                print("\n\n=============================   For tree %d   ==================================\n" % t)
                print("\nAll class names: " + str(self._all_trees[t]._class_names))
                print("\nEach sample data record:")
                for item in sorted(self._all_trees[t]._training_data_dict.items(), key = lambda x: sample_index(x[0]) ):
                    print(item[0]  + "  =>  "  + str(item[1]))
                print("\nclass label for each data sample:")
                for item in sorted(self._all_trees[t]._samples_class_label_dict.items(), key=lambda x: sample_index(x[0])):
                    print(item[0]  + "  =>  "  + str(item[1]))
                print("\nfeatures and the values taken by them:")
                for item in sorted(self._all_trees[t]._features_and_values_dict.items()):
                    print(item[0]  + "  =>  "  + str(item[1]))
                print("\nnumeric features and their ranges:")
                for item in sorted(self._all_trees[t]._numeric_features_valuerange_dict.items()):
                    print(item[0]  + "  =>  "  + str(item[1]))
                print("\nunique values for the features:")
                for item in sorted(self._all_trees[t]._features_and_unique_values_dict.items()):
                    print(item[0]  + "  =>  "  + str(item[1]))
                print("\nnumber of unique values in each feature:")
                for item in sorted(self._all_trees[t]._feature_values_how_many_uniques_dict.items()):
                    print(item[0]  + "  =>  "  + str(item[1]))

    def get_number_of_training_samples(self):
        return self._number_of_training_samples

    def show_training_data_for_all_trees(self):
        for i in range(self._how_many_trees):
            print("\n\n=============================   For Tree %d   ==================================\n" % i)
            self._all_trees[i].show_training_data()            

    def calculate_first_order_probabilities(self):            
        list(map(lambda x: self._all_trees[x].calculate_first_order_probabilities(), range(self._how_many_trees)))

    def calculate_class_priors(self):            
        list(map(lambda x: self._all_trees[x].calculate_class_priors(), range(self._how_many_trees)))
        
    def construct_all_decision_trees(self):            
        self._root_nodes = \
             list(map(lambda x: self._all_trees[x].construct_decision_tree_classifier(), range(self._how_many_trees)))

    def display_all_decision_trees(self):
        for i in range(self._how_many_trees):
            print("\n\n=============================   For Tree %d   ==================================\n" % i)
            self._root_nodes[i].display_decision_tree("     ")

    def classify_with_all_trees(self, test_sample):
        self._classifications = list(map(lambda x: self._all_trees[x].classify(self._root_nodes[x], test_sample),
                                   range(self._how_many_trees)))

    def display_classification_results_for_all_trees(self):         
        classifications = self._classifications
        if classifications is None:
            raise Exception('''You must first call "classify_with_all_trees()" before invoking "display_classification_results_for_all_trees()" ''')
        solution_paths = list(map(lambda x: x['solution_path'], classifications))
        for t in range(self._how_many_trees):
            print("\n\n=============================   For Tree %d   ==================================\n" % t)
            print("\nnumber of training samples used: %d\n" % self._how_many_training_samples_per_tree)
            classification = classifications[t]
            del classification['solution_path']
            which_classes = list( classification.keys() )
            which_classes = sorted(which_classes, key=lambda x: classification[x], reverse=True)
            print("\nClassification:\n")
            print("     "  + str.ljust("class name", 30) + "probability")
            print("     ----------                    -----------")
            for which_class in which_classes:
                if which_class is not 'solution_path':
                    print("     "  + str.ljust(which_class, 30) +  str(classification[which_class]))
            print("\nSolution path in the decision tree: " + str(solution_paths[t]))
            print("\nNumber of nodes created: " + str(self._root_nodes[t].how_many_nodes()))

    def get_majority_vote_classification(self):
        classifications = self._classifications
        if classifications is None:
            raise Exception('''You must first call "classify_with_all_trees()" before invoking "get_majority_vote_classification()" ''')
        decision_classes = {class_label : 0 for class_label in self._all_trees[0]._class_names}
        for t in range(self._how_many_trees):
            classification = classifications[t]
            if 'solution_path' in classification:
                del classification['solution_path']
            sorted_classes = sorted(list(classification.keys()), key=lambda x: classification[x], reverse=True)  
            decision_classes[sorted_classes[0]] += 1
        sorted_by_votes_decision_classes = \
                         sorted(list(decision_classes.keys()), key=lambda x: decision_classes[x], reverse=True)
        return sorted_by_votes_decision_classes[0]

    def get_all_class_names(self):
        return self._all_trees[0]._class_names
