from DecisionTree import DecisionTree

import re
import operator
import math
import sys
import string

def convert(value):
    try:
        answer = float(value)
        return answer
    except:
        return value

def sample_index(sample_name):
    '''
    When the training data is read from a CSV file, we assume that the first column of
    each data record contains a unique integer identifier for the record in that
    row. This training data is stored in a dictionary whose keys are the prefix
    'sample_' followed by the identifying integers.  `xx' is a unique integer.  In
    both cases, the purpose of this function is to return the identifying integer
    associated with a data record.
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
    field in the first record that defines the feature names is the empty string ("")
    and the same field in all other records is an ID number for the record.
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
    
class BoostedDecisionTree(DecisionTree):

    def __init__(self, *args, **kwargs ):
        if kwargs and args:
            raise SyntaxError(  
                   '''BoostedDecisionTree constructor can only be called with keyword arguments for
                      the following keywords: training_datafile, entropy_threshold,
                      max_depth_desired, csv_class_column_index,
                      symbolic_to_numeric_cardinality_threshold,
                      number_of_histogram_bins, csv_columns_for_features,
                      number_of_histogram_bins, how_many_stages, debug1''') 
        allowed_keys = 'training_datafile','entropy_threshold','max_depth_desired','csv_class_column_index',\
                       'symbolic_to_numeric_cardinality_threshold','csv_columns_for_features',\
                       'number_of_histogram_bins', 'how_many_stages','debug1','stagedebug'
        keywords_used = kwargs.keys()
        for keyword in keywords_used:
            if keyword not in allowed_keys:
                raise SyntaxError(keyword + ":  Wrong keyword used --- check spelling") 
        training_datafile=entropy_threshold=max_depth_desired=csv_class_column_index=number_of_histogram_bins= None
        symbolic_to_numeric_cardinality_threshold=csv_columns_for_features=how_many_stages=stagedebug=None
        if kwargs and not args:
            if 'how_many_stages' in kwargs      :      how_many_stages = kwargs.pop('how_many_stages')
        DecisionTree.__init__(self, **kwargs)    
        if how_many_stages is not None: 
            self._how_many_stages               =      how_many_stages
        else:
            self._how_many_stages               =      4
        self._all_trees                         =      {i:DecisionTree(**kwargs) for i in range(how_many_stages)}
        self._training_samples                  =      {i:[]for i in range(how_many_stages)}
        self._root_nodes                        =      {i:None for i in range(how_many_stages)}
        self._sample_selection_probs            =      {i:{} for i in range(how_many_stages)}
        self._trust_factors                     =      {i:None for i in range(how_many_stages)}
        self._misclassified_samples             =      {i:[] for i in range(how_many_stages)}
        self._classifications                   =      None
        self._trust_weighted_decision_classes   =      None
        self._stagedebug                        =      0

    def get_training_data_for_base_tree(self):
        if not self._training_datafile.endswith('.csv'): 
            TypeError("Aborted. get_training_data_from_csv() is only for CSV files")
        class_names = []
        all_record_ids_with_class_labels = {}
        firstline = None
        data_dict = {}
        with open(self._training_datafile) as f:
            for i,line in enumerate(f):
                record = cleanup_csv(line)
                if i == 0:
                    firstline = record
                    continue
                parts = record.rstrip().split(r',')
                data_dict[parts[0].strip('"')] = parts[1:]
                class_names.append(parts[self._csv_class_column_index])
                all_record_ids_with_class_labels[parts[0].strip('"')] = parts[self._csv_class_column_index]
                if i%10000 == 0:
                    print('.'),
                    sys.stdout.flush()
                sys.stdout = sys.__stdout__
            f.close() 
        self._how_many_total_training_samples = i   # i is less by 1 from total num of records; but that's okay
        unique_class_names = list(set(class_names))
        if self._debug1:
            print("\n\nTotal number of training samples: %d\n" % self._how_many_total_training_samples)
        all_feature_names = firstline.rstrip().split(',')[1:]
        class_column_heading = all_feature_names[self._csv_class_column_index - 1]
        feature_names = [all_feature_names[i-1] for i in self._csv_columns_for_features]
        class_for_sample_dict = { "sample_" + key : 
               class_column_heading + "=" + data_dict[key][self._csv_class_column_index - 1] for key in data_dict}
        sample_names = ["sample_" + key for key in data_dict]
        feature_values_for_samples_dict = {"sample_" + key :         
                  list(map(operator.add, list(map(operator.add, feature_names, "=" * len(feature_names))), 
           [str(convert(data_dict[key][i-1])) for i in self._csv_columns_for_features])) 
                           for key in data_dict}
        features_and_values_dict = {all_feature_names[i-1] :
            [convert(data_dict[key][i-1]) for key in data_dict] for i in self._csv_columns_for_features}
        all_class_names = sorted(list(set(class_for_sample_dict.values())))
        if self._debug1: print("\n All class names: "+ str(all_class_names))
        numeric_features_valuerange_dict = {}
        feature_values_how_many_uniques_dict = {}
        features_and_unique_values_dict = {}
        for feature in features_and_values_dict:
            unique_values_for_feature = list(set(features_and_values_dict[feature]))
            unique_values_for_feature = sorted(list(filter(lambda x: x != 'NA', unique_values_for_feature)))
            feature_values_how_many_uniques_dict[feature] = len(unique_values_for_feature)
            if all(isinstance(x,float) for x in unique_values_for_feature):
                numeric_features_valuerange_dict[feature] = \
                              [min(unique_values_for_feature), max(unique_values_for_feature)]
                unique_values_for_feature.sort(key=float)
            features_and_unique_values_dict[feature] = sorted(unique_values_for_feature)
        # set the parameters for the tree for the base classifier:
        self._all_trees[0]._class_names = all_class_names
        self._all_trees[0]._feature_names = feature_names
        self._all_trees[0]._samples_class_label_dict = class_for_sample_dict
        self._all_trees[0]._training_data_dict  =  feature_values_for_samples_dict
        self._all_trees[0]._features_and_values_dict    =  features_and_values_dict
        self._all_trees[0]._features_and_unique_values_dict    =  features_and_unique_values_dict
        self._all_trees[0]._numeric_features_valuerange_dict = numeric_features_valuerange_dict
        self._all_trees[0]._feature_values_how_many_uniques_dict = feature_values_how_many_uniques_dict
        self._all_training_data = feature_values_for_samples_dict
        self._all_sample_names = sorted(feature_values_for_samples_dict.keys(), key = lambda x: sample_index(x))
        if self._debug1:
            print("\n\n=======================   For the base tree   ==================================\n")
            print("\nAll class names: " + str(self._all_trees[0]._class_names))
            print("\nEach sample data record:")
            for item in sorted(self._all_trees[0]._training_data_dict.items(), key = lambda x: sample_index(x[0]) ):
                print(item[0]  + "  =>  "  + str(item[1]))
            print("\nclass label for each data sample:")
            for item in sorted(self._all_trees[0]._samples_class_label_dict.items(), key=lambda x: sample_index(x[0])):
                print(item[0]  + "  =>  "  + str(item[1]))
            print("\nfeatures and the values taken by them:")
            for item in sorted(self._all_trees[0]._features_and_values_dict.items()):
                print(item[0]  + "  =>  "  + str(item[1]))
            print("\nnumeric features and their ranges:")
            for item in sorted(self._all_trees[0]._numeric_features_valuerange_dict.items()):
                print(item[0]  + "  =>  "  + str(item[1]))
            print("\nunique values for the features:")
            for item in sorted(self._all_trees[0]._features_and_unique_values_dict.items()):
                print(item[0]  + "  =>  "  + str(item[1]))
            print("\nnumber of unique values in each feature:")
            for item in sorted(self._all_trees[0]._feature_values_how_many_uniques_dict.items()):
                print(item[0]  + "  =>  "  + str(item[1]))

    def show_training_data_for_base_tree(self):
        self._all_trees[0].show_training_data()            

    def calculate_first_order_probabilities_and_class_priors(self):            
        self._all_trees[0].calculate_first_order_probabilities()
        self._all_trees[0].calculate_class_priors()
        self._sample_selection_probs[0] =  {sample : 1.0/len(self._all_sample_names) for sample in self._all_sample_names}
        
    def construct_base_decision_tree(self):            
        self._root_nodes[0] = self._all_trees[0].construct_decision_tree_classifier()

    def display_base_decision_tree(self):
        self._root_nodes[0].display_decision_tree("     ")

    def classify_with_base_decision_tree(self, test_sample):
        return self._all_trees[0].classify(self._root_nodes[0], test_sample)

    def get_all_class_names(self):
        return self._all_trees[0]._class_names

    def construct_cascade_of_trees(self):
        self._training_samples[0] = self._all_sample_names
        self._misclassified_samples[0] = self.evaluate_one_stage_of_cascade(self._all_trees[0], self._root_nodes[0])
        if self._stagedebug:
            self.show_class_labels_for_misclassified_samples_in_stage(0)
            print("\nSamples misclassified by base classifier: %s" % str(self._misclassified_samples[0]))
            print("\nNumber of misclassified samples: %d" % len(self._misclassified_samples[0]))
        misclassification_error_rate = sum([self._sample_selection_probs[0][x] for x in self._misclassified_samples[0]])
        if self._stagedebug:
            print("\nMisclassification_error_rate for base classifier: %g" % misclassification_error_rate)
        self._trust_factors[0] = 0.5 * math.log((1-misclassification_error_rate)/misclassification_error_rate)
        if self._stagedebug:
            print("\nBase class trust factor: %s" % str(self._trust_factors[0]))
        for stage_index in range(1,self._how_many_stages):
            if self._stagedebug:
                print("\n\n==========================Constructing stage indexed %d=========================\n" % stage_index)
            self._sample_selection_probs[stage_index] =  \
                {sample : self._sample_selection_probs[stage_index - 1][sample] * math.exp(-1.0 * self._trust_factors[stage_index - 1] * (-1.0 if sample in self._misclassified_samples[stage_index - 1] else 1.0)) for sample in self._all_sample_names} 
            normalizer = sum(self._sample_selection_probs[stage_index].values())
            if self._stagedebug:
                print("\nThe normalizer is: ", normalizer)
            self._sample_selection_probs[stage_index].update((sample,prob/normalizer) for sample,prob in
                                                              self._sample_selection_probs[stage_index].items())
            prob_distribution = sorted(self._sample_selection_probs[stage_index].items(), key=lambda x: x[1], reverse=True)
            if self._stagedebug:
                print("\nProbability distribution: %s" % str([(sample_index(x), "%.3f"%y) for x, y in prob_distribution]))
            training_samples_this_stage = []
            sum_of_probs = 0.0
            for sample in [x[0] for x in prob_distribution]:
                sum_of_probs += self._sample_selection_probs[stage_index][sample]
                if sum_of_probs > 0.5:
                    break
                else:
                    training_samples_this_stage.append(sample)
            self._training_samples[stage_index] = sorted(training_samples_this_stage, key=lambda x: sample_index(x))
            if self._stagedebug:
                print("\nTraining samples this stage: %s" % str(self._training_samples[stage_index]))
                print("\nNumber of training samples this stage %d" % len(self._training_samples[stage_index]))
            training_samples_selection_check = set(self._misclassified_samples[stage_index-1]).intersection(set(self._training_samples[stage_index]))
            if self._stagedebug:            
                print("\nTraining samples in the misclassified set: %s" %
                                  str(sorted(training_samples_selection_check, key=lambda x: sample_index(x))))
                print("\nNumber_of_miscalssified_samples_in_training_set: %d" % len(training_samples_selection_check))
            dt_this_stage = DecisionTree('boostingmode')            
            training_data_this_stage = { x : self._all_training_data[x] for x in self._training_samples[stage_index]}
            dt_this_stage._training_data_dict = training_data_this_stage
            dt_this_stage._class_names = self._all_trees[0]._class_names
            dt_this_stage._feature_names = self._all_trees[0]._feature_names
            dt_this_stage._entropy_threshold = self._all_trees[0]._entropy_threshold
            dt_this_stage._max_depth_desired = self._all_trees[0]._max_depth_desired
            dt_this_stage._symbolic_to_numeric_cardinality_threshold =   \
                                                self._all_trees[0]._symbolic_to_numeric_cardinality_threshold
            dt_this_stage._samples_class_label_dict = \
                     {sample_name : self._all_trees[0]._samples_class_label_dict[sample_name] 
                                                     for sample_name in dt_this_stage._training_data_dict.keys()}
            dt_this_stage._features_and_values_dict = \
                                 {feature : [] for feature in self._all_trees[0]._features_and_values_dict}
            pattern = r'(\S+)\s*=\s*(\S+)'        
            for item in sorted(dt_this_stage._training_data_dict.items(), key = lambda x: sample_index(x[0])):
                for feature_and_value in item[1]:
                    m = re.search(pattern, feature_and_value)
                    feature,value = m.group(1),m.group(2)
                    if value != 'NA':
                        dt_this_stage._features_and_values_dict[feature].append(convert(value))
            dt_this_stage._features_and_unique_values_dict = {feature : 
                                      sorted(list(set(dt_this_stage._features_and_values_dict[feature]))) for 
                                                            feature in dt_this_stage._features_and_values_dict}
            dt_this_stage._numeric_features_valuerange_dict = {feature : [] 
                                         for feature in self._all_trees[0]._numeric_features_valuerange_dict}
            dt_this_stage._numeric_features_valuerange_dict = {feature : 
                                   [min(dt_this_stage._features_and_unique_values_dict[feature]), 
                                       max(dt_this_stage._features_and_unique_values_dict[feature])] 
                                           for feature in self._all_trees[0]._numeric_features_valuerange_dict}
            if self._stagedebug:
                print("\n\nPrinting features and their values in the training set:\n")
                for item in sorted(dt_this_stage._features_and_values_dict.items()):
                    print(item[0]  + "  =>  "  + str(item[1]))
                print("\n\nPrinting unique values for features:\n")
                for item in sorted(dt_this_stage._features_and_unique_values_dict.items()):
                    print(item[0]  + "  =>  "  + str(item[1]))
                print("\n\nPrinting unique value ranges for features:\n")
                for item in sorted(dt_this_stage._numeric_features_valuerange_dict.items()):
                    print(item[0]  + "  =>  "  + str(item[1]))
            dt_this_stage._feature_values_how_many_uniques_dict = {feature : [] 
                                       for  feature in self._all_trees[0]._features_and_unique_values_dict}
            dt_this_stage._feature_values_how_many_uniques_dict = {feature :
                 len(dt_this_stage._features_and_unique_values_dict[feature]) 
                                        for  feature in self._all_trees[0]._features_and_unique_values_dict}
#            if stagedebug: dt_this_stage._debug2 = 1
            dt_this_stage.calculate_first_order_probabilities()
            dt_this_stage.calculate_class_priors()
            if self._stagedebug:
                print("\n\n>>>>>>>Done with the initialization of the tree for this stage<<<<<<<<<<\n")
            root_node_this_stage = dt_this_stage.construct_decision_tree_classifier()
            if self._stagedebug:
                root_node_this_stage.display_decision_tree("     ")
            self._all_trees[stage_index] = dt_this_stage
            self._root_nodes[stage_index] = root_node_this_stage
            self._misclassified_samples[stage_index] = \
              self.evaluate_one_stage_of_cascade(self._all_trees[stage_index], self._root_nodes[stage_index])
            if self._stagedebug:           
                print("\nSamples misclassified by this stage classifier: %s" %
                                                           str(self._misclassified_samples[stage_index]))
                print("\nNumber of misclassified samples: %d" % len(self._misclassified_samples[stage_index]))
                self.show_class_labels_for_misclassified_samples_in_stage(stage_index)
            misclassification_error_rate = sum( [self._sample_selection_probs[stage_index][x]
                                                              for x in self._misclassified_samples[stage_index]] )
            if self._stagedebug:           
                print("\nMisclassification_error_rate: %g" % misclassification_error_rate)
            self._trust_factors[stage_index] = \
                                   0.5 * math.log((1-misclassification_error_rate)/misclassification_error_rate)
            if self._stagedebug:
                print("\nThis stage trust factor: %g" % self._trust_factors[stage_index])

    def show_class_labels_for_misclassified_samples_in_stage(self, stage_index):
        classes_for_misclassified_samples = []
        just_class_labels = []
        for sample in self._misclassified_samples[stage_index]:
            true_class_label_for_sample = self._all_trees[0]._samples_class_label_dict[sample]            
            classes_for_misclassified_samples.append( "%s => %s"% (sample,true_class_label_for_sample))
            just_class_labels.append(true_class_label_for_sample) 
        print("\nShowing class labels for samples misclassified by stage %d:" % stage_index)
        print("\nClass labels for samples: %s" % str(classes_for_misclassified_samples))
        class_names_unique = set(just_class_labels)            
        print("\nClass names (unique) for misclassified samples: %s" % str(list(class_names_unique)))
        print("\nFinished displaying class labels for samples misclassified by stage %d\n\n" % stage_index)

    def display_decision_trees_for_different_stages(self):
        print("\nDisplaying the decisions trees for all stages:")
        for i in range(self._how_many_stages):
            print("\n\n=============================   For stage %d   ==================================\n" % i)
            self._root_nodes[i].display_decision_tree("     ")
        print("\n==================================================================================\n\n")

    def classify_with_boosting(self, test_sample):
        self._classifications = list(map(lambda x: self._all_trees[x].classify(self._root_nodes[x], test_sample),
                                   range(self._how_many_stages)))

    def display_classification_results_for_each_stage(self):         
        classifications = self._classifications
        if classifications is None:
            raise Exception('''You must first call "classify_with_boosting()" before invoking "display_classification_results_for_each_stage()" ''')
        solution_paths = list(map(lambda x: x['solution_path'], classifications))
        for i in range(self._how_many_stages):
            print("\n\n=============================   For stage %d   ==================================\n" % i)
            classification = classifications[i]
            del classification['solution_path']
            which_classes = list( classification.keys() )
            which_classes = sorted(which_classes, key=lambda x: classification[x], reverse=True)
            print("\nClassification:\n")
            print("Classifier trust: %g\n" % self._trust_factors[i])
            print("     "  + str.ljust("class name", 30) + "probability")
            print("     ----------                    -----------")
            for which_class in which_classes:
                if which_class is not 'solution_path':
                    print("     "  + str.ljust(which_class, 30) +  str(classification[which_class]))
            print("\nSolution path in the decision tree: " + str(solution_paths[i]))
            print("\nNumber of nodes created: " + str(self._root_nodes[i].how_many_nodes()))
        print("\n=================================================================================\n\n")

    def trust_weighted_majority_vote_classifier(self):
        classifications = self._classifications
        if classifications is None:
            raise Exception('''You must first call "classify_with_boosting()" before invoking "trust_weighted_majority_vote_classifier()" ''')
        decision_classes = {class_label : 0 for class_label in self._all_trees[0]._class_names}
        for i in range(self._how_many_stages):
            classification = classifications[i]
            if 'solution_path' in classification:
                del classification['solution_path']
            sorted_classes = sorted(list(classification.keys()), key=lambda x: classification[x], reverse=True)  
            decision_classes[sorted_classes[0]] += self._trust_factors[i]
        sorted_by_weighted_votes_decision_classes = \
                         sorted(list(decision_classes.keys()), key=lambda x: decision_classes[x], reverse=True)
        self._trust_weighted_decision_classes = sorted(decision_classes.items(), key=lambda x: x[1], reverse=True)
        return sorted_by_weighted_votes_decision_classes[0]

    def display_trust_weighted_decision_for_test_sample(self):
        if self._trust_weighted_decision_classes is None:
            raise Exception('''You must first call "trust_weighted_majority_vote_classifier()" before invoking "display_trust_weighted_decision_for_test_sample()"''')
        print("\nClassifier labels for the test sample sorted by trust weights (The greater the trust weight, the greater the confidence we have in the classification label):\n")        
        for item in self._trust_weighted_decision_classes:
            print("%s   =>    %s" % (item[0], item[1]))
    
    def evaluate_one_stage_of_cascade(self, trainingDT, root_node):
        misclassified_samples = []
        for test_sample_name in self._all_sample_names:
            test_sample_data = self._all_trees[0]._training_data_dict[test_sample_name]
            if self._stagedebug: 
                print("original data in test sample:", str(test_sample_data))  
            test_sample_data = [x for x in test_sample_data if not x.endswith('=NA')]
            if self._stagedebug: 
                print("data in test sample:", str(test_sample_data))  
            classification = trainingDT.classify(root_node, test_sample_data)
            solution_path = classification['solution_path']                                  
            del classification['solution_path']                                              
            which_classes = list( classification.keys() )                                    
            which_classes = sorted(which_classes, key=lambda x: classification[x], reverse=True)
            most_likely_class_label = which_classes[0]
            if self._stagedebug:
                print("\nClassification:\n")                                                     
                print("     "  + str.ljust("class name", 30) + "probability")                    
                print("     ----------                    -----------")                          
                for which_class in which_classes:                                                
                    if which_class is not 'solution_path':                                       
                        print("     "  + str.ljust(which_class, 30) +  str(classification[which_class])) 
                print("\nSolution path in the decision tree: " + str(solution_path))             
                print("\nNumber of nodes created: " + str(root_node.how_many_nodes()))
            true_class_label_for_test_sample = self._all_trees[0]._samples_class_label_dict[test_sample_name]
            if self._stagedebug:
                print("%s:   true_class: %s    estimated_class: %s\n" % \
                         (test_sample_name, true_class_label_for_test_sample, most_likely_class_label))
            if true_class_label_for_test_sample != most_likely_class_label:
                misclassified_samples.append(test_sample_name)
        return sorted(misclassified_samples, key = lambda x: sample_index(x))

