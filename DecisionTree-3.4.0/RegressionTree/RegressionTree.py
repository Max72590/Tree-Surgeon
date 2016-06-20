from DecisionTree import DecisionTree
import re
import operator
import math
import functools
import sys
import os
import os.path
import string
import numpy
import pylab
from mpl_toolkits.mplot3d import Axes3D

def convert(value):
    try:
        answer = float(value)
        return answer
    except:
        return value

def deep_copy_array(array_in):
    '''
    Meant only for an array of scalars (no nesting):
    '''
    array_out = []
    for i in range(len(array_in)):
        array_out.append( array_in[i] )
    return array_out

def minimum(arr):
    '''
    Returns simultaneously the minimum value and its positional index in an
    array. [Could also have used min() and index() defined for Python's
    sequence types.]
    '''
    minval,index = None,None
    for i in range(0, len(arr)):  
        if minval is None or arr[i] < minval:
            index = i
            minval = arr[i]
    return minval,index

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
    
class RegressionTree(DecisionTree):
    def __init__(self, *args, **kwargs ):
        if kwargs and args:
            raise SyntaxError( '''DecisionTree constructor can only be called with keyword arguments '''
                               '''for the following keywords: training_datafile, entropy_threshold, '''   
                               '''max_depth_desired, csv_class_column_index, number_of_histogram_bins, '''
                               '''symbolic_to_numeric_cardinality_threshold, csv_columns_for_features, '''
                               '''number_of_histogram_bins, dependent_variable_column, predictor_columns, '''
                               '''mse_threshold, need_data_normalization, debug1, debug2, and debug3'''
                               '''debug1_r,debug2_r,debug3_r''') 
        allowed_keys = 'training_datafile','entropy_threshold','max_depth_desired','csv_class_column_index',\
                       'symbolic_to_numeric_cardinality_threshold','csv_columns_for_features',\
                       'number_of_histogram_bins','dependent_variable_column','predictor_columns',\
                       'mse_threshold', 'need_data_normalization','debug1','debug2','debug3',\
                       'debug1_r','debug2_r','debug3_r'
        keywords_used = kwargs.keys()
        for keyword in keywords_used:
            if keyword not in allowed_keys:
                raise SyntaxError(keyword + ":  Wrong keyword used --- check spelling") 
        dependent_variable_column=predictor_columns=mse_threshold=need_data_normalization=None
        debug1_r=debug2_r=debug3_r=None
        if kwargs and not args:
            if 'dependent_variable_column' in kwargs: 
                               dependent_variable_column = kwargs.pop('dependent_variable_column')
            if 'predictor_columns' in kwargs:  predictor_columns = kwargs.pop('predictor_columns')
            if 'mse_threshold' in kwargs:              mse_threshold = kwargs.pop('mse_threshold')
            if 'need_data_normalization' in kwargs:    \
                                   need_data_normalization = kwargs.pop('need_data_normalization')
            if 'debug1_r' in kwargs  :  debug1_r = kwargs.pop('debug1_r')
            if 'debug2_r' in kwargs  :  debug2_r = kwargs.pop('debug2_r')
            if 'debug3_r' in kwargs  :  debug3_r = kwargs.pop('debug3_r')
            DecisionTree.__init__(self, **kwargs)    
        if dependent_variable_column:
            self._dependent_variable_column                 =      dependent_variable_column
        else:
            self._dependent_variable_column                 =      None
        if predictor_columns:
            self._predictor_columns                         =      predictor_columns
        else:
            self._predictor_columns                         =      None
        if mse_threshold:
            self._mse_threshold                             =      mse_threshold
        else:
            self._mse_threshold                             =      0.01
        if need_data_normalization:
            self._need_data_normalization                   =      1
        else:
            self._need_data_normalization                   =      0
        self._dependent_var                                 =      None
        self._dependent_var_values                          =      None
        self._samples_dependent_var_val_dict                =      {}
        self._root_node                                     =      None
        if debug1_r:
            self._debug1_r                                  =      debug1_r
        else:
            self._debug1_r                                  =      0
        if debug2_r:
            self._debug2_r                                  =      debug2_r
        else:
            self._debug2_r                                  =      0
        if debug3_r:
            self._debug3_r                                  =      debug3_r
        else:
            self._debug3_r                                  =      0
        self._sampling_points_for_dependent_var             =      []
        self._output_for_plots                              =      {}
        self._output_for_surface_plots                      =      {}

    def get_training_data_for_regression(self):
        if not self._training_datafile.endswith('.csv'): 
            TypeError("Aborted. get_training_data_from_csv() is only for CSV files")
        dependent_var_values = []
        all_record_ids_with_dependent_var_values = {}
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
                dependent_var_values.append(convert(parts[self._dependent_variable_column]))
                all_record_ids_with_dependent_var_values[parts[0].strip('"')] = parts[self._dependent_variable_column]
                if i%10000 == 0:
                    print('.'),
                    sys.stdout.flush()
                sys.stdout = sys.__stdout__
            f.close() 
        self._how_many_total_training_samples = i   # i is less by 1 from total num of records; but that's okay
        if self._debug1_r:
            print("\n\nTotal number of training samples: %d\n" % self._how_many_total_training_samples)
        all_feature_names = firstline.rstrip().split(',')[1:]
        dependent_var_column_heading = all_feature_names[self._dependent_variable_column - 1]
        feature_names = [all_feature_names[i-1] for i in self._predictor_columns]
        dependent_var_value_for_sample_dict = { "sample_" + key : 
          dependent_var_column_heading + "=" + data_dict[key][self._dependent_variable_column - 1] for key in data_dict}
        sample_names = ["sample_" + key for key in data_dict]
        feature_values_for_samples_dict = {"sample_" + key :         
                  list(map(operator.add, list(map(operator.add, feature_names, "=" * len(feature_names))), 
           [str(convert(data_dict[key][i-1])) for i in self._predictor_columns])) for key in data_dict}
        features_and_values_dict = {all_feature_names[i-1] :
            [convert(data_dict[key][i-1]) for key in data_dict] for i in self._predictor_columns}
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
        if self._debug1_r:
            print("\nDependent var values: " + str(dependent_var_values))
            print("\nEach sample data record:")
            for item in sorted(feature_values_for_samples_dict.items(), key = lambda x: sample_index(x[0]) ):
                print(item[0]  + "  =>  "  + str(item[1]))
            print("\ndependent var value for each data sample:")
            for item in sorted(dependent_var_value_for_sample_dict.items(), key=lambda x: sample_index(x[0])):
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
        self._XMatrix  =  None
        self._YVector  =  None
        self._dependent_var = dependent_var_column_heading
        self._dependent_var_values = dependent_var_values
        self._feature_names = feature_names
        self._samples_dependent_var_val_dict    =  dependent_var_value_for_sample_dict
        self._training_data_dict          =  feature_values_for_samples_dict
        self._features_and_values_dict    =  features_and_values_dict
        self._features_and_unique_values_dict    =  features_and_unique_values_dict
        self._numeric_features_valuerange_dict = numeric_features_valuerange_dict
        self._feature_values_how_many_uniques_dict = feature_values_how_many_uniques_dict
        self.calculate_first_order_probabilities()

    def construct_XMatrix_and_YVector_all_data(self):
        matrix_rows_as_lists = [ [ convert(elem[elem.find('=')+1:]) for elem in item[1] ] 
               for item in sorted(self._training_data_dict.items(), key = lambda x: sample_index(x[0]) ) ]
        for row in matrix_rows_as_lists: row.append(1)
        if self._debug1_r:
            print(matrix_rows_as_lists)
        XMatrix = numpy.matrix(matrix_rows_as_lists)
        if self._debug1_r:
            print("\n\nX matrix: ")
            print(XMatrix)
        self._XMatrix = XMatrix
        dependent_var_values = [ convert(item[1][item[1].find('=')+1:]) for item in 
             sorted(self._samples_dependent_var_val_dict.items(), key = lambda x: sample_index(x[0]) ) ]
        if self._debug1_r:
            print(dependent_var_values)
        YVector = numpy.matrix(dependent_var_values).T
        if self._debug1_r:
            print(YVector)
        self._YVector = YVector
        return XMatrix, YVector

    def estimate_regression_coefficients(self, X_matrix, Y_vector, display=None):
        (nrows, ncols) = X_matrix.shape
        if self._debug2_r:
            print("nrows=", nrows, "   ncols=", ncols)    
        X = numpy.zeros(shape=(nrows,ncols))
        X = numpy.asmatrix(X)
        numpy.copyto(X, X_matrix)
        y = numpy.zeros(shape=(nrows,1))
        y = numpy.asmatrix(y)
        numpy.copyto(y, Y_vector)
        if self._need_data_normalization:
            means = []
            stds  = []
            X_normalized = X
            for col in range(ncols-1):
                mean = numpy.mean(X[:,col])
                std = numpy.std(X[:,col])
                X_normalized[:,col] = (X_normalized[:,col] - mean) / std
                means.append(mean)
                stds.append(std)
            X = X_normalized
        beta0 = (X.T * X) ** (-1) * X.T * y
        if self._debug2_r:
            print("\n\ninitial value for beta:")
            print(beta0)
        gamma = 0.1
        iterate_again_flag = 1
        while True:
            if not iterate_again_flag: 
                break
            else:
                gamma *= 0.1
                beta0 = 0.99 * beta0
            if self._debug2_r:
                print("\n\n======== starting iterations with gamma= %0.9f ===========\n\n" % gamma)
            beta = beta0
            beta_old = numpy.zeros((ncols, 1))
            beta_old = numpy.asmatrix(beta_old)
            error_old = numpy.linalg.norm(y - (X * beta))
            error = None
            for iteration in range(1500):
                if beta is None: beta = beta0
                numpy.copyto(beta_old, beta)
                beta =  beta_old + 2 * gamma * X.T * (y - (X * beta))
                error = numpy.linalg.norm(y - (X * beta))
                if error > error_old: 
                    if numpy.linalg.norm(beta - beta_old) < (0.00001 * numpy.linalg.norm(beta_old)): 
                        iterate_again_flag = 0
                        break
                    else:
                        break
                if self._debug2_r:
                    print("\n\niteration: %d   gamma: %0.9f   current error: %0.9f" %(iteration,gamma,error))
                    print("\nnew beta:")
                    print(beta)
                if numpy.linalg.norm(beta - beta_old) < (0.00001 * numpy.linalg.norm(beta_old)): 
                    if self._debug2_r:
                        print("iterations used: %d with gamma: %0.9f" % (iteration,gamma))
                    iterate_again_flag = 0
                    break
                error_old = error
        if self._debug2_r:
            print("\n\nfinal beta:")
            print(beta)         
        predictions = X * beta
        error_distribution = y - (X * beta)
        error_distribution = numpy.asarray(error_distribution)
        squared_error = (error_distribution ** 2).T
        error = sum(squared_error.tolist()[0]) / X.shape[0]
        if display:
            if ncols == 2:
                self._output_for_plots[len(self._output_for_plots) + 1] = [X[:,0],predictions]
            elif ncols == 3:
                xvalues = X[:,0:2].tolist()
                yvalues = predictions[:,0].flatten().tolist()[0]
                self._output_for_surface_plots[len(self._output_for_surface_plots) + 1] = [xvalues,yvalues]
            else:
                # no display when the overall dimensionality of the data exceeds 3
                pass
        return error,beta

##-------------------------------  Construct Regression Tree  ------------------------------------

    def construct_regression_tree(self):
        '''
        At the root node, you do ordinary linear regression for the entire dataset so that you
        can later compare the linear regression fit with the results obtained through the 
        regression tree.  Subsequently, you call the recursive_descent() method to construct
        the tree.
        '''
        print("\nConstructing regression tree...")
        root_node = RTNode(None, None, None, [], self, 'root')
        XMatrix,YVector = self.construct_XMatrix_and_YVector_all_data()
        error,beta = self.estimate_regression_coefficients(XMatrix, YVector,1) 
        root_node.set_node_XMatrix(XMatrix)
        root_node.set_node_YVector(YVector)
        root_node.set_node_error(error)
        root_node.set_node_beta(beta)
        root_node.set_num_data_points(XMatrix.shape[0])
        print("\nerror at root: ", error)
        print("\nbeta at root: ", beta)
        self._root_node = root_node
        if self._max_depth_desired > 0:
            self.recursive_descent(root_node)
        return root_node        

    def recursive_descent(self, node):
        '''
        We first look for a feature, along with its partitioning point, that yields the 
        largest reduction in MSE compared to the MSE at the parent node.  This feature and
        its partitioning point are then used to create two child nodes in the tree.
        '''
        if self._debug3_r:
            print("\n==================== ENTERING RECURSIVE DESCENT ==========================")
        node_serial_number = node.get_serial_num()
        features_and_values_or_thresholds_on_branch = node.get_branch_features_and_values_or_thresholds()
        copy_of_path_attributes = deep_copy_array(features_and_values_or_thresholds_on_branch)
        if len(features_and_values_or_thresholds_on_branch) > 0:
            error,beta,XMatrix,YVector = self._error_for_given_sequence_of_features_and_values_or_thresholds(\
                                                                                  copy_of_path_attributes)
            node.set_node_XMatrix(XMatrix)
            node.set_node_YVector(YVector)
            node.set_node_error(error)
            node.set_node_beta(beta)
            node.set_num_data_points(XMatrix.shape[0])
            print("\nNODE SERIAL NUMBER: "+ str(node_serial_number))
            print("\nFeatures and values or thresholds on branch: "+str(features_and_values_or_thresholds_on_branch))
            if error <= self._mse_threshold: 
                return
        best_feature,best_minmax_error_at_partitioning_point,best_feature_partitioning_point = \
                                                         self.best_feature_calculator(copy_of_path_attributes)
        if best_feature_partitioning_point is None:
            return
        print("\nBest feature found: %s" % best_feature)
        print("Best feature partitioning_point: %0.9f" % best_feature_partitioning_point)
        print("Best minmax error at partitioning point: %0.9f" % best_minmax_error_at_partitioning_point)
        node.set_feature(best_feature)
        if self._debug2_r: 
            node.display_node() 
        if self._max_depth_desired is not None and len(features_and_values_or_thresholds_on_branch) >= self._max_depth_desired:
           return
        if best_minmax_error_at_partitioning_point > self._mse_threshold:
            extended_branch_features_and_values_or_thresholds_for_lessthan_child = \
                                  deep_copy_array(features_and_values_or_thresholds_on_branch)
            extended_branch_features_and_values_or_thresholds_for_greaterthan_child  = \
                                  deep_copy_array(features_and_values_or_thresholds_on_branch)
            feature_threshold_combo_for_less_than = \
                                   "".join([best_feature,"<",str(convert(best_feature_partitioning_point))])
            feature_threshold_combo_for_greater_than = \
                                   "".join([best_feature,">",str(convert(best_feature_partitioning_point))])
            extended_branch_features_and_values_or_thresholds_for_lessthan_child.append( 
                                                              feature_threshold_combo_for_less_than)
            extended_branch_features_and_values_or_thresholds_for_greaterthan_child.append( 
                                                           feature_threshold_combo_for_greater_than)

            left_child_node = RTNode(None, None, None, 
                          extended_branch_features_and_values_or_thresholds_for_lessthan_child, self)
            node.add_child_link(left_child_node)
            self.recursive_descent(left_child_node)
            right_child_node = RTNode(None, None, None, 
                          extended_branch_features_and_values_or_thresholds_for_greaterthan_child, self)
            node.add_child_link(right_child_node)
            self.recursive_descent(right_child_node)

    def best_feature_calculator(self, features_and_values_or_thresholds_on_branch):
        '''
        This is the heart of the regression tree constructor.  Its main job is to figure
        out the best feature to use for partitioning the training data samples at the
        current node.  The partitioning criterion is that the largest of the MSE's in 
        the two partitions should be smaller than the error associated with the parent
        node.
        '''
        print("\n\nfeatures_and_values_or_thresholds_on_branch: %s" % str(features_and_values_or_thresholds_on_branch))
        if len(features_and_values_or_thresholds_on_branch) == 0:
            best_partition_point_for_feature_dict = {feature : None for feature in self._feature_names}
            best_minmax_error_for_feature_dict = {feature : None for feature in self._feature_names}
            for i in range(len(self._feature_names)):
                feature_name = self._feature_names[i]
                values = self._sampling_points_for_numeric_feature_dict[feature_name]
                partitioning_errors = []
                partitioning_error_dict = {}
                for value in values[10:-10]:
                    feature_and_less_than_value_string = "".join([feature_name,"<",str(convert(value))]) 
                    feature_and_greater_than_value_string = "".join([feature_name,">",str(convert(value))])
                    for_left_child = for_right_child = None
                    if features_and_values_or_thresholds_on_branch:
                        for_left_child = deep_copy_array(features_and_values_or_thresholds_on_branch)
                        for_left_child.append(feature_and_less_than_value_string)
                        for_right_child = deep_copy_array(features_and_values_or_thresholds_on_branch)
                        for_right_child.append(feature_and_greater_than_value_string)
                    else:
                        for_left_child = [feature_and_less_than_value_string]
                        for_right_child = [feature_and_greater_than_value_string]
                    error1,beta1,XMatrix1,YVector1 = \
                             self._error_for_given_sequence_of_features_and_values_or_thresholds(for_left_child)
                    error2,beta2,XMatrix2,YVector2 = \
                             self._error_for_given_sequence_of_features_and_values_or_thresholds(for_right_child)
                    partitioning_error = max(error1, error2)
                    partitioning_errors.append(partitioning_error)
                    partitioning_error_dict[partitioning_error] = value
                min_max_error_for_feature = min(partitioning_errors)
                best_partition_point_for_feature_dict[feature_name] = partitioning_error_dict[min_max_error_for_feature]
                best_minmax_error_for_feature_dict[feature_name] = min_max_error_for_feature
            best_feature_name = None
            best_feature_paritioning_point = None
            best_minmax_error_at_partitioning_point = None
            for feature in best_minmax_error_for_feature_dict:
                if best_minmax_error_at_partitioning_point is None:
                    best_minmax_error_at_partitioning_point = best_minmax_error_for_feature_dict[feature]
                    best_feature_name = feature
                elif best_minmax_error_at_partitioning_point > best_minmax_error_for_feature_dict[feature]:
                    best_minmax_error_at_partitioning_point = best_minmax_error_for_feature_dict[feature]
                    best_feature_name = feature               
            best_feature_partitioning_point =  best_partition_point_for_feature_dict[best_feature_name]
            return best_feature_name,best_minmax_error_at_partitioning_point,best_feature_partitioning_point
        else:    
            pattern1 = r'(.+)=(.+)'
            pattern2 = r'(.+)<(.+)'
            pattern3 = r'(.+)>(.+)'
            true_numeric_types = []
            symbolc_types = []
            true_numeric_types_feature_names = []
            symbolic_types_feature_names = []
            for item in features_and_values_or_thresholds_on_branch:
                if re.search(pattern2, item):
                    true_numeric_types.append(item)
                    m = re.search(pattern2, item)
                    feature,value = m.group(1),m.group(2)
                    true_numeric_types_feature_names.append(feature)
                elif re.search(pattern3, item): 
                    true_numeric_types.append(item)
                    m = re.search(pattern3, item)
                    feature,value = m.group(1),m.group(2)
                    true_numeric_types_feature_names.append(feature)
                else:
                    symbolic_types.append(item) 
                    m = re.search(pattern1, item)
                    feature,value = m.group(1),m.group(2)
                    symbolic_types_feature_names.append(feature)
            true_numeric_types_feature_names = list(set(true_numeric_types_feature_names))
            symbolic_types_feature_names = list(set(symbolic_types_feature_names))
            bounded_intervals_numeric_types = self.find_bounded_intervals_for_numeric_features(true_numeric_types)
            # Calculate the upper and the lower bounds to be used when searching for the best
            # threshold for each of the numeric features that are in play at the current node:
            upperbound = {feature : None for feature in true_numeric_types_feature_names}
            lowerbound = {feature : None for feature in true_numeric_types_feature_names}
            for item in bounded_intervals_numeric_types:
                if item[1] == '>':
                    lowerbound[item[0]] = float(item[2])
                else:
                    upperbound[item[0]] = float(item[2])
            best_partition_point_for_feature_dict = {feature : None for feature in self._feature_names}
            best_minmax_error_for_feature_dict = {feature : None for feature in self._feature_names}
            for i in range(len(self._feature_names)):
                feature_name = self._feature_names[i]
                values = self._sampling_points_for_numeric_feature_dict[feature_name]
                newvalues = []
                if feature_name in true_numeric_types_feature_names:
                    if upperbound[feature_name] is not None and lowerbound[feature_name] is not None and \
                                                   lowerbound[feature_name] >= upperbound[feature_name]:
                        continue
                    elif upperbound[feature_name] is not None and lowerbound[feature_name] is not None and \
                                                           lowerbound[feature_name] < upperbound[feature_name]:
                        newvalues = [x for x in values if lowerbound[feature_name] < x <= upperbound[feature_name]]
                    elif upperbound[feature_name] is not None:
                        newvalues = [x for x in values if x <= upperbound[feature_name]]
                    elif lowerbound[feature_name] is not None:
                        newvalues = [x for x in values if x > lowerbound[feature_name]]
                    else:
                        raise Exception("Error in bound specifications in best feature calculator")
                else:
                    newvalues = values
                if len(newvalues) < 30:
                    continue
                partitioning_errors = []
                partitioning_error_dict = {}
                for value in newvalues[10:-10]:
                    feature_and_less_than_value_string = "".join([feature_name,"<",str(convert(value))]) 
                    feature_and_greater_than_value_string = "".join([feature_name,">",str(convert(value))])
                    for_left_child = for_right_child = None
                    if features_and_values_or_thresholds_on_branch:
                        for_left_child = deep_copy_array(features_and_values_or_thresholds_on_branch)
                        for_left_child.append(feature_and_less_than_value_string)
                        for_right_child = deep_copy_array(features_and_values_or_thresholds_on_branch)
                        for_right_child.append(feature_and_greater_than_value_string)
                    else:
                        for_left_child = [feature_and_less_than_value_string]
                        for_right_child = [feature_and_greater_than_value_string]
                    error1,beta1,XMatrix1,YVector1 = \
                     self._error_for_given_sequence_of_features_and_values_or_thresholds(for_left_child)
                    error2,beta2,XMatrix2,YVector2 = \
                     self._error_for_given_sequence_of_features_and_values_or_thresholds(for_right_child)
                    partitioning_error = max(error1, error2)
                    partitioning_errors.append(partitioning_error)
                    partitioning_error_dict[partitioning_error] = value
                min_max_error_for_feature = min(partitioning_errors)
                best_partition_point_for_feature_dict[feature_name] = partitioning_error_dict[min_max_error_for_feature]
                best_minmax_error_for_feature_dict[feature_name] = min_max_error_for_feature
            best_feature_name = None
            best_feature_partitioning_point = None
            best_minmax_error_at_partitioning_point = None
            for feature in best_minmax_error_for_feature_dict:
                if best_minmax_error_at_partitioning_point is None:
                    best_minmax_error_at_partitioning_point = best_minmax_error_for_feature_dict[feature]
                    best_feature_name = feature
                elif best_minmax_error_at_partitioning_point > best_minmax_error_for_feature_dict[feature]:
                    best_minmax_error_at_paritioning_point = best_minmax_error_for_feature_dict[feature]
                    best_feature_name = feature               
            best_feature_partitioning_point =  best_partition_point_for_feature_dict[best_feature_name]
            return best_feature_name,best_minmax_error_at_partitioning_point,best_feature_partitioning_point

    def _error_for_given_sequence_of_features_and_values_or_thresholds(self, array_of_features_and_values_or_thresholds):
        '''
        This method requires that all truly numeric types only be expressed as '<' or '>'
        constructs in the array of branch features and thresholds
        '''
        if len(array_of_features_and_values_or_thresholds) == 0: 
            XMatrix, YVector = self.construct_XMatrix_and_YVector_all_data()
            errors,beta = self.estimate_regression_coefficients(XMatrix,YVector)
            return errors,beta,XMatrix,YVector
        pattern1 = r'(.+)=(.+)'
        pattern2 = r'(.+)<(.+)'
        pattern3 = r'(.+)>(.+)'
        true_numeric_types = []        
        true_numeric_types_feature_names = []
        symbolic_types = []
        symbolic_types_feature_names = []
        for item in array_of_features_and_values_or_thresholds:
            if re.search(pattern2, item):
                true_numeric_types.append(item)
                m = re.search(pattern2, item)
                feature,value = m.group(1),m.group(2)
                true_numeric_types_feature_names.append(feature)
            elif re.search(pattern3, item): 
                true_numeric_types.append(item)
                m = re.search(pattern3, item)
                feature,value = m.group(1),m.group(2)
                true_numeric_types_feature_names.append(feature)
            else:
                symbolic_types.append(item) 
                m = re.search(pattern1, item)
                feature,value = m.group(1),m.group(2)
                symbolic_types_feature_names.append(feature)
        true_numeric_types_feature_names = list(set(true_numeric_types_feature_names))
        symbolic_types_feature_names = list(set(symbolic_types_feature_names))
        bounded_intervals_numeric_types = self.find_bounded_intervals_for_numeric_features(true_numeric_types)
        # Calculate the upper and the lower bounds to be used when searching for the best
        # threshold for each of the numeric features that are in play at the current node:
        upperbound = {feature : None for feature in true_numeric_types_feature_names}
        lowerbound = {feature : None for feature in true_numeric_types_feature_names}
        for item in bounded_intervals_numeric_types:
            if item[1] == '>':
                lowerbound[item[0]] = float(item[2])
            else:
                upperbound[item[0]] = float(item[2])
        training_samples_at_node = set()
        for feature_name in true_numeric_types_feature_names:
            if lowerbound[feature_name] is not None and upperbound[feature_name] is not None \
                          and upperbound[feature_name] <= lowerbound[feature_name]:
                return None,None,None,None
            elif lowerbound[feature_name] is not None and upperbound[feature_name] is not None:
                for sample in self._training_data_dict:
                    feature_val_pairs = self._training_data_dict[sample]
                    for feature_and_val in feature_val_pairs:
                        value_for_feature = float( feature_and_val[feature_and_val.find('=')+1:] )
                        feature_involved = feature_and_val[:feature_and_val.find('=')]
                        if feature_name == feature_involved and \
                                  lowerbound[feature_name] < value_for_feature <= upperbound[feature_name]:
                            training_samples_at_node.add(sample)
                            break
            elif upperbound[feature_name] is not None and lowerbound[feature_name] is None:
                for sample in self._training_data_dict:
                    feature_val_pairs = self._training_data_dict[sample]
                    for feature_and_val in feature_val_pairs:
                        value_for_feature = float( feature_and_val[feature_and_val.find('=')+1:] )
                        feature_involved = feature_and_val[:feature_and_val.find('=')]
                        if feature_name == feature_involved and value_for_feature <= upperbound[feature_name]:
                            training_samples_at_node.add(sample)
                            break
            elif lowerbound[feature_name] is not None and upperbound[feature_name] is None:
                for sample in self._training_data_dict:
                    feature_val_pairs = self._training_data_dict[sample]
                    for feature_and_val in feature_val_pairs:
                        value_for_feature = float( feature_and_val[feature_and_val.find('=')+1:] )
                        feature_involved = feature_and_val[:feature_and_val.find('=')]
                        if feature_name == feature_involved and value_for_feature > lowerbound[feature_name]:
                            training_samples_at_node.add(sample)
                            break
            else:
                raise SyntaxError("Ill formatted call to the '_error_for_given_sequence_...' method")
        for feature_and_value in symbolic_types:
            if re.search(pattern1, feature_and_value):      
                m = re.search(pattern1, feature_and_value)
                feature,value = m.group(1),m.group(2)
                for sample in self._training_data_dict:
                    feature_val_pairs = self._training_data_dict[sample]
                    for feature_and_val in feature_val_pairs:
                        feature_in_sample = feature_and_val[:feature_and_val.find('=')]
                        value_in_sample = float( feature_and_val[feature_and_val.find('=')+1:] )
                        if (feature == feature_in_sample) and (value == value_in_sample):
                            training_samples_at_node.add(sample)
                            break
        matrix_rows_as_lists = [ [ convert(elem[elem.find('=')+1:]) for elem in item[1] ] 
               for item in sorted(self._training_data_dict.items(), key = lambda x: sample_index(x[0]) )
               if item[0] in list(training_samples_at_node) ]
        for row in matrix_rows_as_lists: row.append(1)
        if self._debug1_r:
            print(matrix_rows_as_lists)
        XMatrix = numpy.matrix(matrix_rows_as_lists)
        if self._debug2_r:
            print("\n\nX matrix at node: ")
            print(XMatrix)
        dependent_var_values_at_node = [ convert(item[1][item[1].find('=')+1:])  
             for item in 
             sorted(self._samples_dependent_var_val_dict.items(), key = lambda x: sample_index(x[0]) ) 
             if item[0] in list(training_samples_at_node) ]
        if self._debug2_r:
            print("\n\ndependent var values at node: ")
            print(dependent_var_values_at_node)
        YVector = numpy.matrix(dependent_var_values_at_node).T
        if self._debug2_r:
            print("\n\ndependent var vector node: ")
            print(YVector)
        error,beta = self.estimate_regression_coefficients(XMatrix, YVector)
        if self._debug2_r:
            print("\n\nbeta at node: ", beta)
            print("\n\nerror distribution at node: ", errors)
        return error,beta,XMatrix,YVector

#-----------------------------    Predict with Regression Tree   ------------------------------

    def predictions_for_all_data_used_for_regression_estimation(self, root_node):
        predicted_values = {}
        leafnode_for_values = {}
        ncols = self._XMatrix.shape[1]
        if ncols == 2:
            for sample in self._training_data_dict:
                pattern = r'(\S+)\s*=\s*(\S+)'
                m = re.search(pattern, self._training_data_dict[sample][0])
                feature,value = m.group(1),m.group(2)
                newvalue = convert(value)
                new_feature_and_value = feature + " = " + str(newvalue)
                answer = self.prediction_for_single_data_point(root_node, [new_feature_and_value])
                predicted_values[newvalue] = answer['prediction'][0]
                leafnode_for_values[hash(round(predicted_values[newvalue],9))] = answer['solution_path'][-1]
            leaf_nodes_used = set(leafnode_for_values.values())
            for leaf in leaf_nodes_used:
                xvalues,yvalues = [],[]
                for x in sorted(predicted_values):
                    if leaf == leafnode_for_values[hash(round(predicted_values[x],9))]:
                        xvalues.append(x)
                        yvalues.append(predicted_values[x])
                self._output_for_plots[len(self._output_for_plots) + 1] = [xvalues,yvalues]
        elif ncols == 3:
            for sample in self._training_data_dict:
                pattern = r'(\S+)\s*=\s*(\S+)'
                features_and_vals = []
                newvalues = []
                for feature_and_val in self._training_data_dict[sample]:
                    m = re.search(pattern, feature_and_val)
                    feature,value = m.group(1),m.group(2)
                    newvalue = convert(value)
                    newvalues.append(newvalue)
                    new_feature_and_value = feature + " = " + str(newvalue)
                    features_and_vals.append(new_feature_and_value)
                answer = self.prediction_for_single_data_point(root_node, features_and_vals)
                predicted_values[tuple(newvalues)] = answer['prediction'][0]
                leafnode_for_values[predicted_values[tuple(newvalues)]] = answer['solution_path'][-1]
            leaf_nodes_used = set(leafnode_for_values.values())
            for leaf in leaf_nodes_used:
                xvalues,yvalues = [],[]
                for x in predicted_values:
                    if leaf == leafnode_for_values[predicted_values[x]]:
                        xvalues.append(list(x))
                        yvalues.append(predicted_values[x])
                self._output_for_surface_plots[len(self._output_for_surface_plots) + 1] = [xvalues,yvalues]
        else:
            sys.exit("arrived here for more than 3D data")

    def bulk_predictions_for_data_in_a_csv_file(self, root_node, filename, columns):
        if not filename.endswith('.csv'): 
            TypeError("Aborted. bulk_predictions_for_data_in_a_csv_file() is only for CSV files")
        out_file_name = filename[:filename.find(".")] + "_output.csv"
        outfile = open(out_file_name, 'w')
        with open(filename) as f:
            for i,line in enumerate(f):
                record = cleanup_csv(line)
                if i == 0:
                    firstline = record
                    fieldnames = record.rstrip().split(r',')
                    continue
                feature_vals = list(map(lambda x: convert(x), record.rstrip().split(r',')))
                features_and_vals = []
                for col_index in columns:
                    features_and_vals.append(fieldnames[col_index] + "=" + str(feature_vals[col_index]))
                answer = self.prediction_for_single_data_point(root_node, features_and_vals)
                outfile.write(record + "         =>        " + str(answer['prediction'][0]) + "\n")
        f.close()
        outfile.close()

    def mse_for_tree_regression_for_all_training_samples(self, root_node):
        predicted_values = {}
        dependent_var_values = {}
        leafnode_for_values = {}
        total_error = 0.0
        samples_at_leafnode = {}
        for sample in self._training_data_dict:
            features_and_values = []
            pattern = r'(\S+)\s*=\s*(\S+)'
            for feature_and_val in self._training_data_dict[sample]:
                m = re.search(pattern, feature_and_val)
                feature,value = m.group(1),m.group(2)
                if feature in self._feature_names:
                    newvalue = convert(value)
                    new_feature_and_value = feature + " = " + str(newvalue)
                    features_and_values.append(new_feature_and_value)
            answer = self.prediction_for_single_data_point(root_node, features_and_values)
            predicted_values[newvalue] = answer['prediction'][0]
            m = re.search(pattern, self._samples_dependent_var_val_dict[sample])
            dep_feature,dep_value = m.group(1),m.group(2)
            dependent_var_values[newvalue] = convert(dep_value)
            leafnode_for_sample = answer['solution_path'][-1]
            leafnode_for_values[hash(round(predicted_values[newvalue],9))] = leafnode_for_sample
            error_for_sample = abs(predicted_values[newvalue] - dependent_var_values[newvalue]) ** 2
            total_error += error_for_sample
            if leafnode_for_sample in samples_at_leafnode:
                samples_at_leafnode[leafnode_for_sample].append(sample)
            else:
                samples_at_leafnode[leafnode_for_sample] = [sample]
        leafnodes_used = set(leafnode_for_values.values())
        errors_at_leafnode = {leafnode : 0.0 for leafnode in leafnodes_used}
        for x in sorted(predicted_values):
            for leaf in leafnodes_used:
                if leaf == leafnode_for_values[hash(round(predicted_values[x],9))]:
                    errors_at_leafnode[leaf] += abs(predicted_values[x] - dependent_var_values[x])** 2
        print("\n\nTotal MSE per sample with tree regression: %0.9f" % (total_error / len(self._training_data_dict)))
        for leafnode in leafnodes_used:
            print("MSE per sample at leafnode %d: %0.9f" % (leafnode, errors_at_leafnode[leafnode] / len(samples_at_leafnode[leafnode])))
        error_with_linear_regression = self._root_node.get_node_error()
        print("For comparision, the MSE per sample error with Linear Regression: %0.9f" % error_with_linear_regression)

    def prediction_for_single_data_point(self, root_node, features_and_values):
        '''
        Calculated the predicted value for the dependent variable from a given value for all
        the predictor variables.
        '''
        if not self._check_names_used(features_and_values):
            raise SyntaxError("Error in the names you have used for features and/or values") 
        new_features_and_values = []
        pattern = r'(\S+)\s*=\s*(\S+)'
        for feature_and_value in features_and_values:
            m = re.search(pattern, feature_and_value)
            feature,value = m.group(1),m.group(2)
            newvalue = convert(value)
            new_features_and_values.append(feature + " = " + str(newvalue))
        features_and_values = new_features_and_values       
        answer = {}
        answer['solution_path'] = []
        prediction = self.recursive_descent_for_prediction(root_node,features_and_values, answer)
        answer['solution_path'].reverse()
        return answer

    def recursive_descent_for_prediction(self, node, feature_and_values, answer):
        children = node.get_children()
        if len(children) == 0:
            leaf_node_prediction = node.node_prediction_from_features_and_values(feature_and_values) 
            answer['prediction'] = leaf_node_prediction
            answer['solution_path'].append(node.get_serial_num())
            return answer
        feature_tested_at_node = node.get_feature()
        if self._debug3: print("\nCLRD1 Feature tested at node for prediction: " + feature_tested_at_node)
        value_for_feature = None
        path_found = None
        pattern = r'(\S+)\s*=\s*(\S+)'
        feature,value = None,None
        for feature_and_value in feature_and_values:
            m = re.search(pattern, feature_and_value)
            feature,value = m.group(1),m.group(2)
            if feature == feature_tested_at_node:
                value_for_feature = convert(value)
        if value_for_feature is None:
            leaf_node_prediction = node.node_prediction_from_features_and_values(feature_and_values) 
            answer['prediction'] = leaf_node_prediction
            answer['solution_path'].append(node.get_serial_num())
            return answer
        for child in children:
            branch_features_and_values = child.get_branch_features_and_values_or_thresholds()
            last_feature_and_value_on_branch = branch_features_and_values[-1] 
            pattern1 = r'(.+)<(.+)'
            pattern2 = r'(.+)>(.+)'
            if re.search(pattern1, last_feature_and_value_on_branch):
                m = re.search(pattern1, last_feature_and_value_on_branch)
                feature,threshold = m.group(1),m.group(2)
                if value_for_feature <= float(threshold):
                    path_found = True
                    answer = self.recursive_descent_for_prediction(child, feature_and_values, answer)
                    answer['solution_path'].append(node.get_serial_num())
                    break
            if re.search(pattern2, last_feature_and_value_on_branch):
                m = re.search(pattern2, last_feature_and_value_on_branch)
                feature,threshold = m.group(1),m.group(2)
                if value_for_feature > float(threshold):
                    path_found = True
                    answer = self.recursive_descent_for_prediction(child, feature_and_values, answer)
                    answer['solution_path'].append(node.get_serial_num())
                    break
        if path_found: return answer 

    #--------------------------------------  Utility Methods   ----------------------------------------
    def display_all_plots(self):
        ncols = self._XMatrix.shape[1]
        if os.path.isfile("regression_plots.png"):
            os.remove("regression_plots.png")
        if ncols == 2:
            pylab.scatter(self._XMatrix[:,0], self._YVector[:,0], marker='o', c='b')
            for plot in sorted(self._output_for_plots):
                clr = 'r' if plot == 1 else 'b'
                pylab.plot(self._output_for_plots[plot][0], self._output_for_plots[plot][1], linewidth=4.0, color=clr)
            pylab.title('red: linear regression        blue: tree regression')
            pylab.show()
            fig = pylab.figure()
            ax = fig.add_subplot(111)
            ax.scatter(self._XMatrix[:,0], self._YVector[:,0], marker='o', c='b')
            for plot in sorted(self._output_for_plots):
                clr = 'r' if plot == 1 else 'b'
                ax.plot(self._output_for_plots[plot][0], self._output_for_plots[plot][1], linewidth=4.0, color=clr)
            ax.set_title('red: linear regression        blue: tree regression')
            fig.savefig("regression_plots.png")
        elif ncols == 3:
            fig = pylab.figure()
            ax = fig.add_subplot(111, projection='3d')    
            X1coords = self._XMatrix[:,0].flatten().tolist()[0]
            X2coords = self._XMatrix[:,1].flatten().tolist()[0]
            yvalues = self._YVector[:,0].flatten().tolist()[0]
            ax.scatter(X1coords, X2coords, yvalues, c='k', marker='x')          
            for plot in sorted(self._output_for_surface_plots):
                xvalues,yvalues = self._output_for_surface_plots[plot]
                X1coords = [item[0] for item in xvalues]
                X2coords = [item[1] for item in xvalues]
                clr = 'r' if plot == 1 else 'b'
                ax.scatter(X1coords, X2coords, yvalues, c=clr, marker='o')          
                ax.set_xlabel(self._feature_names[0])
                ax.set_ylabel(self._feature_names[1])
                ax.set_zlabel(self._dependent_var)
            pylab.title('plot: ' + str(plot) + '   red: linear regression        blue: tree regression')
            pylab.show()
            fig.savefig("regression_plots.png")
        else:
           sys.exit("plotting not available for more than 3D data")

#-------------------------------------------  Class RTNode   ------------------------------------------
class RTNode(object):
    '''
    The nodes of a regression tree are instances of this class:
    '''
    def __init__(self, feature, error, beta, branch_features_and_values_or_thresholds, rt, root_or_not=None):
        if root_or_not == 'root':
             rt.nodes_created = -1 
             rt.class_names = None
        self._rt                          = rt
        self._serial_number               = self.get_next_serial_num()
        self._feature                     = feature
        self._error                       = error
        self._beta                        = beta
        self._branch_features_and_values_or_thresholds = branch_features_and_values_or_thresholds
        self._num_data_points             = None
        self._XMatrix                     = None
        self._YVector                     = None
        self._linked_to                   = []

    def node_prediction_from_features_and_values(self, feature_and_values):
        ncols = self._XMatrix.shape[1]
        pattern = r'(\S+)\s*=\s*(\S+)'
        feature,value = None,None
        Xlist = []
        for feature_name in self._rt._feature_names:
            for feature_and_value in feature_and_values:
                m = re.search(pattern, feature_and_value)
                feature,value = m.group(1),m.group(2)
                if feature_name == feature:
                    Xlist.append(convert(value))
        Xlist.append(1)
        dataXMatrix = numpy.matrix(Xlist)
        prediction = dataXMatrix * self.get_node_beta()
        return  prediction.flatten().tolist()[0]            

    def node_prediction_from_data_as_matrix(self, dataXMatrix):
        prediction = dataXMatrix * self.get_node_beta()
        return prediction.flatten().tolist()[0]

    def node_prediction_from_data_as_list(self, data_as_list):
        ncols = self._XMatrix.shape[1]
        if len(data_as_list) != ncols - 1:
            sys.exit("wrong number of elements in data list")
        data_as_one_row_matrix = numpy.matrix(data_as_list.append(1.0))
        prediction = data_as_one_row_matrix * self.get_node_beta()
        return prediction.flatten().tolist()[0]

    def get_num_data_points(self):
        return self._num_data_points

    def set_num_data_points(self, how_many):
        self._num_data_points = how_many

    def how_many_nodes(self):
        return self._rt.nodes_created + 1

    def set_node_XMatrix(self, xmatrix):
        self._XMatrix = xmatrix

    def get_node_XMatrix(self):
        return self._XMatrix

    def set_node_YVector(self, yvector):
        self._YVector = yvector

    def get_node_YVector(self):
        return self._YVector

    def set_node_error(self, error):
        self._error = error

    def get_node_error(self):
        return self._error

    def set_node_beta(self, beta):
        self._beta = beta

    def get_node_beta(self):
        return self._beta

    def get_next_serial_num(self):
        self._rt.nodes_created += 1
        return self._rt.nodes_created

    def get_serial_num(self):
        return self._serial_number

    def get_feature(self):
        '''
        Returns the feature test at the current node
        '''
        return self._feature

    def set_feature(self, feature):
        self._feature = feature

    def get_branch_features_and_values_or_thresholds(self):
        return self._branch_features_and_values_or_thresholds

    def get_children(self):
        return self._linked_to

    def add_child_link(self, new_node):
        self._linked_to.append(new_node)                  

    def delete_all_links(self):
        self._linked_to = None

    def display_node(self):
        feature_at_node = self.get_feature() or " "
        serial_num = self.get_serial_num()
        branch_features_and_values_or_thresholds = self.get_branch_features_and_values_or_thresholds()
        print("\n\nNODE " + str(serial_num) + 
              ":\n   Branch features and values to this node: " + 
              str(branch_features_and_values_or_thresholds) + 
              "\n   Best feature test at current node: " + feature_at_node + "\n\n")
        self._rt.estimate_regression_coefficients(self.get_node_XMatrix(), self.get_node_YVector(), 1)

    def display_regression_tree(self, offset):
        serial_num = self.get_serial_num()
        if len(self.get_children()) > 0:
            feature_at_node = self.get_feature() or " "
            branch_features_and_values_or_thresholds = self.get_branch_features_and_values_or_thresholds()
            print("NODE " + str(serial_num) + ":  " + offset +  "BRANCH TESTS TO NODE: " +
                                           str(branch_features_and_values_or_thresholds))
            second_line_offset = offset + " " * (8 + len(str(serial_num)))
            print(second_line_offset +  "Decision Feature: " + feature_at_node)
            offset += "   "
            for child in self.get_children():
                child.display_regression_tree(offset)
        else:
            branch_features_and_values_or_thresholds = self.get_branch_features_and_values_or_thresholds()
            print("NODE " + str(serial_num) + ":  " + offset +  "BRANCH TESTS TO LEAF NODE: " +
                                           str(branch_features_and_values_or_thresholds))
            second_line_offset = offset + " " * (8 + len(str(serial_num)))
