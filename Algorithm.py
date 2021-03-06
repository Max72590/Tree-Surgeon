#!/usr/bin/env python
from __future__ import print_function
import sys
sys.path.insert(0, '/home/maximino/Desktop/TFG/files/DecisionTree-3.4.0/DecisionTree')
from sys import argv
from math import exp, expm1
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import Imputer
from scipy.io import arff
from flask import Flask, render_template, redirect, url_for,request
from flask import make_response
from sklearn.cross_validation import train_test_split
#app = Flask(__name__)
from app import app
from copy import deepcopy

import ast
import os
import subprocess
import pandas as pd
import numpy as np
import arff
import math
import DecisionTree 
import json


#------------------------------------------------#
#			    Global Variables				 #
#------------------------------------------------#
#  

#Data = pd.DataFrame()
#names = []
#decisionTree = {}
#algorithmName = ""
#index = 0
#jsonTreeDict = None

#------------------------------------------------#
#			    Exception class					 #
#------------------------------------------------#

class MyError(Exception):
	def __init__(self,message):
		self.message = message
	def __str__(self):
		return repr(self.message)

#------------------------------------------------#
#					Functions					 #
#------------------------------------------------#


# Read data from a path to the file and checks the filetype to choose the reader.
def read_data(path, filetype):
	if (filetype == "csv"):
		data =  pd.read_csv(path) 
		return data
	if (filetype == "arff"):
		dataset = arff.load(open(path, 'r'))
		data = intopandas(dataset)
		return data
	else:
		raise MyError("Tipo de archivo no reconocido.")	 
# Transforms the arff data into pandas format
def intopandas(dataset):
	cols =[]
	data = np.array(dataset['data'])
	numberOfAttributes = len(dataset['attributes'])
	attributes = []
	for a in dataset['attributes']:
		attributes.append(a[0])
	df= pd.DataFrame(dataset['data'],index=range(len(dataset['data'])),columns=attributes)
	return df

# Draws into a png the Decision Tree
def print_tree(tree,names,fname,dotname):
	with open(dotname, 'w') as f:
		export_graphviz(tree, out_file=f,feature_names=names)
	command = ["dot", "-Tpng", dotname, "-o", fname]
	try:
	    subprocess.check_call(command)
	except:
	    exit("Could not run dot, ie graphviz, to produce visualization")
 
# Marks the non-numeric attributes that are going to be encoded into integers
def mark_for_replacement(attributes):
	count = 0
	marked = []
	for e in attributes:
		values = e[1]
		all_digits = True
		if (values != "REAL" ):
			for v in values:							
					if not v.isdigit() or v!='.' or v!=',':
						all_digits = False
		if not all_digits:
			marked.append(attributes[count])
		count = count + 1
	return marked

# Same as mark_for_replacement but for csv (must change so it only 1 function for both)
def mark_for_replacement_csv(data,names):
	count = 0
	marked = []
	for n in names:
		values = data[n].unique()
		all_digits = True
		for v in values:							
				if (isinstance(v,basestring)):
					all_digits = False
		if not all_digits:
			marked.append(n)
		count = count + 1
	return marked

# Splits the data (some datasets are too large and the time required to draw the tree is
# too much, this function can be omitted if desired)
def partitionate(data):
	#amount =int( math.floor(len(data.index) * 0.15))
	data = data.head(5000)
	return data

#------------------------------------------------#
#				Tree processing Functions		 #
#------------------------------------------------#

# Encodes the non-numeric values into integers of the target and adds it to the data
def encode_target(name, data):
	targets = data[name].unique()
	mapping_to_int = {name: n for n, name in enumerate(targets)}

	data["Target"] = data[name].replace(mapping_to_int)
	return data

# Encodes the non-numeric values into integers
def encode(attribute, data):				
	attname= attribute
	targets = data[attribute].unique()
	mapping_to_int = {name: n for n, name in enumerate(targets)}
	data[attname] = data[attname].replace(mapping_to_int)
	return data

# Builds the tree and sends it to be drawn by print_tree
def build_tree(data,names,fname,dotname):
	imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
	Y = data["Target"]
	X = data[names]
	imp.fit(X)
	X = imp.transform(X)
	clf = DecisionTreeClassifier(min_samples_split=20, random_state=99)
	clf = clf.fit(X,Y)
	print_tree(clf,names,fname,dotname)

# Auxliar function to check if the parameter is and argument or not.
def check_if_number(parameter):
	try:
		float(parameter)
		return True
	except ValueError:
		return False

# Get the data that follows this rules. The parameter "data" MUST be a copy of the program's input data.
def reprocess_data(data, rules):
	for r in rules:
		rule = r[0]
		ruleval = r[1]
		if (check_if_number(ruleval)):
			if (r[2]):
				data = data[data[rule] <= float(ruleval)]
			else:
				data = data[data[rule] > float(ruleval)]
		else:
			data = data[data[rule] == ruleval]
	return data

# Used in the modification function, generates a decision tree from the data recieved.
def create_new_tree(data2,names,fname):
	Y = data2["Target"]
	X = data2[names]
	clf = DecisionTreeClassifier(min_samples_split=20, random_state=99)
	clf = clf.fit(X,Y)
	print_tree(clf,names,fname,"data1.dot")

# Creates a new tree from the data provided as an argument.
def create_new_C45_tree(data2,name):
	names2 = list(data2.columns[:len(data2.columns)])
	lindex = list(range(1,len(names2)+1))
	indexval = 0
	for i in lindex:
		if (names2[i-1] == name):
			indexval = i
			del lindex[i-1]
			break
	if (indexval == 0):
		print ("Error, target name not correct.")
		return([],[],"",[])
	data2.to_csv('inputData.csv')
	dt = DecisionTree.DecisionTree( training_datafile = "inputData.csv",
			                        csv_class_column_index = indexval,
			                        csv_columns_for_features = lindex,
			                        entropy_threshold = 0.01,
			                        max_depth_desired = 3,
			                      )
	dt.get_training_data()
	dt.calculate_first_order_probabilities()
	dt.calculate_class_priors()
	root_node = dt.construct_decision_tree_classifier()
	arbol =  get_C45_dict(root_node,-1,names2[indexval-1])
	return arbol
	
# Currently not being used. Keep it as a reminder of how sklearn trees work.
def get_node_and_parents(tree, feature_names, nodedest):		
		left      = tree.tree_.children_left[0]
		right     = tree.tree_.children_right[0]
		threshold = tree.tree_.threshold[0]
		features  = [feature_names[i] for i in tree.tree_.feature]
		value = tree.tree_.value[0]
		parents = [[0,threshold,value]]
		found = False		

		def recurse(left, right, threshold, value, node,parents,found):       
				if nodeid == nodedest:
					left = tree.tree_.children_left[node]
					right = tree.tree_.children_right[node]       
					threshold = tree.tree_.threshold[node]
					value = tree.tree_.value[node]
					node1 = [left,threshold, value]
					node2 = [right,threshold, value]
					if left != -1: 			
						parents.append(node1)				                     
						res = recurse (left, right, threshold, value,left,parents,found)						 
						if (found):
							return res
						parents.append(node2)
						res = recurse (left, right, threshold, value,right,parents,found)						
						if (found):
							return res
				else:					
					found = True
					node1 = [left,threshold, value]
					parents.append(node1)
					return parents
		return recurse(left, right, threshold, value, 0,parents,found)

# Gets all the rules from the root node to the objective node.
def return_node_rules(nodeid, arbol,algorithmName):
	parent = arbol[nodeid][1]
	actual = nodeid
	list_r = []
	while (parent != -1):
		if (algorithmName == "CART"):
			name_thres = arbol[parent][3].split(' & ')
			branch_side =  arbol[actual][0]		
			elem = (name_thres[0],branch_side)
			list_r.append(elem)
		elif (algorithmName == "C4.5"):
			name_thres = arbol[actual][3].split(' & ')	
			elem = (name_thres[0])
			list_r.append(elem)		
		actual = parent
		parent = arbol[parent][1]				
	return list_r

# Special function used in the modification of a node to get the information
# of the root node.
def return_root_rules(arbol,algorithmName):
	list_r = []
	if (algorithmName == "CART"):
		name_thres = arbol[0][3].split(' & ')
		branch_side =  arbol[0][0]		
		elem = (name_thres[0],branch_side)
		list_r.append(elem)
	elif (algorithmName == "C4.5"):
		if (len(arbol[0][2]) > 0):
			name_thres = arbol[1][3].split(' & ')	# Arbol[0][0] returns the string "root", the name of the splitting attribute it is in the childs.
			elem = (name_thres[0])
			list_r.append(elem)
	return list_r

# Gets the values of the rules extracted from the nodes.
def rules_values(rules,algorithmName):
	rlist = []	
	if (algorithmName == "CART"):
		for r in rules:
			direction = r[1]
			aux = r[0].replace("label=",'')
			aux = aux.replace(" <= ",' ')
			rule = aux.split(' ')
			rule[0] = rule[0].replace('"','')
			if direction == 'd':
				elem = (rule[0],rule[1],False) 				
			else:
				elem = (rule[0],rule[1],True)
			rlist.append(elem)
	elif (algorithmName == "C4.5"):
		for r in rules:
			r = r.replace('label="','')
			if ('<' in r):
				r = r.replace('<', ';; < ;;')
			elif ('='in r):
				r = r.replace('=', ';; = ;;')
			else: r = r.replace('>', ';; > ;;')
			aux = r.split(';;')		
			if '>' in aux[1]:
				elem = (aux[0],aux[2],False)
			elif  '=' in aux[1]:
				elem = (aux[0],aux[2],False)
			else:
				elem = (aux[0],aux[2],True)
			rlist.append(elem)				
	return rlist

# Process the data.dot file from the CART algorithm and transforms the tree into a dictionary
# out of it.
def process(fdataname):
	with open(fdataname) as f:
		datadot = f.readlines()
		arbol = {}
		datadot.pop(0)
		datadot.pop(0)
		datadot.pop()
		for row in datadot:
			rowaux = row.split('[')
			firstelement = rowaux[0].replace(' ','')
			if "->" in rowaux[0]:
				if(firstelement != "0"):				
					vals = rowaux[0].replace(' ','')				
					vals = vals.split("->")
					val1 = int(vals[0])
					val2 = vals[1].split(';')
					arbol[int(val2[0])][1] = val1   
					arbol[val1][2].append(int(val2[0]))				
					if (len(arbol[val1][2]) > 1):
						arbol[int(val2[0])][0] = 'd'
				else:
					val2 = rowaux[0].split("->")
					arbol[0][2].append(val2[1])	
			else:
				if len(rowaux) < 3:					#Check this asap
					aux = rowaux[1].split("=")
					rowaux.append(aux[-1])
				valores = rowaux[1]+' '+rowaux[2]				
				valores = valores.replace("\\",' & ')
				valores = valores.replace(']','')
				valores = valores.replace(';','')
				if (firstelement == "0"):										
					arbol[0] = ['r',-1,[],valores]											
				elif (firstelement != "0"):	
					arbol[int(firstelement)] = ['i',-1,[],valores]
		return arbol

# Auxiliar function used in C4.5 to make the name take the same format used 
# throughout the project.
def processName(node):
	s = ""
	childs = node.get_children()
	if len(childs) > 0:
		name = childs[0].get_branch_features_and_values_or_thresholds()[-1]
		if ("<" in name):
			name = name.split("<")
		else:
			name = name.split(">")
		s = s + name[0]
	else:
		s = s + "gini"
	return s

# Processes the output of the C4.5 algorithm and turns it into a dictionary.
def get_C45_dict(node,parent,targetName):
	tree = {}
	tree[node.get_serial_num()] = [processName(node),parent]	# The first element is the name of the atribute in which we partitionate our data.
	node_childs = []									# Used later for the json that represents our tree.
	if (node.get_children > 0):
		childs = node.get_children()
		aux = tree.copy()
		for c in childs:
			node_childs.append(c.get_serial_num())
			aux.update(get_C45_dict(c,node.get_serial_num(),targetName))
		tree = aux
	tree[node.get_serial_num()].append(node_childs)
	s = 'label="'
	if (len(node.get_branch_features_and_values_or_thresholds())>0):
		s = s + node.get_branch_features_and_values_or_thresholds()[-1]
	else:
		s = s + "root"
	s = s+' & entropy = '+str(node.get_node_entropy())+' & nvalue = '
	first = True
	class_probabilities = node.get_class_probabilities()
	class_names = node.get_class_names();
	print_class_probabilities = ["%.3f" % x for x in class_probabilities]
	print_class_probabilities_with_class = [class_names[i] + " => " + print_class_probabilities[i] for i in range(len(class_names))]
	s = s + str(print_class_probabilities_with_class) + '"'
	tree[node.get_serial_num()].append(s)
	return tree

# Reads the training data and returns it.
def getData(url,targetName,alName):
	
	filetype = url.split('.')
	filetype = filetype[-1]
	Data = read_data(url,filetype)
	names = list(Data.columns[:len(Data.columns)]) 
	if (alName == "CART"):
		names.remove(targetName)	
		Data = encode_target(targetName,Data)
		marked = []
		if filetype == "arff":
			dataset = arff.load(open(url, 'r'))
			elemlist = mark_for_replacement(dataset['attributes'])
			marked = []
			for e in elemlist:
				marked.append(e[0])
		else:
			marked = mark_for_replacement_csv(Data,names)
		for m in marked:
			if (m != targetName):
				Data = encode(m,Data)	
	return [Data,names]

# Updates a rule with the new info.
def updateRule(nodeInfo,newnodeinfo):
	oldRule = nodeInfo.split(" & ")
	oldRule = oldRule[0]
	print (newnodeinfo)
	actualInfo = newnodeinfo.split(" & ")
	actualInfo.pop(0)
	newRule = oldRule
	for r in actualInfo:
		newRule = newRule  + " & " + r 
	return newRule

# Gets the value and atribute in which a node
# splits the data into.
def get_value_and_atribute(string):	
	string = string.replace('label="',"")
	print(string)
	if ("=" in string):		
		string = string.split("=")	
	elif (">" in string):
		string = string.split(">")
	elif ("<" in string):
		string = string.split("<")
	for e in string:
		e=e.replace(" ","")
	print(string)
	value = string[-1]
	atribute = string[0]	
	return [atribute,value]

# Returns the entropy of a node. Currently not being used due to
# the changes made to the function that joins nodes by percentage
# of cases that falls in them.
def get_entropy(string):
	elements= string.split(" & ")
	return float((elements[1].replace("entropy = ","")))

# This funcion us used in case tht a new tree is generated but the data does
# not have all the classes in which can be classified, maybe because on entire 
# class has been classified leaving only the others in another branch. When
# generating a new tree but the data doesn't have all the classes the algorithm
# does not acknowledge the ones that have been left out. Therefore this function
# "fills" the blanks by putting the ones left out but with a total of cases classified
# of zero (which is correct).
def update_classname(origdataclasseslist,classeslist,tree,nodeid):
	if (len(tree[nodeid][2]) > 0):
		for c in tree[nodeid][2]:
			update_classname(origdataclasseslist,classeslist,tree,c)
	valueslist = tree[nodeid][3].split(" & nvalue = ")
	
	elems_remaining = valueslist[0]
	valueslist = valueslist[-1].split(",")
	#print("orig: "+" "+str(origdataclasseslist))
	#print("class: "+" "+str(classeslist))
	auxlist = []
	for e in valueslist:
		if "&" in e:
			auxlist.append(e.split(" & ")[0])
			auxlist.append(e.split(" & ")[1])
		else:
			auxlist.append(e)
	valueslist = auxlist
	s = ""
	cont = 0
	first = True
	for c in range(0,len(origdataclasseslist)):
		if (first):
			s = s + " & nvalue =  "
			first = False
		else:
			s = s + " , "
		if (origdataclasseslist[c] in classeslist):
			if (origdataclasseslist[c] != classeslist[cont]):
				s = s + str(origdataclasseslist[c]) + " :  0"
			else:
				s = s + str(classeslist[cont]) + " : "+ str(valueslist[cont])
				cont = cont + 1
		else:
			s = s + str(origdataclasseslist[c]) + " :  0"
	tree[nodeid][3] = elems_remaining + s
	
# Calcuates the accuracy, recall and precision given the confusion matrix, the index of the 
# objective class and the total count of instances in the node.	
def get_recall_prec_acc(confusionMatrix, classindex,nodeClass, objectiveclassindex,datacount):
	count = 0
	acumulate_accuracy = 0
	acumulate_recall = 0
	acumulate_precision = 0
	truepositive = confusionMatrix[classindex][1][objectiveclassindex]	
	for c in confusionMatrix:
		acumulate_accuracy  = acumulate_accuracy  + confusionMatrix[c][1][count]
		count = count + 1			
		for listelem in range(0,len(confusionMatrix[c][1])):			
			if confusionMatrix[c][0] == nodeClass:			
				acumulate_precision = acumulate_precision + confusionMatrix[c][1][listelem]
				if listelem == objectiveclassindex :
					acumulate_recall = acumulate_recall + confusionMatrix[c][1][listelem]
			elif listelem == objectiveclassindex :
				acumulate_recall = acumulate_recall + confusionMatrix[c][1][listelem]
	if datacount <= 0:
		#print("div por zero - accuracy")
		return [0,0,0]
	accuracy = (float(acumulate_accuracy) / float(datacount)) * 100
	if acumulate_recall <= 0 :
		#print("div por zero - recall")
		return [0,0,0]
	recall = (float(truepositive) / float(acumulate_recall)) * 100
	if acumulate_precision <= 0:
		#print("div por zero - precision")
		return [0,0,0]
	precision = (float(truepositive) / float(acumulate_precision)) * 100	
	return [math.ceil(accuracy*100)/100,math.ceil(recall*100)/100,math.ceil(precision*100)/100]

# Orders the first list elements like the second list.
def order_list1_like_list2(list1,list2,algorithmName):
	reslist = []
	list1names =[]
	for cv in list1:
		if (algorithmName == "CART"):
			cv = cv.split(" :  ")
			cv[0] = cv[0].replace(" ","")
		elif (algorithmName == "C4.5"):
			cv = cv.split(" => ")
			cv[0] = cv[0].split("=")[-1]
		list1names.append(cv[0])
	for l2 in range(0,len(list2)):
		if (str(list2[l2]) in list1names):	
			for l1 in (list1):	
				val = l1
				if (algorithmName == "CART"):
					l1 = l1.split(" :  ")
					l1[0] = l1[0].replace(" ","")
				elif (algorithmName == "C4.5"):
					l1 = l1.split(" => ")
					l1[0] = l1[0].split("=")[-1]
				if (l1[0] == str(list2[l2])):
					reslist.append(val)
	return reslist			
#------------------------------------------------#
#				Tree operations					 #
#------------------------------------------------#

# Generates a new tree decsion tree from the arguments passed.
def make_and_print_tree(arguments):
	path = "trainData.csv"
	filetype = path.split('.')
	filetype = filetype[-1]
	Data = read_data(path,filetype)
	data_copy = Data.copy()
	# Partitionate data if the dataset is too big (find a partitioner and use it instead)
	arbol = {}
	if (int( math.floor(len(data_copy.index) * 0.15)) > 5000):		
		data_copy = partitionate(data_copy)
	if arguments[-1] == "C4.5":

		names = list(data_copy.columns[:len(data_copy.columns)])
		lindex = list(range(1,len(names)+1))
		indexval = 0
		for i in lindex:
			if (names[i-1] == arguments[1]):
				indexval = i
				del lindex[i-1]
				break
		if (indexval == 0):
			raise MyError("Nombre del atributo objetivo no valido")
		data_copy.to_csv('inputData.csv')

		dt = DecisionTree.DecisionTree( training_datafile = "inputData.csv",
				                        csv_class_column_index = indexval,
				                        csv_columns_for_features = lindex,
				                        entropy_threshold = 0.01,
				                        max_depth_desired = 3,
				                      )
		dt.get_training_data()
		dt.calculate_first_order_probabilities()
		dt.calculate_class_priors()
		root_node = dt.construct_decision_tree_classifier()	
		#root_node.display_decision_tree("     ")				# Uncomment to print the decision tree in the console.
		arbol =  get_C45_dict(root_node,-1,names[indexval-1])		
	elif arguments[-1] == "CART":
		names = list(data_copy.columns[:len(data_copy.columns)]) 
		names.remove(arguments[1])	
		data_copy = encode_target(arguments[1],data_copy)
		marked = []
		if filetype == "arff":
			dataset = arff.load(open(path, 'r'))
			elemlist = mark_for_replacement(dataset['attributes'])
			marked = []
			for e in elemlist:
				marked.append(e[0])
		else:
			marked = mark_for_replacement_csv(Data,names)
		for m in marked:
			if (m != arguments[1]):
				data_copy = encode(m,data_copy)
		# Build tree with sci-kit and print it using graphviz
		build_tree(data_copy,names,"data.png","data.dot")
		clfclasses = data_copy[arguments[1]].unique()
		arbol = process("data.dot")
		update_classname(clfclasses.tolist(),clfclasses.tolist(),arbol,0)
	get_statistics(arbol,0,arguments[1],arguments[-1])
	return (data_copy,names,arguments[-1],arbol)

# Modifies the node's atribute with the id being passed as an argument
# with the new atribute also being passed.
def modify_tree(numbernode, attributename):
	if (os.path.isfile("datadict.txt")):
		datafile = open("datadict.txt")
		info = readTreeDict(datafile)
		decisionTree = info[0]		
		algorithmName = info[3].replace("\n","")
		nodename = ""
		if (algorithmName == "CART"):
			nodename = get_value_and_atribute(decisionTree[numbernode][3].split(" & ")[0])[0].split(" ")[0]
		elif (algorithmName == "C4.5"):
			nodename =decisionTree[numbernode][0]
		print(nodename,attributename)
		if (nodename == attributename):
			raise MyError("No se puede modificar el atributo del nodo con su mismo atributo.")	
		targetClassName = info[2]
		urlFile = info[1]
		datafile.closed
		res = getData("trainData.csv",targetClassName,algorithmName)
		Data = res[0]	
		names = res[1]
		if not (attributename in names):
			raise MyError("No se puede realizar la accion. El atributo escogido ha sido eliminado del arbol.")	
		if (len(decisionTree) == 0):				
			raise MyError("No se puede realizar la accion. No se ha cargado ningun arbol.")					
		else:
			if (len(decisionTree[numbernode][2]) == 0):
				raise MyError("No se puede escoger como nodo a modificar una hoja del arbol.")
			l = return_node_rules(int(numbernode), decisionTree,algorithmName)
			r =rules_values(l,algorithmName)
			newdata = reprocess_data(Data.copy(), r)
			newdata2 = pd.DataFrame()
			values = newdata[attributename].unique()
			if (len(values) < 2):
				raise MyError("No se puede escoger este atributo, en los datos de este nodo solo hay un solo tipo clase, es decir no hay clasificacion en este nodo con este atributo.")
			if (algorithmName == "CART"):				
				for v in values:
					if not(check_if_number(v)):
						raise MyError("No se puede realizar la accion. La modificacion con CART requiere atributos no numericos.")
				newdata2= pd.DataFrame({attributename:newdata[attributename],"Target":newdata["Target"]},index = newdata.index)
			else:
				newdata2= pd.DataFrame({attributename:newdata[attributename],targetClassName:newdata[targetClassName]},index = newdata.index)				
			# Get atribute split value
			arbol2 = {}
			if (algorithmName == "CART"):
				create_new_tree(newdata2,[attributename],"data1.png")
				clfclasses = newdata[targetClassName].unique()
				origdataclasses = Data[targetClassName].unique()
				arbol2 = process("data1.dot")	
				update_classname(origdataclasses.tolist(),clfclasses.tolist(),arbol2,0)				
			elif (algorithmName == "C4.5"):
				arbol2 =create_new_C45_tree(newdata2,targetClassName)	
			l = return_root_rules(arbol2,algorithmName)				
			r = rules_values(l,algorithmName)
			trees= []
			if (len(r)>0):
				rule = r[0]											
				if (algorithmName == "CART"):
					if (rule[0] != "gini"):
						origdataclasses = Data[targetClassName].unique()
						left_data = newdata[newdata[rule[0]] <= float(rule[1])]								
						build_tree(left_data,names,"data2.png","data2.dot")
						clfclasses = left_data[targetClassName].unique()				
						tree = process("data2.dot")
						update_classname(origdataclasses.tolist(),clfclasses.tolist(),tree,0)
						trees.append(tree)

						right_data = newdata[newdata[rule[0]] > float(rule[1])]
						build_tree(right_data ,names,"data3.png","data3.dot")
						clfclasses = right_data[targetClassName].unique()
						tree = process("data3.dot")
						update_classname(origdataclasses.tolist(),clfclasses.tolist(),tree,0)
						trees.append(tree)
					else:
						raise MyError("El algoritmo no ha sabido sacar una conclusion con este atributo, por favor escoga otro y disculpe las molestias.")
				elif (algorithmName == "C4.5"):					
					if (len(arbol2[0][2]) > 2 ):
						cont = 0
						for c in arbol2[0][2]:
							l = return_node_rules(c, arbol2,algorithmName)		
							r = rules_values(l,algorithmName)
							rule = r[0]
							if (check_if_number(rule[1])):
								res_data = newdata[newdata[rule[0]] == float(rule[1])]
							else:
								res_data = newdata[newdata[rule[0]] == rule[1]]
							trees.append(create_new_C45_tree(res_data,targetClassName))
							trees[cont][0][3] = updateRule( (str(rule[0])+"="+str(rule[1])),trees[cont][0][3])
							cont = cont +1
					else:	
						if (check_if_number(rule[1])):					
							left_data = newdata[newdata[rule[0]] <= float(rule[1])]				
							right_data = newdata[newdata[rule[0]] > float(rule[1])]
							trees.append(create_new_C45_tree(left_data,targetClassName))
							trees[0][0][3] = updateRule((str(rule[0])+"<"+str(rule[1])),trees[0][0][3])
							trees.append(create_new_C45_tree(right_data,targetClassName))
							trees[1][0][3] = updateRule((str(rule[0])+">"+str(rule[1])),trees[1][0][3])
						else:
							left_data = newdata[newdata[rule[0]] == rule[1]]				
							right_data = newdata[newdata[rule[0]] == rule[1]]
							trees.append(create_new_C45_tree(left_data,targetClassName))
							trees[0][0][3] = updateRule((str(rule[0])+"="+str(rule[1])),trees[0][0][3])
							trees.append(create_new_C45_tree(right_data,targetClassName))
							trees[1][0][3] = updateRule((str(rule[0])+"="+str(rule[1])),trees[1][0][3])
			else:
				if (algorithmName == "CART"):
					raise MyError("El algoritmo no ha sabido sacar una conclusion con este atributo, por favor escoga otro y disculpe las molestias.")
				else:	
					values = newdata2[attributename].unique()
					arbol2[0][0] = attributename
					print ("values: "+str(values))
					numeric = False
					cont = 0
					for v in values:
						if (check_if_number(v)):
							res_data = newdata[newdata[attributename] == float(v)]
						else:
							res_data = newdata[newdata[attributename] == v]
						trees.append(create_new_C45_tree(res_data,targetClassName))
						trees[cont][0][3] = updateRule( (str(attributename)+"="+str(v)),trees[cont][0][3])
						cont = cont +1
			# Trees built, now add them to the main tree dict
					
			for c in decisionTree[numbernode][2]:
				remove_subtree(decisionTree, c)
			index = decisionTree.keys()[-1] + 1
			parent = decisionTree[numbernode][1]				
			if (algorithmName == "C4.5"):
				arbol2[0][3] = updateRule(decisionTree[numbernode][3],arbol2[0][3])					
			decisionTree[numbernode]= arbol2[0]
			decisionTree[numbernode][1] = parent	
			childs = []
			val = index								
			for t in trees:
				indexcopy = index
				childs.append(val)
				val = val + len(t)
				for e in t:																						
					decisionTree[index] = t[e]
					decisionTree[index][1] = parent
					newchilds = []
					for c in decisionTree[index][2]:
						c = c + indexcopy
						newchilds.append(c)
					decisionTree[index][2] = newchilds
					index = index + 1											
			decisionTree[numbernode][2] = childs
			for c in decisionTree[numbernode][2]:
				update_subtrees_parents(decisionTree,c, numbernode)
			get_statistics(decisionTree,0,targetClassName,algorithmName)
			return [decisionTree,algorithmName,urlFile,targetClassName]
	else:
		raise MyError("No se ha cargado ningun arbol")

# Removes a subtree given the id of the node from the tree.
def remove_subtree(arbol, numbernode):
	if (len(arbol[numbernode][2]) > 0):
		cont = 0
		for c in arbol[numbernode][2]:
			remove_subtree(arbol, c)
			cont = cont + 1
	del arbol[numbernode]

# Updates the parent of the subtree added to the main tree.
def update_subtrees_parents(arbol,actual, parent):
	arbol[actual][1] = parent
	for c in arbol[actual][2]:
		update_subtrees_parents(arbol, c,actual)

# This function removes the subtree of the node passed as an argument.
def remove_children(numbernode):
	datafile = open("datadict.txt")
	info = readTreeDict(datafile)
	decisionTree = info[0]
	for d in decisionTree:
		print(d)
		decisionTree[d][4] = ast.literal_eval(decisionTree[d][4])
	url = info [1]
	algorithmName = info[3].replace("\n","")
	targetClassName = info[2]
	if (len(decisionTree[numbernode][2]) == 0):
		raise MyError("No se puede realizar la accion, el nodo no tiene hijos.")
	for c in decisionTree[numbernode][2]:
		remove_subtree(decisionTree,c)
	decisionTree[numbernode][2] = []
	if (algorithmName == "CART"):
		listinfo =  decisionTree[numbernode][3].split(" & ")
		aux = listinfo[1].split(" = ")[1]
		listinfo.pop(1)
		listinfo[0] = ""
		listinfo[0] = 'label="gini = '+str(aux)
		first = True
		for l in listinfo:
			if first:
				newinfo = str(l)
				first = False
			else:
				newinfo = newinfo + " & "+str(l)
		decisionTree[numbernode][3] = newinfo
	return [decisionTree,url,targetClassName,algorithmName]

# Function that joins the two nodes in a new one with a new name.
def join_selected_branches(nodebranch1,nodebranch2, newname):
	datafile = open("datadict.txt")
	info = readTreeDict(datafile)
	tree = info[0]
	urlFile = info[1]
	targetClassName = info[2]
	algorithmName = info[3].replace("\n","")	
	if (algorithmName != "CART"):
		if not( nodebranch1 in tree.keys()):
			raise MyError("El nodo "+str(nodebranch1)+" no esta dentro del arbol.")
		elif not( nodebranch2 in tree.keys()):
			raise MyError("El nodo "+str(nodebranch2)+" no esta dentro del arbol.")
		if (len(tree[tree[nodebranch1][1]][2]) == 2):
			raise MyError("No se pueden juntar dos ramas si las dos son las unicas que tiene el nodo.")
		if (tree[nodebranch1][1] == tree[nodebranch2][1]):		# The branches have to belong to the same parent.
			previousname = tree[nodebranch1][3].split(" & ")[0].replace('label="',"").split("=")[0]
			parent = tree[nodebranch1][1]
			res = getData("trainData.csv",targetClassName,algorithmName)
			Data = res[0]
			names = res[1]
			l = return_node_rules(parent, tree,algorithmName)
			r =rules_values(l,algorithmName)
			newdata = reprocess_data(Data.copy(), r)
			varname1 = get_value_and_atribute(tree[nodebranch1][3].split(" & ")[0])
			varname2 = get_value_and_atribute(tree[nodebranch2][3].split(" & ")[0])
			all_digits_val1 = True

			newdata[varname1[0]] = newdata[varname1[0]].replace(varname1[1], "Otros")
			newdata[varname2[0]] = newdata[varname2[0]].replace(varname2[1], "Otros")
			newbranch = create_new_C45_tree(newdata ,targetClassName)	
			index = tree.keys()[-1] + 1	
			indexcopy = index
			tree[parent][2].append(index)
			newbranch[0][1]= parent

			#if (tree[nodebranch1][1]):
			#	previousname = tree[0][0]
			
			remove_subtree(tree,nodebranch1)
			remove_subtree(tree,nodebranch2)
			tree[parent][2].remove(nodebranch1)
			tree[parent][2].remove(nodebranch2)
			for e in newbranch:											
				if (e==0):
					branchrule = newbranch[0][3].split(" & ")

					
					del branchrule[0]
					newbranch[0][3] = 'label="'+previousname+'='+newname
					for elem in range(0,len(branchrule)):
						newbranch[0][3] = newbranch[0][3] +" & "+str(branchrule[elem])							
				tree[index] = newbranch[e]
				newchilds = []
				for c in tree[index][2]:
					c = c + indexcopy
					newchilds.append(c)
				tree[index][2] = newchilds
				index = index + 1						
			update_subtrees_parents(tree,indexcopy, parent)

			get_statistics(tree,0,targetClassName,algorithmName)
			return [tree,urlFile,targetClassName,algorithmName]
		else:
			raise MyError("Las ramas escogidas deben proceder del mismo nodo")
	else:
		raise MyError ("No se pueden juntar ramas con CART")

# Function that joins all the child nodes of "numbernode"
# whose number of cases are below the threshold.
def join_by_entropy(numbernode, entropy_threshold):
	datafile = open("datadict.txt")
	info = readTreeDict(datafile)
	tree = info[0]
	urlFile = info[1]
	algorithmName = info[3].replace("\n","")
	targetClassName = info[2]
	res = getData("trainData.csv",targetClassName,algorithmName)
	Data = res[0]
	totaldatacount = len(Data.index)
	names = res[1]
	if (algorithmName != "CART"):
		if not( numbernode in tree.keys()):
			raise MyError("El nodo "+str(numbernode)+" no esta dentro del arbol.")
		if (len(tree[numbernode][2]) == 2):
			raise MyError("No se pueden juntar dos ramas si las dos son las unicas que tiene el nodo.")
		result_data = pd.DataFrame()
		first = True
		l = return_node_rules(numbernode, tree,algorithmName)
		r =rules_values(l,algorithmName)
		newdata = reprocess_data(Data.copy(), r)
		first = True
		s = ""
		nodenewchilds = []
		for a in tree[numbernode][2]:
			l = return_node_rules(a, tree,algorithmName)
			r =rules_values([l[0]],algorithmName)
			newdata2 = reprocess_data(newdata.copy(), r)
			newdata2count = len(newdata2.index)	
			percentage = (float(newdata2count)/float(totaldatacount))*100
			print(float(newdata2count),float(totaldatacount))
			print (percentage)
			if (percentage < entropy_threshold):
				varname = get_value_and_atribute(tree[a][3].split(" & ")[0])
				if (first):
					s = varname[0]+"="+varname[1]
					first = False
				else:
					s = s+" "+varname[1]
				newdata[varname[0]] = newdata[varname[0]].replace(varname[1], "Otros")
				remove_subtree(tree,a)
			else:
				nodenewchilds.append(a)
		tree[numbernode][2] = nodenewchilds
		if (s == ""): 
			raise MyError("El porcentaje de discriminacion es demasiado bajo, todos los nodos tienen un porcentaje superior al dado." )
		#elif (len(tree[numbernode][2]) == 1):
			#raise MyError("El porcentaje de discriminacion es demasiado alto, todos los nodos se juntaran en uno." )
		elif (len(tree[numbernode][2]) == 0):
			raise MyError("El porcentaje de discriminacion es demasiado alto, todos los nodos tienen un porcentaje inferior al dado." )
		newbranch = create_new_C45_tree(newdata ,targetClassName)	
		index = tree.keys()[-1] + 1	
		tree[numbernode][2].append(index)
		indexcopy = index
		for e in newbranch:											
			if (e==0):
				branchrule = newbranch[0][3].split(" & ")
				del branchrule[0]
				newbranch[0][3] = 'label="'+s
				for elem in range(0,len(branchrule)):
					newbranch[0][3] = newbranch[0][3] +" & "+str(branchrule[elem])
			tree[index] = newbranch[e]
			newchilds = []
			for c in tree[index][2]:
				c = c + indexcopy
				newchilds.append(c)
			tree[index][2] = newchilds
			index = index + 1						
		update_subtrees_parents(tree,indexcopy, numbernode)
		get_statistics(tree,0,targetClassName,algorithmName)
		return [tree,urlFile,targetClassName,algorithmName]
	else:
		raise MyError("No se pueden juntar ramas con CART")

# This function generates a new tree ignoring the atributes being passed
# as an argument.
def gen_ignoring_att(attributes):
	datafile = open("datadict.txt")
	info = readTreeDict(datafile)
	tree = {}
	url = info [1]
	algorithmName = info[3].replace("\n","")
	targetClassName = info[2]
	res = getData("trainData.csv",targetClassName,algorithmName)
	res2 = getData("testData.csv",targetClassName,algorithmName)
	Data = res[0]
	Data2 = res2[0]
	names = res[1]	
	if len(attributes) <= (len(names)-2):
		for a in attributes:
			if not(a in names):
				raise MyError ("El atributo "+str(a)+" ya ha sido eliminado o ya no se encuentra en los datos")				 
			if (a == targetClassName ): 
				raise MyError("No se puede ignorar la clase objetivo.")
			Data = Data.drop(a,1)
			Data2 = Data2.drop(a,1)
		if (algorithmName == "CART"):
			build_tree(Data,names,"data1.png","data.dot")
			clfclasses = Data[targetClassName].unique()
			origdataclasses = Data[targetClassName].unique()
			tree = process("data.dot")
			update_classname(origdataclasses.tolist(),clfclasses.tolist(),tree,0)
		elif (algorithmName == "C4.5"):
			tree = create_new_C45_tree(Data,targetClassName)
		Data.to_csv('trainData.csv',index = False)
		Data2.to_csv('testData.csv',index = False)
		get_statistics(tree,0,targetClassName,algorithmName)
		return [tree,url,targetClassName,algorithmName]
	else:
		raise MyError("No se pueden ignorar mas atributos de los datos")


# Generates the statistics and the confusion matrix of the tree.
def get_statistics(tree,nodeid,targetClassName,algorithmName):
	res = getData("testData.csv",targetClassName,algorithmName)
	childs= tree[nodeid][2]
	confusionMatrix = {}
	values = tree[nodeid][3].split(" nvalue = ")[1]
	values = values.replace("[","")
	values = values.replace("]","")
	classes_and_values = values.split(",")	
	maxval = 0
	nodeclass = ""
	l = return_node_rules(nodeid, tree,algorithmName)
	r =rules_values(l,algorithmName)
	origdataclasses = res[0][targetClassName].unique()
	origdataclasses = list(reversed(origdataclasses))
	newdata = reprocess_data(res[0].copy(), r)
	datacount = len(newdata.index)
	class_count = []
	count = 0
	classindex = 0
	objectiveclassindex = 0		
	for val in range(0,len(origdataclasses)):
		if (check_if_number(origdataclasses[val])):
			origdataclasses[val] = round(origdataclasses[val],1)
	classes_and_values = order_list1_like_list2(classes_and_values,origdataclasses,algorithmName)
	if (len(childs)>0):		
		for c in range(0,len(origdataclasses)):	
			if count < len(classes_and_values):			
				cv = classes_and_values[count]	
				if (algorithmName == "CART"):
					cv = cv.split(" :  ")
					cv[0] = cv[0].replace(" ","")
				elif (algorithmName == "C4.5"):
					cv = cv.split(" => ")
					cv[0] = cv[0].split("=")[-1]
				if (cv[0] == origdataclasses[c]):
					if (cv[1]> maxval):
						objectiveclassindex = c
						maxval = cv[1]
						nodeclass = cv[0]
						classindex = c
					count = count + 1
		# Get the class with highest prob.
		matrixList= []
		for c in childs:
			matrixList.append(dict(get_statistics(tree,c,targetClassName,algorithmName)))
		first = True
		for e in matrixList:
			keys = e.keys()		
			if first:
				for r in keys:
					val = list(e[r])					# Make a copy, or else it is referencing the list of another node.
					confusionMatrix[r] = val
				first = False
			else:
				for r in keys:
					valuesList = list(e[r])			# Make a copy, or else it is referencing the list of another node.
					for it in range(0,len(valuesList[1])):					
						confusionMatrix[r][1][it] = confusionMatrix[r][1][it] + valuesList[1][it]
				
	else:	
		for c in range(0,len(origdataclasses)):
			if (count < len(classes_and_values)):
				cv = classes_and_values[count]		
				if (algorithmName == "CART"):
					cv = cv.split(" :  ")
					cv[0] = cv[0].replace(" ","")
				elif (algorithmName == "C4.5"):
					cv = cv.split(" => ")
					cv[0] = cv[0].split("=")[-1]
				if (origdataclasses[c] != cv[0]):
					confusionMatrix[c]= [origdataclasses[c],[0]*len(origdataclasses)]	
					if (check_if_number(cv[0])):			
						class_count.append(len((newdata.loc[newdata[targetClassName] == float(cv[0])]).index))
					else:
						class_count.append(len((newdata.loc[newdata[targetClassName] == cv[0]]).index))
				else:
					if (cv[1]> maxval):
						objectiveclassindex = c
						maxval = cv[1]
						nodeclass = cv[0]
						classindex = c	
					confusionMatrix[c]= [cv[0],[0]*len(origdataclasses)]
					if (check_if_number(cv[0])):			
						class_count.append(len((newdata.loc[newdata[targetClassName] == float(cv[0])]).index))
					else:
						class_count.append(len((newdata.loc[newdata[targetClassName] == cv[0]]).index))		
					count = count + 1
			else:
				confusionMatrix[c]= [origdataclasses[c],[0]*len(origdataclasses)]  
				if (check_if_number(origdataclasses[c])):	#<---------- arreglar		
						class_count.append(len((newdata.loc[newdata[targetClassName] == float(origdataclasses[c])]).index))
				else:
					class_count.append(len((newdata.loc[newdata[targetClassName] == origdataclasses[c]]).index))
		confusionMatrix[classindex] = [nodeclass,class_count]
		
	elem= tree[nodeid]
	if (len(elem)> 4):				# Check if the old statistics are there
		elem = elem[:4]		# Expunge the old statistics in the node
	elem.append(deepcopy(confusionMatrix))
	statistics = get_recall_prec_acc(deepcopy(confusionMatrix),classindex, nodeclass, objectiveclassindex,datacount)
	elem.append(statistics)
	tree[nodeid] = elem 
	return deepcopy(confusionMatrix)
	

#------------------------------------------------#
#				   Main Program		  		     #
#------------------------------------------------#

# Print a few lines explaining how to use this program.
def print_help():
	print("Para generar un arbol escriba 'generar', luego cuando se lo pida el programa indique el nombre del archivo, el nombre de la clase objetivo y el algoritmo a utilizar.")
	print("Para modificar un arbol escriba 'modificar', luego cuando se lo pida el programa indique en nodo del arbol y el nombre del atributo a cambiar.")
	print("Para cerrar el programa escribir 'cerrar'.")

# Write the tree in a JSON file. 
def writeJSON(outfile,ntabs,nodeid,decisionTree,algName):
	if len(decisionTree[nodeid][2]) >0:
		for i in range(0,ntabs):
			outfile.write("\t")
		outfile.write("{\n")
		for i in range(0,ntabs):
			outfile.write("\t")
		outfile.write(" ")
		data = decisionTree[nodeid][3].split("&")
		name = ""
		entropy = ""
		if (algName == "CART"):
			name = data[0].replace('label="',"")
			entropy = data[1].replace(" ngini = ","")	
		elif (algName == "C4.5"):
			name = decisionTree[nodeid][0]
			entropy = data[1].replace(" entropy = ","")
			if ('=' in name):
				name= name.split('=')[0]
		outfile.write('"name": ')
		outfile.write('"'+str(name)+'"'+",\n")
		for i in range(0,ntabs):
			outfile.write("\t")
		outfile.write('"entropy": ')
		outfile.write('"'+str(entropy)+'"'+",\n")
		for i in range(0,ntabs):
			outfile.write("\t")
		outfile.write(" ")
		rule = ""
		if nodeid == 0:
			rule = "null"
		else:
			if (algName == "CART"):
				if decisionTree[nodeid][0] == 'i':
					rule = "TRUE"
				else:
					rule = "FALSE"			
			elif (algName == "C4.5"):
				rule = data[0].replace('label="',"")			
		outfile.write('"rule": ')
		outfile.write('"'+str(nodeid)+": "+str(rule)+'"'+",\n")
		for i in range(0,ntabs):
			outfile.write("\t")
		outfile.write('"id": "'+str(nodeid) +'",\n')
		for i in range(0,ntabs):
			outfile.write("\t")

		confmat = decisionTree[nodeid][4]
		
		confmatstring = ""
		if (isinstance(confmat,basestring)):
			confmatstring = confmat
		else:
			first = True
			for c in confmat:
				if first:
					print (c)
					confmatstring = confmatstring + str(confmat[c][0]) + " & " + str(confmat[c][1])
					first = False
				else:
					confmatstring = confmatstring+ " &-& " + str(confmat[c][0]) + " & " + str(confmat[c][1])
		outfile.write('"confMat": "'+ confmatstring +'",\n')
		for i in range(0,ntabs):
			outfile.write("\t")
		#jsonstats = json.dumps(decisionTree[nodeid][5])					#JSON dump puts the row's name in double quotations which messes the parsing in JS. 
		outfile.write('"stats": "'+str(decisionTree[nodeid][5]).replace("\n","") +'",\n')
		for i in range(0,ntabs):
			outfile.write("\t")
		outfile.write('"children": [\n')
		cont = 0
		for c in decisionTree[nodeid][2]:
			cont = cont +1
			writeJSON(outfile,ntabs+1,c,decisionTree,algName)
		for i in range(0,ntabs):
			outfile.write("\t")
		outfile.write("]\n")
		for i in range(0,ntabs):
			outfile.write("\t")
		if (nodeid == 0):
			outfile.write("}\n")
		else:
			lastchildcheck= decisionTree[decisionTree[nodeid][1]][2][-1] # check if this node is the last of the list, in that case we don't put the comma at the end
			if (lastchildcheck == nodeid):
				outfile.write("}\n")
			else:
				outfile.write("},\n")			
	else:
		for i in range(0,ntabs):
			outfile.write("\t")
		outfile.write("{ ")
		outfile.write(" ")
		data = decisionTree[nodeid][3].split("&")
		name = ""
		entropy = ""
		if (algName == "CART"):
			entropy = data[0].replace('label="gini = ',"")
			#if ("ngini =")
			name = decisionTree[nodeid][3].split(" & ")[2].replace('"',"")
			name = name.replace("nvalue = ","")
			name = name.replace("\n","")
		
		elif (algName == "C4.5"):
			#name = decisionTree[nodeid][0]
			name = decisionTree[nodeid][3].split(" & ")[2].replace('"',"")
			name = name.replace("nvalue = ","")
			name = name.replace("\n","")
			entropy = data[1].replace(" entropy = ","")
		outfile.write('"name": ')
		outfile.write('"'+str(name)+'"'+",\n")
		for i in range(0,ntabs):
			outfile.write("\t")
		outfile.write('"entropy": ')
		outfile.write('"'+str(entropy)+'"'+",\n")
		for i in range(0,ntabs):
			outfile.write("\t")
		rule = ""
		if (algName == "CART"):
			if decisionTree[nodeid][0] == 'i':
				rule = "TRUE"
			else:
				rule = "FALSE"			
		elif (algName == "C4.5"):
			rule = data[0].replace('label="',"")				
		outfile.write('"rule": ')
		outfile.write('"'+str(nodeid)+": "+str(rule)+'",\n')
		for i in range(0,ntabs):
			outfile.write("\t")
		outfile.write('"id": "'+str(nodeid) +'",\n')
		for i in range(0,ntabs):
			outfile.write("\t")

		confmat = decisionTree[nodeid][4]
		confmatstring = ""
		if (isinstance(confmat,basestring)):
			confmatstring = confmat
		else:
			first = True
			for c in confmat:
				if first:
					confmatstring = confmatstring + str(confmat[c][0]) + " & " + str(confmat[c][1])
					first = False
				else:
					confmatstring = confmatstring+ " &-& " + str(confmat[c][0]) + " & " + str(confmat[c][1])
		outfile.write('"confMat": "'+ confmatstring +'",\n')
		for i in range(0,ntabs):
			outfile.write("\t")
		outfile.write('"stats": "'+str(decisionTree[nodeid][5]).replace("\n","") +'"\n')
		for i in range(0,ntabs):
			outfile.write("\t")
		if (nodeid == 0):
			outfile.write("}\n")
		else:
			lastchildcheck= decisionTree[decisionTree[nodeid][1]][2][-1] 	# check if this node is the last of the list, in that case we don't put the comma at the end
			if (lastchildcheck == nodeid):								
				outfile.write("}\n")
			else:
				outfile.write("},\n")

# Write the tree and all his information.
def writeDict(treeDict,url,targetName,algorName):
	with open('datadict.txt', 'w') as outfile:
		outfile.write(str(treeDict.keys()))
		outfile.write(" &&-&& "+url+" &&-&& ")
		outfile.write(targetName+" &&-&& ")
		outfile.write(algorName)
		outfile.write("\n")
		for a in treeDict:
			outfile.write(str(treeDict[a][0]))
			treeDict[a].pop(0)			
			for e in treeDict[a]:
				s = str(e)
				if "\n" in s:
					s = s.replace("\n","")
				outfile.write( " &&-&& "+ s)			
			outfile.write("\n")
	outfile.closed

# Read the file that contains the dictionary that represents the tree
# and the information such as the targetclass and the algorith which
# was generated.
def readTreeDict(dfile):
	datafile = dfile.readlines()
	indiceslist = []
	tree = {}
	cont = 0
	first = True
	alName=""
	targetName= ""
	url = ""
	for d in datafile:
		if first:
			info = d.split(" &&-&& ")
			alName = info[3]
			targetName = info[2]
			url = info[1]
			indicesTree= info[0]
			first = False
			indicesTree = indicesTree.replace("[","")
			indicesTree = indicesTree.replace("]","")
			indicesTree = indicesTree.replace("\n","")
			indiceslist = indicesTree.split(", ")
		else:
			tree[int(indiceslist[cont])] = d.split(" &&-&& ")
			tree[int(indiceslist[cont])][1] = int(tree[int(indiceslist[cont])][1])
			newchilds = []
			if tree[int(indiceslist[cont])][2] != "[]":
				tree[int(indiceslist[cont])][2] = tree[int(indiceslist[cont])][2].replace("[","")
				tree[int(indiceslist[cont])][2] = tree[int(indiceslist[cont])][2].replace("]","")
				childs = tree[int(indiceslist[cont])][2].split(", ")			
				for c in childs:
					newchilds.append(int(c))
			tree[int(indiceslist[cont])][2] = newchilds
			cont = cont + 1
	return [tree,url,targetName,alName]

# This function reads the data and splits it into two datasets,
# one for training and one for testig.
@app.route("/read", methods=['POST']) 
def readData():
	try:
		jsonData =  request.get_json()
		url = jsonData['url']
		if (len(url.split('.')) < 2 ):
			raise MyError("La ubicacion del archivo no es correcta.")
		filetype = url.split('.')[-1]
		Data = read_data(url,filetype)
		if (int( math.floor(len(Data.index))) > 1000):
			Data = Data.head(1000)
		train, test = train_test_split(Data, test_size = 0.3)
	
		train.to_csv('trainData.csv',index = False)
		test.to_csv('testData.csv',index = False)
		names = list(Data.columns[:len(Data.columns)])
		return json.dumps({'success':True,'names':names}), 200, {'ContentType':'application/json'} 
	except MyError as e:
		print ("Error: ", e.message)
		return json.dumps({'success':False,'msg':e.message}), 200, {'ContentType':'application/json'}
 
# Main function. Reads the message from the graphic interface and executes de operation.
# If the operation is successful returns a boolean to the interface stating that the result
# can be retrieved, read and represented graphically.
# If there's a problem a message is returned with the boolean set to false and the erro message.
@app.route("/main", methods=['POST']) 
def main():
	jsonData =  request.get_json()
	#print (jsonData)
	command = []
	if (jsonData['op'] == 'generar'):
		command = ["removableplaceholder",jsonData['op'],jsonData['fileName'],jsonData['targetName'],jsonData['algorithmName']]
	elif (jsonData['op'] == 'modificar'):
		command = ["removableplaceholder",jsonData['op'],jsonData['NodeId'],jsonData['AttName']]
	elif (jsonData['op'] == 'eliminarRama'):
		command = ["removableplaceholder",jsonData['op'],jsonData['NodeId']]
	elif (jsonData['op'] == 'juntarramasselec'):
		command = ["removableplaceholder",jsonData['op'],jsonData['BranchId1'],jsonData['BranchId2'],jsonData['BranchName']]
	elif (jsonData['op'] == 'juntarramasentropia'):
		command = ["removableplaceholder",jsonData['op'],jsonData['NodeId'],jsonData['EntropyThreshold']]
	elif (jsonData['op'] == 'ignoreattribute'):
		command = ["removableplaceholder",jsonData['op'],jsonData['AttributeName']]

	if (command[1] == "ayuda"):
		print_help()
	elif (command[1] == "modificar"):
		try:
			res = modify_tree(int(command[2]),str(command[3]))
			decisionTree = res[0]
			algorName = res[1]
			with open('data.json', 'w') as outfile:		# ALWAYS write JSON first!! Then write the dictionary
				writeJSON(outfile,1,0,decisionTree,algorName)
			outfile.closed
			writeDict(decisionTree,res[2],res[3],algorName)	
		except MyError as e:
			print ("Error: ", e.message)
			return json.dumps({'success':False,'msg':e.message}), 200, {'ContentType':'application/json'} 

	elif (command[1] == "generar"):
		try:
			res = make_and_print_tree([command[2],command[3],command[4]])
			Data = res[0]
			with open('data.json', 'w') as outfile:		# ALWAYS write JSON first!! Then write the dictionary
				writeJSON(outfile,1,0,res[3],command[4])
			outfile.closed
			writeDict(res[3],command[2],command[3],command[4])
		except MyError as e:
			print ("Error: ", e.message)
			return json.dumps({'success':False,'msg':e.message}), 200, {'ContentType':'application/json'} 
	elif (command[1] == "eliminarRama"):
		try:
		#if (int(command[2]) == 0) raise "No se puede eliminar el nodo raiz"
			res = remove_children(int(command[2]))
			with open('data.json', 'w') as outfile:			# ALWAYS write JSON first!! Then write the dictionary
					writeJSON(outfile,1,0,res[0],res[3])	
			writeDict(res[0],res[1],res[2],res[3])
		except MyError as e:
			print ("Error: ", e.message)
			return json.dumps({'success':False,'msg':e.message}), 200, {'ContentType':'application/json'}
	elif (command[1] == "juntarramasselec"):
		try:
			res = join_selected_branches(int(command[2]),int(command[3]), command[4])
			with open('data.json', 'w') as outfile:			# ALWAYS write JSON first!! Then write the dictionary
					writeJSON(outfile,1,0,res[0],res[3])
			writeDict(res[0],res[1],res[2],res[3])
		except MyError as e:
			print ("Error: ", e.message)
			return json.dumps({'success':False,'msg':e.message}), 200, {'ContentType':'application/json'}	
	elif (command[1] == "juntarramasentropia"):
		try:
			res = join_by_entropy(int(command[2]),float(command[3]))
			with open('data.json', 'w') as outfile:			# ALWAYS write JSON first!! Then write the dictionary
					writeJSON(outfile,1,0,res[0],res[3])
			writeDict(res[0],res[1],res[2],res[3])	
		except MyError as e:
			print ("Error: ", e.message)
			return json.dumps({'success':False,'msg':e.message}), 200, {'ContentType':'application/json'}
	elif (command[1] == "ignoreattribute"):
		try:
			res = gen_ignoring_att(command[2])
			with open('data.json', 'w') as outfile:			# ALWAYS write JSON first!! Then write the dictionary
					writeJSON(outfile,1,0,res[0],res[3])	
			writeDict(res[0],res[1],res[2],res[3])
		except MyError as e:
			print ("Error: ", e.message)
			return json.dumps({'success':False,'msg':e.message}), 200, {'ContentType':'application/json'}
	else:
		print ("Orden no valida")
		return json.dumps({'success':False,'msg':'Orden no valida'}), 200, {'ContentType':'application/json'}
	return json.dumps({'success':True}), 200, {'ContentType':'application/json'} 

#------------------------------------------------#
#			  Debugging 						 #
#------------------------------------------------#



#print("* data.head()", data.head(), sep="\n", end="\n\n")
#print("* data.tail()", data.tail(), sep="\n", end="\n\n")

#print "----------------------------------------------------"

#print("* data.head()", data_copy.head(), sep="\n", end="\n\n")
#print("* data.tail()", data_copy.tail(), sep="\n", end="\n\n")ules
