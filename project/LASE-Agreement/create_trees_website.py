import warnings
import pickle, json, os, pyconll, sys
import utils, dataloader
import argparse
import numpy as np
np.random.seed(1)
import sklearn
from sklearn.externals.six import StringIO
from sklearn.model_selection import GridSearchCV
from sklearn.tree.export import export_text
import pydotplus
from copy import deepcopy
from collections import defaultdict

def printTreeWithExamplesPDF(model, treerules, treelines, leaves, feature, leafcount, feature_names, test_samples, train_samples, dev_samples):
	dot_data = StringIO()
	sklearn.tree.export_graphviz(model.best_estimator_, out_file=dot_data,
								 feature_names=feature_names, node_ids=True
								 , class_names=["disagree", "agree"], proportion=False, rounded=True, filled=True,
								 leaves_parallel=False, impurity=False)

	nodes = []
	else_nodes = []

	for tree_rule, treeline in zip(treerules.split("\n"), treelines.split("\n")):
		header = ""
		if feature in tree_rule:
			header = feature
			if "head" in tree_rule:
				if "<=" in tree_rule:
					nodes.append(header + " not in [head]")
				else:
					else_nodes.append(header + " in [head]")
			elif "child" in tree_rule:
				if "<=" in tree_rule:
					nodes.append(header + " not in [child]")
				else:
					else_nodes.append(header + " in [child]")
		else:
			if "relation" in tree_rule:
				header  = "relation"
			elif "head" in tree_rule:
				header="head-pos"
			elif "child" in tree_rule:
				header="child-pos"

			info = "[" + treeline.split("[")[-1]

			if "<=" in tree_rule:
				nodes.append(header + " in " + info.lstrip().rstrip().lower())
			if "else" in treeline:
				else_nodes.append(header + " in " +info.lstrip().rstrip().lower())

	if args.hard:
		threshold = 0.9
	else:
		threshold = 0.01

	editedgraph = deepcopy(dot_data.getvalue()).split("\n")
	tree_dictionary, top_nodes = {}, []
	leafnodes = []
	leafedges = {}
	leafvalues = {}
	relabeled_leaves = getTree(dot_data, editedgraph, else_nodes, feature, leaves, nodes,
														tree_dictionary, top_nodes, leafnodes, leafedges, leafvalues,
														threshold=threshold)
	automated_acc, _ = utils.automated_metric(relabeled_leaves, test_path, feature,
																data_loader.feature_distribution[feature],
																test_samples, threshold=threshold, hard=args.hard,
																traindata=data)

	print("test" + percent + ", " + str(train_samples) + ", " + lang + ", " + feature + ", " + str(	automated_acc) )

	if dev_samples > 0:
		automated_acc, _ = utils.automated_metric(relabeled_leaves, dev_path, feature,
																	data_loader.feature_distribution[feature],
																	dev_samples, threshold=threshold, hard=args.hard,
																	traindata=data)

		print("dev" + percent + ", " +  str(train_samples) + ", " + lang + ", " + feature + ", " + str(automated_acc)  )

	if not args.inTh:
		return

def getTree(dot_data, editedgraph, else_nodes, feature, leaves, nodes, tree_dictionary, topnodes, leafnodes, leafedges, leafvalues, threshold):
	i = 0
	leaf_num = 0
	leftstart = 0
	rightstart = 0
	relabeled_leaves = {}
	for linenum, line in enumerate(dot_data.getvalue().split("\n")):
		if "<=" in line:
			info = line.split("<=")
			info_index = info[-1].find("\\")
			nodenum = line.split("[")[0]
			textinfo = info[-1][info_index + 2:].split("fillcolor=")[0]
			edge = info[0] + " in " + nodes[i]
			values = info[-1].split("\\nclass")[0].split("value = ")[1].replace("[", "").replace("]", "").replace("\'", "").split(
				",")
			disagree, agree = int(values[0]), int(values[1])
			color = utils.colorRetrival(agree, disagree, data_loader.feature_distribution[feature], threshold, args.hard)
			editedgraph[linenum] = line.split("[")[0] + "[label=\"node - " + nodenum + "\\n" + textinfo.replace("\\n","\\l").replace("class = agree", "").replace("class = disagree", "") + 'fillcolor=\"{0}\"] ;'.format(color)
			tree_dictionary[int(nodenum)] = {"children": [], "edge": nodes[i], "info": editedgraph[linenum]}
			i += 1
			topnodes.append(int(nodenum))

		elif "->" in line:
			lefttext = "[labeldistance={0},labelangle=50, headlabel=\"{1}\",labelfontsize=10];"
			righttext = "[labeldistance={0},labelangle=-50, headlabel=\"   {1}\", labelfontsize=10];"
			info = line.replace('\'', '').replace(";", "").split("->")
			leftnode, rightnode = int(info[0]), int(info[-1].split("[")[0])
			if rightnode - leftnode == 1:
				edge = nodes[leftstart]
				input = edge.split("[")[-1].replace("]", "").split(",")
				edge = edge.split("[")[0] + utils.printMultipleLines(input, t=7)
				leftstart += 1
				newtext = line.split(str(rightnode))[0] + " " + str(rightnode) + " " + lefttext.format(3.5, edge)
			else:
				edge = else_nodes[rightstart]
				input = edge.split("[")[-1].replace("]", "").split(",")
				edge = edge.split("[")[0] + utils.printMultipleLines(input, t=7)
				rightstart += 1
				newtext = line.split(str(rightnode))[0] + " " + str(rightnode) + " " + righttext.format(3.5, edge)
			editedgraph[linenum] = newtext
			tree_dictionary[leftnode]["children"].append(rightnode)
			tree_dictionary[rightnode]["top"] = leftnode
			leafedges[rightnode] = edge

		elif ">" in line:
			info = line.split(">")
			info_index = info[-1].find("\\")
			nodenum = line.split("[")[0]
			textinfo = info[-1][info_index + 2:].split("fillcolor=")[0]
			edge = info[0] + " in " + nodes[i]
			values = info[-1].split("\\nclass")[0].split("value = ")[1].replace("[", "").replace("]", "").replace("\'","").split(
				",")
			disagree, agree = int(values[0]), int(values[1])
			color = utils.colorRetrival(agree, disagree, data_loader.feature_distribution[feature], threshold, args.hard)
			editedgraph[linenum] = line.split("[")[0] + "[label=\"node - " + nodenum + "\\n" + textinfo.replace("\\n",
																												"\\l").replace(
				"class = agree", "").replace("class = disagree", "") + 'fillcolor=\"{0}\"] ;'.format(color)
			tree_dictionary[int(nodenum)] = {"children": [], "edge": nodes[i], "info": editedgraph[linenum]}
			i += 1
			topnodes.append(int(nodenum))

		else:
			if "class" in line:

				info = line.split("label=\"")
				info[-1] = "\\n".join(info[-1].split("\\n")[1:])
				leafvalues[leaf_num] = info[-1].split("\\n")[1].split("value = ")[1].replace("[", "").replace("]",
																											  "").replace(
					"\'", "").split(",")

				disagree, agree = int(leafvalues[leaf_num][0]), int(leafvalues[leaf_num][1])
				# t = agree * 1.0 / (disagree + agree)
				agreement = "chance-agreement\\n"
				if utils.isAgreement(data_loader.feature_distribution[feature], agree, disagree, threshold, args.hard):# t >= threshold:
					agreement = "agreement\\n"

				color = utils.colorRetrival(agree, disagree, data_loader.feature_distribution[feature], threshold, args.hard)
				text_position = info[-1].split("agree")
				classinfo = text_position[0].replace("dis", "") + agreement + "\",fillcolor=\"{0}\"] ;".format(color)

				nodenum = line.split("[")[0]
				data_info = ""
				(leaf_node_class, leaf_node_data) = leaves[leaf_num]
				relabeled_leaves[leaf_num] = (leaf_node_data, agree, disagree)
				if leaf_node_data["head_feature"] != None:
					data_info = leaf_node_data["head_feature"] + "\\n\\n"

				if leaf_node_data["child_feature"] != None:
					data_info += leaf_node_data["child_feature"] + "\\n\\n"

				if leaf_node_data["relation"] == None:
					data_info += "relation = *" + "\\l\\l"
				else:
					class_relations = data_loader.class_relations[leaf_node_class]
					input = set(
						leaf_node_data["relation"].replace("\'", "").replace("[", "").replace("]", "").split(","))
					extra = input - class_relations
					actual = input - extra
					leaf_node_data["relation"] = ",".join(list(actual))
					data_info += "relation = " + utils.printMultipleLines(actual) + "\\l"

				if leaf_node_data["head"] == None:
					data_info += "head-pos = *" + "\\l\\l"
				else:
					class_pos = data_loader.class_headpos[leaf_node_class]
					input = set(
						leaf_node_data["head"].replace("\'", "").replace("[", "").replace("]", "").split(","))
					extra = input - class_pos
					actual = input - extra
					leaf_node_data["head"] = ",".join(list(actual))
					data_info += "head-pos = " + utils.printMultipleLines(actual) + "\\l"

				if leaf_node_data["child"] == None:
					data_info += "child-pos = *" + "\\l\\l"
				else:
					class_pos = data_loader.class_childpos[leaf_node_class]
					input = set(
						leaf_node_data["child"].replace("\'", "").replace("[", "").replace("]", "").split(","))
					extra = input - class_pos
					actual = input - extra
					leaf_node_data["child"] = ",".join(list(actual))
					data_info += "child-pos = " + utils.printMultipleLines(actual) + "\\l"
				textinfo = info[0] + "label=" + "\"Leaf- " + str(
					leaf_num) + "\\n" + data_info.lower() + classinfo.replace("\\n", "\\l")
				editedgraph[linenum] = textinfo

				tree_dictionary[int(nodenum)] = {"children": [], "edge": data_info.lower(),
												 "info": editedgraph[linenum]}
				leaf_num += 1
				leafnodes.append(int(nodenum))

	if args.prune:
		editedgraph, tree_dictionary, leafnodes, topleafnodes, removednodes, topnodes = utils.pruneTree(editedgraph, tree_dictionary, topnodes, leafnodes, leafedges, feature)

		_, _, relabeled_leaves = utils.collateTree(data_loader.feature_distribution[feature], leafedges, editedgraph, topleafnodes, tree_dictionary, leaves, threshold, topnodes, removednodes, args.hard)

	return relabeled_leaves

def train(feature):
	x_train, _, y_train, y_test = train_features[feature] , test_features[feature], \
													 train_output_labels[feature], test_output_labels[feature]
	if dev_path:
		x_dev, y_dev = dev_features[feature], dev_output_labels[feature]
		x = np.concatenate([x_train, x_dev])
		y = np.concatenate([y_train, y_dev])
		test_fold = np.concatenate([
			np.full(x_train.shape[0], -1, dtype=np.int8),
			np.zeros(x_dev.shape[0], dtype=np.int8)
		])
		cv = sklearn.model_selection.PredefinedSplit(test_fold)
	else:
		x,y = x_train, y_train
		cv = None

	criterion = ['gini', 'entropy']
	parameters = {'criterion':criterion, 'max_depth':np.arange(6, 15), 'min_impurity_decrease':[1e-3]}
	decision_tree = sklearn.tree.DecisionTreeClassifier()
	model = GridSearchCV( decision_tree , parameters, cv=cv)
	model.fit(x, y)

	trainleave_id = model.best_estimator_.apply(x)

	uniqueleaves = set(trainleave_id)
	uniqueleaves = sorted(uniqueleaves)
	leafcount = {}
	for i, leaf in enumerate(uniqueleaves):
		leafcount[i] = round(np.count_nonzero(trainleave_id == leaf) * 100 / len(trainleave_id), 2)

	feature_names = []
	for i in range(len(data_loader.pos_dictionary)):
		feature_names.append("head@" + data_loader.pos_id2tag[i])
	for i in range(len(data_loader.pos_dictionary)):
		feature_names.append("child@" + data_loader.pos_id2tag[i])
	for i in range(len(data_loader.relation_dictionary)):
		feature_names.append("relation@" + data_loader.relation_id2tag[i])

	feature_names.append(feature + "@child")
	feature_names.append(feature + "@head")
	tree_rules = export_text(model.best_estimator_, feature_names= feature_names, max_depth=model.best_params_["max_depth"])
	treelines = utils.printTreeForBinaryFeatures(tree_rules, data_loader.pos_id2tag, data_loader.relation_id2tag,  data_loader.used_relations, data_loader.used_head_pos, data_loader.used_child_pos, feature)
	leaves = utils.constructTree(treelines, feature)
	assert len(leaves) == len(leafcount)
	dev_samples = len(x_dev) if dev_path else 0
	printTreeWithExamplesPDF(model, tree_rules, treelines, leaves, feature, leafcount, feature_names, len(y_test), len(x_train), dev_samples)

if __name__ == "__main__":
	warnings.filterwarnings("ignore")
	parser = argparse.ArgumentParser()
	parser.add_argument("--file", type=str, default="./decision_tree_files.txt")
	parser.add_argument("--features", type=str, default="Gender+Person+Number+Tense+Mood+Case+Degree+Aspect+Voice+PronType+NumType", nargs='+')
	parser.add_argument("--prune", action="store_true", default=True)
	parser.add_argument("--binary", action="store_true", default=True)
	parser.add_argument("--debug_folder", type=str, default="./")
	parser.add_argument("--percent", type=float, default=1.0)
	parser.add_argument("--relation_map", type=str, default="./relation_map")
	parser.add_argument("--inTh", action="store_true", default=False)
	parser.add_argument("--seed", type=int, default=1)
	parser.add_argument("--simulate", action="store_true", default=False)
	parser.add_argument("--hard", action="store_true", default=False)

	args = parser.parse_args()

	with open(args.file, "r") as inp:
		files = []
		for file in inp.readlines():
			if file.startswith("#"):
				continue
			files.append(file)

	d = {}
	relation_map = {}
	with open(args.relation_map, "r") as inp:
		for line in inp.readlines():
			relation_map[line.split(";")[0]] = (line.split(";")[1].lstrip().rstrip(), line.split(";")[-1].lstrip().rstrip())

	fnum = 0
	args.features = args.features.split("+")
	while fnum < len(files):

		treebank = files[fnum].strip()
		fnum += 1
		train_path, dev_path, test_path = None, None, None
		for [path, dir, inputfiles] in os.walk(treebank):
			for file in inputfiles:
				if "-train_sm.conllu" in file:
					train_path = treebank + "/" + file
					if args.simulate:
						percent = treebank.split("-")[-2] + "-" + treebank.split("-")[-1]
						lang = train_path.strip().split('/')[-1].split("_")[0]
						lang += "-" + percent
					else:
						percent = 'all'
						lang = train_path.strip().split('/')[-1].split("-")[0]

				if "dev.conllu" in file:
					dev_path = treebank + "/" + file

				if "test.conllu" in file:
					test_path = treebank + "/" + file

		if train_path is None:
			continue
		language_fullname = "_".join(os.path.basename(treebank).split("_")[1:])
		lang_full = lang
		f = train_path.strip()

		i = 0
		data = pyconll.load_from_file(f"{f}")
		data_loader = dataloader.DataLoader(args, relation_map)

		inputFiles = [train_path, dev_path, test_path]
		data_loader.readData(inputFiles)

		train_features, train_output_labels = data_loader.getBinaryFeatures(train_path,type="train", p=args.percent, shuffle=True)
		if dev_path:
			dev_features, dev_output_labels = data_loader.getBinaryFeatures(dev_path, type="dev", p=1.0, shuffle=False)
		test_features, test_output_labels = data_loader.getBinaryFeatures(test_path, type="test", p=1.0, shuffle=False)

		for feature in args.features:
			if feature in test_features and feature in train_features:
				try:
					train(feature)
				except:
					print("error processing ", feature, lang)