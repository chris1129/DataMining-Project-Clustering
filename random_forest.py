# created by Xiaoxin Wu
# https://machinelearningmastery.com/implement-random-forest-scratch-python/
# http://syllabus.cs.manchester.ac.uk/pgt/2017/COMP61011/goodProjects/Sazonau.pdf
from random import randrange

import numpy as np
import decision_tree as DT
import perform_eval as PE
import kfold as KF
import time

start_time = time.time()
# gini = 1 - p(1)^2 - p(2)^2
def gini(data):
    count0 = count1 = 0.0
    for entry in data:
        if entry[-1] == '1':
            count1 += 1
        else:
            count0 += 1
    length = count1 + count0
    return 1 - (count0/length) * (count0/length) - (count1/length) * (count1/length)

# loop trough all the columns and all possible values on a column and choose the one with the greatest gain
def get_best_split(data, features_num):
    max_gain = 0
    selected_col = 0
    selected_pivot = None
    features = list()
    parent_gini = gini(data)
    length = len(data)
    # randomly pick a certain number of features
    while len(features) < features_num:
        col = randrange(len(data[0]) - 1)   # the last column are true/false label
        if col not in features:
            features.append(col)
    for col in features:
        vals = set()
        for entry in data:
            vals.add(entry[col])
        for pivot in vals:                  # extract all the unique vals
            trues, falses = DT.split_data(data, col, pivot)
            if len(trues) == 0 or len(falses) == 0:
                continue
            cur_gain = DT.gain(trues, falses, parent_gini, length)
            if cur_gain >= max_gain:
                max_gain = cur_gain
                selected_col = col
                selected_pivot = pivot
    return max_gain, selected_col, selected_pivot

# recursively build the tree
# for each layer, choose the best split
def build_tree(data, depth, features_num):
    gain, col, pivot = get_best_split(data, features_num)
    if gain == 0 or len(data) <= 3 or depth > len(data[0]) - 1:
        return DT.Decision_TreeNode(data, True, col, pivot)
    trues, falses = DT.split_data(data, col, pivot)
    root = DT.Decision_TreeNode(data, False, col, pivot)
    root.true_children = build_tree(trues, depth + 1, features_num)
    root.false_children = build_tree(falses, depth + 1, features_num)
    return root

# resample the data based on the weight
def resample(data, sample_num):
    indices = [i for i in np.random.choice(len(data), sample_num)]
    new_training = []
    for index in indices:
        new_training.append(data[index])
    return new_training

# build trees and return all the trees as a forest
def build_random_forest(tree_num, training, sample_num, features_num):
    forest = list()
    while len(forest) < tree_num:
        cur_training = resample(training, sample_num)
        cur_tree = build_tree(cur_training, 0, features_num)
        forest.append(cur_tree)
    return forest

# used to predict the label for the testing
def predict(testing, forest):
    pred = list()
    for row in testing:
        count_0 = 0
        count_1 = 0
        for tree in forest:
            cur_pred = DT.predict_val(row, tree)
            if cur_pred == '0':
                count_0 += 1
            else:
                count_1 += 1
        if count_1 > count_0:
            pred.append('1')
        elif count_1 < count_0:
            pred.append('0')
        else:
            pred.append(str(randrange(2)))  # randomly return '1' or '0'
    return pred

tree_num = 50
accuracy = []
precision = []
recall = []
f_1 = []
for i in range(len(KF.res)):
    sample_num = (int)(len(KF.res[i].train_set) / 5)
    features_num = (int)(np.sqrt(len(KF.res[i].train_set[0]) - 1))
    forest = build_random_forest(tree_num, KF.res[i].train_set, sample_num, features_num)
    test_label = []
    for row in KF.res[i].test_set:
        test_label.append(row[-1])
    print('--------------------------------------------round: %i------------------------------------------------' % (i))
    # DT.print_tree(my_tree, "")
    # print("original labels:", test_label)
    pred = predict(KF.res[i].test_set, forest)
    # print("predicted labels:", pred)
    a, p, r, F = PE.evaluate(pred, test_label)
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    f_1.append(F)

print("\nresult:")
print("accurcay = %0.4f, precision = %0.4f, recall = %0.4f, f-1 score = %0.4f" %(np.mean(accuracy), np.mean(precision), np.mean(recall), np.mean(f_1)))

print("\n--- %s seconds ---" % (time.time() - start_time))



