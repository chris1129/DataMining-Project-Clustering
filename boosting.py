# created by Xiaoxin Wu
# reference: https://simsicon.wordpress.com/2016/08/05/understand-decision-tree-learning-4/
# reference: http://blog.csdn.net/GYQJN/article/details/45501185

import numpy as np
import decision_tree as DT
import perform_eval as PE
import kfold as KF
import time

start_time = time.time()
# just create a tree stump, the depth is at most two layers.
def build_BDT(data):
    gain, col, pivot = DT.get_best_split(data)
    if gain == 0:
        return DT.Decision_TreeNode(data, True, col, pivot)
    trues, falses = DT.split_data(data, col, pivot)
    root = DT.Decision_TreeNode(data, False, col, pivot)

    root.true_children = DT.Decision_TreeNode(trues, True, -1, None)
    root.false_children = DT.Decision_TreeNode(falses, True, -1, None)
    return root

# resample the data based on the weight
def resample(data, weights, sample_num):
    indices = [i for i in np.random.choice(len(data), sample_num, p = weights)]
    new_training = []
    for index in indices:
        new_training.append(data[index])
    return new_training

# train the classifiers certain times
def train(itr_num, training, sample_num):
    BDT_list = list()
    BDT_alpha_list = list()
    N = len(training)
    weights = [1.0 / N for i in range(N)]
    y = []
    for row in training:
        y.append(row[-1])
    for i in range(itr_num):
        cur_training = resample(training, weights, sample_num)
        cur_tree = build_BDT(cur_training)
        y_head = DT.predict(training, cur_tree)
        errors = np.array([1 if a != b else 0 for a, b in zip(y_head, y)])
        epsilon = sum(errors * weights)
        alpha = np.log((1 - epsilon) * 1.0 / epsilon) / 2
        # y_head == y, C[i] = 1 else C[i] = -1
        C = [-1 if error == 1 else 1 for error in errors]
        for j in range(N):
            weights[j] *= np.exp(-1 * alpha * C[j])
        weights /= sum(weights)
        BDT_list.append(cur_tree)
        BDT_alpha_list.append(alpha)
    return BDT_list, BDT_alpha_list

# used to predict the label for the testing
def predict(testing, BDT_list, BDT_alpha_list):
    pred = list()
    for row in testing:
        score = 0
        for i in range(len(BDT_list)):
            cur_pred = DT.predict_val(row, BDT_list[i])
            if cur_pred == '0':
                score -= BDT_alpha_list[i]
            else:
                score += BDT_alpha_list[i]
        if score >= 0:
            pred.append('1')
        else:
            pred.append('0')
    return pred

accuracy = []
precision = []
recall = []
f_1 = []
for i in range(len(KF.res)):
    sample_num = (int)(len(KF.res[i].train_set) / 20)
    tree_num = 200
    BDT_list, BDT_alpha_list = train(tree_num, KF.res[i].train_set, sample_num)
    test_label = []
    for row in KF.res[i].test_set:
        test_label.append(row[-1])
    print('--------------------------------------------round: %i------------------------------------------------' % (i))
    # DT.print_tree(my_tree, "")
    # print("original labels:", test_label)
    pred = predict(KF.res[i].test_set, BDT_list, BDT_alpha_list)
    # print("predicted labels:", pred)
    a, p, r, F = PE.evaluate(pred, test_label)
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    f_1.append(F)

print("\nresult:")
print("accurcay = %0.4f, precision = %0.4f, recall = %0.4f, f-1 score = %0.4f" %(np.mean(accuracy), np.mean(precision), np.mean(recall), np.mean(f_1)))

print("\n--- %s seconds ---" % (time.time() - start_time))


