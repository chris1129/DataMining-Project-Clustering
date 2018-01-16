import numpy as np
import decision_tree as DT
import perform_eval as PE
import kfold as KF
import time

start_time = time.time()
depth = 4
accuracy = []
precision = []
recall = []
f_1 = []
for i in range(len(KF.res)):
    my_tree = DT.build_tree(KF.res[i].train_set, depth)
    test_label = []
    for row in KF.res[i].test_set:
        test_label.append(row[-1])
    print('--------------------------------------------round: %i------------------------------------------------' % (i))
    # DT.print_tree(my_tree, "")
    # print("original labels:", test_label)
    predict = []
    count = 0
    for j in range(len(KF.res[i].test_set)):
        predict.append(DT.predict_val(KF.res[i].test_set[j], my_tree))
    # print("predicted labels:", predict)
    a, p, r, F = PE.evaluate(predict, test_label)
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    f_1.append(F)

print("\nresult:")
print("accurcay = %0.4f, precision = %0.4f, recall = %0.4f, f-1 score = %0.4f" %(np.mean(accuracy), np.mean(precision), np.mean(recall), np.mean(f_1)))

print("\n--- %s seconds ---" % (time.time() - start_time))

