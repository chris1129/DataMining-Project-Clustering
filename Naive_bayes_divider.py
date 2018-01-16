import NaiveBayes as NB
import perform_eval as PE
import kfold  as KF
import numpy as np
import time

start_time = time.time()
accuracy = []
precision = []
recall = []
f_1 = []
#kf=kfold_test("project3_dataset2.txt")


for i in range(len(KF.res)):
    nb = NB.NaiveBayes(KF.res[i].train_set)
    # DT.print_tree(my_tree, "")
    test_label = []#np.array(KF.res[i].test_set[:, -1])
    for row in KF.res[i].test_set:
        test_label.append(row[-1])
    print('--------------------------------------------round: %i------------------------------------------------' % (i))
    # print("original labels:", test_label)
    predict = []
    for j in range(len(KF.res[i].test_set)):
        #print("Predict data")
        # print(KF.res[i].test_set[j][:-1])
        predict.append(nb.predict(KF.res[i].test_set[j][:-1]))
    # print("predicted labels:", np.array(predict))
    a, p, r, F = PE.evaluate(predict, test_label)
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    f_1.append(F)
 
print("\nresult:")
print("accurcay = %0.4f, precision = %0.4f, recall = %0.4f, f-1 score = %0.4f" %(np.mean(accuracy), np.mean(precision), np.mean(recall), np.mean(f_1)))

print("\n--- %s seconds ---" % (time.time() - start_time))