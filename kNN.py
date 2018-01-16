import heapq
import numpy as np
import perform_eval as PE
from sklearn.model_selection import KFold
import time

start_time = time.time()

# dist_label class is used to associate the distance with label
class dist_label:
    def __init__(self, dist, label):
        self.dist = dist
        self.label = label

    def __lt__(self, other):
        return self.dist > other.dist

# conver a string to a number
def to_numeric(value):
    try:
        return (float)(value)
    except Exception:
        return value

# check if a value is numeric or not
def is_numeric(value):
    try:
        (float)(value)
    except Exception:
        return False
    return True

# get distance between p and q
# if they are continous type, use euclid distance
# if they are categorical type, label 0 if they are the same, otherwise 1
def distance(p, q):
    dist = 0
    for i in range(len(p) - 1):
        if is_numeric(p[i]):
            dist += pow(to_numeric(p[i]) - to_numeric(q[i]), 2)
        elif p[i] != q[i]:
            dist += 1
    return dist_label(np.sqrt(dist), q[-1:])

# normailize the continous data by (val - min) / (max - min)
def prepare(raw_data):
    prepared_data = list()
    N = len(raw_data)
    M = len(raw_data[0])
    for row in raw_data:
        tmp = list()
        for i in range(M - 1):
            tmp.append(to_numeric(row[i]))
        tmp.append(row[M - 1])
        prepared_data.append(tmp)
    #normalize the data
    for col in range(M - 1):
        if isinstance(prepared_data[0][col], int) or isinstance(prepared_data[0][col], float):
            min_val = 1e6
            max_val = -1e6
            for row in range(N):
                max_val = max(prepared_data[row][col], max_val)
                min_val = min(prepared_data[row][col], min_val)
            for row in range(N):
                prepared_data[row][col] = (prepared_data[row][col] - min_val) / (float)(max_val - min_val)
    return prepared_data

def predict(testing, traning, k):
    pred = list()
    for testing_row in testing:
        q = []
        for training_row in traning:
            cur = distance(testing_row, training_row)
            if len(q) < k :
                heapq.heappush(q, cur)
            elif q[0].dist > cur.dist:
                heapq.heappop(q)
                heapq.heappush(q, cur)
        score = 0
        while len(q) > 0:
            top = heapq.heappop(q)
            if top.label == '1':
                score += 1.0 / (top.dist + 0.0000001)
            else:
                score -= 1.0 / (top.dist + 0.0000001)
        if score > 0:
            pred.append('1')
        else:
            pred.append('0')
    return pred

K = 15
filename="project3_dataset1.txt"
file=open(filename,'r')
raw_data=[line.split() for line in file.readlines()]
data=np.array(prepare(raw_data))

kf = KFold(n_splits=10)
kf.get_n_splits(data)

class train_test_pair:
    def __init__(self, train,test):
        self.train_set = train
        self.test_set=test

res=[]
for train_index, test_index in kf.split(data):
    X_train, X_test = data[train_index], data[test_index]
    temp=train_test_pair(X_train,X_test)
    res.append(temp)


accuracy = []
precision = []
recall = []
f_1 = []
for i in range(len(res)):
    test_label = np.array(res[i].test_set[:, -1])
    print('--------------------------------------------round: %i------------------------------------------------' %(i))
    # print("original labels:", test_label)
    pred = predict(res[i].test_set, res[i].train_set, K)
    # print("predicted labels:", np.array(pred))
    a, p, r, F = PE.evaluate(pred, test_label)
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    f_1.append(F)

print("\nresult:")
print("accurcay = %0.4f, precision = %0.4f, recall = %0.4f, f-1 score = %0.4f" %(np.mean(accuracy), np.mean(precision), np.mean(recall), np.mean(f_1)))

print("\n--- %s seconds ---" % (time.time() - start_time))
