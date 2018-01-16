from sklearn.model_selection import KFold
import numpy as np

# check if a value is numeric or not
def is_numeric(value):
    try:
        (float)(value)
    except Exception:
        return False
    return True

# conver a string to a number
def to_numeric(value):
    try:
        return (float)(value)
    except Exception:
        return value

def prepare(raw_data):
    prepared_data = list()
    M = len(raw_data[0])
    for row in raw_data:
        tmp = list()
        for i in range(M - 1):
            tmp.append(to_numeric(row[i]))
        tmp.append(row[M - 1])
        prepared_data.append(tmp)
    return prepared_data

filename="project3_dataset1.txt"
file=open(filename,'r')
raw_data=[line.split() for line in file.readlines()]
data=np.array(raw_data)
# data=raw_data.astype(np.float)
# print(len(data))

kf = KFold(n_splits=10)
kf.get_n_splits(data)
# print(kf)
class train_test_pair:
 	"""docstring for train_test_psir"""
 	def __init__(self, train,test):
 		self.train_set = train
 		self.test_set=test


res=[]
for train_index, test_index in kf.split(data):
	#print("TRAIN:", train_index, "TEST:", test_index)
	X_train, X_test = prepare(data[train_index]), prepare(data[test_index])
	temp=train_test_pair(X_train,X_test)
	res.append(temp)


# for i in range(len(res)):
# 	print("train_size",len(res[i].train_set),"test_size",len(res[i].test_set))
# 	print("train:",res[i].train_set," test:",res[i].test_set)
# 	print("X_train:",X_train," X_test:",X_test)
#
# print("data size",len(data))
