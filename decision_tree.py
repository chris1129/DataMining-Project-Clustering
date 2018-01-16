# created by Xiaoxin Wu

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

# gain = gini(parent) - P(left) * gini(left) - P(right) * gini(right)
def gain(left, right, parent_gini, length):
    p = 1.0 * len(left) / length
    return parent_gini - p * gini(left) - (1 - p) * gini(right)

# if two numbers are numeric
# conditions are >= (true) and < (false)
# otherwise conditions are == (true) and != (false)
def greaterOrEqual(val, target):
    if isinstance(target, float):
        return val >= target
    else:
        return val == target

# for a certain column, choose a target value and do partition
# split the data into two halves, one set is all true, one set is all false
def split_data(data, col, target):
    trues, falses = list(), list()
    for entry in data:
        if greaterOrEqual(entry[col], target):
            trues.append(entry)
        else:
            falses.append(entry)
    return trues, falses

# loop trough all the columns and all possible values on a column and choose the one with the greatest gain
def get_best_split(data):
    max_gain = 0
    selected_col = 0
    selected_pivot = None
    parent_gini = gini(data)
    length = len(data)
    for col in range(len(data[0]) - 1):     # the last column are true/false label
        vals = set()
        for entry in data:
            vals.add(entry[col])
        for pivot in vals:                  # extract all the unique vals
            trues, falses = split_data(data, col, pivot)
            if len(trues) == 0 or len(falses) == 0:
                continue
            cur_gain = gain(trues, falses, parent_gini, length)
            if cur_gain >= max_gain:
                max_gain = cur_gain
                selected_col = col
                selected_pivot = pivot
    return max_gain, selected_col, selected_pivot

# recursively build the tree
# for each layer, choose the best split
def build_tree(data, depth):
    gain, col, pivot = get_best_split(data)
    if gain == 0 or depth <= 0:
        return Decision_TreeNode(data, True, col, pivot)
    trues, falses = split_data(data, col, pivot)
    root = Decision_TreeNode(data, False, col, pivot)
    root.true_children = build_tree(trues, depth - 1)
    root.false_children = build_tree(falses, depth - 1)
    return root

# tree node, if it is a leaf, it has its true or false label
class Decision_TreeNode:
    def __init__(self, data, isLeaf, col, pivot):
        self.data = data
        self.isLeaf = isLeaf
        self.col = col
        self.pivot = pivot

    def get_label(self):
        count1 = count0 = 0
        for entry in self.data:
            if entry[-1] == '1':
                count1 += 1
            else:
                count0 += 1
        if count1 > count0:
            return '1'
        else:
            return '0'

    def __repr__(self):
        if self.isLeaf:
            return ('Leaf with label of %s' % (repr(self.get_label())))
        if isinstance(self.pivot, float):
            return ('feature %s >= %s?' % (repr(self.col), repr(self.pivot)))
        else:
            return ('feature %s = %s?' % (repr(self.col), repr(self.pivot)))

# predict true or false for the test data
def predict_val(data, node):
    if(node.isLeaf):
        return node.get_label()
    if greaterOrEqual(data[node.col], node.pivot):
        return predict_val(data, node.true_children)
    else:
        return predict_val(data, node.false_children)

# print decision tree
def print_tree(node, tab):
    print(tab + str(node))
    if not node.isLeaf:
        print (tab + '>> True:')
        print_tree(node.true_children, tab + "\t")
        print (tab + '>> False:')
        print_tree(node.false_children, tab + "\t")

def predict(testing, node):
    pred = list()
    for i in range(len(testing)):
        pred.append(predict_val(testing[i], node))
    return pred


