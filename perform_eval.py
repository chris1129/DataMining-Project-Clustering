def evaluate(pred, gt):
    a = b = c = d = 0.0
    for i in range(len(pred)):
        if pred[i] == '1' and gt[i] == '1':
            a += 1
        elif pred[i] == '1' and gt[i] == '0':
            c += 1
        elif pred[i] == '0' and gt[i] == '1':
            b += 1
        else:
            d += 1
    accu = (a + d) / (a + b + c + d)
    p = a / (a + c)
    r = a / (a + b)
    F = 2 * r * p / (r + p)
    print("accurcay = %0.4f, precision = %0.4f, recall = %0.4f, f-1 score = %0.4f" %(accu, p, r, F))
    return accu, p, r, F
