import numpy as np
import skfeature.utility.entropy_estimators as ees


def cwjrFS (X, y, k):
    """
    :param X:  the input features numpy.array [n_sample,n_feature]
    :param y: the class label numpy.array [n_sample,1]
    :param k: the number of selected features
    :return:F the indexs of selected-features list
    """
    t1 = []
    n_samples, n_features = X.shape
    for i in range(n_features):  # the mutual information between feature and the class label
        f = X[:, i]
        t1.append(ees.midd(f, y))
    F = []
    red = np.zeros(n_features)
    rrd = np.zeros(n_features)
    while len(F) < k:
        if len(F) == 0:   # select the first feature
            index = t1.index(max(t1))
            F.append(index)
            f_select = X[:, index]
        j_mim = -1000000000000
        for i in range(n_features):
            if i not in F:
                fi = X[:, i]
                miff = ees.midd(fi, f_select)
                ent1=ees.entropyd(fi)-miff
                ent2=ees.entropyd(f_select)-miff
                if ent1 == 0 or ent2 == 0:
                    cw=0
                else:
                    cmi1 = ees.cmidd(fi,y,f_select)
                    cmi2 = ees.cmidd(f_select,y,fi)
                    cw = cmi1/ent1 + cmi2/ent2
                jointMI = t1[i]+ees.cmidd(f_select,y,fi)
                b = cw*jointMI
                red[i] += b
                rrd[i] += miff
                t = red[i]-rrd[i]  # calculate the J(fk)
                if t > j_mim:
                    j_mim = t
                    idx = i
        F.append(idx)
        f_select = X[:, idx]
    return F