from sklearn.metrics import auc, roc_curve, precision_recall_curve, accuracy_score, recall_score, f1_score, \
    precision_score
import numpy as np
import pandas as pd
from multi_adaboost_CNN import AdaBoostClassifier as Ada_CNN
import test2_CNN
from sklearn.model_selection import KFold
import lightgbm as lgb
def kfold(data, k, row=0, col=0, cv=3):
    dlen = len(data)
    if cv != 4:
        if cv == 1:
            lens = row
        elif cv == 2:
            lens = col
        else:
            lens = dlen
        test_res = []
        train_res = []
        d = list(range(lens))
        kf = KFold(5, shuffle=True)
        d = kf.split(d)
        for i in d:
            test_res.append(list(i[1]))
            train_res.append(list(i[0]))
        if cv == 3:
            return train_res, test_res
        else:
            train_s = []
            test_s = []
            for i in range(k):
                train_ = []
                test_ = []
                for j in range(dlen):
                    if data[j][cv - 1] in test_res[i]:
                        test_.append(j)
                    else:
                        train_.append(j)
                train_s.append(train_)
                test_s.append(test_)
            return train_s, test_s
    else:
        r = list(range(row))
        c = list(range(col))
        kf = KFold(5, shuffle=True)
        r = kf.split(r)
        c = kf.split(c)
        r_test_res = []
        r_train_res = []
        c_test_res = []
        c_train_res = []
        for i in r:
            r_test_res.append(list(i[1]))
            r_train_res.append(list(i[0]))
        for i in c:
            c_test_res.append(list(i[1]))
            c_train_res.append(list(i[0]))
        train_s = []
        test_s = []
        for i in range(k):
            train_ = []
            test_ = []
            for m in range(dlen):
                flag_1 = False
                flag_2 = False
                if data[m][0] in r_test_res[i]:
                    flag_1 = True
                if data[m][1] in c_test_res[i]:
                    flag_2 = True
                if flag_1 and flag_2:
                    test_.append(m)
                if (not flag_1) and (not flag_2):
                    train_.append(m)
            train_s.append(train_)
            test_s.append(test_)
        return train_s, test_s
for mm in (1,2,3,4):
    f_1 = open('res.txt', 'a')
    time = 5
    k = 5
    cv = 4
    batch_size = 10
    PREs = np.array([])
    ACCs = np.array([])
    RECs = np.array([])
    AUCs = np.array([])
    AUPRs = np.array([])
    F1s = np.array([])
    Label = pd.read_csv('F:/learning/LRI/data/data' + str(mm) + '/label.csv', header=None, index_col=None).to_numpy()
    ldata = pd.read_csv('F:/learning/LRI/data/data' + str(mm) + '/ldata.csv', header=None, index_col=None).to_numpy()
    rdata = pd.read_csv('F:/learning/LRI/data/data' + str(mm) + '/rdata.csv', header=None, index_col=None).to_numpy()
    row, col = Label.shape
    p = np.array([(i, j) for i in range(row) for j in range(col) if Label[i][j]])
    n = np.array([(i, j) for i in range(row) for j in range(col) if Label[i][j] == 0])
    np.random.shuffle(n)
    sample = len(p)
    n = n[:sample]
    for j in range(time):
        p_tr, p_te = np.array(kfold(p, k=5), dtype=object)
        n_tr, n_te = np.array(kfold(n, k=5), dtype=object)
        for i in range(k):
            train_sample = np.vstack([np.array(p[p_tr[i]]), np.array(n[n_tr[i]])])
            test_sample = np.vstack([np.array(p[p_te[i]]), np.array(n[n_te[i]])])
            X_train = np.hstack([ldata[train_sample[:, 0]], rdata[train_sample[:, 1]]])
            y_tr = Label[train_sample[:, 0], train_sample[:, 1]]
            X_test = np.hstack([ldata[test_sample[:, 0]], rdata[test_sample[:, 1]]])
            y_te = Label[test_sample[:, 0], test_sample[:, 1]]
            label = y_te
            y_train = list(map(int, y_tr))
            y_train = np.array(y_train)
            y_test = list(map(int, y_te))
            y_test = np.array(y_test)
            X_train_r = test2_CNN.reshape_for_CNN(X_train)
            X_test_r = test2_CNN.reshape_for_CNN(X_test)
            bdt_real_test_CNN = Ada_CNN(base_estimator=test2_CNN.baseline_model(n_features=200), n_estimators=10,
                                        learning_rate=1, epochs=1)
            bdt_real_test_CNN.fit(X_train_r, y_train, batch_size)
            cpre_label = bdt_real_test_CNN.predict(X_test_r)
            cscore = bdt_real_test_CNN.predict_proba(X_test_r)
            cscore = cscore[:, 1]
            model = lgb.LGBMClassifier(learning_rate=0.1, n_estimators=1000)
            model.fit(X_train, y_tr)
            lscore = model.predict_proba(X_test)
            lscore = lscore[:, 1]
            score = 0.4 * cscore + 0.6 * lscore
            pre_label = np.zeros(score.shape)
            for m in range(len(score)):
                if score[m] >= 0.5:
                    pre_label[m] = 1
                else:
                    pre_label[m] = 0

            acc = accuracy_score(label, pre_label)
            rec = recall_score(label, pre_label)
            f1 = f1_score(label, pre_label)
            pre = precision_score(label, pre_label)
            fp, tp, threshold = roc_curve(label, score)
            pre_, rec_, _ = precision_recall_curve(label, score)
            au = auc(fp, tp)
            aupr = auc(rec_, pre_)
            PREs = np.append(PREs, pre)
            ACCs = np.append(ACCs, acc)
            RECs = np.append(RECs, rec)
            AUCs = np.append(AUCs, au)
            AUPRs = np.append(AUPRs, aupr)
            F1s = np.append(F1s, f1)
    PRE = PREs.mean()
    ACC = ACCs.mean()
    REC = RECs.mean()
    AUC = AUCs.mean()
    AUPR = AUPRs.mean()
    F1 = F1s.mean()
    PRE_err = np.std(PREs)
    ACC_err = np.std(ACCs)
    REC_err = np.std(RECs)
    AUC_err = np.std(AUCs)
    AUPR_err = np.std(AUPRs)
    F1_err = np.std(F1s)
    f_1.write('data' + str(mm) + 'cv' + str(cv) + ':\n')
    f_1.write(str(round(PRE, 4)) + '±' + str(round(PRE_err, 4)) + '\t')
    f_1.write(str(round(REC, 4)) + '±' + str(round(REC_err, 4)) + '\t')
    f_1.write(str(round(ACC, 4)) + '±' + str(round(ACC_err, 4)) + '\t')
    f_1.write(str(round(F1, 4)) + '±' + str(round(F1_err, 4)) + '\t')
    f_1.write(str(round(AUC, 4)) + '±' + str(round(AUC_err, 4)) + '\t')
    f_1.write(str(round(AUPR, 4)) + '±' + str(round(AUPR_err, 4)) + '\t\n\n')
    f_1.close()
