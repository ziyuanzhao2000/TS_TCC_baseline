from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import OneHotEncoder
import sklearn
import os
import torch
import torch.nn.functional as F

alias = 'pFD_B'
base_dir = os.path.join('data', alias)
train_dict = torch.load(os.path.join(base_dir, 'train.pt'))
test_dict = torch.load(os.path.join(base_dir, 'test.pt'))
train_X = train_dict['samples'][:,0,:]
train_y = train_dict['labels']
test_X = test_dict['samples'][:,0,:]
test_y = test_dict['labels']
clf = KNeighborsClassifier(n_neighbors = 1)
clf.fit(train_X, train_y)
# enc = OneHotEncoder(handle_unknown = 'ignore')
# enc.fit([[0,0],[1,1]])

metrics_dict = {}
pred_prob = clf.predict_proba(test_X)
pred = pred_prob.argmax(axis=1)
target = test_y
target_prob = (F.one_hot(torch.tensor(test_y), num_classes=3)).numpy()
metrics_dict['Accuracy'] = sklearn.metrics.accuracy_score(target, pred)
metrics_dict['Precision'] = sklearn.metrics.precision_score(target, pred, average='macro')
metrics_dict['Recall'] = sklearn.metrics.recall_score(target, pred, average='macro')
metrics_dict['F1'] = sklearn.metrics.f1_score(target, pred, average='macro')
metrics_dict['AUROC'] = sklearn.metrics.roc_auc_score(target_prob, pred_prob, multi_class='ovr')
metrics_dict['AUPRC'] = sklearn.metrics.average_precision_score(target_prob, pred_prob)
print(metrics_dict)