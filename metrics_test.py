import csv
import numpy as np
import sklearn.metrics
from kaggle_auc import auc


csv_filepath = 'answer.csv'
clip2label, clip2usage = {}, {}
clips_csv, labels_csv = [], []
with open(csv_filepath, 'rb') as f:
    reader = csv.reader(f)
    reader.next()
    for row in reader:
        clip = row[0]
        label = int(row[1])
        usage = row[2]
        clip2label[clip] = label
        clip2usage[clip] = usage
        clips_csv.append(clip)
        labels_csv.append(label)

clip2prediction = {}
with open('test_submission.csv', 'rb') as f:
    reader = csv.reader(f)
    reader.next()
    for row in reader:
        clip = row[0]
        prediction = np.float64(row[1])
        clip2prediction[clip] = prediction

usage = ['Public', 'Private']
for u in usage:
    targets, predictions = [], []
    for k, v in clip2label.iteritems():
        if clip2usage[k] == u:
            targets.append(v)
            predictions.append(clip2prediction[k])

    targets, predictions = np.array(targets), np.array(predictions)
    print u, 'Kaggle AUC:', auc(targets, predictions)
    print u, 'Sklearn AUC', sklearn.metrics.roc_auc_score(targets, predictions)
    print
