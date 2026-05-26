#!/usr/bin/env python3
"""
Load `features_X.csv` and `features_meta.csv`, read labels filled by user
in the `label` column of `features_meta.csv`, map labels to indices using
the config classes, save `features_Y.csv`, then load the pretrained model and
produce evaluation metrics (accuracy, precision, recall, f1, confusion matrix).

Run from `src/` directory.
"""
import os
import csv
import pickle
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

from utils.config import Config
from utils.drawer import plot_confusion_matrix


def main():
    cfg = Config(config_file='../configs/train_action_recogn_pipeline.yaml')
    cfg_stage = cfg['s4_train_classifier.py']
    get_path = lambda x: os.path.join(*x) if isinstance(x, (list, tuple)) else x
    features_X_path = get_path(cfg_stage.input.features_x)
    features_Y_path = get_path(cfg_stage.input.features_y)
    model_path = cfg_stage.output.model_path

    # Features and meta
    out_dir = os.path.abspath(os.path.join('..', 'datasets', 'realtime_action_recognition', 'extracted_data'))
    meta_path = os.path.join(out_dir, 'features_meta.csv')
    if not os.path.exists(features_X_path) or not os.path.exists(meta_path):
        print('Missing features or metadata. Run extract_features_mediapipe.py first.')
        return

    X = np.loadtxt(features_X_path, dtype=float)
    # Read labels from meta
    labels = {}
    with open(meta_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row['feature_idx'])
            label = row.get('label','').strip()
            labels[idx] = label

    # Build Y array mapping label names to indices using config classes
    classes = np.array(cfg.classes)
    name2idx = {c:i for i,c in enumerate(classes)}

    Y = []
    missing = 0
    for i in range(len(X)):
        lab = labels.get(i,'')
        if lab == '' or lab not in name2idx:
            Y.append(-1)
            missing += 1
        else:
            Y.append(name2idx[lab])
    Y = np.array(Y, dtype=int)
    print(f'Labels assigned: {len(X)-missing}, missing: {missing}')

    # Save Y to features_Y_path if at least one label present
    if len(X)-missing == 0:
        print('No labeled samples found. Fill `label` column in features_meta.csv with class names from config.')
        return

    # Save features_Y.csv next to features_X
    features_Y_out = os.path.join(out_dir, 'features_Y.csv')
    np.savetxt(features_Y_out, Y, fmt='%i')
    print('Saved labels to', features_Y_out)

    # Load pretrained model
    p = os.path.join('..', model_path) if not os.path.isabs(model_path) else model_path
    if not os.path.exists(p):
        print('Pretrained model not found at', p)
        return
    with open(p, 'rb') as f:
        model = pickle.load(f)

    # Select the labeled subset for evaluation
    labeled_idx = np.where(Y != -1)[0]
    X_l = X[labeled_idx]
    Y_l = Y[labeled_idx]

    # Predict using model (model expects to run predict on raw features, using model.pca internally)
    Y_pred = model.predict(X_l)

    # Metrics
    acc = accuracy_score(Y_l, Y_pred)
    prec = precision_score(Y_l, Y_pred, average='weighted', zero_division=0)
    rec = recall_score(Y_l, Y_pred, average='weighted', zero_division=0)
    f1 = f1_score(Y_l, Y_pred, average='weighted', zero_division=0)

    print('Accuracy:', acc)
    print('Precision (weighted):', prec)
    print('Recall (weighted):', rec)
    print('F1-score (weighted):', f1)
    print('\nClassification report:')
    print(classification_report(Y_l, Y_pred, target_names=classes, zero_division=0))

    # Confusion matrix
    ax, cm = plot_confusion_matrix(Y_l, Y_pred, classes, normalize=False)
    plt.show()


if __name__ == '__main__':
    main()
