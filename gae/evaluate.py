import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
import optuna
from sklearn.decomposition import PCA

from util import read_cosmic_labels, node_id_to_entrez_id, sigmoid, CANCER_CODES, NODE_FEATURES_WITH_DEGREE

"""
Loads feature vectors.
"""
def load_feature_vector():
    df = pd.read_csv(NODE_FEATURES_WITH_DEGREE).sort_values('node_idx')
    label_cols = [f'p_{code}' for code in CANCER_CODES]
    feat_vec_cols = [f'zstat_{code}' for code in CANCER_CODES]

    feature_vector = df[feat_vec_cols].to_numpy()
    labels_df = df[label_cols]
    node_degrees = df[['degree']].to_numpy()

    mask = labels_df.values < 2e-6
    labels = (mask.sum(axis=1) >= 2).astype(int)

    return feature_vector, node_degrees, labels, labels_df.to_numpy()

"""
Returns different combinations of labeled data for experimentation
"""
def get_final_labels(eid_to_nodeid, gwas_labels):
    multi_cancer, one_cancer, no_cancer = read_cosmic_labels(eid_to_nodeid)
    gwas_copy = gwas_labels.copy()
    gwas_labels[multi_cancer] = 1
    gwas_labels[one_cancer + no_cancer] = 0
    cosmic_indices = list(sorted(multi_cancer + one_cancer + no_cancer))
    cosmic_only = np.zeros_like(gwas_labels)
    cosmic_only[multi_cancer] = 1
    return gwas_labels, cosmic_only, cosmic_indices, gwas_copy

"""
Optimize hyperparameters for the XGB Classifier Using Optuna
"""
def opt_baseline(emb, labels):
    def objective(trial):
        params = {
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 100),
            'eval_metric': 'aucpr',
        }
        warnings.filterwarnings('ignore', category=FutureWarning)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)
        ap_scores = []

        for train_idx, test_idx in cv.split(emb, labels):
            train_x, train_y = emb[train_idx], labels[train_idx]
            test_x, test_y = emb[test_idx], labels[test_idx]
            x, y = train_x, train_y
            clf = XGBClassifier(**params, learning_rate= 0.13795829071334384,
                            reg_alpha= 1.881513201135225,
                            reg_lambda=3.9115331313601316, random_state=17)
            clf.fit(x, y)

            pred_probs = clf.predict_proba(test_x)[:, 1]
            ap = average_precision_score(test_y, pred_probs)
            ap_scores.append(ap)
        return np.mean(ap_scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=200)
    print(study.best_params)


"""
Run XGBClassifier to evaluate GAE embeddings on CGC labels
"""
def run_model(emb, labels, vec_name='feat_vec'):
    warnings.filterwarnings('ignore', category=FutureWarning)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)
    roc_scores, f1_scores, ap_scores = [], [], []

    for train_idx, test_idx in cv.split(emb, labels):
        train_x, train_y = emb[train_idx], labels[train_idx]
        test_x, test_y = emb[test_idx], labels[test_idx]

        x, y = train_x, train_y

        weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=train_y)
        clf = XGBClassifier(scale_pos_weight=weights[1] / weights[0],
                            random_state=17,
                            learning_rate= 0.13795829071334384,
                            reg_alpha= 1.881513201135225,
                            reg_lambda=3.9115331313601316,
                            eval_metric='aucpr')
        clf.fit(x, y)

        pred_probs = clf.predict_proba(test_x)[:, 1]
        pred = (pred_probs >= 0.5).astype(int)

        f1, roc, ap = f1_score(test_y, pred), roc_auc_score(test_y, pred_probs), average_precision_score(test_y, pred_probs)
        f1_scores.append(f1)
        roc_scores.append(roc)
        ap_scores.append(ap)

    print(f"{vec_name} AUC: {100 * np.mean(roc_scores):.3f}")
    print(f"{vec_name} F1: {100 * np.mean(f1_scores):.3f}")
    print(f"{vec_name} Average Precision: {100 * np.mean(ap_scores):.3f}\n")


def standardize(emb):
    # emb = decorrelate_from_degree(emb, node_degrees)
    emb = MinMaxScaler().fit_transform(emb)
    # emb = normalize(emb, norm='l2')
    return emb


feature_vector, node_degrees, gwas_labels, pvalues = load_feature_vector()
gae_embeddings = np.load('../data/output/model_best.npy')
adj_rec = sigmoid(np.dot(gae_embeddings, gae_embeddings.T))
nodeid_to_eid, eid_to_nodeid = node_id_to_entrez_id()
labels, cosmic_only, cosmic_indices, gwas_labels = get_final_labels(eid_to_nodeid, gwas_labels)

pca = PCA(n_components=0.90, random_state=17)
zstat_pca = pca.fit_transform(MinMaxScaler().fit_transform(feature_vector))
emb_enriched = np.hstack([standardize(gae_embeddings), zstat_pca])

final_embeddings = np.hstack((standardize(gae_embeddings), feature_vector))

run_model(gae_embeddings, cosmic_only, 'gae embeddings ONLY')
run_model(final_embeddings, cosmic_only, 'gae embeddings + raw feature vector')
run_model(emb_enriched, cosmic_only, 'gae embeddings + pca transformed feature vector')

# opt_baseline(gae_embeddings, cosmic_only)
