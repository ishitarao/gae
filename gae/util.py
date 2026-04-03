import pandas as pd
import numpy as np

DATA_DIR = '../data/'
RAW_DATA_DIR = '../data/raw_input/'

NODE_ID_MAPPING = RAW_DATA_DIR + 'string/final_node_id_mapping.csv'
EDGE_LIST = RAW_DATA_DIR + 'string/final_edge_list.csv'
NODE_FEATURES = RAW_DATA_DIR + 'string/final_node_features.csv'
NODE_FEATURES_WITH_DEGREE = RAW_DATA_DIR + 'string/final_node_features_with_degree.csv'

COSMIC_LABELS = RAW_DATA_DIR + 'cgc_tier1_broad_pleiotropic.csv'
CANCER_CODES = {'brca', 'coca', 'luca', 'ovca', 'prca'}


"""
Uses Linear Regression to decorrelate a node from its degree.
The degree of a gene measures the extent to which that gene has been studied. 
"""
def decorrelate_from_degree(embeddings, degrees):
    log_deg = np.log1p(degrees).reshape(-1, 1)
    X = np.hstack([log_deg, np.ones((len(log_deg), 1))])  # (n, 2) — slope + intercept
    beta, _, _, _ = np.linalg.lstsq(X, embeddings, rcond=None)  # (2, 15737)
    return embeddings - X @ beta


def node_id_to_entrez_id():
    df = pd.read_csv(NODE_ID_MAPPING)
    nodeid_to_eid = dict(zip(df['node_idx'], df['entrez_id']))
    eid_to_nodeid = dict(zip(df['entrez_id'], df['node_idx']))

    return nodeid_to_eid, eid_to_nodeid


# sorted by node idx, returns numpy array
def get_degrees():
    df = pd.read_csv(NODE_FEATURES_WITH_DEGREE)
    df = df.iloc[:, -3:-1]
    df = df.sort_values(by=df.columns[0])
    return df.iloc[:, -1].to_numpy()


def read_cosmic_labels(eid_to_nodeid):
    df = pd.read_csv(COSMIC_LABELS)
    get_node_id = lambda x: x["Entrez_GeneId"].map(eid_to_nodeid).dropna().astype(int).tolist()

    multi_cancer = df[df["n_project_cancers"] > 1]
    one_cancer = df[df["n_project_cancers"] == 1]
    no_cancer = df[df["n_project_cancers"] == 0]

    return get_node_id(multi_cancer), get_node_id(one_cancer), get_node_id(no_cancer)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))