import numpy as np
import networkx as nx
import csv
from sklearn.preprocessing import normalize, MinMaxScaler

from gae.util import decorrelate_from_degree, CANCER_CODES, DATA_DIR, EDGE_LIST, NODE_FEATURES_WITH_DEGREE

NUM_FEATURES = len(CANCER_CODES)

"""
If create=False, then it loads a pre-existing graph file. 
If create=True, then this function builds a numpy graph from CSV edge list and saves it to file.
"""
def load_adjacency_matrix(create=False):
    if not create:
        return np.load(DATA_DIR + 'input/ind.ca.graph', allow_pickle=True)

    edges, nodes = set(), set()
    with open(EDGE_LIST) as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = int(row['source'])
            tgt = int(row['target'])
            weight = float(row['combined_score_scaled'])

            edges.add((src, tgt, weight))
            nodes.add(src)
            nodes.add(tgt)

    n = len(nodes)
    adj_matrix = np.zeros((n, n))
    for src, tgt, weight in edges:
        adj_matrix[src, tgt] = weight
        adj_matrix[tgt, src] = weight

    np.fill_diagonal(adj_matrix, 1)
    np.save(DATA_DIR + 'input/ind.ca.graph', adj_matrix)
    return adj_matrix

"""
If create=False, then it loads a pre-existing feature vector file of GWAS z-scores. 
If create=True, then this function builds a standardized numpy feature vector from a CSV of node features. 
"""
def load_feature_vector(n, create=True):
    if not create:
         return np.load(DATA_DIR + 'input/ind.ca.allx', allow_pickle=True)
    node_degrees = np.zeros(n)
    with open(NODE_FEATURES_WITH_DEGREE) as f:
        reader = csv.DictReader(f)
        feature_vector = np.zeros((n, NUM_FEATURES))
        for row in reader:
            features = [row[f'zstat_{code}'] for code in CANCER_CODES]
            for i in range(len(features)):
                feature_vector[int(row['node_idx']), i] = float(features[i])
            node_degrees[int(row['node_idx'])] = int(row['degree'])

    # Use residualized scores as features instead
    feature_vector = decorrelate_from_degree(feature_vector, node_degrees)

    scaler = MinMaxScaler()
    feature_vector = scaler.fit_transform(feature_vector)
    feature_vector = normalize(feature_vector, norm='l2')
    np.save(DATA_DIR + 'input/ind.ca.allx', feature_vector)
    return feature_vector, node_degrees

"""
Returns the feature vector and graph to be input to the GAE.
"""
def load_data():
    graph = load_adjacency_matrix()
    graph[graph != 0] = 1
    features, node_degrees = load_feature_vector(graph.shape[0])
    adj = nx.adjacency_matrix(nx.from_numpy_array(graph))
    return adj, features
