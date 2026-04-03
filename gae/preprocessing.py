import numpy as np
import scipy.sparse as sp


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(adj_normalized, adj, features, placeholders):
    # construct feed dictionary
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adj']: adj_normalized})
    feed_dict.update({placeholders['adj_orig']: adj})
    return feed_dict

def create_test_edges_false(edges_all_set, test_edges, adj_rows):
    test_edges_false = set()
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj_rows)
        idx_j = np.random.randint(0, adj_rows)
        if idx_i == idx_j:
            continue
        if (idx_i, idx_j) in edges_all_set:
            continue
        if (idx_j, idx_i) in test_edges_false:
            continue
        test_edges_false.add((idx_i, idx_j))

    return [list(tup) for tup in test_edges_false]

def create_val_edges_false(train_edges, val_edges, edges_all_set, adj_rows):
    train_set = {(min(a, b), max(a, b)) for a, b in train_edges}
    val_set = {(min(a, b), max(a, b)) for a, b in val_edges}
    forbidden = train_set | val_set

    val_edges_false = set()
    while len(val_edges_false) < len(val_edges):
        i = np.random.randint(0, adj_rows)
        j = np.random.randint(0, adj_rows)
        if i == j:
            continue
        if (i, j) in edges_all_set or (j, i) in edges_all_set:
            continue
        if (i, j) in forbidden or (j, i) in forbidden or (j, i) in val_edges_false:
            continue

        val_edges_false.add((i, j))

    return [list(edge) for edge in val_edges_false]

def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    edges_all_set = set(map(tuple, edges_all))
    test_edges_false = create_test_edges_false(edges_all_set, test_edges, adj.shape[0])
    val_edges_false = create_val_edges_false(train_edges, val_edges, edges_all_set, adj.shape[0])

    def ismember(a, b):
        return set(map(tuple, a)) & set(map(tuple, b))

    assert not ismember(test_edges_false, edges_all)
    assert not ismember(val_edges_false, edges_all)
    assert not ismember(val_edges, val_edges_false)
    assert not ismember(val_edges, train_edges)
    assert not ismember(test_edges, train_edges)
    assert not ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false
