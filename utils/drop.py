import os
os.environ['CUDA_VISIBLE_DEVICES']='4,5'
import torch
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_sparse import coalesce
from utils.num_nodes import maybe_num_nodes
from itertools import combinations,product
# from num_nodes import maybe_num_nodes


def add_edge(edge_index,p):
    print(type(edge_index))
    exit()
    all_nodes=set(edge_index[0])^set(edge_index[1])
    existing_pair=set([tuple(item) for item in list(edge_index.T)])
    non_existing_pair=existing_pair-set(combinations(all_nodes,2))
    non_existing_pair_list=sorted(list(non_existing_pair))
    
    length_non=len(non_existing_pair)
    randn=torch.randn(1,length_non)
    selected_pairs=[non_existing_pair_list[idx] for idx,prob in enumerate(randn) if prob >p]
    added_pairs=torch.tensor(selected_pairs,dtype=torch.int)
    return added_pairs

def drop_nodes(tensor_edge_index):

    node_num=max(tensor_edge_index[0].max(),tensor_edge_index[1].max())+1
    _, edge_num = tensor_edge_index.shape
    drop_num = int(node_num / 10)

    idx_drop = np.random.choice(node_num, drop_num, replace=False)
    idx_nondrop = [n for n in range(node_num) if not n in idx_drop]
    idx_dict = {idx_nondrop[n]:n for n in list(range(node_num - drop_num))}

    # data.x = data.x[idx_nondrop]
    edge_index = tensor_edge_index.cpu().numpy()

    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    edge_index = adj.nonzero().t()


    # edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)]
    # edge_index = [[edge_index[0, n], edge_index[1, n]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)] + [[n, n] for n in idx_nondrop]
    # data.edge_index = torch.tensor(edge_index).transpose_(0, 1)

    return edge_index


def permute_edges(tensor_edge_index):

    node_num, edge_num = tensor_edge_index.shape
    permute_num = int(edge_num / 10)

    edge_index = tensor_edge_index.T.cpu().numpy()

    idx_add = np.random.choice(node_num, (permute_num, 2))
    # idx_add = [[idx_add[0, n], idx_add[1, n]] for n in range(permute_num) if not (idx_add[0, n], idx_add[1, n]) in edge_index]

    # edge_index = np.concatenate((np.array([edge_index[n] for n in range(edge_num) if not n in np.random.choice(edge_num, permute_num, replace=False)]), idx_add), axis=0)
    # edge_index = np.concatenate((edge_index[np.random.choice(edge_num, edge_num-permute_num, replace=False)], idx_add), axis=0)
    edge_index = edge_index[np.random.choice(edge_num, edge_num-permute_num, replace=False)]
    # edge_index = [edge_index[n] for n in range(edge_num) if not n in np.random.choice(edge_num, permute_num, replace=False)] + idx_add
    

    return torch.tensor(edge_index).T.to(DEVICE)


def edge_permutate_me(edge_index,p_add,p_drop):
    # print('edge aug start')
    all_nodes=product(list(set(edge_index[0])), list(set(edge_index[1])))
    rand_nodes=list(all_nodes)[:20]
    existing_pair=set([tuple(item) for item in list(edge_index.T)])
    existing_pair_list=list(existing_pair)
    len_existing=len(existing_pair)
    randn=torch.randn(1,len_existing).squeeze_()
    selected_pairs=[existing_pair_list[idx] for idx,prob in enumerate(randn) if prob > p_drop]
    remaining_pairs=torch.tensor(selected_pairs,dtype=torch.int)
    # print('edge aug start2')
    non_existing_pair=set(combinations([[],[]],2))-existing_pair
    non_existing_pair_list=list(non_existing_pair)
    # print('edge aug start3')
    length_non=len(non_existing_pair)
    randn=torch.randn(1,length_non).squeeze_()
    selected_pairs=[non_existing_pair_list[idx] for idx,prob in enumerate(randn) if prob <p_add]
    added_pairs=torch.tensor(selected_pairs,dtype=torch.int)
    
    auged_pairs=torch.cat((remaining_pairs,added_pairs))
    # print('edge aug done')
    return auged_pairs.T


def filter_adj(row, col, edge_attr, mask):
    return row[mask], col[mask], None if edge_attr is None else edge_attr[mask]


def dropout_adj(edge_index, edge_attr=None, p=0.5, force_undirected=False,
                num_nodes=None, training=True):
    r"""Randomly drops edges from the adjacency matrix
    :obj:`(edge_index, edge_attr)` with probability :obj:`p` using samples from
    a Bernoulli distribution.
    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        p (float, optional): Dropout probability. (default: :obj:`0.5`)
        force_undirected (bool, optional): If set to :obj:`True`, will either
            drop or keep both edges of an undirected edge.
            (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        training (bool, optional): If set to :obj:`False`, this operation is a
            no-op. (default: :obj:`True`)
    """

    if p < 0. or p > 1.:
        raise ValueError('Dropout probability has to be between 0 and 1, '
                         'but got {}'.format(p))

    if not training:
        return edge_index, edge_attr

    N = maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index

    if force_undirected:
        row, col, edge_attr = filter_adj(row, col, edge_attr, row < col)

    mask = edge_index.new_full((row.size(0), ), 1 - p, dtype=torch.float)
    mask = torch.bernoulli(mask).to(torch.bool)

    row, col, edge_attr = filter_adj(row, col, edge_attr, mask)

    if force_undirected:
        edge_index = torch.stack(
            [torch.cat([row, col], dim=0),
             torch.cat([col, row], dim=0)], dim=0)
        if edge_attr is not None:
            edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
        edge_index, edge_attr = coalesce(edge_index, edge_attr, N, N)
    else:
        edge_index = torch.stack([row, col], dim=0)

    return edge_index, edge_attr


def dropout_node(edge_index,p):
    N=int(edge_index.max())
    rands=np.random.rand(N)
    all_nums=set(edge_index[0])| set(edge_index[1])
    is_in={num:prob>p for num,prob in zip(all_nums,rands)}
    for i in range(edge_index.shape[1]):
        if is_in.get(edge_index[:,i][0]) and is_in.get(edge_index[:,i][1]):
            pass
        else:
            np.delete(edge_index,i,axis=1)
    
    deleted_node_idx=list(is_in.values())
    return edge_index,deleted_node_idx


def torch_delete(arr,index,axis):
    if axis==0:
        arr1 = arr[0:index]
        arr2 = arr[index+1:]
    elif axis==1:
        arr1 = arr[:,:index]
        arr2 = arr[:,index+1:]
    return torch.cat((arr1,arr2),dim=axis)

def drop_feature(x, drop_prob=0.5):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x

def dropout_edge_with_p(edge_index, p_mask, edge_attr=None, force_undirected=False, num_nodes=None, training=True):
    r"""Randomly drops edges
    :obj:`(edge_index, edge_attr)` with probability mask.
    Args:
        edge_index (LongTensor): The edge indices.
        p_mask (Tensor, optional): Dropout probability mask matrix.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        force_undirected (bool, optional): If set to :obj:`True`, will either
            drop or keep both edges of an undirected edge.
            (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        training (bool, optional): If set to :obj:`False`, this operation is a
            no-op. (default: :obj:`True`)
    """
    p_mask=probablize_mask(p_mask,'linear')
    if sum(sum(p_mask>1))>0 or sum(sum(p_mask<0))>0 :
        raise ValueError('Dropout probability has to be between 0 and 1')

    if not training:
        return edge_index, edge_attr

    N = maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index

    if force_undirected:
        row, col, edge_attr = filter_adj(row, col, edge_attr, row < col)

    # mask = edge_index.new_full((row.size(0), ), 1 - p, dtype=torch.float)
    mask = torch.bernoulli(p_mask).to(torch.bool)

    row, col, edge_attr = filter_adj(row, col, edge_attr, mask)

    if force_undirected:
        edge_index = torch.stack(
            [torch.cat([row, col], dim=0),
             torch.cat([col, row], dim=0)], dim=0)
        if edge_attr is not None:
            edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
        edge_index, edge_attr = coalesce(edge_index, edge_attr, N, N)
    else:
        edge_index = torch.stack([row, col], dim=0)

    return edge_index, edge_attr

def drop_feature_with_p(x, p_mask):
    p_mask = probablize_mask(p_mask, 'linear')
    mask = torch.bernoulli(p_mask).to(torch.bool)
    # drop_mask = torch.empty(
    #     (x.size(1), ),
    #     dtype=torch.float32,
    #     device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    # x[:, mask] = 0
    x[mask] = 0

    return x

def convert(p,k=None,b=None):
    '''
    convert non-zero-one to zero-one with linear-clip function
    :param p: original value
    :param k: k in [0, +∞]
    :param b: b in [0,1], 0 if normal drop, 1 preserve all edges.
    :return: converted probability
    '''

    # y=k*p+b
    y=0.1*p+0
    y[y>1]=1
    y[y<0]=0
    return y

def probablize_mask(p_mask, convert_func):
    if convert_func=='linear':
        func=convert
    elif convert_func=='sigmoid':
        func=nn.Sigmoid()
    elif convert_func=='tanh':
        func = nn.Tanh()
    tmp=p_mask.clone()
    p_mask=func(tmp.detach())

    return p_mask

def shuffle_corrupt(features):
    # shuffled features (corruption operation)
    nb_nodes=len(features)
    idx = np.random.permutation(nb_nodes)
    shuf_fts = features[idx]
    return shuf_fts