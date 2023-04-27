import os
os.environ['CUDA_VISIBLE_DEVICES']='7'
from itertools import combinations
import copy
import torch
import pickle
from torch_geometric.data import InMemoryDataset, Data
import numpy as np
import scipy.sparse as sp
import networkx as nx
from collections import defaultdict

class HeteroTCMDataSet(InMemoryDataset):
    '''
    edge_index： all edge in graph
    '''

    def __init__(self, root, dataset_name, feature_size, transform=None, pre_transform=None,
                 show=False):
        self.show = show
        print(f'feature size: {feature_size}')

        super(HeteroTCMDataSet, self).__init__(root, transform, pre_transform)
        

        # required: data.x_dict, data.edge_index_dict
        # key type: (a,rel,b)
        base = f"datasets/{dataset_name}/"
        
        try:            
            self.data=torch.load(base+'data_package.pt')
            print('successfully loaded from data package.')
            return
        except:
            print('regenerating data.')
        
        x_dict = pickle.load(open(base + 'x_dict.pkl', 'rb'))
        # tensor_x_dict={'herb':torch.tensor(x_dict['herb'],dtype=torch.float),
        #             'syndrome':torch.tensor(x_dict['syndrome'],dtype=torch.float),
        #             'symptom':torch.tensor(x_dict['symptom'],dtype=torch.float),
        #             }
        tensor_x_dict={key:torch.tensor(x_dict[key],dtype=torch.float) for key in x_dict}
        l1_labels=pickle.load(open(base+'l1_labels.pkl','rb'))
        l_id2g_id=pickle.load(open(base+'l_id2g_id.pkl','rb'))
        directed_meta_edge_pairs=pickle.load(open(base+'meta_edge_pairs.pkl','rb'))
        
        # 删除herb与symptom的
        # if dataset_name=='longhua':
        #     del directed_meta_edge_pairs[('herb', 'to', 'symptom')]
            
        if dataset_name=='longhua' or dataset_name=='qihuang':
            idx_prescriptions=pickle.load(open(base+'prescriptions.pkl','rb'))
        else:
            idx_prescriptions=None
            

        meta_edge_pairs=copy.deepcopy(directed_meta_edge_pairs)
        # meta= a-b
        for meta in directed_meta_edge_pairs:
            left_cls,_,right_cls=meta
            if left_cls=='herb' and right_cls=='symptom':continue
            if left_cls=='symptom' and right_cls=='herb':continue
            reversed_meta=(right_cls,'to',left_cls)
            
            local_edge_pairs=meta_edge_pairs[meta]
            # 双向化
            meta_edge_pairs[reversed_meta]=[pair[::-1] for pair in local_edge_pairs]
        
        
        
        # transposed
        np_meta_edge_pairs={key:torch.tensor(np.array(edge_pairs).T,dtype=torch.long) for key,edge_pairs in meta_edge_pairs.items()}
        
        edge_weight_dict={}
        for key,edge_pairs in meta_edge_pairs.items():
            g=nx.from_edgelist(edge_pairs)
            pr_dict=nx.pagerank(g)
            edge_weights=[(pr_dict[n1]+pr_dict[n2])/2 for n1,n2 in edge_pairs]
            edge_weight_dict[key]=edge_weights
            
        # 形成global graph edge pair
        g_edge_pairs=[]
        for edge_type,local_edge_pairs in meta_edge_pairs.items():
            # print(edge_type)
            l_cls,_,r_cls=edge_type
            left_dict=l_id2g_id[l_cls]
            right_dict=l_id2g_id[r_cls]
            g_edge_pairs.extend([(left_dict[l_idx],right_dict[r_idx]) for l_idx,r_idx in local_edge_pairs])
        # global weight
        
        g=nx.from_edgelist(g_edge_pairs)
        pr_dict=nx.pagerank(g)
        global_edge_weights=[(pr_dict[n1]+pr_dict[n2])/2 for n1,n2 in g_edge_pairs]

        g_edge_pairs=torch.tensor(g_edge_pairs,dtype=torch.long).T
        # label的顺序，就是最后得到embeddings的顺序。
        node_types=list(x_dict.keys()) # 此处应当与gen_data.py 中node_storage同序

        # herb used
        # herb_label_mask=pickle.load(open(base+'label_mask.pkl','rb'))
        raw_herb_labels_with_unlabel=pickle.load(open(base+'herb_labels.pkl','rb'))
        labeled_idx=pickle.load(open(base+'herb_labeled_idx.pkl','rb'))
        # name_herb_labels=raw_herb_lables[herb_label_mask]
        
        # label名称id化
        diff_labels=sorted(list(set(raw_herb_labels_with_unlabel)))
        label_name2label_idx={name:idx for idx,name in enumerate(diff_labels)}
        list_herb_label=[label_name2label_idx[name] for name in raw_herb_labels_with_unlabel]
        raw_herb_labels=np.array(list_herb_label,dtype=np.int8)
        # metadata_info, 2 dim list, first is all node type, second is all relation key list.
        # [['author', 'paper', 'term', 'conference']
        #  [('author', 'to', 'paper'), ('paper', 'to', 'author'), ('paper', 'to', 'term'), ('paper', 'to', 'conference'), ('term', 'to', 'paper'), ('conference', 'to', 'paper')]
        #   ]
        
        
        meta_data_info=[{n_typ:len(feat) for n_typ,feat in x_dict.items()},list(meta_edge_pairs.keys()),]
        data = Data(
            x_dict=tensor_x_dict,
            edge_index_dict=np_meta_edge_pairs,
            edge_weight_dict=edge_weight_dict, # useless for now?
            global_edge_weight=global_edge_weights,
            l1_labels=l1_labels,
            herb_labels=raw_herb_labels,
            herb_labeled_idx=labeled_idx,
            meta_data_info=meta_data_info,
            g_prescription_pairs=idx_prescriptions,
            global_edge_index=g_edge_pairs,#全局edge index，里面的id全都是全局id
        )
        torch.save(data,base+'data_package.pt')
        self.data = data

    def gen_node2pos_sample(self, edge_index,node_type1,node_type2,):
        x2y_dict = dict(edge_index)
        y2x_dict={b:a for a,b in x2y_dict.items()}
        xyx_edge_index = [(x,y,y2x_dict[y]) for x,y in x2y_dict.items()]
        


        return xyx_edge_index

    def convert_to_xyx_all_edge_index(self, edge_index,node_type1,node_type2):
        cvt_edge_index=edge_index[::-1]
        y_dict=defaultdict(set)
        for x,y in edge_index:
            y_dict[y].add(x)
        
        mp_instance_idxs = []
        for y,xs in y_dict.items():
            mp_instance_idxs.extend([(x1,y,x2) for x1,x2 in combinations(xs,2)])
        

        return mp_instance_idxs
    
    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def download(self):
        return

    @staticmethod
    def show_graph(net):
        import networkx as nx
        import matplotlib.pyplot as plt
        G = nx.from_edgelist(net)
        nx.draw(G, pos=nx.spring_layout(G),
                node_color='b',
                edge_color='r',
                with_labels=False,
                font_size=1,
                node_size=2)
        plt.show()

    def process(self):
        return
