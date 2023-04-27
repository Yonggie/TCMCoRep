import GCL.augmentors as A
import os
import copy
from utils.self_made import acc_plot_boundary
os.environ['CUDA_VISIBLE_DEVICES']='8'
import numpy as np
np.random.seed(0)
from sklearn.metrics import normalized_mutual_info_score, f1_score,adjusted_rand_score, pairwise
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score,pairwise
from sklearn.model_selection import train_test_split
from utils.drop import *
import torch.nn.functional as F
import torch
from utils.self_made import *

import torch.nn as nn
from torch.nn import Parameter
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from torch_geometric.nn import GCNConv,GATConv
from utils.initialization import reset, uniform
from torch_geometric.nn import Linear,HANConv#, HGTConv

from hgt_conv_me import HGTConv_me
# from utils.evaluate import *

EPS = 1e-15

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
make_global=lambda feat_dict:torch.vstack([x for x in feat_dict.values()])

class EvaluationOperator():
    def __init__(self,embeds,labels,classification_rate,cluster_k,solver:str='lbfgs') -> None:
        if type(embeds)!=np.ndarray:
            embeds=embeds.detach().cpu().numpy()
        if type(labels)!=np.ndarray:
            labels=labels.detach().cpu().numpy()
        
        self.embeds=embeds
        self.labels=labels
        self.classification_rate=classification_rate
        self.cluster_k=cluster_k
        self.solver=solver
    
    def do_cluster(self):
        k=self.cluster_k
        x=self.embeds
        y=self.labels
        
        estimator = KMeans(n_clusters=k)

        NMI_list = []
        ARI_list=[]
        sample_w=compute_sample_weight(class_weight='balanced', y=y)
        for i in range(10):
            estimator.fit(x,sample_weight=sample_w)
            y_pred = estimator.predict(x)

            NMI_score = normalized_mutual_info_score(y, y_pred, average_method='arithmetic')
            ARI_score=adjusted_rand_score(y,y_pred)
            NMI_list.append(NMI_score)
            ARI_list.append(ARI_score)

        nmi = sum(NMI_list) / len(NMI_list)
        ari=sum(ARI_list) / len(ARI_list)
        return nmi,ari

    def do_classification(self):
        x=self.embeds
        y=self.labels
        train_z,test_z,train_y,test_y=train_test_split(x,y,train_size=self.classification_rate)
        mif1s=[]
        maf1s=[]
        
        for _ in range(10):
            # clf=SVC().fit(train_z,train_y)
            clf = LogisticRegression(solver=self.solver, multi_class='auto',max_iter=10000).fit(train_z,train_y)
            y_pred=clf.predict(test_z)
            microf1_v=f1_score(test_y,y_pred,average='micro')
            macrof1_v=f1_score(test_y,y_pred,average='macro')
            
            mif1s.append(microf1_v)
            maf1s.append(macrof1_v)
        
        return np.mean(mif1s),np.mean(maf1s)
    
    def do_eval(self):
        # do clustering
        nmi,ari=self.do_cluster()
        # do classification
        microf1,macrof1=self.do_classification()
        
        return nmi,ari,microf1,macrof1    



class NodeInfoAttention(nn.Module):
    def __init__(self,meta_info,att_scehme):
        super().__init__()
        self.att_scheme=att_scehme
        if att_scehme=='mlp':
            self.lin_dict=nn.ModuleDict()
            all_len=sum(meta_info[0].values())
            for n_typ,length in meta_info[0].items():
                other_len=all_len-length
                self.lin_dict[n_typ]=nn.Linear(length,1)
                self.lin_dict[n_typ+'other']=nn.Linear(other_len,1)

    def forward(self,feat_me,feat_ohter,node_type):
        if self.att_scheme=='mlp':
            # 不用mean就意味着节点级别
            att1=self.lin_dict[node_type](feat_me).mean()
            att2=self.lin_dict[node_type+'other'](feat_ohter).mean()
            att1,att2=torch.softmax(torch.stack([att1,att2]),dim=0)
        
        return att1,att2

class MetaPathAttention(nn.Module):
    def __init__(self,meta_info,att_scheme):
        super().__init__()
        self.att_scheme=att_scheme
        if att_scheme=='mlp':
            self.lin=nn.Linear(1,1)

    def forward(self,dis,mp_type):
        if self.att_scheme=='mlp':
            x=self.lin(dis).relu_()
            att1,att2=torch.softmax(x,dim=0)
        return att1,att2


class GATEncoder(nn.Module):
    def __init__(self, in_channels, out_channels,drop_prob, heads):
        super(GATEncoder, self).__init__()
        self.drop_prob=drop_prob
        self.conv = GATConv(in_channels, out_channels,heads,)
        self.activation = nn.PReLU() #F.relu()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        x = F.dropout(x, self.drop_prob)
        return self.activation(self.conv(x, edge_index))

class TCMHGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, metadata_info,):
        super().__init__()
        # metadata_info, 2 dim list, first is all node type, second is all relation key list.
        # [['author', 'paper', 'term', 'conference']
        #  [('author', 'to', 'paper'), ('paper', 'to', 'author'), ('paper', 'to', 'term'), ('paper', 'to', 'conference'), ('term', 'to', 'paper'), ('conference', 'to', 'paper')]
        #   ]
        
        self.node_types=metadata_info[0]
        # print(f'node type order: {self.node_types}')
        
        self.conv1 = HGTConv_me(2048, 2*hidden_channels, metadata_info,num_heads, group='sum')
        self.conv2 = HGTConv_me(2*hidden_channels, hidden_channels, metadata_info,num_heads, group='sum')
        
        # self.lin_dict = torch.nn.ModuleDict()
        self.lin=nn.Linear(hidden_channels,out_channels)
        

    def forward(self, x_dict, edge_index_dict):
        
        new_dict:dict=self.conv2(self.conv1(x_dict,edge_index_dict),edge_index_dict)
        out=torch.vstack([self.lin(feat) for feat in new_dict.values()])
        out=F.softmax(out,dim=1)

        return out
    
    def embed(self, x_dict, edge_index_dict):
        new_dict:dict=self.conv2(self.conv1(x_dict,edge_index_dict),edge_index_dict)
        embeds=make_global(new_dict)
        
        return embeds

    def test(self,embeds,labels,classification_rate,cluster_k,solver='lbfgs'):        
        evaler=EvaluationOperator(embeds,labels,classification_rate,cluster_k,solver)
        nmi,ari,microf1,macrof1=evaler.do_eval()
        return nmi,ari,microf1,macrof1

class TCMHAN(nn.Module):
    def __init__(self,hidden_channels,out_channels, num_heads, metadata_info):
        super().__init__()
        self.han_conv=HANConv(2048,hidden_channels,metadata_info,num_heads,)
        self.lin=nn.Linear(hidden_channels,out_channels)
    
    def forward(self,x_dict,edge_index_dict):
        out = self.han_conv(x_dict, edge_index_dict)
        unsoftmax=torch.vstack([self.lin(feat) for feat in out.values()])
        ret=torch.softmax(unsoftmax,dim=1)
        return ret

    def embed(self,x_dict,edge_index_dict):
        out = self.han_conv(x_dict, edge_index_dict)
        embeds=make_global(out)
        return embeds

    def test(self,embeds,labels,classification_rate,cluster_k,solver='lbfgs'):        
        evaler=EvaluationOperator(embeds,labels,classification_rate,cluster_k,solver)
        nmi,ari,microf1,macrof1=evaler.do_eval()
        return nmi,ari,microf1,macrof1


class Deheterofy(nn.Module):
    def __init__(self,de_type,bridge_dim1,bridge_dim2,metadata_info,num_heads,group_type):
        super().__init__()
        self.de_convs = torch.nn.ModuleList()
        self.de_type=de_type
        if de_type=='hgt':
            self.conv1 = HGTConv_me(bridge_dim1, bridge_dim2, metadata_info,
                                num_heads, group=group_type)
            self.conv2 = HGTConv_me(bridge_dim2, bridge_dim2, metadata_info,
                                num_heads, group=group_type)
            
        elif de_type=='han':
            self.conv=HANConv(bridge_dim1,bridge_dim2,metadata_info,num_heads)

    def forward(self,x_dict,edge_index_dict):
        if self.de_type=='hgt':
            return self.conv2(self.conv1(x_dict,edge_index_dict),edge_index_dict)
        elif self.de_type=='han':
            return self.conv(x_dict,edge_index_dict)


class DGCL(torch.nn.Module):
    def __init__(self, feature_size,embed_size,estimator_type,metadata_info,
                encoder_type,dehetero_type,aug,
                bridge_dim1, bridge_dim2, num_heads, group_type,
                activation,dropout,
                summary_type,edge_drop_prob,momentum,
                nce_mean,tau,
                ):
        super(DGCL,self).__init__()
        
        self.sim_type='dot'

        
        
        self.momentum=momentum

        self.aug_type=aug

        self.node_types=metadata_info[0] # come from datase x_dict.keys()
        
        self.lin=Linear(feature_size,bridge_dim1,weight_initializer='glorot')
        # self.lin_dict = torch.nn.ModuleDict()
        # for node_type in self.node_types:
        #     self.lin_dict[node_type] = Linear(feature_size, bridge_dim1)

        # batch normalization
        self.bn_layer=nn.BatchNorm1d(bridge_dim1)
        
        # self.BN_dict1 = torch.nn.ModuleDict()
        # for node_type in self.node_types:
        #     self.BN_dict1[node_type] = nn.BatchNorm1d(bridge_dim1)
        
        # useless for now
        self.BN_dict2 = torch.nn.ModuleDict()
        for node_type in self.node_types:
            self.BN_dict2[node_type] = nn.BatchNorm1d(bridge_dim1)

        self.none_trans=False
        if dehetero_type in ['hgt','han']:
            self.deheterofy=Deheterofy(dehetero_type,bridge_dim1,bridge_dim2,metadata_info,num_heads,group_type)
        elif dehetero_type=='none':
            self.none_trans=True
            bridge_dim2=2048
        elif dehetero_type=='linear':
            self.no_hetero=True
            self.lin=Linear(feature_size,bridge_dim2,weight_initializer='glorot')
            self.bn_layer=nn.BatchNorm1d(bridge_dim2)

        

        # for encoder
        if encoder_type=='gat':
            self.encoder = GATEncoder(bridge_dim2, embed_size,heads=1,drop_prob=dropout)
            self.encoder_neg = GATEncoder(bridge_dim2, embed_size,heads=1,drop_prob=dropout)
        elif encoder_type=='gcn':
            self.encoder=GCNEncoder(bridge_dim2, embed_size, activation, dropout)
            self.encoder_neg=GCNEncoder(bridge_dim2, embed_size, activation, dropout)
        else:
            print('specify encoder type.')
            raise Exception
        
        for param_q, param_k in zip(self.encoder.parameters(), self.encoder_neg.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False 
        
        self.corrupt_type='shuffle'
        self.edge_drop_prob=edge_drop_prob
        
        # for summary
        if summary_type=='avg':
            self.summary_func=AvgReadout()
        else:
            print('specify summary type.')
            raise Exception

        # for loss estimator
        # self.tau: float = 0.07
        self.tau=tau
        self.fc1 = Linear(embed_size, bridge_dim2,weight_initializer='glorot')
        self.fc2 = Linear(bridge_dim2, bridge_dim2,weight_initializer='glorot')
        self.estimator_type=estimator_type
        self.nce_mean=nce_mean

        # for discriminator
        self.embed_size=embed_size
        self.weight=Parameter(torch.Tensor(embed_size, embed_size))
    
    def reset_parameters(self):
        uniform(self.embed_size, self.weight)
        
    def forward(self,x_dict:dict,edge_index_dict,global_edge_index,global_edge_weight):
        self._momentum_update_neg_encoder()
        
        # de-heterofy
        if self.none_trans:
            new_dict=copy.deepcopy(x_dict)
        else:
            new_dict={}
            for node_type, x in x_dict.items():
                new_dict[node_type] = self.lin(x).relu_()
            for node_type, x in new_dict.items():
                new_dict[node_type] = self.bn_layer(x).relu_()

            if not self.no_hetero:
                new_dict=self.deheterofy(new_dict,edge_index_dict)
        
        
        # no linear version
        # out_tmp=[self.lin1(x_dict[node_type]) for node_type in self.node_types]
        out_tmp=[new_dict[node_type] for node_type in self.node_types]

        unauged_features=torch.vstack(out_tmp)
        # ======================deheterofy finished===============================================

        pos_feat=unauged_features
        
        # if self.corrupt_type=='shuffle':
        #     idxx = np.random.permutation(pos_feat.shape[0])
        #     corrupted_feat = pos_feat[idxx]
        # else:
        #     print('corruption type not specified!')
        #     raise Exception

        pos_embed=self.encoder(pos_feat,global_edge_index)

        
        aug_feat1,aug_feat2,aug1_g_edge_index,aug2_g_edge_index=\
            self._glo_augmentation(unauged_features,global_edge_index,global_edge_weight,self.aug_type)
        # corrupted_edge,_=dropout_adj(global_edge_index,p=self.edge_drop_prob)
        
        
        v1_embed=self.encoder(aug_feat1,aug1_g_edge_index)
        # v1_embed=self.encoder(corrupted_feat,aug2_g_edge_index)
        v2_embed=self.encoder_neg(aug_feat2,aug2_g_edge_index)
        

        summary=self.summary_func(pos_embed)
        
        
        return v1_embed,v2_embed,summary
    
    @torch.no_grad()
    def _momentum_update_neg_encoder(self):
        """
        Momentum update of the key encoder
        """
        for pos_param,neg_param in zip(self.encoder.parameters(),self.encoder_neg.parameters()):
            neg_param.data = neg_param.data * self.momentum + pos_param.data * (1. - self.momentum)
            
    @torch.no_grad()
    def _glo_augmentation(self,g_feat,global_edge_index,global_edge_weight,scheme):
        '''
            edge_weight是只有page rank才用的
        '''
        if scheme=='pagerank':
            # pagerank概率化删global图的边
            t_g_edge_index=global_edge_index.t()
            aug1_g_edge_index=prob_drop_edge(t_g_edge_index,global_edge_weight).t().to(DEVICE)
            aug2_g_edge_index=prob_drop_edge(t_g_edge_index,global_edge_weight).t().to(DEVICE)
        elif scheme=='permute':
            aug1_g_edge_index=permute_edges(global_edge_index)
            aug2_g_edge_index=permute_edges(global_edge_index)
        elif scheme=='gcl': # 用GCL
            # pos neg模式 feature masking 用了急剧跌下！
            aug=A.Compose([A.EdgeRemoving(pe=0.5),A.NodeDropping(pn=0.5),A.FeatureMasking(pf=0.5)])
            
            aug_feat1,aug1_g_edge_index,_=aug(g_feat,global_edge_index)
            aug_feat2,aug2_g_edge_index,_=aug(g_feat,global_edge_index)
        else:
            # no aug
            aug1_g_edge_index=global_edge_index
            aug2_g_edge_index=global_edge_index
        
        return aug_feat1,aug_feat2,aug1_g_edge_index,aug2_g_edge_index

    def discriminate(self, z, summary, sigmoid=True):
        r"""Given the patch-summary pair :obj:`z` and :obj:`summary`, computes
        the probability scores assigned to this patch-summary pair.

        Args:
            z (Tensor): The latent space.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        value = torch.matmul(z, torch.matmul(self.weight, summary))
        return torch.sigmoid(value) if sigmoid else value

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def loss(self,pos_embed,neg_embed,summary):
        if self.estimator_type=='jsd':
            # dgi version
            pos_loss = -torch.log(self.discriminate(pos_embed, summary, sigmoid=True) + EPS).mean()
            neg_loss = -torch.log(1 - self.discriminate(neg_embed, summary, sigmoid=True) + EPS).mean()

            distance_loss=0
            
            
            return pos_loss+neg_loss+distance_loss
            
        elif self.estimator_type=='nce':
            # grace version
            
            # h1 = self.projection(pos_embed)
            # h2 = self.projection(neg_embed)
            
            h1=pos_embed
            h2=neg_embed
            
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
            

            ret = (l1 + l2) * 0.5
            # ret=l1
            ret = ret.mean() if self.nce_mean else ret.sum()

            return ret
        else:
            print('loss type not specified!')
            raise Exception
        
    def embed(self,x_dict:dict,edge_index_dict:dict,global_edge_index, global_edge_weight):
        new_dict={}
        # de-heterofy
        if self.none_trans:
            new_dict=copy.deepcopy(x_dict)
        else:
            new_dict={}
            for node_type, x in x_dict.items():
                new_dict[node_type] = self.lin(x).relu_()
            for node_type, x in new_dict.items():
                new_dict[node_type] = self.bn_layer(x).relu_()

            if not self.no_hetero:
                new_dict=self.deheterofy(new_dict,edge_index_dict)
        
            
        
        global_feature=make_global(new_dict)
        final_embed=self.encoder(global_feature,global_edge_index)
        return final_embed
    
    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        if self.sim_type=='dot':
            dis=torch.mm(z1, z2.t())
        elif self.sim_type=='cos':
            # dis_line=[[torch.cosine_similarity(h1,h2,dim=0) for h2 in z2] for h1 in z1] # too slow!
            dis=pairwise.cosine_distances(z1.cpu().detach().numpy(),z2.cpu().detach().numpy())
            dis=torch.tensor(dis).to(DEVICE)
        else:
            dis=torch.mm(z1, z2.t())
        return dis

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
        
        # if only between
        # f = lambda x: torch.exp(x / self.tau)
        # refl_sim = f(self.sim(z1, z1))
        # between_sim = f(self.sim(z1, z2))

        # return -torch.log(between_sim.diag()/ (between_sim.sum(1)))
    
    def test(self,herb_feat,herb_labels,cls_task_train_rate,cluster_k,solver='lbfgs'):
        evaler=EvaluationOperator(herb_feat,herb_labels,cls_task_train_rate,cluster_k,solver)
        nmi,ari,microf1,macrof1=evaler.do_eval()
        return nmi,ari,microf1,macrof1
        

class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk=None):
        if msk is None:
            return torch.mean(seq, 0)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 0) / torch.sum(msk)

class GCNEncoder(torch.nn.Module):
    '''
    GCN with k layers.
    '''
    def __init__(self, in_channels: int, out_channels: int, activation, drop_prob,
                 base_model=GCNConv, k: int = 2):
        '''

        :param in_channels:
        :param out_channels:
        :param activation:
        :param base_model:
        :param k: depth of base model.
        '''
        super(GCNEncoder, self).__init__()
        self.base_model = base_model

        assert k >= 2
        self.k = k
        self.drop_prob=drop_prob
        self.conv = [base_model(in_channels, 2 * out_channels)]
        for _ in range(1, k-1):
            self.conv.append(base_model(2 * out_channels, 2 * out_channels))
        self.conv.append(base_model(2 * out_channels, out_channels))
        self.conv = nn.ModuleList(self.conv)

        self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        
        x = F.dropout(x, self.drop_prob)
        for i in range(self.k):
            x = self.activation(self.conv[i](x, edge_index))
        return x

class GCNEncoder1(torch.nn.Module):
    '''
    GCN with k layers.
    '''
    def __init__(self, in_channels: int, out_channels: int, activation, drop_prob,
                 base_model=GCNConv,):
        '''

        :param in_channels:
        :param out_channels:
        :param activation:
        :param base_model:
        :param k: depth of base model.
        '''
        super(GCNEncoder1, self).__init__()
        self.base_model = base_model

        self.drop_prob=drop_prob
        self.conv = base_model(in_channels, out_channels)
        self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        
        x = F.dropout(x, self.drop_prob)
        x = self.activation(self.conv(x, edge_index))
        return x


class GATEncoder2(nn.Module):
    def __init__(self, in_channels, out_channels, heads):
        super(GATEncoder2, self).__init__()
        self.conv1 = GATConv(in_channels, out_channels,heads,)
        self.conv2 = GATConv(out_channels, out_channels,heads,)
        self.activation1 = nn.PReLU() #F.relu()
        self.activation2 = nn.PReLU()
        

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        tmp=self.activation1(self.conv1(x, edge_index))
        return self.activation2(self.conv2(tmp, edge_index))





class Grace(torch.nn.Module):
    def __init__(self, encoder: GCNEncoder, num_hidden: int, num_proj_hidden: int,
                 tau: float = 0.5):
        super(Grace, self).__init__()
        self.encoder: GCNEncoder = encoder
        self.tau: float = tau

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(between_sim.diag()/ (between_sim.sum(1)))
        
        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = 0):
        # only distance between
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret

class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k_bilinear = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = torch.unsqueeze(c, 1) # c: summary vector, h_pl: positive, h_mi: negative
        c_x = c_x.expand_as(h_pl)

        sc_1 = torch.squeeze(self.f_k_bilinear(h_pl, c_x), 2) # sc_1 = 1 x nb_nodes
        sc_2 = torch.squeeze(self.f_k_bilinear(h_mi, c_x), 2) # sc_2 = 1 x nb_nodes

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2
        logits = torch.cat((sc_1, sc_2), 1)

        return logits

class NodeLevelAttentionLayer(nn.Module):
    def __init__(self, embed_size,nb_features,nb_nodes):
        super(NodeLevelAttentionLayer, self).__init__()
        self.Z = nn.Parameter(torch.FloatTensor( nb_nodes, embed_size))
        self.init_weight()
        self.embed_size=embed_size
        self.nb_features=nb_features
        self.nb_nodes=nb_nodes
        self.mlps = nn.ModuleList([nn.Linear(embed_size, 1) for _ in range(nb_features)])
        self.my_coefs=None

    def forward(self, features):
        aggregated_feat, atten_coef = self.attn_feature(features)
        self.my_coefs=atten_coef
        return aggregated_feat

    def attn_feature(self, features):
        # print('I am in attention!!'+'+'*100)
        att_coef = []
        for i in range(self.nb_features):
            att_coef.append((self.mlps[i](features[i].squeeze())))
        att_coef = F.softmax(torch.cat(att_coef, 1), -1)
        features = torch.cat(features, 0)#.squeeze(0)
        attn_coef_reshaped = att_coef.transpose(1, 0).contiguous().view(-1, 1)

        aggregated_feat = features * attn_coef_reshaped.expand_as(features)
        aggregated_feat = aggregated_feat.view(self.nb_features, self.nb_nodes, self.embed_size)
        aggregated_feat = aggregated_feat.mean(dim=0)

        return aggregated_feat, att_coef

    def init_weight(self):
        nn.init.xavier_normal_(self.Z)

    def loss(self,pos,neg):
        agg_loss = F.triplet_margin_loss(self.Z, pos, neg)
        return agg_loss

class GraphLevelAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features):
        super(GraphLevelAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.my_coefs=None
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.b = nn.Parameter(torch.zeros(size=(1, out_features)))
        nn.init.xavier_uniform_(self.b.data, gain=1.414)
        self.q = nn.Parameter(torch.zeros(size=(1, out_features)))
        nn.init.xavier_uniform_(self.q.data, gain=1.414)
        self.Tanh = nn.Tanh()
        self.leakyrelu = nn.LeakyReLU()

    # input (PN)*F
    def forward(self, total_embeds, P=2):
        if type(total_embeds)==list:
            total_embeds=torch.vstack(total_embeds)
        h = torch.mm(total_embeds, self.W)
        # h=(PN)*F'
        h_prime = self.Tanh(h + self.b.repeat(h.size()[0], 1))
        # h_prime=(PN)*F'
        semantic_attentions = torch.mm(h_prime, torch.t(self.q)).view(P, -1)
        # semantic_attentions = P*N
        N = semantic_attentions.size()[1]
        semantic_attentions = semantic_attentions.mean(dim=1, keepdim=True)
        # semantic_attentions = P*1
        semantic_attentions = F.softmax(semantic_attentions, dim=0)
        self.my_coefs=semantic_attentions
        # print(semantic_attentions)
        semantic_attentions = semantic_attentions.view(P, 1, 1)
        semantic_attentions = semantic_attentions.repeat(1, N, self.in_features)
        #        print(semantic_attentions)

        # input_embedding = P*N*F
        input_embedding = total_embeds.view(P, N, self.in_features)

        # h_embedding = N*F
        h_embedding = torch.mul(input_embedding, semantic_attentions)
        h_embedding = torch.sum(h_embedding, dim=0).squeeze()

        return h_embedding

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

def l1_regularization(model:nn.Module, l1_alpha):
    l1_loss = []
    for module in model.modules():
        if type(module) is GCNEncoder:
            for param in module.parameters():
                l1_loss.append(torch.abs(param).sum())
        # if type(module) is HGTConv_me:
        #     for param in module.parameters():
        #         l1_loss.append(torch.abs(param).sum())
        
    return l1_alpha * sum(l1_loss)

def l2_regularization(model, l2_alpha):
    l2_loss = []
    for module in model.modules():
        if type(module) is GCNEncoder:
            l2_loss.append((module.weight ** 2).sum() / 2.0)
    return l2_alpha * sum(l2_loss)

