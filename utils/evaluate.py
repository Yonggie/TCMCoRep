import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np
np.random.seed(0)
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, pairwise,adjusted_rand_score
from sklearn.utils.class_weight import compute_sample_weight


# def link_prediction(symptom_embeds,herb_embeds):


def l2_sim_search(test_embs, test_lbls):
    if type(test_embs)!=np.ndarray:
        test_embs=test_embs.cpu()
    if type(test_lbls)!=np.ndarray:
        test_lbls = test_lbls.cpu().numpy()
    numRows = test_embs.shape[0]
    # cos sim
    cos_sim_array = pairwise.cosine_similarity(test_embs) - np.eye(numRows)
    # dot_sim_array=test_embs@test_embs.T
    st = []
    for N in [5, 10]:
        indices = np.argsort(cos_sim_array, axis=1)[:, -N:]
        tmp = np.tile(test_lbls, (numRows, 1))
        selected_label = tmp[np.repeat(np.arange(numRows), N), indices.ravel()].reshape(numRows, N)
        original_label = np.repeat(test_lbls, N).reshape(numRows,N)
        st.append(np.round(np.mean(np.sum((selected_label == original_label), 1) / N),4))

    return st
    


def run_kmeans(x, y, k):
    if type(x)!=np.ndarray:
        x=x.cpu()
    if type(y)!=np.ndarray:
        y=y.cpu().numpy()
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

    NMI_score = sum(NMI_list) / len(NMI_list)
    ARI_score=sum(ARI_list) / len(ARI_list)

    return NMI_score,ARI_score


def l1_sim_search(embeds:torch.Tensor,prescriptions:dict,syndrome_idxs,herb_idxs,high_nmi):
    # 说辞：（除去锚点的）所有点都计算距离，
    # prescriptions={(test_symptom_idxs):[[syndrome_idx],[herb_idxs]],...}
    if type(embeds)!=np.ndarray:
        embeds=embeds.cpu().numpy()
    embeds_len=len(embeds)
    
    cos_sim_array = pairwise.cosine_similarity(embeds) - np.eye(embeds_len) # 减掉eye，不要再次找到自己了。

    # cos_sim_array=embeds@embeds.T
    # pairwise.p

    # common distance array
    distances=[sum(cos_sim_array[sympt_idx] \
                        for sympt_idx in group_idx)\
                            for group_idx in prescriptions.keys()]
    
    
    distances=np.array(distances)
    sorted_idxs=np.argsort(-distances,axis=1)# 数值由大到小，相似性也由大到小

    in_targets={idx:True for idx in syndrome_idxs}
    sorted_syndrome_idx=[[idx for idx in single_sorted_idxs if in_targets.get(idx)]\
                                 for single_sorted_idxs in sorted_idxs]
    sorted_syndrome_idx=np.array(sorted_syndrome_idx)
    
    herb_recalls=[]
    for N in [5,10,20]:
        herb_distances=distances[:,herb_idxs]
        herb_sorted_idxs=np.argsort(-herb_distances,axis=1)# 数值由大到小，相似性也由大到小
        result=herb_sorted_idxs[:,:N]
        # 对herb来说，recall更重要。
        all_recalls=[len(set(herb_idx)&set(res_idxs))/len(set(herb_idx)) for (herb_idx,_),res_idxs in zip(prescriptions.values(),result)]
        herb_recalls.append(np.mean(all_recalls))
    
    # syndrome aware
    syndrome_hit_rates=[]
    for N in [1,2,3]: 
        result=sorted_syndrome_idx[:,:N]
        # 对syndrome来说，hit就可以了。
        # all_hit=[int(syndrome_idx in res_idxs) for (_,syndrome_idx),res_idxs in zip(prescriptions.values(),result) ]
        all_hit=[int(len(set(syndrome_idx)&set(res_idxs))!=0) \
            for (_,syndrome_idx),res_idxs in zip(prescriptions.values(),result) ]
        
        syndrome_hit_rates.append(sum(all_hit)/len(all_hit))
        # print("hit rate frac:{}".format(len(all_hit)))
    return syndrome_hit_rates,herb_recalls


def l1_sim_search2step(embeds:torch.Tensor,prescriptions:dict,syndrome_idxs,herb_idxs,method='cos'):
    # 说辞：（除去锚点的）所有点都计算距离，
    # prescriptions={(test_symptom_idxs):[[syndrome_idx],[herb_idxs]],...}
    if type(embeds)!=np.ndarray:
        embeds=embeds.cpu().numpy()
    embeds_len=len(embeds)
    if method=='cos':
        # sim_matrix=pairwise.cosine_distances(embeds)- np.eye(embeds_len) # 减掉eye，不要再次找到自己了。
        sim_matrix = pairwise.cosine_similarity(embeds) - np.eye(embeds_len)
    elif method=='dot':
        sim_matrix = embeds@embeds.T - np.eye(embeds_len) # 减掉eye，不要再次找到自己了。

    # common distance array
    symptom2other_distances=[sum(sim_matrix[sympt_idx] for sympt_idx in group_idx)\
                                for group_idx in prescriptions.keys()]
    
    
    symptom2other_distances=np.array(symptom2other_distances)
    sorted_idxs=np.argsort(-symptom2other_distances,axis=1)# 数值由大到小，相似性也由大到小

    in_targets={idx:True for idx in syndrome_idxs}
    is_herb_idx={idx:True for idx in herb_idxs}
    sorted_syndrome_idx=[[idx for idx in single_sorted_idxs if in_targets.get(idx)]\
                                 for single_sorted_idxs in sorted_idxs]
    sorted_syndrome_idx=np.array(sorted_syndrome_idx)
    
    syndrome_hit_rates=[]

    for N in [1,2,3]: 
        result=sorted_syndrome_idx[:,:N]
        # 对syndrome来说，hit就可以了。
        all_hit=[int(syndrome_idx in res_idxs) for (_,syndrome_idx),res_idxs in zip(prescriptions.values(),result) ]

        # syndrome也有多个时
        # all_hit=[len(set(syndrome_idx)&set(res_idxs))/len(set(syndrome_idx)) \
        #     for (_,syndrome_idx),res_idxs in zip(prescriptions.values(),result) ]
        
        
        syndrome_hit_rates.append(sum(all_hit)/len(all_hit))

    # syndrome找其他
    top3syndrome=sorted_syndrome_idx[:,:3]
    syndrome2other_distances=[sum(sim_matrix[syndrome_idx] \
                                for syndrome_idx in group_idx)\
                                    for group_idx in top3syndrome]
    syndrome2herb_distances=np.array(syndrome2other_distances)
    sorted_syndrome2other_idxs=np.argsort(-syndrome2herb_distances,axis=1)# 数值由大到小，相似性也由大到小
    tmp=[[idx for idx in idxs if is_herb_idx.get(idx)]\
                for idxs in sorted_syndrome2other_idxs]
    sorted_syndrome2herb_idxs=np.array(tmp )
    herb_recalls=[]
    for N in [5,10,20]:
        result=sorted_syndrome2herb_idxs[:,:N]
        # 对herb来说，recall更重要。
        all_recalls=[len(set(herb_idx)&set(res_idxs))/len(set(herb_idx)) for (herb_idx,_),res_idxs in zip(prescriptions.values(),result)]
        herb_recalls.append(np.mean(all_recalls))
        
    
    return syndrome_hit_rates,herb_recalls

def semantic_cluster(embeds:torch.Tensor,prescriptions:dict):
    # 聚在一起的这些点有直接连边的概率
    pass
