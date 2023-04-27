import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import numpy as np
import copy
def shining_print(content):
    print(f'\033[0;37;45m{content} \033[0m')

def seed_everything():
    return
    
def acc_plot_boundary(X,Y,y_pred,pic_name):
    if type(X)!=np.ndarray:
        X=X.cpu().numpy()
    x0=X[:,0]
    x1=X[:,1]
    len_data=len(X)
    x0_min, x0_max = min(x0) - 0.1, max(x0) + 0.1
    x1_min, x1_max = min(x1) - 0.1, max(x1) + 0.1
    xx0=np.linspace(x0_min, x0_max, len_data)
    xx1=np.linspace(x1_min, x1_max, len_data)


    # xx0, xx1 = np.meshgrid(xx0, xx1)
    plt.scatter(x0, x1, y_pred, cmap=plt.cm.Spectral, alpha=0.8)

    plt.scatter(x0, x1, c=Y, s=40)

    plt.xlim(xx0.min(), xx0.max())
    plt.ylim(xx1.min(), xx1.max())
    
    plt.savefig(f'plots/acc/{pic_name}.png')



# y_pred=np.array([1,0])
# Y=np.array([0,1])
# X=np.random.rand(2,2)
# acc_plot_boundary(X,Y,y_pred,'test')

def plot_embedding(embeds,y,pic_name):
    xx=embeds
    if type(xx)!=np.ndarray:
        xx=xx.cpu().numpy()
    
    x_copy=copy.deepcopy(xx)
    
    tsne = TSNE()
    down_dimed = tsne.fit_transform(x_copy)


    plt.scatter(down_dimed[:, 0], down_dimed[:, 1], c=y, s=50, cmap='viridis')  # 颜色是彩色
    plt.savefig(f'plots/{pic_name}.png')

def plot_kmeans(embeds,y):
    xx=embeds
    xx=xx.cpu().numpy()
    y_copy=copy.deepcopy(y)
    x_copy=copy.deepcopy(xx)
    selected_label=[]
    selected_embeds=[]
    for i in range(K):
        value=np.argmax(np.bincount(y_copy))
        # print(f"{value} to be removed.")
        group_idx=np.where(y_copy==value)[0]
        selected_label.append(y[group_idx]) 
        selected_embeds.append(x_copy[group_idx])
        remain_idx=np.where(y_copy!=value)[0]
        y_copy=y_copy[remain_idx]
        
        # print(y_copy)
        x_copy=x_copy[remain_idx]
        print(f'{i}th done.')


    selected_embeds=np.concatenate(selected_embeds)
    selected_label=np.concatenate(selected_label)
    xx=selected_embeds




    km = KMeans(n_clusters=K, init='k-means++', max_iter=30)
    km.fit(xx)
    # 获取簇心
    centroids = km.cluster_centers_
    # 获取归集后的样本所属簇对应值
    y_kmean = km.predict(xx)
    # print(y_kmean)
    # 呈现未归集前的数据
    # plt.scatter(xx[:, 0], xx[:, 1], s=50)  # 颜色默认所有点是蓝色
    # plt.yticks(())
    # plt.show()

    from sklearn.manifold import TSNE
    tsne = TSNE()
    down_dimed = tsne.fit_transform(xx)


    # 呈现分类后的结果
    plt.scatter(down_dimed[:, 0], down_dimed[:, 1], c=y_kmean, s=50, cmap='viridis')  # 颜色是彩色
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=100, alpha=0.5)
    plt.savefig('kmeans.png')

    plt.scatter(down_dimed[:, 0], down_dimed[:, 1], c=selected_label, s=50, cmap='viridis')  # 颜色是彩色
    plt.savefig('original.png')

def probabilize(edge_weights:torch.tensor,method='max-min')->torch.tensor:
    if method=='softmax':
        return torch.softmax(edge_weights)
    if method=='max-min':
        edge_weights=(edge_weights-edge_weights.min())/edge_weights.max()
        return edge_weights
    else:
        raise Exception

def prob_drop_edge(edge_indexs:np.array,edge_weights:torch.tensor):
    '''
        edge_indexs:[[n1,n2],[n3,n2],...]
    '''
    
    edge_indexs=torch.tensor(edge_indexs)
    edge_weights=torch.tensor(edge_weights)

    probs=probabilize(edge_weights)
    remain_mask = torch.bernoulli(1.0-probs).to(torch.bool)
    return edge_indexs[remain_mask]


# example
# edge_indexs=torch.tensor([[1,2],[2,3]])
# x=torch.randn(4,16)
# # 生成有向图
# g=nx.DiGraph(edge_indexs.numpy())
# values=nx.pagerank(g)
# auged_edge_indexs=prob_drop_edge(compute_edge_weight(edge_indexs,values))
# exit()