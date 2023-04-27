import os
os.environ['CUDA_VISIBLE_DEVICES']='7'
from datetime import datetime
from torch.optim.lr_scheduler import MultiStepLR
import itertools
import torch.optim as optim
from HeteroTCMDataSet import *
from models import *
from utils.self_made import *
from torch.utils.tensorboard import SummaryWriter
import itertools
from utils.evaluate import l1_sim_search2step,l2_sim_search

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



PIC_NAME='pa60keep2_no_res'
TRAIN_SAFE=False
EVAL='cos' # cos/dot
TRAIN_SAFE_AIM='0421_tcm_check'
embed_aim='ablation/'

taus=[0.07] # exp tau=0.07
MOMENTUM=0.9 # exp mo=0.9
# for training control
PATIENCE=20000 # bigger than epoch, useless for now.
G_EPOCH_MAX=10000 # tcm 1500, chp/longhua 10000

META_SELECT=['herb-syndrome', 'herb-symptom', 'syndrome-symptom']
FEATURE_SIZE=2048

ATT_SCHEME='mlp'

SUMMARY_TYPE='avg'

if TRAIN_SAFE:
    embeds=[16]#*pow(2,i) for i in range(5)]
    
else:
    embeds=[16*pow(2,i) for i in range(5)]

estimators=['nce']#,'nce']#,'jsd','nce',]
bridge_dim_pairs=[[512,128]]#,[1024,512]]
datasets=['longhua']#,'chp','tcm']#'longhua']#,'chp','tcm']
augs=['gcl']#'pagerank']# 'gcl',

# confirmed options: encoder=gcn, dehetero=hgt, 
# adjustable options:  bridge, dropout, hgt_heads, hgt_group_type, l1/l2, 
encoders=['gcn','gat']#,'gat']#,'gat']
dropout_probs=[0.5]#[0,0.3,0.5]


deheteros=['none','linear','hgt','han']#,'han']# 'linear','hgt','han','gat','gcn'    
hgt_n_heads=[2]#,4,8]

if deheteros[0]=='hgt':
    group_types=['mean']#,'sum']#'max','min']#,'mean','min','max']
else:
    group_types=['_']#'sum','mean','min','max']


reg_types=['l1']#,'l2']

reg_coefs=[0]#,0.05]#,0.01]#,0.7]#,0.9]#[0.1,0.01,0.001]
lrs=[0.001,]#0.001]#[0.1,0.001,0.005,0.0001]

# edge permutation的时候用
if estimators[0]=='permute':
    edge_drop_prob=[0.1,0.3,0.5,0.7,0.9]
else:
    edge_drop_probs=[0]

time_stamp = "{0:%Y-%m-%d %H-%M-%S}".format(datetime.now())
for opt in itertools.product(datasets,group_types,lrs,edge_drop_probs,estimators,bridge_dim_pairs,hgt_n_heads,encoders,dropout_probs,reg_types,reg_coefs,embeds,augs,taus,deheteros):
    dataset,group_type,lr,edge_drop_prob,estimator,bridge_dim_pair,hgt_n_head,encoder,dropout_prob,reg_type,reg_coef,embed_size,aug,tau,de_type=opt
    
    TAU=tau
    DEHETERO=de_type
    LR=lr

    DATA_SET=dataset
    EMBED_SIZE=embed_size
    EDGE_DROP_PROB=edge_drop_prob


    ESTIMATOR=estimator
    NCE_MEAN=True # if ESTIMATOR is not nce, then this has no use.

    # for HGT
    GROUP_TYPE=group_type
    BRIDGE_DIM_PAIR=bridge_dim_pair
    bridge_dim1,bridge_dim2=BRIDGE_DIM_PAIR
    N_HEADS=hgt_n_head
    

    # for encoder
    ENCODE=encoder
    ACT= nn.PReLU()#F.relu
    DROPOUT=dropout_prob

    # regularization
    REG_TYPE=reg_type
    REG_COEF=reg_coef
    base = f"datasets/{DATA_SET}/"
    hetero_data_class=HeteroTCMDataSet(base,DATA_SET,FEATURE_SIZE)
    
    hetero_data=hetero_data_class.data
    hetero_data.to(device)
    options=f"bridge={bridge_dim_pair}/lr={lr}/dropout={DROPOUT}/mo={MOMENTUM}/epoch={G_EPOCH_MAX}"
    tmp=f'nhead{hgt_n_head}'

    print_str=f"{DATA_SET}: aug method{aug}, {embed_size}embed, {estimator} {encoder} dehetero{DEHETERO}{hgt_n_head} {options} {reg_type}={reg_coef} bridge={bridge_dim_pair} lr={lr} dropout={DROPOUT}"
    print(print_str)
    # 网络结构优先，控制参数次之。
    
    # useless for now
    save_model_name=f"best_{ENCODE}_head{hgt_n_head}_group_type={group_type}_l1{reg_coef}_dropout{DROPOUT}_{EMBED_SIZE}"

    if TRAIN_SAFE :
        tb_save=f"runs/train_safe/{TRAIN_SAFE_AIM}/{DEHETERO}/{DATA_SET}/{encoder}/{options}/{reg_type}={reg_coef}_{GROUP_TYPE}/tau={tau}"
    else:
        # tb_save=f"runs/search_best_high_nmi/{estimator}/{encoder}/hgt{hgt_n_head}+{hgt_layer}/{options}/{reg_type}={reg_coef}"
        if embed_aim=='ablation':
            tb_save=f"runs/{embed_aim}/{dataset}/{DEHETERO}/{encoder}_{group_type}/"
        else:
            tb_save=f"runs/{embed_aim}/pa={PATIENCE}/{dataset}/{encoder}/nhead{hgt_n_head}/{GROUP_TYPE}/{reg_type}={reg_coef}_dropout={DROPOUT}"
    tb=SummaryWriter(log_dir=tb_save)
    
    print('model constructing...')
    dgcl=DGCL(bridge_dim1=bridge_dim1,bridge_dim2=bridge_dim2,num_heads=N_HEADS,
            metadata_info=hetero_data.meta_data_info,edge_drop_prob=EDGE_DROP_PROB,group_type=GROUP_TYPE,
            feature_size=FEATURE_SIZE,embed_size=EMBED_SIZE,
            dehetero_type=DEHETERO,encoder_type=ENCODE,activation=ACT,dropout=DROPOUT,
            summary_type=SUMMARY_TYPE,estimator_type=ESTIMATOR,nce_mean=NCE_MEAN,
            aug=aug,
            momentum=MOMENTUM,tau=TAU)
    dgcl.to(device)

    # optimizer = optim.SparseAdam(list(dmgi_model.parameters()), lr=LR)
    
    optimizer = optim.Adam(list(dgcl.parameters()), lr=LR)
    scheduler = MultiStepLR(optimizer, milestones=[50,200], gamma=1) # lr不变


    # source code example=============================================================
    print('on training...')
    dgcl.train()
    best = 1e9
    cnt_wait=0
    real_epoch=-1
    
    save_model_path=f'./saved_models/{time_stamp}/'
    if not os.path.exists(save_model_path) and not TRAIN_SAFE:
        os.mkdir(save_model_path)
    for epoch in range(G_EPOCH_MAX):
        optimizer.zero_grad()

        v1_embed, v2_embed, summary_vec=\
            dgcl(hetero_data.x_dict, hetero_data.edge_index_dict,hetero_data.global_edge_index,hetero_data.global_edge_weight)

        loss = dgcl.loss(v1_embed, v2_embed,summary_vec)

        # 只对gcn有用
        if REG_TYPE=='l1':
            loss=loss+l1_regularization(dgcl,REG_COEF)
        elif REG_TYPE=='l2':
            loss=loss+l2_regularization(dgcl,REG_COEF)
        
        tb.add_scalar('loss',loss.item(),epoch)
        if TRAIN_SAFE and epoch%500==0:
            with torch.no_grad():
                dgcl.eval()
                # l1 level 
                pos_embed = dgcl.embed(hetero_data.x_dict, hetero_data.edge_index_dict,hetero_data.global_edge_index,hetero_data.global_edge_weight)
                
                # order: herb syndrome symptom
                
                # labels=hetero_data.l1_labels
                # k = len(set(labels))
                # NMI,ARI,microf1,macrof1 = dgcl.test(pos_embed,labels,0.8,k,MAX_ITER)
                # tb.add_scalar('Classification/l1 micro',microf1,epoch)
                # tb.add_scalar('Classification/l1 macro',macrof1,epoch)

                # tb.add_scalar('Cluster/l1 NMI',NMI,epoch)
                # tb.add_scalar('Cluster/l1 ARI',ARI,epoch)
                
                if DATA_SET in ['longhua','qihuang']:
                    len_dict={node_type:len(nodes) for node_type,nodes in hetero_data.x_dict.items()} #herb,syndrome,symptom
                    herb_idxs=list(range(len_dict['herb']))
                    total_len=sum(val for val in len_dict.values())
                    syndrome_idxs=list(range(len_dict['herb'],len_dict['herb']+len_dict['syndrome']))
                    # syndrome_hit_rates,herb_recalls=l1_sim_search(pos_embed,hetero_data.g_prescription_pairs,syndrome_idxs,herb_idxs,EVAL)
                    syndrome_hit_rates,herb_recalls=l1_sim_search2step(pos_embed,hetero_data.g_prescription_pairs,syndrome_idxs,herb_idxs,EVAL)
                    
                    tb.add_scalar('hit/@1',syndrome_hit_rates[0],epoch)
                    tb.add_scalar('hit/@2',syndrome_hit_rates[1],epoch)
                    tb.add_scalar('hit/@3',syndrome_hit_rates[2],epoch)

                    tb.add_scalar('recall/@5',herb_recalls[0],epoch)
                    tb.add_scalar('recall/@10',herb_recalls[1],epoch)
                    tb.add_scalar('recall/@20',herb_recalls[2],epoch)
                    syndrome_hit_rates=[round(item,4) for item in syndrome_hit_rates]
                    herb_recalls=[round(item,4) for item in herb_recalls]

                # l2 level

                herb_len=len(hetero_data.x_dict['herb'])
                pos_embed=pos_embed[:herb_len]
                pos_embed=pos_embed[hetero_data.herb_labeled_idx]
                labels=hetero_data.herb_labels[hetero_data.herb_labeled_idx]
                k = len(set(labels))
                
                NMI,ARI,microf1,macrof1 = dgcl.test(pos_embed,labels,0.8,k)
                
                tb.add_scalar('Classification/l2 micro f1',microf1,epoch)
                tb.add_scalar('Classification/l2 macro f1',macrof1,epoch)

                tb.add_scalar('Cluster/l2 NMI',NMI,epoch)
                tb.add_scalar('Cluster/l2 ARI',ARI,epoch)

                sim5,sim10=l2_sim_search(pos_embed,labels)
                tb.add_scalar('l2 sim@5',sim5,epoch)
        
            
            print(f'Epoch: {epoch}, Loss: {loss.item():.4f}, lr = {optimizer.param_groups[0]["lr"]}')
        tb.add_scalar('loss',loss.item(),epoch)
        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            
        else:
            cnt_wait += 1

        if cnt_wait == PATIENCE:
            print('Early stopping!')
            real_epoch=epoch
            # torch.save(hegemim.state_dict(), save_model_path+f'{save_model_name}.pkl')
            break

        loss.backward()
        nn.utils.clip_grad_norm_(dgcl.parameters(), max_norm=20, norm_type=2)
        optimizer.step()
        scheduler.step()

    if TRAIN_SAFE:
        continue
    # hegemim.load_state_dict(torch.load(save_model_path+f'{save_model_name}.pkl'))
    
    # EMBED_SIZE=400
    # evaluation
    # time_stamp = "{0:%Y-%m-%d %H-%M-%S} ".format(datetime.now())
    detail='\t[Detail] '+time_stamp +f"dataset {DATA_SET}, feature_size {FEATURE_SIZE}, epoch {real_epoch}, encoder {ENCODE} " \
                                        f"learning rate {LR}, augmentation {aug}, embed size {EMBED_SIZE}, dropout {DROPOUT}," \
                                        f" {REG_TYPE}={REG_COEF} \n"

    with torch.no_grad():
        opt=f"bridge={bridge_dim_pair} lr={lr} edge drop={edge_drop_prob} dropout={DROPOUT}"
        tmp_str=f'exp: {DATA_SET} {opt} {EMBED_SIZE}embed size'
        print(tmp_str)
        record=open('auto_exp_record.md','a')
        record.write(tmp_str+'\n')

        dgcl.eval()
        # l1 level 
        print('level 1 scores:')
        pos_embed = dgcl.embed(hetero_data.x_dict, hetero_data.edge_index_dict,hetero_data.global_edge_index,hetero_data.global_edge_weight)
        
        # 下面的是l1 的，没有用。
        # order: herb syndrome symptom
        # labels=hetero_data.l1_labels
        # plot_embedding(pos_embed,labels,PIC_NAME)
        # exit()

        # k = len(set(labels))
        # NMI,ARI,microf1,macrof1= dgcl.test(pos_embed,labels,0.8,k,MAX_ITER)
        # tb.add_scalar('Classification/l1 micro',microf1,EMBED_SIZE)
        # tb.add_scalar('Classification/l1 macro',macrof1,EMBED_SIZE)
        # cls_str='\t[Classification] Accuracy: Micro {:.4f} Macro {:.4f} '.format(microf1,macrof1)
        # print(cls_str)
        # record.write(cls_str)

        # tb.add_scalar('Cluster/l1 NMI',NMI,EMBED_SIZE)
        # tb.add_scalar('Cluster/l1 ARI',ARI,EMBED_SIZE)
        # cluster_str='\t[Clustering] NMI: {:.4f} ARI {:.4f}'.format(NMI,ARI)
        # print(cluster_str)
        # record.write(cluster_str)
        
        if DATA_SET in ['longhua','qihuang']:
            len_dict={node_type:len(nodes) for node_type,nodes in hetero_data.x_dict.items()}
            herb_idxs=list(range(len_dict['herb']))
            total_len=sum(val for val in len_dict.values())
            syndrome_idxs=list(range(len_dict['herb'],len_dict['herb']+len_dict['syndrome']))
            syndrome_hit_rates,herb_recalls=l1_sim_search2step(pos_embed,hetero_data.g_prescription_pairs,syndrome_idxs,herb_idxs,EVAL)
            
            tb.add_scalar('hit/@1',syndrome_hit_rates[0],EMBED_SIZE)
            tb.add_scalar('hit/@2',syndrome_hit_rates[1],EMBED_SIZE)
            tb.add_scalar('hit/@3',syndrome_hit_rates[2],EMBED_SIZE)

            tb.add_scalar('recall/@5',herb_recalls[0],EMBED_SIZE)
            tb.add_scalar('recall/@10',herb_recalls[1],EMBED_SIZE)
            tb.add_scalar('recall/@20',herb_recalls[2],EMBED_SIZE)
            syndrome_hit_rates=[round(item,4) for item in syndrome_hit_rates]
            herb_recalls=[round(item,4) for item in herb_recalls]
            sim_str=f'\t[Sim Search] hit rate123: {syndrome_hit_rates} recalls {herb_recalls}'
            record.write(sim_str)
            print(sim_str)

        # l2 level
        print('level 2 scores:')

        herb_len=len(hetero_data.x_dict['herb'])
        pos_embed=pos_embed[:herb_len]
        pos_embed=pos_embed[hetero_data.herb_labeled_idx]
        labels=hetero_data.herb_labels[hetero_data.herb_labeled_idx]
        k = len(set(labels))
        NMI,ARI,microf1,macrof1 = dgcl.test(pos_embed,labels,0.8,k)
        
        tb.add_scalar('Classification/l2 micro f1',microf1,EMBED_SIZE)
        tb.add_scalar('Classification/l2 macro f1',macrof1,EMBED_SIZE)
        l2_cls_str='\t[Classification] Accuracy: Micro {:.4f} Macro {:.4f} '.format(microf1,macrof1)
        print(l2_cls_str)
        record.write(l2_cls_str)

        tb.add_scalar('Cluster/l2 NMI',NMI,EMBED_SIZE)
        tb.add_scalar('Cluster/l2 ARI',ARI,EMBED_SIZE)
        l2_cluster_str='\t[Clustering] NMI: {:.4f} ARI {:.4f}'.format(NMI,ARI)
        print(l2_cluster_str)
        record.write(l2_cluster_str)

        sim5,sim10=l2_sim_search(pos_embed,labels)
        tb.add_scalar('l2 sim@5',sim5,EMBED_SIZE)
        record.write('\t[Sim Search] sim@5: {:.4f}'.format(sim5))

    print(detail)
    record.write(detail)
    record.write('\n\n')

    tb.close()
    dgcl.eval()
    if not TRAIN_SAFE and embed_aim!='ablation':
        torch.save(dgcl, save_model_path+f'{save_model_name}.pt')