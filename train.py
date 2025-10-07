import torch
import numpy as np
import random
from logger import get_logger
from utils import get_transform
from dataset import Dataset
from prompt_learner import ICA
from prompt_learner import FGP
from loss import FocalLoss, BinaryDiceLoss
from tqdm import tqdm
import torch.nn.functional as F
import os
import argparse
from utils import get_similarity_map
import VIP2CLIP_lib

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic=True #使用确定性算法
    torch.backends.cudnn.benchmark=False #禁用自动化


def train(args):

    logger=get_logger(args.save_path)

    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    VIP2CLIP_parameters={"Prompt_length":args.n_ctx,"Prompt_cls_length":args.n_cls}

    #data
    preprocess,target_transform = get_transform(args) #target_transform指mask的
    train_data=Dataset(root=args.train_data_path,transform=preprocess,target_transform=target_transform,dataset_name=args.dataset)
    train_dataloader=torch.utils.data.DataLoader(train_data,batch_size=args.batch_size,shuffle=True)

    #model
    model,_=VIP2CLIP_lib.load("ViT-L/14@336px",device=device,design_details=None)
    ica_prompt_learner=ICA(train_data.obj_list,model.to('cpu'),VIP2CLIP_parameters)
    ica_prompt_learner.to(device)
    fgp_learner=FGP(dim_v=768,dim_t=768,dim_out=768)
    fgp_learner.to(device)
    model.to(device)

    optimizer=torch.optim.Adam([
        {"params":ica_prompt_learner.parameters(),"lr":args.learning_rate},
        {"params":fgp_learner.parameters(),"lr":args.learning_rate*0.01}
    ])

    loss_focal=FocalLoss()
    loss_dice=BinaryDiceLoss()

    model.eval()
    ica_prompt_learner.train()
    fgp_learner.train()
    for epoch in tqdm(range(args.epoch)):
        loss_list=[]
        image_loss_list=[]

        for items in tqdm(train_dataloader):
            image=items['img'].to(device)
            label=items['anomaly'].to(device)
            gt=items['img_mask'].squeeze().to(device)
            gt[gt>0.5]=1
            gt[gt<=0.5]=0

            with torch.no_grad():
                image_features,patch_features=model.encode_image(image,args.features_list)
                image_features=image_features/image_features.norm(dim=-1,keepdim=True)

            prompts,tokenized_prompts=ica_prompt_learner(image_features)
            #2n,d
            text_embeddings=model.encode_text_learn(prompts,tokenized_prompts).float()
            text_embeddings=torch.stack(torch.chunk(text_embeddings,dim=0,chunks=2),dim=1)
            text_embeddings=text_embeddings/text_embeddings.norm(dim=-1,keepdim=True)

            similarity_map_list=[]
            anomaly_score_list=[]
            for idx,patch_feature in enumerate(patch_features):
                if idx>=args.feature_map_layer[0]:
                    #n,l+1,d
                    patch_feature=patch_feature/patch_feature.norm(dim=-1,keepdim=True)
                    #n,2,d
                    aux_text_embeddings=fgp_learner(text_embeddings,patch_feature[:,1:,:])
                    similarity=torch.matmul(patch_feature[:,1:,:],aux_text_embeddings.transpose(1,2))
                    scores=similarity
                    similarity=(similarity/0.07).softmax(dim=-1)
                    similarity_map=get_similarity_map(similarity,args.image_size).permute(0,3,1,2)
                    similarity_map_list.append(similarity_map)

                    topk_values,topk_indices=torch.topk(similarity[:,:,1],k=args.topk,dim=-1,largest=True,sorted=False)
                    batch_indices=torch.arange(scores.size(0),device=scores.device).unsqueeze(1).expand(-1,args.topk)
                    topk_scores=scores[batch_indices,topk_indices]
                    #4 ,n topk 2
                    anomaly_score_list.append(topk_scores)
            
            mean_score=torch.mean(torch.cat(anomaly_score_list,dim=1),dim=1)
            mean_score=(mean_score/0.07)
            anomaly_loss=F.cross_entropy(mean_score,label.long().cuda())
            image_loss_list.append(anomaly_loss.item())

            loss=0
            for i in range(len(similarity_map_list)):
                loss+=loss_focal(similarity_map_list[i],gt)
                loss+=loss_dice(similarity_map_list[i][:,1,:,:],gt)
                loss+=loss_dice(similarity_map_list[i][:,0,:,:],1-gt)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            (loss+anomaly_loss).backward()
            optimizer.step()

        if (epoch+1)%args.print_freq==0:
            logger.info('epoch [{}/{}], loss:{:.4f}, image_loss:{:.4f}'.format(epoch+1,args.epoch,np.mean(loss_list),np.mean(image_loss_list)))
        
        if (epoch+1)%args.save_freq==0 and (epoch+1)>=5:
            ckpt_path=os.path.join(args.save_path,f'epoch_{epoch+1}.pth')
            torch.save({"prompt_learner": ica_prompt_learner.state_dict(), "Zero_try": fgp_learner.state_dict()}, ckpt_path)

if __name__=='__main__':
    parser=argparse.ArgumentParser("VIP2CLIP",add_help=False)
    parser.add_argument("--train_data_path", type=str, default="/ssd2/yangziteng/anomaly_detect/data/mvtec",help="train dataset path")
    parser.add_argument("--save_path", type=str, default='./checkpoint', help='path to save results')
    parser.add_argument("--dataset", type=str, default='mvtec', help="train dataset name")

    parser.add_argument("--n_ctx", type=int, default=9, help="learnable prompt length")
    parser.add_argument("--n_cls", type=int, default=3, help="learnable cls prompt length")
    parser.add_argument("--topk", type=int, default=50, help="topk value for similarity scores")
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3], help="the number of feature map used")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")

    parser.add_argument("--epoch", type=int, default=10, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--print_freq", type=int, default=1, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=1, help="save frequency")
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    args=parser.parse_args()
    setup_seed(args.seed)
    train(args)


            
            


                    
                    

















