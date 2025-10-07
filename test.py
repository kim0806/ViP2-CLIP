import torch 
import numpy as np
import random
from logger import get_logger
from utils import get_similarity_map,get_transform
from scipy.ndimage import gaussian_filter
from dataset import Dataset
import VIP2CLIP_lib
from prompt_learner import ICA,FGP
from tqdm import tqdm
import argparse
from visualization import visualizer
from metrics import image_level_metrics, pixel_level_metrics
from tabulate import tabulate

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

def test(args):
    logger=get_logger(args.save_path)
    device="cuda:0" if torch.cuda.is_available() else 'cpu'
    VIP2CLIP_parameters={"Prompt_length":args.n_ctx,"Prompt_cls_length":args.n_cls}

    preprocess,target_transform=get_transform(args)
    test_data=Dataset(root=args.data_path,transform=preprocess,target_transform=target_transform,dataset_name=args.dataset)
    test_dataloader=torch.utils.data.DataLoader(test_data,shuffle=False,batch_size=1)
    obj_list=test_data.obj_list

    model,_=VIP2CLIP_lib.load("ViT-L/14@336px",device=device,design_details=None)
    
    ckpt=torch.load(args.checkpoint_path)
    ica_prompt_learner=ICA(obj_list,model.to('cpu'),VIP2CLIP_parameters)
    ica_prompt_learner.load_state_dict(ckpt['prompt_learner'])
    ica_prompt_learner.to(device)
    fgp_learner=FGP(dim_v=768,dim_t=768,dim_out=768)
    fgp_learner.load_state_dict(ckpt['Zero_try'])
    fgp_learner.to(device)
    model.to(device)

    results={}
    metrics={}
    for obj in obj_list:
        results[obj] = {}
        results[obj]['gt_sp'] = []
        results[obj]['pr_sp'] = []
        results[obj]['imgs_masks'] = []
        results[obj]['anomaly_maps'] = []
        results[obj]['img_paths'] = []
        metrics[obj] = {}
        metrics[obj]['pixel-auroc'] = 0
        metrics[obj]['pixel-aupro'] = 0
        metrics[obj]['image-auroc'] = 0
        metrics[obj]['image-ap'] = 0
    
    model.eval()
    ica_prompt_learner.eval()
    fgp_learner.eval()


    for items in tqdm(test_dataloader):

        image = items['img'].to(device)
        cls_name = items['cls_name']
        gt_mask = items['img_mask']

        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results[cls_name[0]]['imgs_masks'].append(gt_mask.squeeze(1))  # px
        results[cls_name[0]]['gt_sp'].extend(items['anomaly'].detach().cpu())
        results[cls_name[0]]['img_paths'].extend(items['img_path'])

        with torch.no_grad():
            image_features,patch_features=model.encode_image(image,args.features_list)
            image_features=image_features/image_features.norm(dim=-1,keepdim=True)
        
            prompts,tokenized_prompts=ica_prompt_learner(image_features)
            text_embeddings=model.encode_text_learn(prompts,tokenized_prompts).float()
            text_embeddings=torch.stack(torch.chunk(text_embeddings,dim=0,chunks=2),dim=1)
            text_embeddings=text_embeddings/text_embeddings.norm(dim=-1,keepdim=True)

            anomaly_score_list=[]
            similarity_map_list=[]
            for idx,patch_feature in enumerate(patch_features):
                if idx>=args.feature_map_layer[0]:
                    patch_feature=patch_feature/patch_feature.norm(dim=-1,keepdim=True)
                    aux_text_embeddings=fgp_learner(text_embeddings,patch_feature[:,1:,:])
                    similarity=torch.matmul(patch_feature[:,1:,:],aux_text_embeddings.transpose(1,2))
                    scores=similarity
                    similarity = (similarity / 0.07).softmax(-1)
                    similarity_map=get_similarity_map(similarity,args.image_size)
                    similarity_map=(similarity_map[...,1]+1-similarity_map[...,0])/2.0
                    similarity_map_list.append(similarity_map)

                    topk_values,topk_indices=torch.topk(similarity[:,:,1],k=args.topk,dim=-1,largest=True,sorted=False)
                    batch_incides=torch.arange(scores.size(0),device=device).unsqueeze(1).expand(-1,args.topk)
                    topk_scores=scores[batch_incides,topk_indices]
                    anomaly_score_list.append(topk_scores)
            
            mean_scores=torch.mean(torch.cat(anomaly_score_list,dim=1),dim=1)
            mean_scores=(mean_scores/0.07).softmax(dim=-1)
            text_probs=mean_scores[:,1]

            anomaly_map=torch.stack(similarity_map_list,dim=0)
            anomaly_map=anomaly_map.mean(dim=0)

            results[cls_name[0]]['pr_sp'].extend(text_probs.detach().cpu())
            anomaly_map=torch.stack(
                [torch.from_numpy(gaussian_filter(i,sigma=args.sigma,axes=(0,1))) for i in anomaly_map.detach().cpu()],dim=0
            )
            results[cls_name[0]]['anomaly_maps'].append(anomaly_map)
    
    table_ls = []
    image_auroc_list = []
    image_ap_list = []
    image_f1_list=[]
    pixel_auroc_list = []
    pixel_aupro_list = []
    pixel_f1_list=[]
    for obj in obj_list:
        table = []
        table.append(obj)
        results[obj]['imgs_masks'] = torch.cat(results[obj]['imgs_masks']).detach().cpu().numpy()
        results[obj]['anomaly_maps'] = torch.cat(results[obj]['anomaly_maps']).detach().cpu().numpy()
        if args.metrics == 'image-level':
            image_auroc = image_level_metrics(results, obj, "image-auroc")
            image_ap = image_level_metrics(results, obj, "image-ap")
            table.append(str(np.round(image_auroc * 100, decimals=1)))
            table.append(str(np.round(image_ap * 100, decimals=1)))
            image_auroc_list.append(image_auroc)
            image_ap_list.append(image_ap)
        elif args.metrics == 'pixel-level':
            pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
            pixel_aupro = pixel_level_metrics(results, obj, "pixel-aupro")
            table.append(str(np.round(pixel_auroc * 100, decimals=1)))
            table.append(str(np.round(pixel_aupro * 100, decimals=1)))
            pixel_auroc_list.append(pixel_auroc)
            pixel_aupro_list.append(pixel_aupro)
        elif args.metrics == 'image-pixel-level':
            image_auroc = image_level_metrics(results, obj, "image-auroc")
            image_ap = image_level_metrics(results, obj, "image-ap")
            image_f1 = image_level_metrics(results, obj, "image-f1")
            pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
            pixel_aupro = pixel_level_metrics(results, obj, "pixel-aupro")
            pixel_f1,thresholds=pixel_level_metrics(results, obj, "pixel-f1")
            table.append(str(np.round(pixel_auroc * 100, decimals=1)))
            table.append(str(np.round(pixel_aupro * 100, decimals=1)))
            table.append(str(np.round(pixel_f1 * 100, decimals=1)))
            table.append(str(np.round(image_auroc * 100, decimals=1)))
            table.append(str(np.round(image_ap * 100, decimals=1)))
            table.append(str(np.round(image_f1 * 100, decimals=1)))
            image_auroc_list.append(image_auroc)
            image_ap_list.append(image_ap)
            image_f1_list.append(image_f1)
            pixel_auroc_list.append(pixel_auroc)
            pixel_aupro_list.append(pixel_aupro)
            pixel_f1_list.append(pixel_f1)
        table_ls.append(table)
        #vis
        if args.visualize:
            visualizer(results[obj]['img_paths'], results[obj]['anomaly_maps'], args.image_size, args.save_path, obj,results[obj]['imgs_masks'])

    if args.metrics=='image-level':
        table_ls.append(['mean',str(np.round(np.mean(image_auroc_list)*100,decimals=1)),str(np.round(np.mean(image_ap_list) * 100, decimals=1))])
        results=tabulate(table_ls,headers=['objects','image_auroc','image_ap'],tablefmt='pipe')
    elif args.metrics == 'pixel-level':
        # logger
        table_ls.append(['mean', str(np.round(np.mean(pixel_auroc_list) * 100, decimals=1)),
                        str(np.round(np.mean(pixel_aupro_list) * 100, decimals=1))
                        ])
        results = tabulate(table_ls, headers=['objects', 'pixel_auroc', 'pixel_aupro'], tablefmt="pipe")
    elif args.metrics == 'image-pixel-level':
        # logger
        table_ls.append(['mean', str(np.round(np.mean(pixel_auroc_list) * 100, decimals=1)),
                        str(np.round(np.mean(pixel_aupro_list) * 100, decimals=1)),
                        str(np.round(np.mean(pixel_f1_list) * 100, decimals=1)),
                        str(np.round(np.mean(image_auroc_list) * 100, decimals=1)),
                        str(np.round(np.mean(image_ap_list) * 100, decimals=1)),
                        str(np.round(np.mean(image_f1_list) * 100, decimals=1))])
        results = tabulate(table_ls, headers=['objects', 'pixel_auroc', 'pixel_aupro', 'pixel_f1','image_auroc', 'image_ap', 'image_f1'],
                        tablefmt="pipe")
    logger.info("\n%s", results)

if __name__ == '__main__':
    parser=argparse.ArgumentParser('VIP2CLIP',add_help=True)

        # paths
    parser.add_argument("--data_path", type=str, default="./data/visa", help="path to test dataset")
    parser.add_argument("--save_path", type=str, default='./results/', help='path to save results')
    parser.add_argument("--checkpoint_path", type=str, default='./checkpoint/', help='path to checkpoint')
    parser.add_argument("--visualize",action='store_true',help='start vis')
    # model
    parser.add_argument("--dataset", type=str, default='mvtec')
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--n_ctx", type=int, default=9, help="learnable prompt length")
    parser.add_argument("--n_cls", type=int, default=3, help="learnable cls prompt length")
    parser.add_argument("--topk", type=int, default=50, help="topk value for similarily scores")
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3], help="the number of feature map used")
    parser.add_argument("--metrics", type=str, default='image-pixel-level')
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    parser.add_argument("--sigma", type=int, default=8, help="gaussian filter params")

    args=parser.parse_args()
    setup_seed(args.seed)
    test(args)









