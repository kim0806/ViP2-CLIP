
import cv2
import os
from utils import normalize
import numpy as np

def apply_ad_scoremap(image,scoremap,alpha=0.5):
    np_image=np.asarray(image,dtype=float)
    scoremap=(scoremap*255).astype(np.uint8)
    scoremap=cv2.applyColorMap(scoremap,cv2.COLORMAP_JET) #应用JET色彩映射（蓝-青-黄-红）
    scoremap=cv2.cvtColor(scoremap,cv2.COLOR_BGR2RGB) #转为pil的rgb
    return (alpha*np_image+(1-alpha)*scoremap).astype(np.uint8)


def he_cheng(img_list, size = 256):
    h,w,c = img_list[0].shape
    jian = np.ones((h, 10, 3),dtype=np.uint8) * 255
    vis_con = img_list[0]
    for i in range(1,len(img_list)):
        vis_con = np.concatenate([vis_con, jian, img_list[i]], axis=1)

    vis_con = cv2.resize(vis_con, (size*len(img_list)+ 10*(len(img_list)-1), size)).astype(np.uint8)
    return vis_con


def visualizer(pathes, anomaly_map, img_size, save_path, cls_name, gt_masks=None, the=0.5,he_cheng=False):

    for idx,path in enumerate(pathes):
        cls=path.split('/')[-2]
        filename=path.split('/')[-1].replace('bmp','png')

        raw_img=cv2.cvtColor(cv2.resize(cv2.imread(path),(img_size,img_size)),cv2.COLOR_BGR2RGB)

        map=normalize(anomaly_map[idx])
        map_binary=np.array(anomaly_map[idx]>the,dtype=np.uint8)
        map_ad_binary=map*map_binary

        vis_map=apply_ad_scoremap(raw_img,map)
        vis_map=cv2.cvtColor(vis_map,cv2.COLOR_RGB2BGR)

        vis_map_binary = apply_ad_scoremap(raw_img, map_binary)
        vis_map_binary = cv2.cvtColor(vis_map_binary, cv2.COLOR_RGB2BGR)
        
        vis_map_ad_binary = apply_ad_scoremap(raw_img, map_ad_binary)
        vis_map_ad_binary = cv2.cvtColor(vis_map_ad_binary, cv2.COLOR_RGB2BGR)

        save_vis=os.path.join(save_path,'img',cls_name,cls)
        if not os.path.exists(save_vis):
            os.makedirs(save_vis)
        cv2.imwrite(os.path.join(save_vis,filename),vis_map_ad_binary)


        if he_cheng:

            if gt_masks is not None:
                gt=normalize(gt_masks[idx])
                ground_truth_contours, _ =cv2.findContours(np.array(gt_masks[idx]*255,dtype=np.uint8),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            else:
                gt=np.zeros_like(map)
                ground_truth_contours=[]
            
            vis_gt = apply_ad_scoremap(raw_img, gt)
            vis_gt = cv2.cvtColor(vis_gt, cv2.COLOR_RGB2BGR)

            if gt_masks is not None:
                vis_map_binary = cv2.drawContours(vis_map_binary, ground_truth_contours, -1, (0, 255, 0), 2)
                vis_map_ad_binary = cv2.drawContours(vis_map_ad_binary, ground_truth_contours, -1, (0, 255, 0), 2)

            raw_image_bgr = cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR)
            zong = he_cheng([raw_image_bgr, vis_map, vis_map_ad_binary, vis_gt])

            save_vis=os.path.join(save_path,'imgs_hecheng',cls_name,cls)
            if not os.path.exists(save_vis):
                os.makedirs(save_vis)
            cv2.imwrite(os.path.join(save_vis,filename),zong)


            

