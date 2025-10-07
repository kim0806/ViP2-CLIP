
import torchvision.transforms as transforms
from VIP2CLIP_lib.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from VIP2CLIP_lib.transform import image_transform
import torch

def normalize(pixel,max_pixel=None,min_pixel=None):
    if max_pixel is None or min_pixel is None:
        pixel=(pixel-pixel.min())/(pixel.max()-pixel.min())
    else:
        pixel=(pixel-min_pixel)/(max_pixel-min_pixel)
    return pixel

def get_transform(args):
    preprocess=image_transform(args.image_size,is_train=False,mean=OPENAI_DATASET_MEAN,std=OPENAI_DATASET_STD)
    target_preprocess=transforms.Compose([
        transforms.Resize((args.image_size,args.image_size)),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor()
    ])
    preprocess.transforms[0]=transforms.Resize(size=(args.image_size,args.image_size),interpolation=transforms.InterpolationMode.BICUBIC,max_size=None,antialias=None)
    preprocess.transforms[1]=transforms.CenterCrop(size=(args.image_size, args.image_size))
    return preprocess,target_preprocess

def get_similarity_map(sm,shape):
    side=int(sm.shape[1]**0.5)
    sm=sm.reshape(sm.shape[0],side,side,-1).permute(0,3,1,2)
    sm=torch.nn.functional.interpolate(sm,shape,mode='bilinear')
    sm=sm.permute(0,2,3,1)
    return sm



