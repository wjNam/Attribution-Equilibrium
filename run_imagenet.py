import torch
from torchvision.transforms import transforms
import numpy as np
import os
import argparse
from modules.vgg import vgg16 as vgg16_hedge
from modules.resnet import resnet50 as resnet50_hedge
from modules.inception import inception_v3 as incept_hedge
from modules.densenet import densenet121 as dense121_hedge
from modules.alexnet import alexnet as alex_hedge
import render
from modules.utils import *
import imageio
from baselines.gradcam import GradCam
from baselines.gradactivation import GradActivation
import torchvision.datasets as datasets
from skimage import img_as_ubyte
def visualize(relevances, img_name, args):
    n = len(relevances)
    PATH = './Results/' + args.method + '/' + args.vis_class +'/' + args.model
    os.makedirs(PATH, exist_ok=True)
    heatmap = np.sum(relevances, axis=3)
    heatmaps = []
    for h, heat in enumerate(heatmap):
        maps = render.hm_to_rgb(heat, scaling=3, sigma=1, cmap='seismic')
        heatmaps.append(maps)

        imageio.imsave(PATH + '/' + img_name + '.jpg', img_as_ubyte(maps), vmax=1, vmin=-1)
    # np.save(PATH + '/' + img_name + '.npy', heatmap)
def generate_visualization_to_hdf5(args):

    for batch_idx, (data, target) in enumerate(sample_loader):
        img_name = sample_loader.dataset.imgs[batch_idx][0].split('\\')[1].split('.')[0]
        target = target.to(device)
        data = normalize(data)
        data = data.to(device)
        data.requires_grad_()
        pred = model(data)

        if args.vis_class == 'top':
            pred_class = pred.data.max(1, keepdim=True)[1].squeeze(1).item()
        elif args.vis_class == 'index':
            pred_class = args.class_id
        elif args.vis_class == 'target':
            pred_class = target.item()
        else:
            raise Exception('Invalid vis-class')
        print(img_name)
        print('pred_class: '+ str(pred_class))
        # print('target class: '+str(target))
        if 'AE' in args.method:
            R_CAM = args.BP.generate_activation(data, pred_class)
            R_CAM = R_CAM / R_CAM.sum().abs()
            att = model.hedge(R=R_CAM.to(device), flag=args.flag, ratio=args.ratio)
            cam = (att).sum(dim=1, keepdim=True)
            print(cam.sum())
        elif args.method == 'gradcam':
            # pred = model(data)
            cam = args.BP.generate_cam(data, pred_class)
        else:
            raise Exception('No method found')
        heatmap = cam[0].permute(1, 2, 0).data.cpu().numpy()
        visualize(heatmap.reshape([args.batch_size, heatmap.shape[0], heatmap.shape[1], 1]), img_name, args)
        # npsave(cam.data.cpu().numpy(), img_name, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a segmentation')
    parser.add_argument('--batch-size', type=int,
                        default=1,
                        help='Batch size')
    parser.add_argument('--method', type=str,
                        default='AE',
                        help='Method: AE/gradcam')
    parser.add_argument('--vis-class', type=str,
                        default='top',
                        choices=['top', 'target', 'index'],
                        help='Propagated class')
    parser.add_argument('--ratio', type=int, default=2, metavar='N',
                        help='positive ratio: 1<=r<=2')
    parser.add_argument('--class-id', type=int,
                        default=0,
                        help='If --vis-class == index, then propagate --class-id')
    parser.add_argument('--model', type=str, default='vgg',
                        help='dense/google/alex/res/vgg')
    args = parser.parse_args()


    # cuda = torch.cuda.is_available()
    device = torch.device('cuda:0')
    # device = torch.device("cuda" if cuda else "cpu")

    # Load pre-trained models

    if  args.model == 'vgg':
        model = vgg16_hedge(pretrained=True).to(device)
        image_size = 224
        args.flag = 29
    elif args.model == 'res':
        model = resnet50_hedge(pretrained=True).to(device)
        image_size = 224
        args.flag = 'layer4'
    elif args.model == 'dense':
        model = dense121_hedge(pretrained=True).to(device)
        image_size = 224
        args.flag = None
    elif args.model == 'google':
        model = incept_hedge(pretrained=True).to(device)
        image_size = 299
        args.flag = None
    elif args.model == 'alex':
        model = alex_hedge(pretrained=True).to(device)
        image_size = 224
        args.flag = None
    else:
        raise Exception('No model found')

    model.eval()
    if args.method == 'gradcam':
        args.BP = GradCam(model, target_layer=args.model)
    elif args.method == 'AE':
        args.BP = GradActivation(model, model_name=args.model)
    else:
        raise Exception('No model found')

    # Dataset loader for sample images

    sample_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder('./sample/',
                             transforms.Compose([
                                 transforms.Scale([image_size, image_size]),
                                 transforms.ToTensor(),
                             ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=4)


    generate_visualization_to_hdf5(args)
