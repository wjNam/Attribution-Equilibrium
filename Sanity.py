import argparse
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from torchray.benchmark.models import get_model, get_transform
from torchvision.datasets import VOCDetection
from torchray.benchmark.datasets import voc_as_class_ids_ex
from baselines.gradactivation import GradActivation
import imageio
from torch.autograd import Variable
import render
from modules.vgg import vgg16, vgg16_bn,vgg19, vgg19_bn



def get_args():
    parser = argparse.ArgumentParser(description='Grad_hedge')
    parser.add_argument('--cuda', type=str, default='cuda:0')
    parser.add_argument('--model', type=str, default='vgg', metavar='N',
                        help='Model architecture: vgg / res')
    parser.add_argument('--method', type=str, default='AE', metavar='N',
                        help='attribution method: AE')
    parser.add_argument('--ratio', type=int, default=1, metavar='N',
                        help='positive ratio: 1<=r<=2')
    parser.add_argument('--target_class', type=str, default='pd',
                        help='target_class: pd/lb')
    args = parser.parse_args()

    return args

def visualize(relevances, img_name, method):
    n = len(relevances)
    PATH = './Sanity_Results/' + method + '/heatmap/'
    os.makedirs(PATH, exist_ok=True)
    heatmap = np.sum(relevances, axis=3)
    heatmaps = []
    for h, heat in enumerate(heatmap):
        maps = render.hm_to_rgb(heat, scaling=3, sigma=1, cmap='seismic')
        heatmaps.append(maps)

        imageio.imsave(PATH + '/' + img_name + '.jpg', maps, vmax=1, vmin=-1)

if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    args = get_args()
    device = torch.device('cuda:1')
    torch.cuda.set_device(device)
    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    model_arch = 'vgg16'
    kwargs = {'num_classes': 20}
    if args.model == 'vgg':
        model = vgg16(pretrained=False, **kwargs)
        args.flag = 29
        checkpoint = torch.load('./run/VOC/VGG/vgg_new.pth.tar')
        for i in range(len(checkpoint)):
            if i == 26:
                checkpoint[list(checkpoint)[i]] = checkpoint[list(checkpoint)[i]].view(4096, -1)
            elif i == 28:
                checkpoint[list(checkpoint)[i]] = checkpoint[list(checkpoint)[i]].view(4096, 4096)
            elif i == 30:
                checkpoint[list(checkpoint)[i]] = checkpoint[list(checkpoint)[i]].view(20, 4096)
        model.load_state_dict(checkpoint)
    best_pred = 0
    batch_size = 1
    cls = ['airplane',
           'bicycle',
           'bird',
           'boat',
           'bottle',
           'bus',
           'car',
           'cat',
           'chair',
           'cow',
           'dining table',
           'dog',
           'horse',
           'motobike',
           'person',
           'potted plant',
           'sheep',
           'sofa',
           'train',
           'tv'
           ]
    trans_caffe = get_transform(dataset='voc', size=(224,224))

    train_ds = VOCDetection('../../../Data/voc', transform=trans_caffe, image_set='test', year='2007')
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
    args.BP = GradActivation(model, model_name=args.model)
    pert_list_num = [30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0]
    checkpoint_org = checkpoint.copy()
    for pert_num in range(len(pert_list_num)):
        task = 'lb'
        method = 'sanity_' + task
        checkpoint[list(checkpoint)[pert_list_num[pert_num]]] = torch.nn.init.xavier_uniform_(
            checkpoint[list(checkpoint)[pert_list_num[pert_num]]])
        checkpoint[list(checkpoint)[pert_list_num[pert_num]+1]] = torch.nn.init.uniform_(
            checkpoint[list(checkpoint)[pert_list_num[pert_num]+1]])

        model.load_state_dict(checkpoint)
        model.cuda()
        model.eval()

        for idx, (input, labels) in enumerate(train_dl):

            if idx<1:

                label = voc_as_class_ids_ex(labels)
                img_name = train_dl.dataset.images[idx].split('\\')[-1]
                labels = np.zeros([1, 20])
                for aq in range(len(label)):
                    labels[0, label[aq]] = 1
                labels = torch.from_numpy(labels).cuda()
                input = Variable(input, volatile=True).cuda()
                input.requires_grad = True

                output = model(input)
                sig = torch.sigmoid(output).ge(0.5)
                valid_cat = torch.nonzero(labels[0])[:, 0]
                pd_cat = torch.nonzero(sig[0])[:, 0]

                if task == 'lb':
                    task_cat = valid_cat
                elif task == 'pd':
                    task_cat = pd_cat
                num_pred = len(task_cat)
                if len(task_cat) == 0:
                    tmp_lb = output.argmax()
                T = sig.eq(True).type(input.type())
                img_n = img_name.split('.')[0]
                if num_pred > 1:
                    T = Variable(T).cuda()
                    pd_cls = ''
                    tmp = []
                    for g in range(num_pred):
                        pd_cls = cls[task_cat[g]]
                        maxindex = task_cat[g]
                        # Tt = torch.zeros_like(T)
                        # Tt[:, task_cat[g]] = 1

                        P_name = img_name.split('.')[0] + '_' + pd_cls
                        R_CAM = args.BP.generate_activation(input, maxindex)
                        R_CAM = R_CAM / R_CAM.sum().abs()
                        G_CAM = (R_CAM).sum(dim=1, keepdim=True)
                        Gheatmap = G_CAM[0].permute(1, 2, 0).data.cpu().numpy()
                        visualize(Gheatmap.reshape([batch_size, Gheatmap.shape[0], Gheatmap.shape[1], 1]), P_name + '_GCAM_' + str(pert_num), method)
                        att = model.hedge(R=R_CAM.to(device), flag=args.flag, ratio=args.ratio)
                        Res = (att).sum(dim=1, keepdim=True)
                        print(Res.sum())
                        tmp.append(Res)
                        heatmap = Res[0].permute(1, 2, 0).data.cpu().numpy()
                        visualize(heatmap.reshape([batch_size, heatmap.shape[0], heatmap.shape[1], 1]), P_name+str(pert_num), method)
                    Res = torch.cat(tmp, 1)
                else:
                    if num_pred == 0:
                        target_index = tmp_lb.detach().cpu().numpy()
                    else:
                        target_index = task_cat[0].detach().cpu().numpy()
                    maxindex = target_index
                    R_CAM = args.BP.generate_activation(input, maxindex)
                    R_CAM = R_CAM / R_CAM.sum().abs()
                    G_CAM = (R_CAM).sum(dim=1, keepdim=True)
                    Gheatmap = G_CAM[0].permute(1, 2, 0).data.cpu().numpy()
                    visualize(Gheatmap.reshape([batch_size, Gheatmap.shape[0], Gheatmap.shape[1], 1]),
                              P_name + '_GCAM_' + str(pert_num), method)
                    att = model.hedge(R=R_CAM.to(device), flag=args.flag, ratio=args.ratio)
                    Res = (att).sum(dim=1, keepdim=True)
                    # save results
                    heatmap = Res[0].permute(1, 2, 0).data.cpu().numpy()
                    visualize(heatmap.reshape([batch_size, heatmap.shape[0], heatmap.shape[1], 1]), img_n+str(pert_num), method)
            else:
                break;