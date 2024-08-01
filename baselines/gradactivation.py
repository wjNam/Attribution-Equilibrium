"""
Created on Thu Oct 26 11:06:51 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

from baselines.misc_functions import get_example_params, save_class_activation_images


def minmax_dims(x, minmax):
    y = x.clone()
    dims = x.dim()
    for i in range(1, dims):
        y = getattr(y, minmax)(dim=i, keepdim=True)[0]
    return y


class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad
    def forward_pass_on_alex(self, x):
        x = self.model.features(x)
        x.register_hook(self.save_gradient)
        conv_output = x.clone()  # Save the convolution output on that layer
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.classifier(x)
        return conv_output, x
    def forward_pass_on_vgg(self, x):
        target_layer = 29
        conv_output = None
        for module_pos, module in self.model.features._modules.items():
            x = module(x)  # Forward
            if int(module_pos) == target_layer:
                x.register_hook(self.save_gradient)
                conv_output = x.clone()  # Save the convolution output on that layer
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.classifier(x)
        return conv_output, x
    def forward_pass_on_res(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x.register_hook(self.save_gradient)
        conv_output = x.clone()  # Save the convolution output on that layer

        x = self.model.avgpool(x)
        ### if imagenet
        # x = x.view(x.size(0), -1)

        x = self.model.fc(x)
        return conv_output, x
    def forward_pass_on_eff(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        x = self.model._swish(self.model._bn0(self.model._conv_stem(x)))

        # Blocks
        for idx, block in enumerate(self.model._blocks):
            drop_connect_rate = self.model._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.model._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = self.model._swish(self.model._bn1(self.model._conv_head(x)))
        x.register_hook(self.save_gradient)
        conv_output = x.clone()

        x = self.model._avg_pooling(x)
        if self.model._global_params.include_top:
            x = x.flatten(start_dim=1)
            x = self.model._dropout(x)
            x = self.model._fc(x)
        return conv_output, x
    def forward_pass_on_dense(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        x = self.model.features(x)
        x.register_hook(self.save_gradient)
        conv_output = x.clone()
        x = F.relu(x, inplace=True)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.classifier(x)
        return conv_output, x
    def forward_pass_on_google(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """

        x = self.model.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.model.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.model.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.model.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.model.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.model.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.model.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.model.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.model.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.model.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.model.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.model.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.model.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.model.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.model.Mixed_6e(x)
        # N x 768 x 17 x 17
        aux: Optional[Tensor] = None
        if self.model.AuxLogits is not None:
            if self.model.training:
                aux = self.model.AuxLogits(x)
        # N x 768 x 17 x 17
        x = self.model.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.model.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.model.Mixed_7c(x)
        # N x 2048 x 8 x 8
        x.register_hook(self.save_gradient)
        conv_output = x.clone()
        # Adaptive average pooling
        x = self.model.avgpool(x)
        # N x 2048 x 1 x 1
        x = self.model.dropout(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        x = self.model.fc(x)
        # N x 1000 (num_classes)
        return conv_output, x
    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """

        # Forward pass on the convolutions
        ##If VGG
        # conv_output, x = self.forward_pass_on_convolutions(x)
        # x = x.view(x.size(0), -1)  # Flatten
        # # Forward pass on the classifier
        # x = self.model.classifier(x)
        ## IF EFF
        if self.model_name == 'alex':
            conv_output, x = self.forward_pass_on_alex(x)
        elif self.model_name == 'vgg':
            conv_output, x = self.forward_pass_on_vgg(x)
        elif self.model_name == 'res':
            conv_output, x = self.forward_pass_on_res(x)
        elif self.model_name == 'google':
            conv_output, x = self.forward_pass_on_google(x)
        elif self.model_name == 'dense':
            conv_output, x = self.forward_pass_on_dense(x)
        else:
            raise Exception('No model found')
        return conv_output, x


class GradActivation():
    """
        Produces class activation map
    """
    def __init__(self, model, model_name):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, model_name)

    def generate_activation(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        # Target for backprop
        # one_hot_output = torch.zeros(1, model_output.size()[-1]).to(input_image.device)
        # one_hot_output[0][target_class] = 1
        one_hot_output = torch.zeros_like(model_output).to(input_image.device)
        one_hot_output[0,target_class] = 1
        self.model.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients
        # Get convolution outputs
        target = conv_output
        # Get weights from gradients
        weights = torch.mean(guided_gradients, dim=[2, 3], keepdim=True)
        cam = (target * weights)
        return cam

