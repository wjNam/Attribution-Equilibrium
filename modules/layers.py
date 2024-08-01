import torch
import torch.nn as nn
import torch.nn.functional as F
__all__ = ['forward_hook', 'Clone', 'Add', 'Cat', 'ReLU', 'Dropout', 'BatchNorm2d', 'Linear', 'MaxPool2d',
           'AdaptiveAvgPool2d', 'AvgPool2d', 'Conv2d', 'Sequential', 'safe_divide']


def safe_divide(a, b):
    return a / (b + b.eq(0).type(b.type()) * 1e-9) * b.ne(0).type(b.type())

def forward_hook(self, input, output):
    if type(input[0]) in (list, tuple):
        self.X = []
        for i in input[0]:
            x = i.detach()
            x.requires_grad = True
            self.X.append(x)
    else:
        self.X = input[0].detach()
        self.X.requires_grad = True

    self.Y = output

def backward_hook(self, grad_input, grad_output):
    self.grad_input = grad_input
    self.grad_output = grad_output


class RelProp(nn.Module):
    def __init__(self):
        super(RelProp, self).__init__()
        # if not self.training:
        self.register_forward_hook(forward_hook)

    def gradprop(self, Z, X, S):
        C = torch.autograd.grad(Z, X, S, retain_graph=True)
        return C

    def hedge(self, R, ratio):
        return R

class RelPropSimple(RelProp):
    def hedge(self, R_p, ratio):
        def backward(R_p):
            if torch.is_tensor(self.X) == False:
                X = torch.clamp(self.X, min=0)
                X[0] = self.X[0] + 1e-9
                X[1] = self.X[1] + 1e-9
            else:
                X = torch.clamp(self.X, min=0) + 1e-9
            Z = self.forward(X)
            Sp = safe_divide(R_p, Z)
            Cp = self.gradprop(Z, X, Sp)[0]
            if torch.is_tensor(self.X) == False:
                Rp = []
                Rp.append(X[0] * Cp)
                Rp.append(X[1] * Cp)
            else:
                Rp = X * (Cp)
            return Rp
        if torch.is_tensor(R_p) == False:
            idx = len(R_p)
            tmp_R_p = R_p
            Rp = []
            for i in range(idx):
                Rp_tmp = backward(tmp_R_p[i])
                Rp.append(Rp_tmp)
        else:
            Rp = backward(R_p)

        return Rp
class ReLU(nn.ReLU, RelProp):
    pass


class Dropout(nn.Dropout, RelProp):
    pass


class MaxPool2d(nn.MaxPool2d, RelPropSimple):
    pass


class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d, RelPropSimple):
    pass


class AvgPool2d(nn.AvgPool2d, RelPropSimple):
    pass


class Add(RelPropSimple):
    def forward(self, inputs):
        return torch.add(*inputs)
    def hedge(self, R_p, ratio):
        def backward(R_p):
            if torch.is_tensor(self.X) == False:
                X = self.X
                X[0] = self.X[0]
                X[1] = self.X[1]
            else:
                X = self.X + 1e-9
            Z = self.forward(X)
            Sp = safe_divide(R_p, Z)
            Cp = self.gradprop(Z, X, Sp)
            a = X[0] * Cp[0]
            b = X[1] * Cp[0]
            a_sum = a.sum()
            b_sum = b.sum()
            a_fact = safe_divide(a_sum.abs(), a_sum.abs() + b_sum.abs()) * R_p.sum()
            b_fact = safe_divide(b_sum.abs(), a_sum.abs() + b_sum.abs()) * R_p.sum()

            a = a * safe_divide(a_fact, a.sum())
            b = b * safe_divide(b_fact, b.sum())

            Rp = []
            Rp.append(a)
            Rp.append(b)

            return Rp
        if torch.is_tensor(R_p) == False:
            idx = len(R_p)
            tmp_R_p = R_p
            Rp = []
            for i in range(idx):
                Rp_tmp = backward(tmp_R_p[i])
                Rp.append(Rp_tmp)
        else:
            Rp = backward(R_p)

        return Rp
class Clone(RelProp):
    def forward(self, input, num):
        self.__setattr__('num', num)
        outputs = []
        for _ in range(num):
            outputs.append(input)

        return outputs

    def hedge(self, R_p, ratio):
        def backward(R_p):
            Z = []
            for _ in range(self.num):
                Z.append(self.X)

            Spp = []
            Spn = []

            for z, rp in zip(Z, R_p):
                Spp.append(safe_divide(torch.clamp(rp, min=0), z))
                Spn.append(safe_divide(torch.clamp(rp, max=0), z))

            Cpp = self.gradprop(Z, self.X, Spp)[0]
            Cpn = self.gradprop(Z, self.X, Spn)[0]

            Rp = self.X * (Cpp * Cpn)

            return Rp

        Rp = backward(R_p)
        return Rp
class Cat(RelProp):
    def forward(self, inputs, dim):
        self.__setattr__('dim', dim)
        return torch.cat(inputs, dim)

    def relprop(self, R, alpha):
        Z = self.forward(self.X, self.dim)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)

        outputs = []
        for x, c in zip(self.X, C):
            outputs.append(x * c)

        return outputs
    def hedge(self, R_p, ratio):
        def backward(R_p):
            Z = self.forward(self.X, self.dim)
            Sp = safe_divide(R_p, Z)

            Cp = self.gradprop(Z, self.X, Sp)

            Rp = []

            for x, cp in zip(self.X, Cp):
                Rp.append(x * (cp))

            return Rp
        if torch.is_tensor(R_p) == False:
            idx = len(R_p)
            tmp_R_p = R_p
            Rp = []
            for i in range(idx):
                Rp_tmp = backward(tmp_R_p[i])
                Rp.append(Rp_tmp)
        else:
            Rp = backward(R_p)
        return Rp
class Sequential(nn.Sequential):
    def relprop(self, R, alpha):
        for m in reversed(self._modules.values()):
            R = m.relprop(R, alpha)
        return R
    def hedge(self, R, ratio):
        for m in reversed(self._modules.values()):
            R = m.hedge(R, ratio)
        return R
class BatchNorm2d(nn.BatchNorm2d, RelProp):
    def hedge(self, R_p, ratio):
        def f(R, w1, x1):
            Z1 = x1 * w1
            S1 = safe_divide(R, Z1) * w1
            C1 = x1 * S1
            return C1
        def backward(R_p):
            X = self.X

            weight = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3) / (
                (self.running_var.unsqueeze(0).unsqueeze(2).unsqueeze(3).pow(2) + self.eps).pow(0.5))
            Rp = f(R_p, weight, X)
            return Rp
        if torch.is_tensor(R_p) == False:
            idx = len(R_p)
            tmp_R_p = R_p
            Rp = []
            for i in range(idx):
                Rp_tmp = backward(tmp_R_p[i])
                Rp.append(Rp_tmp)
        else:
            Rp = backward(R_p)
        return Rp
class Linear(nn.Linear, RelProp):
    def relprop(self, R, alpha):
        beta = alpha - 1
        pw = torch.clamp(self.weight, min=0)
        nw = torch.clamp(self.weight, max=0)
        px = torch.clamp(self.X, min=0)
        nx = torch.clamp(self.X, max=0)

        def f(w1, w2, x1, x2):
            Z1 = F.linear(x1, w1)
            Z2 = F.linear(x2, w2)
            S1 = safe_divide(R, Z1)
            S2 = safe_divide(R, Z2)
            C1 = x1 * self.gradprop(Z1, x1, S1)[0]
            C2 = x2 * self.gradprop(Z2, x2, S2)[0]

            return C1 + C2

        activator_relevances = f(pw, nw, px, nx)
        inhibitor_relevances = f(nw, pw, px, nx)

        R = alpha * activator_relevances - beta * inhibitor_relevances

        return R


class Conv2d(nn.Conv2d, RelProp):

    def gradprop_to_input(self, DY, weight):
        Z = self.forward(self.X)

        output_padding = self.X.size()[2] - (
                (Z.size()[2] - 1) * self.stride[0] - 2 * self.padding[0] + self.kernel_size[0])
        output_padding2 = self.X.size()[3] - (
                (Z.size()[3] - 1) * self.stride[0] - 2 * self.padding[0] + self.kernel_size[0])
        return F.conv_transpose2d(DY, weight, stride=self.stride, padding=self.padding, output_padding=(output_padding,output_padding2))

    def hedge(self, R_p, ratio):
        def shift_rel(R, R_val):
            R_nonzero = torch.ne(R, 0).type(R.type())
            shift = safe_divide(R_val, torch.sum(R_nonzero, dim=[1,2,3], keepdim=True)) * torch.ne(R, 0).type(R.type())
            K = R - shift
            return K
        def att_prop(R_pos, R_neg, Ga, Gb, x_pos):
            S1 = safe_divide(R_pos, Ga)
            C1 = x_pos * self.gradprop(Ga, x_pos, S1)[0]
            S2 = safe_divide(R_neg, Gb)
            C2 = x_pos * self.gradprop(Gb, x_pos, S2)[0]
            C = (C1 + C2)
            return C

        def f(R, w1, w2, x1, x2, ratio):
            ## Modulate Ratio
            R_tar_org = torch.clamp(R, min=0)
            R_oth_org = torch.clamp(R, max=0)
            R_tar_org = R_tar_org / R_tar_org.sum()
            R_oth_org = R_oth_org / R_oth_org.sum() * -1
            R_tar = R_tar_org * ratio
            R_oth = R_oth_org

            R_ge = R_tar.ne(0).type(R.type())
            R_le = R_oth.ne(0).type(R.type())

            wabs = self.weight.abs()
            Zp1 = F.conv2d(x1, wabs, bias=None, stride=self.stride, padding=self.padding) * R_ge
            Zp2 = F.conv2d(x1, wabs, bias=None, stride=self.stride, padding=self.padding) * R_le

            C1 = att_prop(R_tar, R_oth, Zp1, Zp2, x1)
            C1_shift = shift_rel(C1, (R_tar / R_tar.sum()).sum(dim=[1, 2, 3], keepdim=True))

            xabs = torch.ones_like(self.X) * torch.gt(self.X, 0)
            xnabs = torch.ones_like(self.X) * torch.le(self.X, 0)

            xabs.requires_grad_()
            xnabs.requires_grad_()
            Zabs = F.conv2d(xabs, wabs, bias=None, stride=self.stride, padding=self.padding) * R.ge(0).type(
                R.type())
            S = safe_divide(R_tar_org, Zabs)
            grad = torch.autograd.grad(Zabs, xabs, S)
            A = xabs * grad[0]

            Znabs = F.conv2d(xnabs, wabs, bias=None, stride=self.stride, padding=self.padding) * R.le(0).type(
                R.type())
            Sn = safe_divide(R_oth_org, Znabs)
            gradn = torch.autograd.grad(Znabs, xnabs, Sn)
            B = xnabs * gradn[0]

            if w1.shape[2] == 1:
                S = safe_divide(R_tar_org, Zp1)
                C = x1 * self.gradprop(Zp1, x1, S)[0]
                S2 = safe_divide(R_oth_org, Zp2)
                C2 = x1 * self.gradprop(Zp2, x1, S2)[0]
                C = C + C2
                return C + A + B

            C2 = A+B
            C = C1_shift + C2

            return C
        def backward(R_p, px, nx, pw, nw):
            Rp = f(R_p, pw, nw, px, nx, ratio)

            return Rp
        def final_backward(R_p, pw, nw, X1):
            X = X1
            L = X * 0 + \
                torch.min(torch.min(torch.min(X, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0], dim=3,
                          keepdim=True)[0]
            H = X * 0 + \
                torch.max(torch.max(torch.max(X, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0], dim=3,
                          keepdim=True)[0]
            Za = torch.conv2d(X, self.weight, bias=None, stride=self.stride, padding=self.padding) - \
                 torch.conv2d(L, pw, bias=None, stride=self.stride, padding=self.padding) - \
                 torch.conv2d(H, nw, bias=None, stride=self.stride, padding=self.padding)

            Sp = safe_divide(R_p, Za)

            Rp = X * self.gradprop_to_input(Sp, self.weight) - L * self.gradprop_to_input(Sp, pw) - H * self.gradprop_to_input(Sp, nw)
            return Rp

        pw = torch.clamp(self.weight, min=0)
        nw = torch.clamp(self.weight, max=0)

        px = torch.clamp(self.X, min=0)
        nx = torch.clamp(self.X, max=0)
        if self.X.shape[1] == 3:
            Rp = final_backward(R_p, pw, nw, self.X)
        else:

            Rp = backward(R_p, px, nx, pw, nw)
        return Rp
