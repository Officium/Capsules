"""
PyTorch implementation of "Matrix capsules with EM routing"
Hyper-parameters and variables are consist with the paper's settings.
Some details can be found at
`https://openreview.net/forum?id=HJWLfGWRb&noteId=rJUY2VdbM%27`.
"""
import math
import os
import random
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import softmax, conv2d
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from smallnorb_dataset import smallNORB


ln_2pi = math.log(2 * math.pi)
random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True


def truncated_normal_(tensor, std):
    """ Truncated normal initilizer with zero mean """
    with torch.no_grad():
        return tensor.normal_(0, std).clamp(-2 * std, 2 * std)


class CapsuleNet(nn.Module):
    """ Capsule Network """
    def __init__(self):
        super().__init__()
        # Conv Layer
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, 5, 2),
            nn.ReLU(True)
        )
        truncated_normal_(self.conv[0].weight, 0.015)

        # Primary Capsule
        self.cap1 = PrimaryCapsule(32, 32)

        # Conv capsule1
        self.cap2 = ConvCapsule(32, 32, 3, stride=2, iters=3)

        # Conv capsule2
        self.cap3 = ConvCapsule(32, 32, 3, stride=1, iters=3)

        # Class capsule, kernel size is equal to input size
        self.cap4 = ConvCapsule(32, 5, 4, stride=1, iters=3,
                                add_coordinate=True, share_transformation=True)

    def forward(self, x):
        fea = self.conv(x)
        a, M = self.cap1(fea)
        a, M = self.cap2(a, M)
        a, M = self.cap3(a, M)
        a, _ = self.cap4(a, M)
        return a.view(x.size(0), 5)


class PrimaryCapsule(nn.Module):
    """ Primary convolutional capsule layer """
    def __init__(self, A, B):
        """
        Args:
            A (int): channel number of previous conv layer
            B (int): number of capsule types learned in this layer

        1 * 1 convonlution, output size b * h * w * (16 + 1)
        """
        super().__init__()
        self.pose = nn.Conv2d(A, B * 16, 1)
        self.a = nn.Sequential(
            nn.Conv2d(A, B, 1),
            nn.Sigmoid()
        )
        truncated_normal_(self.pose.weight, 0.01)
        truncated_normal_(self.a[0].weight, 0.01)

    def forward(self, x):
        a, M = self.a(x), self.pose(x)
        _, _, h, w = a.size()
        return a.contiguous().view(-1, 1, h, w), M.contiguous().view(-1, 1, h, w)


class ConvCapsule(nn.Module):
    """ Convolutional capsule layer """
    def __init__(self, B, C, K, stride, iters,
                 add_coordinate=False, share_transformation=False):
        """
        Args:
            B: input number of capsule types
            C: output number of capsule types
            K: kernel size of convolution
            stride: stride of convolution
            iters: number of EM iterations
            add_coordinate: use Coordinate Addition or not
            share_transformation: whether share transformation matrices in
                                  different positions of the kernel

        Each capsule have 4 * 4 + 1 dimension and represent one type feature
        in one position. In each input layer, there are h * w positions i.e.
        h * w * B capsules. We use all types capsules in K * K positions to
        calculate one type capsule in next layer. Transformation matrices in
        different positions are shared like convolution kernel.
        So transformation matrices have (B * K * K) * C * (4 * 4) parameters.
        """
        super().__init__()
        self.B = B
        self.C = C
        self.K = K
        self.stride = stride
        self.iters = iters
        self.add_coordinate = add_coordinate
        self.coordinate = None
        self.share_transformation = share_transformation

        self.beta_a = nn.Parameter(torch.Tensor(C).zero_())
        self.beta_u = nn.Parameter(torch.Tensor(C).zero_())
        if share_transformation:
            self.W = nn.Parameter(torch.eye(4)[None, None, :, :] +
                                  torch.Tensor(B, C, 4, 4).uniform_(-0.03, 0.03))
        else:
            self.W = nn.Parameter(torch.eye(4)[None, None, None, :, :] +
                                  torch.Tensor(B, K * K, C, 4, 4).uniform_(-0.03, 0.03))

        self.kernel = torch.zeros((self.K * self.K), 1, self.K, self.K)
        for i in range(self.K):
            for j in range(self.K):
                self.kernel[i * self.K + j, :, i, j] = 1

    def forward(self, a, M):
        device = a.device
        b, _, h, w = a.size()
        b = b // self.B
        oh = int((h - self.K) / self.stride + 1)
        ow = int((w - self.K) / self.stride + 1)

        # calculate votes, the vote size is (b, B, K, K, C, oh, ow, 4, 4),
        # because for each capsule in output layer (oh * ow * C) there is a
        # 4 * 4 vote matrix from its receptive field (K * K * B)
        M = conv2d(M, self.kernel.to(device), stride=self.stride) \
            .contiguous() \
            .view(b, self.B, 4, 4, self.K * self.K, 1, oh * ow) \
            .permute(0, 1, 4, 5, 6, 2, 3)
        if self.share_transformation:
            V = self.W[None, :, None, :, None, :, :] @ M
        else:
            V = self.W[None, :, :, :, None, :, :] @ M

        # add scaled corrdinate
        if self.add_coordinate:
            if self.coordinate is None:
                coordinates = []
                for i in range(self.K):
                    for j in range(self.K):
                        for l in range(oh):
                            pos_y = self.stride * l + j
                            for k in range(ow):
                                pos_x = self.stride * k + i
                                coordinates.append([pos_x / w, pos_y / h])
                coordinates = torch.Tensor(coordinates).view(self.K * self.K, oh * ow, 2)
                self.coordinate = coordinates[None, None, :, None, :, :]
            V[:, :, :, :, :, 0, :2] = self.coordinate.to(device) + V[:, :, :, :, :, 0, :2]

        # EM-routing prepare and process
        a_i = conv2d(a, self.kernel.to(device), stride=self.stride) \
            .contiguous() \
            .view(b, self.B * self.K * self.K, oh * ow) \
            .repeat(1, 1, self.C)
        V = V.view(b, self.B * self.K * self.K, self.C * oh * ow, 16)
        a_j, M_j = em_routing(a_i, V, self.beta_a, self.beta_u, self.iters)
        return a_j.view(b * self.C, 1, oh, ow), \
               M_j.permute(0, 2, 1).contiguous().view(b * self.C * 16, 1, oh, ow)


def em_routing(a, V, beta_a, beta_u, max_iter):
    """ EM routing algorithm """
    assert max_iter >= 1
    device = a.device
    # b, ni, nj = a.size()  # ni is actually B * K * K
    b, ni, nj, _ = V.size()  # nj is actually C * oh * ow
    C = beta_a.size(0)
    beta_a = beta_a[None, None, :].repeat(1, 1, nj // C)
    beta_u = beta_u[None, None, :, None].repeat(1, 1, nj // C, 1)

    R = torch.Tensor(b, ni, nj).to(device).fill_(1.0 / nj)
    # Note that `sigma` below means \sigma ^ 2 in the paper
    for t in range(max_iter):
        lamda = 0.01 * (1 - 0.95 ** (t + 1))

        # M-Step
        R = (R * a)[:, :, :, None]
        coeff = R.sum(1, True).clamp(1e-8)
        R = R / coeff  # (a * b) / a.sum(1, True) == a / a.sum(1, True) * b
        miu = (R * V).sum(1, True)
        V_miu_square = (V - miu).pow(2)
        sigma = (R * V_miu_square).sum(1, True).clamp(1e-8)
        log_sigma = sigma.log()
        cost = (beta_u + 0.5 * log_sigma) * coeff
        a_out = torch.sigmoid(lamda * (beta_a - cost.sum(-1)))

        # E-Step
        if t != max_iter - 1:
            logp_j = -(ln_2pi + log_sigma + V_miu_square / sigma).sum(-1) / 2
            R = softmax(a_out * logp_j, -1)

    return a_out.squeeze(1), miu.squeeze(1)


def common_transform(data):
    """ Common transform applied to the dataset """
    res = np.zeros((data.size(0), 2, 48, 48), dtype='uint8')
    for i, item in enumerate(data):
        item = item.numpy()
        img0 = Image.fromarray(item[0], mode='L').resize((48, 48))
        img1 = Image.fromarray(item[1], mode='L').resize((48, 48))
        res[i] = np.asarray([np.asarray(img0), np.asarray(img1)])
    return torch.from_numpy(res)


def loader_wrapper(loader):
    """ Wrapper DataLoader to a infinite iter

    Args:
        loader (DataLoader)
    """
    loader_copy = deepcopy(loader)
    loader = iter(loader)
    while True:
        try:
            yield next(loader)
        except StopIteration:
            loader = iter(loader_copy)
            yield next(loader)


if __name__ == '__main__':
    data_folder = 'smallNORB'
    log_internal = 1000
    n_iter = 2000000
    n_cpu = 32
    n_device = max(1, torch.cuda.device_count())
    batch_size = 8 * n_device
    train_transform = transforms.Compose([
        transforms.RandomCrop(32),
        transforms.ColorJitter(brightness=0.2, contrast=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.177, 0.174], [0.752, 0.757]),
    ])
    test_transform = transforms.Compose([
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize([0.177, 0.174], [0.752, 0.757]),
    ])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    download = not os.path.exists(data_folder)
    train_data = smallNORB(data_folder, train=True,
                           download=download, transform=train_transform)
    train_data.train_data = common_transform(train_data.train_data)
    test_data = smallNORB(data_folder, train=False,
                          download=download, transform=test_transform)
    test_data.test_data = common_transform(test_data.test_data)
    train_loader = DataLoader(train_data, batch_size, True, num_workers=n_cpu)
    train_loader = loader_wrapper(train_loader)
    test_loader = DataLoader(test_data, batch_size, False, num_workers=n_cpu)

    model = nn.DataParallel(CapsuleNet()).to(device)
    optimizer = Adam(model.parameters(), 3e-3, weight_decay=.0000002)
    scheduler = ExponentialLR(optimizer, 0.96)

    margin = lambda n: 0.2 + .79 / (1 + math.exp(-(min(10.0, n / 50000.0 - 4))))

    model.train()
    i_iter = 0
    while i_iter < n_iter:
        if i_iter % 20000 == 0:
            scheduler.step()

            model.eval()
            with torch.no_grad():
                total = correct = 0
                for xs, ys in test_loader:
                    xs = xs.to(device)
                    total += xs.size(0)
                    pred = model(xs)[:, :5].max(1)[1]
                    correct += (pred == ys.to(device)).sum().float()
            model.train()

            t = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print('{} Iter {} Test Acc {:.2f}%'.format(t, i_iter, correct * 100.0 / total))

        xs, ys = next(train_loader)
        m = margin(i_iter)
        xs = xs.to(device).float()
        ys = ys.to(device).long()
        optimizer.zero_grad()
        pred = model(xs)
        other_index = []
        for y in ys:
            other_index.append([i for i in range(5) if i != y])
        other_index = torch.Tensor(other_index).to(device).long()
        loss = ((m - pred.gather(1, ys.unsqueeze(1))).repeat(1, 4)
                + pred.gather(1, other_index)).clamp(0).pow(2).sum(1).mean(0)
        loss.backward()
        optimizer.step()

        if i_iter % log_internal == 0:
            t = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print('{} Iter {} Train Loss {:.4f} Margin {:.4f}'.format(t, i_iter, loss.item(), m))

        i_iter += n_device
