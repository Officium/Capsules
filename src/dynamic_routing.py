"""
PyTorch implementation of "Dynamic Routing Between Capsules"
Hyper-parameters and variables are consist with the paper's settings.
"""
import os
import random
from datetime import datetime

import torch
import torch.nn as nn
from torch.nn.functional import softmax
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True


class CapsuleNet(nn.Module):
    """
        1st layer: Convolution with ReLU to extract basic features
        2nd layer: Primary capsules to get feature combinations, each capsule
                   represent features in one area extracted by different convs
        3rd layer: Digit capsules to map feature to digit labels, each capsule
                   represent features of one digit
        Decoder: Reconstruct image from digit capsule output
    """
    def __init__(self):
        super().__init__()

        self.first = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=9, stride=1),
            nn.ReLU(True),
        )

        self.second = nn.ModuleList([
            nn.Conv2d(256, 32, kernel_size=9, stride=2) for _ in range(8)
        ])

        std = 1e-2
        self.third = nn.Parameter(torch.Tensor(10, 32 * 36, 8, 16)
                                  .normal_(0, std).clamp(-2 * std, 2 * std))

        self.decoder = nn.Sequential(
            nn.Linear(160, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 28 * 28),
            nn.Sigmoid()
        )

        self.recon_loss = nn.MSELoss(reduction='none')

    def forward(self, images, labels=None):
        num = images.size(0)

        # First Layer
        fea = self.first(images)

        # Second Layer, output size B * 1152(capsule number) * 8(capsule dim)
        u = squash(torch.stack([c(fea).view(num, -1) for c in self.second], 2))

        # Third Layer, output size B * 10(capsule number) * 16(capsule dim)
        u_hat = u[None, :, :, None, :] @ self.third[:, None, :, :, :]
        b = torch.zeros(10, num, 32 * 6 * 6, 1, 1).to(u_hat.device)
        s = dynamic_routing(b, u_hat).squeeze().transpose(0, 1)

        if labels is None:
            return (s ** 2).sum(-1).max(1)[1]
        else:
            # Decoder Module
            lens = (s ** 2).sum(-1) ** 0.5
            pred = lens.max(1, True)[1]
            pred = torch.zeros(num, 10).to(device).scatter_(1, pred, 1)
            recon = self.decoder((s * pred[:, :, None]).view(num, -1))
            labels = torch.zeros(num, 10).to(device).scatter_(1, labels, 1)
            margin_loss = (
                (0.9 - lens).clamp(0) ** 2 * labels +
                (lens - 0.1).clamp(0) ** 2 * 0.5 * (1.0 - labels)
            ).sum(1)
            recon_loss = self.recon_loss(recon, images.view(num, -1)).sum(1)
            return margin_loss + 5e-4 * recon_loss


def squash(x):
    """ Squash operation """
    squared_norm = (x ** 2).sum(dim=-1, keepdim=True)
    return squared_norm / (1 + squared_norm) * x / torch.sqrt(squared_norm)


def dynamic_routing(b, u_hat, max_iter=3):
    """ Dynamic routing between primary capsule (i) and digit capsule (j)

    Args:
        b: initial logit, size N_j * B * N_i * 1 * 1
        u_hat: weighted capsule vectors, size N_j * B * N_i * 1 * D_j
        max_iter (int): routing iterations

    Returns:
        v (torch.Tensor): capsule vectors
    """
    for r in range(max_iter):
        c = softmax(b, 2)  # normalize over all capsules in layer i
        s = (c * u_hat).sum(dim=2, keepdim=True)
        v = squash(s)
        if r == max_iter - 1:
            return v
        else:
            b = b + (u_hat * v).sum(dim=-1, keepdim=True)


if __name__ == '__main__':
    data_folder = 'mnist'
    n_epoch = 200
    n_cpu = 32
    batch_size = 100  # when fine-tuning, we use batchsize 300
    train_transform = transforms.Compose([
        transforms.Pad(2),
        transforms.RandomCrop(28),
        transforms.ToTensor(),
    ])
    test_transform = transforms.ToTensor()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    download = not os.path.exists(data_folder)
    train_data = datasets.MNIST(data_folder, train=True,
                                download=download, transform=train_transform)
    test_data = datasets.MNIST(data_folder, train=False,
                               download=download, transform=test_transform)
    train_loader = DataLoader(train_data, batch_size, True, num_workers=n_cpu)
    test_loader = DataLoader(test_data, batch_size, False, num_workers=n_cpu)

    model = nn.DataParallel(CapsuleNet()).to(device)
    optimizer = Adam(model.parameters())  # when fine-tuning, we use lr 1e-5

    best_acc = 0
    for i_epoch in range(1, n_epoch + 1):
        model.train()
        for xs, ys in train_loader:
            xs = xs.to(device)
            ys = ys.to(device).unsqueeze(1)
            optimizer.zero_grad()
            loss = model(xs, ys).mean()
            loss.backward()
            optimizer.step()
            batch_loss = loss.item()

        with torch.no_grad():
            model.eval()
            total = right = 0
            for xs, ys in test_loader:
                xs = xs.to(device)
                total += xs.size(0)
                pred = model(xs)
                right += (pred == ys.to(device)).sum().float()
            acc = right * 100.0 / total

        t = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        lr = optimizer.param_groups[0]['lr']
        if acc >= best_acc:
            best_acc = acc
            save_path = os.path.join(data_folder, '{}.pkl'.format(i_epoch))
            torch.save([model.state_dict(), optimizer.state_dict()], save_path)
        print('{} Epoch {} Train Loss {:.4f} LR {:.6f} Acc {:.2f}%/{:.2f}%'
              .format(t, i_epoch, batch_loss, lr, acc, best_acc))
