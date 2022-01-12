import time
import torch.backends.cudnn as cudnn
import torch
from torch import nn
from models import SRResNet
from datasets import SRDataset
from utils import *
import os
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio

checkpoint_dir = "results"
if not os.path.isdir(checkpoint_dir):
    os.mkdir(checkpoint_dir)

pth_num = 0
pth_dir = "pth_" + str(pth_num) + "/"
while os.path.isdir(pth_dir):
    pth_num = pth_num + 1
    pth_dir = "pth_" + str(pth_num) + "/"
os.mkdir(pth_dir)

history = []
psnrlist = []

# Data parameters
data_folder = "./"  # folder with JSON data files
crop_size = 150  # crop size of target HR images
scaling_factor = 3  # the scaling factor

# Model parameters
large_kernel_size = 9
small_kernel_size = 3
n_channels = 64  # number of channels in-between
n_blocks = 16  # number of residual blocks

# Learning parameters
checkpoint = None  # path to model checkpoint, None if none
batch_size = 64  # batch size
start_epoch = 0  # start at this epoch
iterations = 1e5  # number of training iterations
workers = 4  # number of workers for loading data in the DataLoader
print_freq = 500  # print training status once every __ batches
lr = 1e-4  # learning rate
grad_clip = None  # clip if gradients are exploding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cudnn.benchmark = True


def main():
    """
    Training.
    """
    global start_epoch, epoch, checkpoint

    # Initialize model or load checkpoint
    if checkpoint is None:
        model = SRResNet(
            large_kernel_size=large_kernel_size,
            small_kernel_size=small_kernel_size,
            n_channels=n_channels,
            n_blocks=n_blocks,
            scaling_factor=scaling_factor,
        )
        # Initialize the optimizer
        optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, model.parameters()), lr=lr
        )

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        model = checkpoint["model"]
        optimizer = checkpoint["optimizer"]

    # Move to default device
    model = model.to(device)
    criterion = nn.MSELoss().to(device)

    # Custom dataloaders
    train_dataset = SRDataset(
        split="train",
        crop_size=crop_size,
        scaling_factor=scaling_factor,
        lr_img_type="imagenet-norm",
        hr_img_type="[-1, 1]",
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
    )  # note that we're passing the collate function here
    val_dataset = SRDataset(
        split="val",
        crop_size=0,
        scaling_factor=scaling_factor,
        lr_img_type="imagenet-norm",
        hr_img_type="[-1, 1]",
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=True,
        num_workers=workers, pin_memory=True
    )

    # Total number of epochs to train for
    epochs = int(iterations // len(train_loader) + 1)
    maxpsnr = 0
    # Epochs
    for epoch in range(start_epoch, epochs):
        # One epoch's training
        train(
            train_loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
        )

        epoch_psnr = psnr_cal(
            val_loader=val_loader,
            model=model, epoch=epoch
        )
        if epoch_psnr.avg > maxpsnr:
            maxpsnr = epoch_psnr.avg
            # Save checkpoint
            torch.save(
                {"epoch": epoch, "model": model, "optimizer": optimizer},
                pth_dir + "checkpoint_srresnet.pth.tar",
            )

    print(len(history))
    # summarize history for accuracy
    plt.figure(1)
    plt.plot(history)
    plt.title("Loss")
    plt.ylabel("loss")
    plt.xlabel("iteration")
    plt.savefig(pth_dir + "loss.png")

    plt.figure(2)
    print(len(psnrlist))
    # summarize history for accuracy
    plt.plot(psnrlist)
    plt.title("psnr")
    plt.ylabel("psnr")
    plt.xlabel("iteration")
    plt.savefig(pth_dir + "psnr.png")


def train(train_loader, model, criterion, optimizer, epoch):
    """
    One epoch's training.
    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: content loss function (Mean Squared-Error loss)
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables batch normalization

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss
    start = time.time()

    # Batches
    for i, (lr_imgs, hr_imgs) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to default device
        lr_imgs = lr_imgs.to(device)
        # (batch_size (N), 3, 24, 24), imagenet-normed
        hr_imgs = hr_imgs.to(device)
        # (batch_size (N), 3, 96, 96), in [-1, 1]

        # Forward prop.
        sr_imgs = model(lr_imgs)
        # (N, 3, 96, 96), in [-1, 1]

        # Loss
        loss = criterion(sr_imgs, hr_imgs)  # scalar

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()

        # Keep track of loss
        losses.update(loss.item(), lr_imgs.size(0))

        # Keep track of batch time
        batch_time.update(time.time() - start)

        # Reset start time
        start = time.time()

        # Print status
        # if i % print_freq == 0:
        print(
            "Epoch: [{0}][{1}/{2}]----"
            "Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})----"
            "Data Time {data_time.val:.3f} ({data_time.avg:.3f})----"
            "Loss {loss.val:.4f} ({loss.avg:.4f})".format(
                epoch,
                i,
                len(train_loader),
                batch_time=batch_time,
                data_time=data_time,
                loss=losses,
            )
        )
        # print((epoch+1)*i)
        # if(((epoch+1)*i)%8==0):
        # print(float(loss.cpu()))
        history.append(float(loss.cpu()))

    del (
        lr_imgs,
        hr_imgs,
        sr_imgs,
    )  # free some memory since their histories may be stored


def psnr_cal(val_loader, model, epoch):
    global psnrlist
    model.eval()
    PSNRs = AverageMeter()
    with torch.no_grad():
        for i, (lr_imgs, hr_imgs) in enumerate(val_loader):
            # Move to default device
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            sr_imgs = model(lr_imgs)
            sr_imgs_y = convert_image(
                sr_imgs, source="[-1, 1]", target="y-channel"
            ).squeeze(0)
            hr_imgs_y = convert_image(
                hr_imgs, source="[-1, 1]", target="y-channel"
            ).squeeze(0)
            psnr = peak_signal_noise_ratio(
                hr_imgs_y.cpu().numpy(),
                sr_imgs_y.cpu().numpy(),
                data_range=255.0
            )
            PSNRs.update(psnr, lr_imgs.size(0))
    print(f"Epoch: {epoch}, PSNR: {PSNRs.avg}")
    psnrlist.append(PSNRs.avg)
    del lr_imgs, hr_imgs, sr_imgs
    return PSNRs


if __name__ == "__main__":
    main()
