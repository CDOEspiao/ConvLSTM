import matplotlib.pyplot as plt
import glob
import os
import torch
import numpy as np
import torchvision.utils as vutils

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def showModelTraining(losses):
    plt.figure(figsize=(10, 5))
    plt.title("Model Loss During Training")
    plt.plot(losses)
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def latestCheckpoint(path, mask="*.pth.tar"):
    list_of_files = glob.glob(os.path.join(path, mask))
    latest_file = os.path.basename(max(list_of_files, key=os.path.getctime))
    print("Latest checkpoint: {}".format(latest_file))
    return latest_file


def collate(batch):
    # Add channel dim, scale pixels between 0 and 1, send to GPU
    batch = torch.tensor(np.array(batch)).unsqueeze(1)
    batch = batch / 255.0
    batch = batch.to(device)

    # Randomly pick 10 frames as input, 11th frame is target
    rand = np.random.randint(10,20)
    return batch[:, :, rand-10:rand], batch[:, :, rand]


# Visualize Test model results
def collate_test(batch):
    # Last 10 frames are target
    target = np.array(batch)[:, 10:]

    # Add channel dim, scale pixels between 0 and 1, send to GPU
    batch = torch.tensor(np.array(batch)).unsqueeze(1)
    batch = batch / 255.0
    batch = batch.to(device)
    return batch, target


def createPredImage(out, batch):
    # Prepare plot
    fig, axarr = plt.subplots(batch.shape[0], batch.shape[2])
    fig.set_size_inches(18, 9)

    for image in batch:
        axarr[0,0].imshow(np.transpose(vutils.make_grid(out.to('cpu')[:64], padding=2, normalize=True), (1, 2, 0)))
        axarr[0,0].set_axis_off()
    # if i == 0:
    #     axarr[i, epoch - 1 - show_epochs * image_counter].title.set_text('Epoch {}'.format(epoch))



    for i, fixed_noise in enumerate(fixed_noises):
        fake = netG(fixed_noise).detach().cpu()
        axarr[i, epoch-1-show_epochs*image_counter].imshow(np.transpose(vutils.make_grid(fake.to('cpu')[:64], padding=2, normalize=True), (1, 2, 0)))
        axarr[i, epoch-1-show_epochs*image_counter].set_axis_off()
        if i == 0:
            axarr[i, epoch-1-show_epochs*image_counter].title.set_text('Epoch {}'.format(epoch))

# 1.Create Image: 5 x 10 Input frames -> Predicted Frame
# 2.Create Image: 10 Input frames -> 10 Recursively Predicted Frames
# 3.GIF: Input frames + 10 Recursively Predicted Frames
