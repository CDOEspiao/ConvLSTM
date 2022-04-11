import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
from termcolor import colored
from model import Seq2Seq

import imageio
from datetime import datetime
import re

from Support import *

# Use GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Device: " + colored(str(device), color='blue', attrs=['bold']))
print(torch.cuda.get_device_name(0))

###########################
LEARNING_RATE = 0.0001
BATCH_SIZE = 8
NUM_EPOCHS = 40
TRAIN = False
LOAD_WEIGHTS = False
TEST = True
LATEST_CHECKPOINT = True
###########################

# Load Data as Numpy Array
MovingMNIST = np.load(r'C:\Users\psurd\PycharmProjects\ConvLSTM\.data\mnist\raw\mnist_test_seq.npy').transpose(1, 0, 2, 3)
np.random.shuffle(MovingMNIST)

# Train, Test, Validation splits
train_data = MovingMNIST[:8000]
val_data = MovingMNIST[8000:9000]
test_data = MovingMNIST[9000:10000]


def save_checkpoint(state, path="checkpoints/convLSTM_0000.pth.tar"):
    print(colored("Saving checkpoints {}".format(path), color='green'))
    torch.save(state, path)


train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE, collate_fn=collate)
val_loader = DataLoader(val_data, shuffle=True, batch_size=BATCH_SIZE, collate_fn=collate)

# Save input examples
inp, _ = next(iter(val_loader))
# Reverse process before displaying
inp = inp.cpu().numpy() * 255.0
print("Input: {}".format(inp.shape))
for i, video in enumerate(inp.squeeze(1)[:3]):          # Loop over videos
    path = r"C:\Users\psurd\PycharmProjects\ConvLSTM\examples\example_{}.gif".format(i)
    imageio.mimsave(path, video.astype(np.uint8), "GIF", fps=5)
    print("Saving example: {}".format(path))

# Training the model: randomly choose 10 frames from among 20 frames, and ask network to predict the 11th frame.
# Use Adam optimizer and Binary Cross Entropy Loss since in original dataset,
# each pixel value is 0 or 255 --> a 2 class classification problem for each pixel value.
model = Seq2Seq(num_channels=1, num_kernels=64, kernel_size=(3, 3), padding=(1, 1),
                activation="relu", frame_size=(64, 64), num_layers=3).to(device)
print(colored("Model:", "blue", attrs=["bold"]))
print(model)

if LOAD_WEIGHTS:
    print(colored("Loading latest checkpoint", color='green'))
    path = latestCheckpoint(r"C:\Users\psurd\PycharmProjects\ConvLSTM\checkpoints")
    model.load_state_dict(torch.load(path)['state_dict'])

if TRAIN:
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    print(colored("Starting Training Loop...", "blue"))
    losses = []
    for epoch in range(11, NUM_EPOCHS + 1):
        print('{:%Y-%m-%d %H:%M:%S}'.format(datetime.now()))
        if epoch > 11:
            checkpoint = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            save_checkpoint(checkpoint, "checkpoints/convLSYM_{}.pth.tar".format("".join("%04d" % epoch)))
        train_loss = 0
        model.train()
        for batch_num, (inp, target) in enumerate(train_loader, 1):
            output = model(inp)
            loss = criterion(output.flatten(), target.flatten())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()

            # Output training stats
            if batch_num % 500 == 0:
                print('[%d/%d][%d/%d]\tLoss: %.4f'
                      % (epoch, NUM_EPOCHS, batch_num, len(train_loader), loss.item()))

            # Save Losses for plotting later
            losses.append(loss.item())

        train_loss /= len(train_loader.dataset)

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for input, target in val_loader:
                output = model(input)
                loss = criterion(output.flatten(), target.flatten())
                val_loss += loss.item()
        val_loss /= len(val_loader.dataset)

        print("Epoch:{} Training Loss:{:.2f} Validation Loss:{:.2f}\n".format(epoch, train_loss, val_loss))

        if epoch % 5 == 0:
            showModelTraining(losses)


if TEST:
    # Test Data Loader
    # Number of examples: batch_size
    test_loader = DataLoader(test_data, shuffle=True, batch_size=3, collate_fn=collate_test)
    batch, target = next(iter(test_loader))

    # Create test examples
    for i, tgt in enumerate(target):  # Loop over samples
        path = r"C:\Users\psurd\PycharmProjects\ConvLSTM\results\target_{}.gif".format(i)
        imageio.mimsave(path, tgt, "GIF", fps=5)

    # Use latest checkpoint:
    checkpoints_path = r"C:\Users\psurd\PycharmProjects\ConvLSTM\checkpoints"
    if LATEST_CHECKPOINT:
        checkpoints = [latestCheckpoint(checkpoints_path)]
    else:
        checkpoints = sorted(os.listdir(checkpoints_path))
        print(colored("Checkpoints count: {}".format(len(checkpoints)), "yellow", attrs=["bold"]))

    for check in checkpoints:
        model.load_state_dict(torch.load(os.path.join(checkpoints_path, check))['state_dict'])

        # Initialize output sequence
        output = np.zeros(target.shape, dtype=np.uint8)
        output_fake = np.zeros(target.shape, dtype=np.uint8)

        # Test with recursive Real
        with torch.no_grad():
            # Loop over timesteps
            for timestep in range(target.shape[1]):
                print("Timestep {}".format(timestep))
                inp_real = batch[:, :, timestep:timestep + 10]
                # model(inp).squeeze(1).cpu() > 0.5 - for binarisation
                output[:, timestep] = model(inp_real).squeeze(1).cpu() * 255

            for timestep in range(target.shape[1]):
                inp_fake = batch[:, :, timestep:10+timestep]
                output_fake[:, timestep] = model(inp_fake).squeeze(1).cpu() * 255
                batch[:, :, 10+timestep] = model(inp_fake)

        for i, out in enumerate(output):  # Loop over samples
            # Write output video as gif
            path = r"C:\Users\psurd\PycharmProjects\ConvLSTM\results\recursReal_{}_epoch_{}.gif".format(i, re.findall("(\d+)", check)[0])
            imageio.mimsave(path, out, "GIF", fps=5)

        for i, out in enumerate(output_fake):
            path = r"C:\Users\psurd\PycharmProjects\ConvLSTM\results\recursFake_{}_epoch_{}.gif".format(i, re.findall("(\d+)", check)[0])
            imageio.mimsave(path, out, "GIF", fps=5)