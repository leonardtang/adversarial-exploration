from __future__ import print_function
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

epsilons = [0, .05, .1, .15, .2, .25, .3]
pretrained_model = "data/lenet_mnist_model.pth"
use_cuda = True


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# MNIST test dataset
test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=False, download=True,
                                                         transform=transforms.Compose([transforms.ToTensor()])),
                                          batch_size=1, shuffle=True)

print("CUDA Available: ", torch.cuda.is_available())
device = torch.device("cuda:1" if (use_cuda and torch.cuda.is_available()) else "cpu")

model = Net().to(device)
model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))
model.eval()

# # Multiple GPUs
# if torch.cuda.device_count() > 1:
#     model = nn.DataParallel(model)
#     print("Multi-GPU")


def fgsm_attack(image, epsilon, data_grad):
    # Getting signs of loss gradient w.r.t. image
    sign_data_grad = data_grad.sign()
    perturbation = epsilon*sign_data_grad
    # perturbed_image = np.ndarray(shape=(1, 1, data_grad.size()[2], data_grad.size()[3]), dtype=float)
    # perturbed_image = torch.from_numpy(perturbed_image).float()  # Model parameters must be floats

    # for i in range(list(data_grad.size())[2]):
    #     for j in range(list(data_grad.size())[3]):
    #         if image[0][0][i][j] != 0:
    #             perturbed_image[0][0][i][j] = image[0][0][i][j] + perturbation[0][0][i][j]

    # nonzero_idx = np.nonzero(image) 4D array for Tensor index
    # for idx in nonzero_idx:
    #     perturbed_image[0][0][idx[2]][idx[3]] = image[0][0][idx[2]][idx[3]] + perturbation[0][0][idx[2]][idx[3]]

    # image = torch.flatten(image)
    # nonzero_idx = torch.nonzero(image)
    # perturbation = np.ndarray.flatten(np.ndarray(perturbation))
    # perturbation[nonzero_idx] = 0
    # image.reshape(1, 1, 28, 28)
    # image = torch.from_numpy(image).float()
    # perturbation(1, 1, 28, 28)
    # perturbation = torch.from_numpy(perturbation).float()
    # perturbed_image = perturbation + image

    perturbation = perturbation.masked_fill(image != 0, 0)
    perturbed_image = image + perturbation

    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def test(model, device, test_loader, epsilon):

    since = time.time()

    # Accuracy counter
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for data, target in test_loader:

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]  # Get the index of the max log-probability

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # Calculate the loss -- equivalent to Cross Entropy Loss (since used LogSoftmax)
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Gradient w.r.t. input images
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)  # image, eps, gradient
        perturbed_data = perturbed_data.to(device)

        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1]
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}\t \tTime Elapsed = {}".format(
        epsilon, correct, len(test_loader), final_acc, time.time() - since))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples


accuracies = []
examples = []

# Run test for each epsilon
for eps in epsilons:
    acc, ex = test(model, device, test_loader, eps)
    accuracies.append(acc)
    examples.append(ex)

plt.figure(figsize=(5, 5))
plt.plot(epsilons, accuracies, "*-")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, .35, step=0.05))
plt.title("Accuracy vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.savefig("Accuracy-vs-Epsilon")
plt.show()

# Plotting adversarial examples (per epsilon)
cnt = 0
plt.figure(figsize=(8, 10))
for i in range(len(epsilons)):
    for j in range(len(examples[i])):
        cnt += 1
        plt.subplot(len(epsilons), len(examples[0]), cnt)
        # No ticks wanted
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
        orig, adv, ex = examples[i][j]
        plt.title("{} -> {}".format(orig, adv))
        plt.imshow(ex, cmap="gray")
plt.tight_layout()
plt.savefig("Adversarial-Examples")
plt.show()

