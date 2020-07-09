from __future__ import print_function
from torchvision import datasets, transforms, models
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

epsilons = [0, .05, .1, .15, .2, .25, .3]
pretrained_model = "model/lenet_mnist_model.pth"
use_cuda = True


""" Flow is test without any additional training OR include feature extraction/finetune """
""" Tasks - 1) Train to see how well pretrained models work with now training 2) Finetune 3) Test again"""


def set_parameter_requires_grad(model, feature_extracting):
    """ Feature extraction: no need for backprop through all layers
        Finetuning: must have backprop (grad) through all layers """

    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


# Setting up ResNet-101 pre-trained on ImageNet
model = models.resnet101(pretrained=True, progress=True)
set_parameter_requires_grad(model, feature_extracting=True)  # Set all layers to not require grad
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 80)  # 80 categories for MS-COCO
input_size = 224

# COCO 2014 val dataset (using here as test)
test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=False, download=True,
                                                         transform=transforms.Compose([transforms.ToTensor()])),
                                          batch_size=1, shuffle=True)

print("CUDA Available: ", torch.cuda.is_available())
device = torch.device("cuda:1" if (use_cuda and torch.cuda.is_available()) else "cpu")

model = model.to(device)
model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))
model.eval()

# # Multiple GPUs
# if torch.cuda.device_count() > 1:
#     model = nn.DataParallel(model)
#     print("Multi-GPU")


def fgsm_attack(image, epsilon, data_grad, context=False):

    if context:
        """ Only perturbing context """
        # Getting signs of loss gradient w.r.t. image
        sign_data_grad = data_grad.sign()
        perturbation = epsilon * sign_data_grad
        perturbation = perturbation.masked_fill(image != 0, 0)
        perturbed_image = image + perturbation

    else:
        """ Traditional approach """
        sign_data_grad = data_grad.sign()
        perturbed_image = image + epsilon * sign_data_grad

    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


def test(model, device, test_loader, epsilon):

    since = time.time()

    # Accuracy counter
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for data, target in test_loader:

        # Send the model and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the model through the model
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

