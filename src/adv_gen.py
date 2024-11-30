import torch
from torchvision import models, transforms, datasets
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import urllib.request

device = "cuda" if torch.cuda.is_available() else "cpu"


def preprocess_image (image):
    """ Preprocesses an input image for model inference.

    Args:
        image (PIL.Image): The input image to preprocess.

    Returns:
        torch.Tensor: The preprocessed image as a tensor, ready for model input.
    """
    transform = transforms.Compose([
    transforms.Resize((256)), 
    transforms.ToTensor(),         
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
    ])  

    preprocessed_image = transform(image)
    image_batch = preprocessed_image.unsqueeze(0)
    image_batch = image_batch.to(device)

    return image_batch


def denorm_image(batch, device, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """ Denormalizes a batch of images by reversing normalization operations.

    Args:
        batch (torch.Tensor): The input image batch to denormalize.
        device (torch.device): The device (CPU or GPU) to perform the operation on.
        mean (list, optional): The mean values used for normalization. Defaults to [0.485, 0.456, 0.406].
        std (list, optional): The standard deviation values used for normalization. Defaults to [0.229, 0.224, 0.225].

    Returns:
        torch.Tensor: The denormalized image batch.
    """
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)

    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

def validate_model(model, image, labels):
    """ Validates the model's prediction on an input image.

    Args:
        model (torch.nn.Module): The trained model to use for prediction.
        image (torch.Tensor): The input image tensor to classify.
        label_map (dict, optional): A mapping from class index to human-readable label. Defaults to None.

    Returns:
        int: The predicted class index.
    """
    model.eval()
    pred = model(image)
    probs = torch.nn.functional.softmax(pred[0], dim=0)
    probs_top5, idx_top5 = torch.topk(probs, 5)

    print("Misclassified prediction / top 5 labels:")
    for i in range(probs_top5.size(0)):
        print(f"{labels[idx_top5[i]]}: {probs_top5[i].item()*100:.2f}%")

    # _, predicted_class = outputs.max(1)
    # class_index = predicted_class.item()
    # if labels:
    #     print(f"Model prediction: {labels[class_index], 'Unknown')} (class {class_index})")
    # else:
    #     print(f"Model prediction: Class {class_index}")
    # return class_index

def generate_adversarial_noise(model, image, target_class, epsilon=0.03, alpha=0.005, iterations=100):
    """ Generates adversarial noise to misclassify the input image.

    Args:
        model (torch.nn.Module): The pretrained model to generate adversarial examples against.
        image (torch.Tensor): The input image tensor to perturb.
        target_class (int): The class index to misclassify the image to.
        epsilon (float, optional): The magnitude of perturbations allowed. Defaults to 0.03.
        alpha (float, optional): The step size for perturbations. Defaults to 0.005.
        iterations (int, optional): The number of iterations to apply adversarial updates. Defaults to 100.

    Returns:
        torch.Tensor: The perturbed adversarial image.
    """
    adv_image = image.clone().detach().requires_grad_(True)
    target = torch.tensor([target_class]).to(device)

    criterion = nn.CrossEntropyLoss()

    for i in range(iterations):
        # forward pass
        output = model(adv_image)
        loss = criterion(output, target)

        # zero the gradients, compute gradients
        model.zero_grad()
        loss.backward()

        # update the adversarial image
        adv_image = adv_image - alpha * adv_image.grad.sign()
        adv_image = torch.clamp(adv_image, image - epsilon, image + epsilon)  
        adv_image = torch.clamp(adv_image, 0, 1).detach().requires_grad_(True)

        if i % 10 == 0:
            print(f"Iteration {i}/{iterations}, Loss: {loss.item()}")

    return adv_image

def load_classes (url):
    """ Loads the ImageNet class labels.

    Returns:
        list: A list of ImageNet class labels.
    """
    urllib.request.urlretrieve(url, "imagenet_classes.txt")
    
    with open("imagenet_classes.txt") as f:
        imagenet_classes = [line.strip() for line in f.readlines()]

    return imagenet_classes    

def initial_predition (model, image, labels):
    """ Makes an initial prediction on the input image and prints top 5 predicted classes.

    Args:
        model (torch.nn.Module): The pretrained model to use for prediction.
        image (torch.Tensor): The input image tensor to classify.
        labels (list): A list of class labels corresponding to class indices.
    """
    pred = model(image)
    probs = torch.nn.functional.softmax(pred[0], dim=0)
    probs_top5, idx_top5 = torch.topk(probs, 5)
    
    print("Initial prediction / top 5 labels:")
    for i in range(probs_top5.size(0)):
        print(f"{labels[idx_top5[i]]}: {probs_top5[i].item()*100:.2f}%")

def main ():

    # load pretrained model 
    model = models.resnet18(pretrained=True) 
    model.eval()  
    model.to(device)
    torch.manual_seed(42)

    # load original image from dataset
    dataset = datasets.CIFAR10(root="./data", train=False, download=True)
    original_image, label = dataset[1]  

    # preprocess image
    original_image_batch = preprocess_image(original_image)

    # load class labels 
    labels = load_classes("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt")
    
    # make intial predictio (optional)
    initial_predition(model, original_image_batch,labels)

    # generate adversarial noise for target class of choice and validate
    target_class = 189 
    adversarial_image = generate_adversarial_noise(model, original_image_batch, target_class)
    validate_model(model, adversarial_image, labels)  

    # print ("predicted class = " ,labels[predicted_class])

    # # visualize the original and adversarial images
    # original_np = denorm_image(image_batch, device)
    # adversarial_np = denorm_image(adversarial_image, device)
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.imshow(original_np)
    # plt.title("Original Image")
    # plt.axis("off")

    # plt.subplot(1, 2, 2)
    # plt.imshow(adversarial_np)
    # plt.title("Adversarial Image")
    # plt.axis("off")
    plt.show()

    
if __name__ == "__main__":
    main()