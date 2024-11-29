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

    # # Visualize the example image (optional)
    # plt.imshow(example_image)
    # plt.title(f"Original Label: {label}")
    # plt.show()

    # Define the image transformation pipeline
    transform = transforms.Compose([
    transforms.Resize((256)), 
    transforms.ToTensor(),         
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
    ])  

    # Transform the image
    preprocessed_image = transform(image)

    return preprocessed_image


def denorm(batch, device, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
  
  if isinstance(mean, list):
    mean = torch.tensor(mean).to(device)
  if isinstance(std, list):
    std = torch.tensor(std).to(device)
  return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

def validate_model(model, image, label_map=None):

    model.eval()
    outputs = model(image)
    _, predicted_class = outputs.max(1)
    class_index = predicted_class.item()
    if label_map:
        print(f"Model prediction: {label_map.get(class_index, 'Unknown')} (class {class_index})")
    else:
        print(f"Model prediction: Class {class_index}")
    return class_index

def generate_adversarial_noise(model, image, target_class, epsilon=0.03, alpha=0.005, iterations=100):

    # Clone the input image and set requires_grad=True
    adv_image = image.clone().detach().requires_grad_(True)
    target = torch.tensor([target_class]).to(device)

    criterion = nn.CrossEntropyLoss()

    for i in range(iterations):
        # Forward pass
        output = model(adv_image)
        loss = criterion(output, target)

        # Zero the gradients, compute gradients
        model.zero_grad()
        loss.backward()

        # Update the adversarial image
        adv_image = adv_image - alpha * adv_image.grad.sign()
        adv_image = torch.clamp(adv_image, image - epsilon, image + epsilon)  
        adv_image = torch.clamp(adv_image, 0, 1).detach().requires_grad_(True)

        if i % 10 == 0:
            print(f"Iteration {i}/{iterations}, Loss: {loss.item()}")

    return adv_image

def load_classes ():
    imagenet_classes_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    urllib.request.urlretrieve(imagenet_classes_url, "imagenet_classes.txt")
    
    # Load class names into a list
    with open("imagenet_classes.txt") as f:
        imagenet_classes = [line.strip() for line in f.readlines()]

    return imagenet_classes    

def initial_predition (model, image_batch, labels):

    pred = model(image_batch)
    # probs = F.softmax(pred[0], dim=0)
    probs = torch.nn.functional.softmax(pred[0], dim=0)
    
    predicted_class = probs.argmax().item()
    print(f"Class Index: {predicted_class}, Class Name: {labels[predicted_class]}")
    probs_top5, idx_top5 = torch.topk(probs, 5)
    print("The top 5 labels of highly probabilies:")
    for i in range(probs_top5.size(0)):
        print(f"{labels[idx_top5[i]]}: {probs_top5[i].item()*100:.2f}%")

def main ():
  
    print ("hello main has been executed")
    model = models.resnet18(pretrained=True) 
    model.eval()  # Set to evaluation mode

    torch.manual_seed(42)
    use_cuda = True

    dataset = datasets.CIFAR10(root="./data", train=False, download=True)
    example_image, label = dataset[1]  

    image_tensor = preprocess_image(example_image)
    image_batch = image_tensor.unsqueeze(0)
    image_batch = image_batch.to(device)
    model = model.to(device)

    labels = load_classes()
    initial_predition(model, image_batch,labels)
    print ('target class ', target_class)

    target_class = 189 

    adversarial_image = generate_adversarial_noise(model, image_batch, target_class)

    predicted_class = validate_model(model, adversarial_image)  
    print ("predicted class = " ,labels[predicted_class])

    # # visualize the original and adversarial images
    # original_np = denorm(image_batch, device)
    # adversarial_np = denorm(adversarial_image, device)
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