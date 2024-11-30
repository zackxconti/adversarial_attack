from adv_gen import*

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
    
    # make intial prediction (optional)
    initial_prediction(model, original_image_batch,labels)

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
    # plt.show()

    
if __name__ == "__main__":
    main()