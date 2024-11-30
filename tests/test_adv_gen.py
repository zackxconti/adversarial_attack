import pytest
import torch
from torchvision import models, datasets

from adversarial_attack.adv_gen import (
    preprocess_image,
    denorm_image,
    generate_adversarial_noise,
)

@pytest.fixture
def model():
    return models.resnet18(pretrained=True)

@pytest.fixture
def sample_image():
    return torch.ones((1, 3, 224, 224))  

def test_preprocess_image():
    dataset = datasets.CIFAR10(root="./data", train=False, download=True)
    image, label = dataset[1]
    try:
        processed_image = preprocess_image(image)
        assert processed_image.shape == (1, 3, 256, 256)
    except FileNotFoundError:
        pytest.skip("Image file not found. Skipping test.")

def test_deprocess_image(sample_image):
    deprocessed = denorm_image(sample_image)
    assert deprocessed.shape == (1, 3, 224, 224) 
    assert deprocessed.min() >= 0 and deprocessed.max() <= 1  


def test_generate_adversarial_noise(model, sample_image):
    target_class = 5  
    adversarial_image = generate_adversarial_noise(
        model, sample_image, target_class,epsilon=0.03, alpha=0.005, iterations=100
    )
    assert adversarial_image.shape == sample_image.shape  
    assert adversarial_image.min() >= 0 and adversarial_image.max() <= 1  
    assert not torch.equal(adversarial_image, sample_image)  