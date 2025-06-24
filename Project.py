import os
import torch
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt


class PotatoDiseaseClassifier:
    """Class to handle training a potato disease classification model."""

    def __init__(self, data_dir, classes, batch_size=16, learning_rate=1e-4):
        self.data_dir = data_dir
        self.classes = classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.dataloader = None

    def prepare_data_loader(self):
        """Prepare the data loader using ImageFolder and transformations."""
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ]
        )

        dataset = datasets.ImageFolder(self.data_dir, transform=transform)
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size,
                                     shuffle=True)

    def initialize_model(self):
        """Load pretrained ResNet18 and modify the final layer."""
        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, len(self.classes))
        self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.learning_rate)

    def train(self, epochs=10):
        """Train the classification model for the specified number of epochs."""
        if self.model is None or self.dataloader is None:
            raise RuntimeError("Model and data loader must be initialized first.")

        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in self.dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(self.dataloader)
            print(f"Epoch {epoch + 1}/{epochs} Loss: {avg_loss:.4f}")


class PotatoDiseaseImageGenerator:
    """Class to generate potato disease images using Stable Diffusion."""

    def __init__(self, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16
        )
        self.pipe = self.pipe.to(self.device)

    def generate_image(self, disease_name):
        """Generate and show an image of a potato with the given disease."""
        prompt = f"a photo of a potato with {disease_name} disease, detailed, high quality"
        image = self.pipe(prompt).images[0]
        plt.imshow(image)
        plt.axis("off")
        plt.title(f"Generated image: {disease_name}")
        plt.show()

    def generate_images_for_all_classes(self, classes):
        """Generate one image per disease class."""
        for disease in classes:
            print(f"Generating image for: {disease}")
            self.generate_image(disease)


def main():
    data_dir = r"C:\Users\ASUS\Desktop\Potato Disease Datasets"
    classes = [
        "Black Scurf",
        "Blackleg",
        "Common Scab",
        "Dry Rot",
        "Healthy Potatoes",
        "Miscellaneous",
        "Pink Rot",
    ]

    # 1. Train the classification model
    classifier = PotatoDiseaseClassifier(data_dir, classes)
    classifier.prepare_data_loader()
    classifier.initialize_model()
    classifier.train(epochs=10)

    # 2. Generate one image per class
    generator = PotatoDiseaseImageGenerator()
    generator.generate_images_for_all_classes(classes)


if __name__ == "__main__":
    main()
