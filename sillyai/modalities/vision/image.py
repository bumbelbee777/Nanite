import math
import torch
import torch.nn as nn

class ImageModality(nn.Module):
    def __init__(self, input_channels, output_dim, input_size=(32, 32), dropout_rate=0.5):
        super().__init__()
        self.input_channels = input_channels
        self.output_dim = output_dim
        self.input_size = input_size

        # Define a more advanced CNN for image feature extraction
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout_rate)

        # Dynamically calculate the number of features after convolutions
        self.flattened_dim = self._calculate_flattened_dim()
        self.fc = nn.Linear(self.flattened_dim, output_dim)

    def _calculate_flattened_dim(self):
        # Simulate a forward pass to calculate the flattened dimension
        with torch.no_grad():
            x = torch.zeros(1, self.input_channels, *self.input_size)
            x = self.pool(torch.relu(self.bn1(self.conv1(x))))
            x = self.pool(torch.relu(self.bn2(self.conv2(x))))
            return x.numel()

    def forward(self, x):
        # x: [batch_size, input_channels, height, width]
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

    @staticmethod
    def load_and_process_image(image_path, input_size):
        """
        Load an image from a file and preprocess it for the ImageModality.

        Args:
            image_path (str): Path to the image file.
            input_size (tuple): Target size (height, width) for resizing the image.

        Returns:
            torch.Tensor: Preprocessed image tensor of shape [1, input_channels, height, width].
        """
        from PIL import Image
        import torchvision.transforms as transforms

        # Define preprocessing pipeline
        preprocess = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
        ])

        # Load and preprocess the image
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        image_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

        return image_tensor