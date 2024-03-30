import os
import shutil
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor
import argparse
from facenet_pytorch import MTCNN
from PIL import Image
import os
from torchvision.transforms import RandomHorizontalFlip, RandomRotation, Resize, Normalize
from torch.utils.data import Dataset
import glob
import torch.nn.functional as F

class CustomImageDataset(Dataset):
    def __init__(self, img_dirs, transform=None):
        self.img_labels = []
        self.img_names = []
        for label, dir_path in img_dirs.items():
            for img_name in os.listdir(dir_path):
                if img_name.lower().endswith(('.jpg', '.png')):
                    self.img_names.append(os.path.join(dir_path, img_name))
                    numeric_label = 0 if label == 'person' else 1
                    self.img_labels.append(numeric_label)
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = self.img_names[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.img_labels[idx]
        return image, label

class SimpleCNN(torch.nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        # Calculate the size of the features after the convolution and pooling layers
        # For an input image of size (160, 160), after two convolutions and poolings, the size is (40, 40)
        self.fc1 = torch.nn.Linear(64 * 40 * 40, 128)  # Adjusted size
        self.fc2 = torch.nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def preprocess_data(training_images_folder):
    dataset = CustomImageDataset(training_images_folder, transform=transforms.Compose([
        Resize((160, 160)),
        RandomHorizontalFlip(),
        RandomRotation(20),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]))
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    return dataloader

def train_model(dataloader, device):
    model = SimpleCNN().to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Train all parameters
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(50):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss}")
    torch.save(model.state_dict(), 'models/model.pth')
    return model

def load_test_images(test_images_folder):
    test_images_paths = glob.glob(os.path.join(test_images_folder, '*.jpg')) + glob.glob(os.path.join(test_images_folder, '*.png')) + glob.glob(os.path.join(test_images_folder, '*.JPG')) + glob.glob(os.path.join(test_images_folder, '*.PNG'))
    test_images = [Image.open(image_path).convert('RGB') for image_path in test_images_paths]
    return test_images, test_images_paths

def load_model(device, model_file):
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_file))
    return model

def main(training_images_folder, test_images_folder, model_file, sort_output, output_folder, score_threshold):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    training_images_folder = {
        'person': os.path.join(training_images_folder, 'person'),
        'not_person': os.path.join(training_images_folder, 'not_person')
    }
    dataloader = preprocess_data(training_images_folder)
    if model_file:
        model = load_model(device, model_file)
        print("Model loaded.")
    else:
        model = train_model(dataloader, device)
        print("Training complete.")

    # Load test images
    test_images, test_images_paths = load_test_images(test_images_folder)

    # Preprocess test images
    transform = transforms.Compose([
        Resize((160, 160)),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    test_images = [transform(image) for image in test_images]

    # Test the model
    model.eval()
    with torch.no_grad():
        for i, test_image in enumerate(test_images):
            test_image = test_image.unsqueeze(0).to(device)
            output = model(test_image)
            _, predicted = torch.max(output, 1)
            score = torch.nn.functional.softmax(output, dim=1)[0][predicted].item()
            label = 'person' if predicted.item() == 0 else 'not_person'
            print(f"Image: {test_images_paths[i]}, Label: {label}, Score: {score}")
            if sort_output and score >= score_threshold:
                dest_folder = os.path.join(output_folder, label)
                os.makedirs(dest_folder, exist_ok=True)
                shutil.copy(test_images_paths[i], dest_folder)

    model.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_images_folder", type=str, default="training_images", help="Folder containing images of the face to train on.")
    parser.add_argument("--test_images_folder", type=str, default="test_images", help="Folder containing images to test.")
    parser.add_argument("--model_file", type=str, help="Model file to load.")
    parser.add_argument("--sort_output", type=bool, default=False, help="Whether to sort the output.")
    parser.add_argument("--output_folder", type=str, default="output", help="Folder to store the output.")
    parser.add_argument("--score_threshold", type=float, default=0, help="Score threshold for sorting output.")
    args = parser.parse_args()
    main(args.training_images_folder, args.test_images_folder, args.model_file, args.sort_output, args.output_folder, args.score_threshold)
