import glob
import sys
import argparse
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor

class CustomImageDataset(Dataset):
    def __init__(self, img_dirs, transform=None):
        self.img_labels = []
        self.img_names = []
        for label, dir_path in img_dirs.items():
            for img_name in os.listdir(dir_path):
                if img_name.endswith('.jpg'):
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

def preprocess_data(training_images_folder):
    dataset = CustomImageDataset(training_images_folder, transform=transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]))
    dataloader = DataLoader(dataset, batch_size=len(dataset)//2, shuffle=True)
    return dataloader

def train_model(dataloader, device):
    model = InceptionResnetV1(pretrained='vggface2').to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(5):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    return model

def detect_faces(model, device, test_images, test_images_paths):
    model.eval()
    mtcnn = MTCNN(keep_all=True, device=device)
    embeddings = []
    for idx, test_image in enumerate(test_images):
        try:
            test_image_cropped = mtcnn(test_image)
            if test_image_cropped is not None and len(test_image_cropped) > 0:
                if isinstance(test_image_cropped, list):
                    test_image_cropped = torch.stack(test_image_cropped).to(device)
                else:
                    test_image_cropped = test_image_cropped.unsqueeze(0).to(device)
                embedding = model(test_image_cropped)
                embeddings.append(embedding)
            else:
                embeddings.append(None)
        except Exception as e:
            embeddings.append(None)
    model.train()
    return embeddings

def load_test_images(test_images_folder):
    test_images_paths = glob.glob(os.path.join(test_images_folder, '*.jpg'))
    test_images = [Image.open(image_path) for image_path in test_images_paths]
    return test_images, test_images_paths

def main(training_images_folder, test_images_folder):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    training_images_folder = {
        'person': os.path.join(training_images_folder, 'person'),
        'not_person': os.path.join(training_images_folder, 'not_person')
    }
    dataloader = preprocess_data(training_images_folder)
    model = train_model(dataloader, device)
    test_images, test_images_paths = load_test_images(test_images_folder)
    embeddings = detect_faces(model, device, test_images, test_images_paths)
    for path, embedding in zip(test_images_paths, embeddings):
        if embedding is not None:
            print(f"Face detected in {path} with embeddings: {embedding}")
        else:
            print(f"No face detected in {path}.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_images_folder", type=str, default="training_images", help="Folder containing images of the face to train on.")
    parser.add_argument("--test_images_folder", type=str, default="test_images", help="Folder containing images to test.")
    args = parser.parse_args()
    main(args.training_images_folder, args.test_images_folder)
