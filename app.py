import glob
import sys
import argparse
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image
import os
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms

def preprocess_data(training_images_folder):
    dataset = datasets.ImageFolder(training_images_folder, transform=transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
    ]))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    return dataloader

def train_model(dataloader, device):
    model = InceptionResnetV1(pretrained='vggface2').to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(5):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}')
    return model

def detect_faces(model, device, test_images):
    mtcnn = MTCNN(keep_all=True, device=device)
    embeddings = []
    for test_image in test_images:
        test_image_cropped = mtcnn(test_image)
        if test_image_cropped is not None:
            test_image_cropped = test_image_cropped.to(device)
            embedding = model(test_image_cropped.unsqueeze(0))
            embeddings.append(embedding)
        else:
            embeddings.append(None)
    return embeddings

def load_test_images(test_images_folder):
    test_images_paths = glob.glob(os.path.join(test_images_folder, '*'))
    test_images = [Image.open(image_path) for image_path in test_images_paths]
    return test_images, test_images_paths

def main(training_images_folder, test_images_folder):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dataloader = preprocess_data(training_images_folder)
    model = train_model(dataloader, device)
    test_images, test_images_paths = load_test_images(test_images_folder)
    embeddings = detect_faces(model, device, test_images)
    for path, embedding in zip(test_images_paths, embeddings):
        if embedding is not None:
            print(f"Face detected in {path} with embeddings: {embedding}")
        else:
            print(f"No face detected in {path}.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_images_folder", type=str, required=True, default="training_images", help="Folder containing images of the face to train on.")
    parser.add_argument("--test_images_folder", type=str, required=True, default="test_images", help="Folder containing images to test.")
    args = parser.parse_args()
    main(args.training_images_folder, args.test_images_folder)
