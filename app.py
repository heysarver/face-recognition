import glob
import sys
import argparse
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image
import os
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.io import read_image
from torchvision.transforms.functional import convert_image_dtype
from torchvision.transforms import ToTensor

class CustomImageDataset(Dataset):
    def __init__(self, img_dirs, transform=None):
        # img_dirs is a dict with keys 'person' and 'not_person' pointing to their respective image folders
        self.img_labels = []
        self.img_names = []
        for label, img_dir in enumerate(['person', 'not_person']):
            dir_path = img_dirs[img_dir]
            for img_name in os.listdir(dir_path):
                if img_name.endswith('.jpg'):
                    self.img_names.append(os.path.join(dir_path, img_name))
                    self.img_labels.append(label)
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = self.img_names[idx]
        image = read_image(img_path)
        image = convert_image_dtype(image, dtype=torch.float32)
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        if self.transform:
            image = self.transform(image)
        label = self.img_labels[idx]
        return image, label

def preprocess_data(training_images_folder):
    dataset = CustomImageDataset(training_images_folder, transform=transforms.Compose([
        transforms.Resize((160, 160)),
    ]))
    print(f"Length of dataset: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=len(dataset)//2, shuffle=True)
    print(f"Length of dataloader: {len(dataloader)}")
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

def detect_faces(model, device, test_images, test_images_paths):
    model.eval()  # set the model to evaluation mode
    mtcnn = MTCNN(keep_all=True, device=device)
    embeddings = []
    for idx, test_image in enumerate(test_images):
        try:
            print(f"Processing image: {test_images_paths[idx]}")  # Log the image being processed
            test_image_cropped = mtcnn(test_image)
            print(f"Output from MTCNN for {test_images_paths[idx]}: {test_image_cropped}")  # Log MTCNN output

            if test_image_cropped is not None:
                if isinstance(test_image_cropped, list):
                    if len(test_image_cropped) > 0:
                        test_image_cropped = torch.stack(test_image_cropped).to(device)
                        print(f"Faces detected and stacked for {test_images_paths[idx]}")
                    else:
                        print(f"No faces detected in {test_images_paths[idx]} after MTCNN processing.")
                        embeddings.append(None)
                        continue
                else:
                    test_image_cropped = test_image_cropped.unsqueeze(0).to(device)
                    print(f"Single face detected and processed for {test_images_paths[idx]}")
                
                embedding = model(test_image_cropped)
                embeddings.append(embedding)
            else:
                print(f"No faces detected in {test_images_paths[idx]} by MTCNN.")
                embeddings.append(None)
        except Exception as e:
            print(f"Error processing image {test_images_paths[idx]}: {e}")
            embeddings.append(None)
    model.train()  # set the model back to training mode
    return embeddings

def load_test_images(test_images_folder):
    test_images_paths = glob.glob(os.path.join(test_images_folder, '*.jpg'))
    to_tensor = ToTensor()
    test_images = [to_tensor(Image.open(image_path)) for image_path in test_images_paths]
    return test_images, test_images_paths

# Update the main function to pass test_images_paths to detect_faces
def main(training_images_folder, test_images_folder):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    training_images_folder = {'person': os.path.join(training_images_folder, 'person'), 
                              'not_person': os.path.join(training_images_folder, 'not_person')}
    dataloader = preprocess_data(training_images_folder)
    model = train_model(dataloader, device)
    test_images, test_images_paths = load_test_images(test_images_folder)
    embeddings = detect_faces(model, device, test_images, test_images_paths)  # Pass test_images_paths
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
