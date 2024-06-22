import argparse
import torch
from lib.utils.data_preprocessing import preprocess_data, load_test_images
from lib.utils.model_operations import train_model, load_model, test_model
from lib.datasets.custom_image_dataset import CustomImageDataset

def main(training_images_folder, test_images_folder, model_file, sort_output, output_folder, score_threshold):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    training_images_folder = {
        'person': f"{training_images_folder}/person",
        'not_person': f"{training_images_folder}/not_person"
    }
    if model_file:
        model = load_model(device, model_file)
        print("Model loaded.")
    else:
        dataloader = preprocess_data(training_images_folder, CustomImageDataset)
        model = train_model(dataloader, device)
        print("Training complete.")

    test_model(model, device, test_images_folder, sort_output, output_folder, score_threshold)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_images_folder", type=str, default="training_images")
    parser.add_argument("--test_images_folder", type=str, default="test_images")
    parser.add_argument("--model_file", type=str)
    parser.add_argument("--sort_output", action='store_true')
    parser.add_argument("--output_folder", type=str, default="output")
    parser.add_argument("--score_threshold", type=float, default=0.0)
    args = parser.parse_args()
    main(args.training_images_folder, args.test_images_folder, args.model_file, args.sort_output, args.output_folder, args.score_threshold)
