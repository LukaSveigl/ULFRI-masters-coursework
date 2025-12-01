import torch
import PIL
import numpy as np
import torchvision.models
import os
from torchvision import transforms


if __name__ == "__main__":
    print("\n")
    folder_dir = '../datasets/ears/images/test'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        model = torchvision.models.resnet50(pretrained=True)
        model.fc = torch.nn.Linear(2048, 136)
        model.load_state_dict(torch.load('../best_model.pth', map_location=device), strict=False)
        model.to(device)
        feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

        features_list = []

        transforms = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # Default values for imagenet.
                std=[0.229, 0.224, 0.225]
            )
        ])

        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()

        for filename in os.listdir(folder_dir):
            image = PIL.Image.open(os.path.join(folder_dir, filename)).convert("RGB")

            input_tensor = transforms(image).unsqueeze(0)
            input_tensor = input_tensor.to(device)

            features = feature_extractor(input_tensor)

            del input_tensor

            features_list.append(features)

            filename = filename.replace('images', 'features-resnet')
            filename = filename.replace('.png', '.txt')

            features_folder_dir = folder_dir.replace('images', 'features-resnet')

            np.savetxt(os.path.join(features_folder_dir, filename), features.cpu().numpy().flatten(), delimiter=',')

            del features

        print(features_list)
