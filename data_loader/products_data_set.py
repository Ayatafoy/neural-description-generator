from __future__ import print_function, division
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import os
from data_loader.data_set_preprocessor import DataSetPreprocessor
from sklearn import model_selection
from data_loader.transformations import Resize, ToTensor


class ProductsDataSet(Dataset):

    def __init__(self, features, path_to_images, transform=transforms.Compose([Resize(224), ToTensor()])):
        self.features = features.values
        self.path_to_images = path_to_images
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        product_id = self.features[idx, 0]
        img_name = self.features[idx, 1]
        product_description = self.features[idx, 2]
        img_path = os.path.join(self.path_to_images, str(img_name) + ".jpg")
        try:
            img = Image.open(img_path)
        except FileNotFoundError:
            img_path = os.path.join(self.path_to_images, str(img_name) + ".png")
            img = Image.open(img_path)
        image = img.convert('RGB')

        sample = {
            'product_id': product_id,
            'image': image,
            'product_description': product_description
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_classes(self):
        classes = set(self.features[:, 1])

        return classes


if __name__ == "__main__":
    data_dir = os.path.join("../input_data")
    processor = DataSetPreprocessor()
    features = processor.get_features(data_dir)
    input_data_set = ProductsDataSet(features, data_dir)
    train_data, test_data = model_selection.train_test_split(input_data_set, test_size=0.1, random_state=0)