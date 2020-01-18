import os
import pandas as pd


def get_vocab(descriptions):
    idx = 1
    vocab = {
        'unknown': 0
    }
    for description in descriptions:
        description_words = description.split()
        for word in description_words:
            word = word.lower()
            if word not in vocab.keys():
                vocab[word] = idx
                idx += 1

    return vocab


class DataSetPreprocessor:
    def __init__(self):
        self.features = None
        self.label_map = None
        self.vocab = None

    def get_features(self, base_dir):

        if self.features is None:
            products = os.listdir(base_dir)
            if '.DS_Store' in products:
                products.remove('.DS_Store')
            features = []
            for i, product in enumerate(products):
                path_to_product_dir = os.path.join(base_dir, product)
                files = os.listdir(path_to_product_dir)
                for file in files:
                    if file.endswith('.csv'):
                        path_to_csv = os.path.join(path_to_product_dir, file)
                        break
                products = pd.read_csv(path_to_csv, engine='python')
                products['path_to_img'] = products['product_id'].apply(lambda img_id: os.path.join(product, str(img_id)))
                features.append(products)
            features = pd.concat(features)
            features = features[['product_id', 'path_to_img', 'product_description']]
            features = features[features['product_id'] != 'productId']

            self.vocab = get_vocab(features['product_description'])

            self.features = features

        return self.features


if __name__ == "__main__":
    data_dir = os.path.join("../input_data")
    processor = DataSetPreprocessor()
    features = processor.get_features(data_dir)