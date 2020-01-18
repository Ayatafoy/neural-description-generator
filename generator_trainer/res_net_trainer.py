import os
import numpy as np
import torch
from torch import optim
from sklearn import model_selection
import torch.utils.data as data
import time
from data_loader.products_data_set import ProductsDataSet
from data_loader.data_set_preprocessor import DataSetPreprocessor
from generator_trainer.focal_loss import FocalLoss
from tqdm import tqdm
from generator_trainer.se_res_net import SeResNet
from utils import models_manager


train_log = open(os.path.join(os.getcwd(), 'Train log'), "w")

def calculate_perplexity(true_product_description, predict_product_description):
    # Need to implement
    return 0

def criterion(logits, batch, device):
    # Need to implement
    product_description_true = batch['product_description_true'].to(device)

    car_brand_loss = FocalLoss(5)(logits, product_description_true)

    train_log.write('car_brand_loss: %s' % car_brand_loss.item() + '\n')

    total_loss = car_brand_loss

    return total_loss


def train(
        model,
        optimizer,
        scheduler,
        model_dir,
        trainloader,
        device,
        num_epochs=5,
        n_epochs_stop=10
):
    model.to(device)
    epochs_no_improve = 0
    best_perplexity = np.inf
    path_for_best_model = ''
    for epoch in tqdm(range(num_epochs)):
        epoch_start_time = time.time()
        print("Epoch %s started..." % epoch)
        true_product_description = []
        predict_product_description = []
        running_loss = 0
        for batch in tqdm(trainloader):
            image = batch['image'].to(device)
            product_description = batch['product_description'].to(device)
            optimizer.zero_grad()
            logits = model.forward(image)
            car_brand_logits, car_model_logits, body_logits, color_logits, engine_type_logits, transmission_logits,\
            loss = criterion(logits, batch, device)

            true_product_description += list(product_description.detach().cpu().numpy())
            predict_product_description += list(np.argmax(car_brand_logits.detach().cpu().numpy(), axis=1))

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print("Loss: {}".format(running_loss / len(trainloader)))
        perplexity = calculate_perplexity(true_product_description, predict_product_description)
        scheduler.step(perplexity)
        if perplexity > best_perplexity:
            epochs_no_improve = 0
            best_perplexity = perplexity
            models_manager.save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optim_dict': optimizer.state_dict()
                },
                checkpoint=model_dir
            )

            path_for_best_model = os.path.join(model_dir, 'DescriptionGenerator')
            path_for_best_model = path_for_best_model + "_accuracy_{}.pth".format(perplexity)
            torch.save(model.state_dict(), path_for_best_model)
            print("Best model \'DescriptionGenerator.pth\' saved!")
        else:
            epochs_no_improve += 1
            if epochs_no_improve == n_epochs_stop:
                print('Early stopping!')
                break
        print("Epoch %s: takes %s seconds" % (epoch, (time.time() - epoch_start_time)))
        path_for_best_model = os.path.join(model_dir, 'DescriptionGenerator.pth')
        torch.save(model.state_dict(), path_for_best_model)
    return path_for_best_model


def run_train(
        trainloader,
        testloader,
        validation_loader,
        model_dir,
        num_classes,
        path_to_pretrained_resnet=None,
        cuda_device=1,
        optimizer_lr=0.01,
        num_epohs_top=5,
        num_epohs_total=50
):
    if cuda_device is not None:
        assert cuda_device == 0 or cuda_device == 1

    device = torch.device("cuda:{}".format(cuda_device) if torch.cuda.is_available() else "cpu")
    resnet = SeResNet(num_classes)
    if path_to_pretrained_resnet is not None:
        resnet_state_dict = torch.load(path_to_pretrained_resnet)
        resnet.load_state_dict(resnet_state_dict)
    resnet.set_gr(False)

    optimizer = optim.Adam(resnet.parameters(), lr=optimizer_lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)

    train(
        model=resnet,
        optimizer=optimizer,
        scheduler=scheduler,
        model_dir=model_dir,
        trainloader=trainloader,
        testloader=testloader,
        validation_loader=validation_loader,
        device=device,
        num_epochs=num_epohs_top,
        n_epochs_stop=num_epohs_total
    )

    resnet.set_gr(True)

    path_for_best_model = train(
        model=resnet,
        optimizer=optimizer,
        scheduler=scheduler,
        model_dir=model_dir,
        trainloader=trainloader,
        testloader=testloader,
        validation_loader=validation_loader,
        device=device,
        num_epochs=num_epohs_total,
        n_epochs_stop=num_epohs_total
    )
    return path_for_best_model


params = {
    'num_epohs_top': 0,
    'num_epohs_total': 200,
    'optimizer_lr': 0.00001,
    'cuda_device': 0,
}

path_to_train_data = '../input_data/phones'
processor = DataSetPreprocessor()
features = processor.get_features(path_to_train_data)

vocab_size = len(features['vocab'])
vocab = features['vocab']

print("Num_classes: {}".format(vocab_size))
criterion_mse = torch.nn.MSELoss()

train_features, validation_features = model_selection.train_test_split(features, random_state=0)
model_dir = os.path.join(os.getcwd(), 'models')
train_features, test_features = model_selection.train_test_split(train_features, test_size=0.1)
train_data = ProductsDataSet(train_features, path_to_train_data)
test_data = ProductsDataSet(test_features, path_to_train_data)
validation_data = ProductsDataSet(validation_features, path_to_train_data)
train_loader = data.DataLoader(train_data, shuffle=True, batch_size=2)
test_loader = data.DataLoader(test_data, shuffle=False, batch_size=32)
validation_loader = data.DataLoader(validation_data, shuffle=False, batch_size=32)
run_train(
    train_loader,
    test_loader,
    validation_loader,
    model_dir,
    vocab_size,
    None,
    params['cuda_device'],
    params['optimizer_lr'],
    params['num_epohs_top'],
    params['num_epohs_total']
)