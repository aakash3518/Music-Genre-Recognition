import torch
from torch.utils.data import random_split, DataLoader
import argparse

from utils.model import AudioClassifier, ModelDimensions
from utils.dataset import NUM_CLASSES, get_dataset, SongDataset
from utils.train import train, valid_epoch, calculate_confusion_matrix, plot_confusion_matrix

def main(l_rate, gamma, n_epochs, batch_size, save_path, seed):
    # replicability
    torch.manual_seed(seed)

    # select device 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # load dataset
    dataset = get_dataset('annotations.csv')
    train_set, valid_set, test_set = random_split(dataset, [0.6,0.2,0.2], generator=torch.Generator().manual_seed(seed))

    # load train & validation loader
    train_set = SongDataset(train_set)
    valid_set = SongDataset(valid_set)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size)
    
    # select model
    dims = ModelDimensions(n_mels=80, n_audio_ctx=1500, n_audio_state=384, n_audio_head=6, n_audio_layer=4)
    model = AudioClassifier(dims).to(device)

    # train
    history = train(model, train_loader, valid_loader, loss_fn, device, save_path, lr=l_rate, n_epochs=n_epochs, gamma=gamma)

    # load best model
    model = AudioClassifier(dims).to(device)
    model.load_state_dict(torch.load(save_path))

    # test
    test_set = SongDataset(test_set)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loss, test_accuracy = valid_epoch(model, test_loader, loss_fn, device)
    print(f"Test loss: {test_loss}; Test accuracy: {test_accuracy}")
    print(f"Training Summary: {history}")

    # get confusion matrix
    m = calculate_confusion_matrix(model, test_loader, device)
    plot_confusion_matrix(m)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--l_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.9, help='gamma parameter for optimizer scheduler')   
    parser.add_argument('--n_epochs', type=int, default=5, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--save_path', type=str, default='models/best_model_transformer.pth', help='best model saved path')
    parser.add_argument('--seed', type=int, default=1234)
    kwargs = parser.parse_args()

    main(**vars(kwargs))
