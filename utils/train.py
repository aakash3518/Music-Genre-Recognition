import torch
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from utils.dataset import NUM_CLASSES, classes

def train_epoch(model, dataloader, loss_fn, optimizer, device):
    train_loss,train_correct=0.0,0
    model.train()

    pbar = tqdm(dataloader)
    for i, (images, labels) in enumerate(pbar):
        images,labels = images.to(device),labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
        predictions = torch.argmax(output.data, 1)
        train_correct += (predictions == labels).sum().item()
        pbar.set_description(f"Training Loss: {train_loss/(i+1):.3f} Training Acc: {train_correct/(i+1):.2f}")

    return train_loss/len(dataloader.sampler), train_correct/len(dataloader.sampler)


def valid_epoch(model,dataloader,loss_fn, device):
    valid_loss, val_correct = 0.0, 0
    model.eval()
    for images, labels in dataloader:

        images,labels = images.to(device),labels.to(device)
        output = model(images)
        loss=loss_fn(output,labels)
        valid_loss+=loss.item()*images.size(0)
        predictions = torch.argmax(output.data,1)
        val_correct+=(predictions == labels).sum().item()

    return valid_loss/len(dataloader.sampler), val_correct/len(dataloader.sampler)


def calculate_confusion_matrix(model, dataloader, device):
    valid_loss, val_correct = 0.0, 0
    model.eval()

    conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))
    for images, labels in dataloader:

        images,labels = images.to(device),labels.to(device)
        output = model(images)
        predictions = torch.argmax(output.data,1)
        conf_matrix += confusion_matrix(labels, predictions, labels=list(range(NUM_CLASSES)))

    return conf_matrix


def train(model, train_loader, valid_loader, loss_fn, device, save_path, lr=1e-4, n_epochs=5, gamma=0.9):
    # optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    history = {'train_loss': [], 'valid_loss': [],'train_acc':[],'valid_acc':[]}

    best_accuracy = -1.0
    for epoch in range(n_epochs):
        train_loss, train_acc=train_epoch(model, train_loader, loss_fn, optimizer, device)
        valid_loss, valid_acc=valid_epoch(model, valid_loader, loss_fn, device)
        
        scheduler.step()
        print(f"Epoch #{epoch:3d}: Training Loss: {train_loss:.3f} Valid Loss: {valid_loss:.3f} Training Acc: {train_acc:.2f} Valid Acc: {valid_acc:.2f}")

        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)
        history['train_acc'].append(train_acc)
        history['valid_acc'].append(valid_acc)

        if best_accuracy < valid_acc:
            best_accuracy = valid_acc
            torch.save(model.state_dict(), save_path)

    return history


def train_final_layer(model, preprocessing, train_loader, valid_loader, loss_fn, device, lr=1e-4, n_epochs=5, gamma=0.9):
    # optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    history = {'train_loss': [], 'valid_loss': [],'train_acc':[],'valid_acc':[]}

    pbar = tqdm(range(n_epochs))
    for epoch in pbar:
        train_loss, train_acc=train_epoch(model, train_loader, loss_fn,optimizer, device)
        valid_loss, valid_acc=valid_epoch(model, valid_loader, loss_fn, device)
        
        scheduler.step()
        pbar.set_description(f"Training Loss: {train_loss:.3f} Valid Loss: {valid_loss:.3f} Training Acc: {train_acc:.2f} Valid Acc: {valid_acc:.2f}")

        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)
        history['train_acc'].append(train_acc)
        history['valid_acc'].append(valid_acc)

    return history


def plot_confusion_matrix(m):
    df_cm = pd.DataFrame(m, index = [i for i in classes],
                  columns = [i for i in classes])
    df_cm = df_cm.rename_axis("predicted")
    df_cm = df_cm.rename_axis("actual", axis="columns")

    plt.figure(figsize = (10,7))
    sns.heatmap(df_cm, annot=True)
    plt.show()