from torch.utils.data import Dataset
import pandas as pd
import json
import os
import numpy as np
import random

NUM_CLASSES = 2
classes = ('No Porg-Rock', 'Prog-Rock')


def get_dataset(path):
    df = pd.read_csv(path)
    return df.values.tolist()


class SongDataset(Dataset):
    def __init__(self, annotations, audio_dir='data/process/', transform=None, target_transform=None):
        self.audio_labels = annotations
        self.audio_dir = audio_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.audio_labels)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_labels[idx][0])
        audio = np.load(audio_path)
        label = self.audio_labels[idx][1]
        if self.transform:
            audio = self.transform(audio)
        if self.target_transform:
            label = self.target_transform(label)
        start = random.randint(0,audio.shape[1]-(3000+1))
        return audio[:,start:3000+start], label