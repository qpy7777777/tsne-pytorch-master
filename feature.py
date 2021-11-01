import numpy as np
import librosa
import librosa.display
import torch
PATH = "test.wav"
def normalization(data):
    max_data = np.max(data)
    min_data = np.min(data)
    new_audio = (data-min_data)/(max_data-min_data)
    return new_audio

data,fs = librosa.load(PATH)

mfcc_data = librosa.feature.mfcc(data,fs,n_mfcc=40)
mfccs = np.mean(mfcc_data.T, axis=0)
data = normalization(mfccs)
print(data.shape)
X = torch.Tensor(data)
print(torch.sum(X))
