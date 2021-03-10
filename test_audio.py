import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch
import torch.nn as nn
from scipy.io import wavfile
from torch.autograd import Variable
from tqdm import tqdm,trange

from data_preprocess import slice_signal, window_size, sample_rate
from model import Generator
from utils import emphasis
from utils1 import eval_composite

def output_enhanced_speech(FILE_NAME,EPOCH_NAME,save_root="WGAN1"):
    clean_file = os.path.join("../SEGAN/data/clean_testset_wav", FILE_NAME)
    noisy_file = os.path.join("../SEGAN/data/noisy_testset_wav", FILE_NAME)

    generator = Generator()
    generator.load_state_dict(torch.load('../save1/epochs/' + EPOCH_NAME, map_location='cpu'))
    if torch.cuda.is_available():
        generator.cuda()

    noisy_slices = slice_signal(noisy_file, window_size, 1, sample_rate)
    clean_slices = slice_signal(clean_file, window_size, 1, sample_rate)

    noisy_speechs = []
    clean_speechs = []
    enhanced_speech = []
    for i in trange(len(noisy_slices), desc='Generate enhanced audio'):
        noisy_slice = noisy_slices[i]
        clean_slice = clean_slices[i]
        z = nn.init.normal_(torch.Tensor(1, 1024, 8))
        noisy_slice = torch.from_numpy(emphasis(noisy_slice[np.newaxis, np.newaxis, :])).type(torch.FloatTensor)
        clean_slice = torch.from_numpy(emphasis(clean_slice[np.newaxis, np.newaxis, :])).type(torch.FloatTensor)
        if torch.cuda.is_available():
            noisy_slice, z = noisy_slice.cuda(), z.cuda()
        noisy_slice, z = Variable(noisy_slice), Variable(z)

        generated_speech = generator(noisy_slice, z).data.cpu().numpy()
        generated_speech = emphasis(generated_speech, emph_coeff=0.95, pre=False)
        generated_speech = generated_speech.reshape(-1)
        enhanced_speech.append(generated_speech)

        noisy_slice = noisy_slice.data.cpu().numpy()
        noisy_speech = emphasis(noisy_slice, emph_coeff=0.95, pre=False)
        noisy_speech = noisy_speech.reshape(-1)
        noisy_speechs.append(noisy_speech)

        clean_slice = clean_slice.data.cpu().numpy()
        clean_speech = emphasis(clean_slice, emph_coeff=0.95, pre=False)
        clean_speech = clean_speech.reshape(-1)
        clean_speechs.append(clean_speech)
    if not os.path.exists(os.path.join(save_root,'enhanced')):
        os.makedirs(os.path.join(save_root,'enhanced'))
    enhanced_speech = np.array(enhanced_speech).reshape(1, -1)
    file_name = os.path.join(os.path.join(save_root,'enhanced'),
                             '{}.wav'.format(os.path.basename(FILE_NAME).split('.')[0]))
    wavfile.write(file_name, sample_rate, enhanced_speech.T)

    if not os.path.exists(os.path.join(save_root,'noisy')):
        os.makedirs(os.path.join(save_root,'noisy'))
    noisy_speechs = np.array(noisy_speechs).reshape(1, -1)
    file_name = os.path.join(os.path.join(save_root,'noisy'),
                             '{}.wav'.format(os.path.basename(FILE_NAME).split('.')[0]))
    wavfile.write(file_name, sample_rate, noisy_speechs.T)

    if not os.path.exists(os.path.join(save_root,'clean')):
        os.makedirs(os.path.join(save_root,'clean'))
    clean_speechs = np.array(clean_speechs).reshape(1, -1)
    file_name = os.path.join(os.path.join(save_root,'clean'),
                             '{}.wav'.format(os.path.basename(FILE_NAME).split('.')[0]))
    wavfile.write(file_name, sample_rate, clean_speechs.T)

if __name__ == '__main__':
    EPOCH_NAME = 'generator-84.pkl'
    file_names = os.listdir("../SEGAN/data/noisy_testset_wav")
    for name in file_names:
        output_enhanced_speech(name,EPOCH_NAME)
