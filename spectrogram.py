import matplotlib.pyplot as plt   #画图用
import librosa.core as lc   #计算stft使用
import numpy as np   #使用了其中的一些工具函数
import librosa.display   #画声谱图用
from scipy.io import wavfile   #获取wav文件的采样率
import os
import tqdm

# path = "./WGAN1/enhanced/p232_001.wav"
def get_piputu(file_path,mode="clean"):
    if not os.path.exists("./PinPuTu/{}".format(mode)):
        os.makedirs("./PinPuTu/{}".format(mode))
    bname = os.path.splitext(os.path.basename(file_path))[0]
    fs, y_ = wavfile.read(file_path)        #读取文件的采样率
    fs = fs
    n_fft = 1024         #FFT长度
    y, sr = librosa.load(file_path, sr=fs)
    plt.figure()
    #获取宽带声谱图
    mag = np.abs(lc.stft(y, n_fft=n_fft, hop_length=10, win_length=40, window='hamming'))        #进行短时傅里叶变换，并获取幅度
    D = librosa.amplitude_to_db(mag, ref=np.max)    #幅度转换为db单位
    librosa.display.specshow(D, sr=fs, hop_length=10, x_axis='s', y_axis='linear')             #画声谱图
    plt.colorbar(format='%+2.0f dB')
    # plt.title('broadband spectrogram')
    plt.savefig("./PinPuTu/{}/{}_broader.png".format(mode,bname))
    # plt.show()
    plt.close()

    plt.figure()
    #获取窄带声谱图
    mag1 = np.abs(lc.stft(y, n_fft=n_fft, hop_length=100, win_length=400, window='hamming'))
    mag1_log = 20*np.log(mag1)
    D1 = librosa.amplitude_to_db(mag1, ref=np.max)
    librosa.display.specshow(D1, sr=fs, hop_length=100, x_axis='s', y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    # plt.title('narrowband spectrogram')
    plt.savefig("./PinPuTu/{}/{}_narrowband.png".format(mode,bname))
    # plt.show()
    plt.close()

modes = ["clean","noisy","enhanced"]
# mode = "clean"
for mode in modes:
    file_names = os.listdir("./WGAN1/{}".format(mode))
    for name in tqdm.tqdm(file_names):
        get_piputu("./WGAN1/{}/{}".format(mode,name),mode)