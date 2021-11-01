import librosa
import librosa.display
import matplotlib.pyplot as plt
import os


def show_melspectrogram(audio,fs,title):
    melspec = librosa.feature.melspectrogram(y=audio, sr=fs, n_fft=1024, hop_length=512, n_mels=128)
    melspec = librosa.power_to_db(melspec)
    # Log-Mel Spectrogram特征是二维数组的形式，(78, 548)
    # 78表示Mel频率的维度（频域），548（时域），Log-Mel Spectrogram特征是音频信号的时频表示特征
    plt.figure(figsize=(4, 4))
    # 可视化原始音频
    librosa.display.specshow(melspec, sr=fs, x_axis='time', y_axis='hz')
    plt.title(title)
    # plt.tight_layout()
    plt.show()
if __name__=="__main__":
    # 读取音频
    path = "data"
    # 数据增强音频文件
    file = os.listdir(path)
    for i in range(len(file)):
        title = ["Adventure_of_the seas_1", "blue whale", "Humpback_Whales", "Killer whales",
                 "Mar_de_Cangas","Maximum_rain","Sea weaves",
                 "Sperm_Whales","White_Whale","Zodiac"]
        file_name = os.path.join(path, file[i])
        audio, fs = librosa.load(file_name, sr=None)
        print(file_name)
        # 可视化mel特征 一阶mel，二阶mel特征谱图
        show_melspectrogram(audio,fs,title[i])