import librosa
import librosa.display
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt

# 提取梅尔谱图
def extract_mel_spectrogram(wav_path, sr=16000, n_mels=128):        # 128 是 梅尔滤波器数量
    # 加载音频文件
    y, sr = librosa.load(wav_path, sr=sr)
    
    # 计算梅尔谱图
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    return mel_spectrogram

# 使用本地权重文件加载 AlexNet 并提取特征
def extract_features_with_alexnet(mel_spectrogram, weights_path):
    # 将梅尔谱图转换为张量
    mel_tensor = torch.tensor(mel_spectrogram).unsqueeze(0).unsqueeze(0).float()  # (1, 1, n_mels, time_steps)
    
    # 加载预训练的 AlexNet 模型
    model = models.alexnet(num_classes=1000)
    
    # 加载本地权重文件
    model.load_state_dict(torch.load(weights_path))
    
    # 修改 AlexNet 的输入层以适应梅尔谱图的形状
    # 原始 AlexNet 输入层为 3 通道（RGB 图像），这里将其改为 1 通道（梅尔谱图）
    model.features[0] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)
    
    # 移除分类器层，只保留卷积层
    model.classifier = nn.Identity()
    
    # 提取特征
    with torch.no_grad():
        features = model(mel_tensor)
    
    return features

# 主程序
if __name__ == "__main__":
    # 替换为你的 WAV 文件路径
    # wav_path = '/home/fu/Desktop/ubuntu_data/nlp/demo_51_qiangnao_voice/dda/material/wav/example.wav'
    wav_path = 'alert.wav'

    
    # 替换为你的 AlexNet 权重文件路径
    weights_path = '/home/fu/Desktop/ubuntu_data/nlp/raw_data/alexnet-owt-7be5be79.pth'
    
    # 提取梅尔谱图
    mel_spectrogram = extract_mel_spectrogram(wav_path)
    print(f"梅尔谱图形状: {mel_spectrogram.shape}")
    
    # 可视化梅尔谱图
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spectrogram, sr=16000, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-Spectrogram')
    plt.tight_layout()
    plt.show()
    
    # 使用本地权重文件加载 AlexNet 并提取特征
    features = extract_features_with_alexnet(mel_spectrogram, weights_path)
    print(f"提取的特征形状: {features.shape}")

    # 梅尔谱图形状: (128, 864)
    # 提取的特征形状: torch.Size([1, 9216])    