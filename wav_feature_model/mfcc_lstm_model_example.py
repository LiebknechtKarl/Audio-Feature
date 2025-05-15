import librosa
import numpy as np
import torch
import torch.nn as nn

# 提取梅尔谱图
def extract_mel_spectrogram(wav_path, sr=16000, n_mels=128):
    # 加载音频文件
    y, sr = librosa.load(wav_path, sr=sr)
    
    # 计算梅尔谱图
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    return mel_spectrogram

# 定义 LSTM 模型
class LSTMFeatureExtractor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMFeatureExtractor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    
    def forward(self, x):
        # x.shape = (batch_size, sequence_length, input_size)
        out, _ = self.lstm(x)
        # out.shape = (batch_size, sequence_length, hidden_size)
        return out

# 主程序
if __name__ == "__main__":
    # 替换为你的 WAV 文件路径
    # wav_path = '/home/fu/Desktop/ubuntu_data/nlp/demo_51_qiangnao_voice/dda/material/wav/example.wav'
    wav_path = 'alert.wav'

    
    # 提取梅尔谱图
    mel_spectrogram = extract_mel_spectrogram(wav_path)
    print(f"梅尔谱图形状: {mel_spectrogram.shape}")
    
    # 将梅尔谱图转换为张量并调整形状以适应 LSTM 输入
    # LSTM 输入形状为 (batch_size, sequence_length, input_size)
    # 这里我们假设每个时间步的特征维度是 n_mels，序列长度是时间步数
    mel_tensor = torch.tensor(mel_spectrogram.T).unsqueeze(0).float()  # (1, time_steps, n_mels)
    
    # 定义 LSTM 模型参数
    input_size = mel_spectrogram.shape[0]  # n_mels
    hidden_size = 128
    num_layers = 2
    
    # 初始化 LSTM 模型
    model = LSTMFeatureExtractor(input_size, hidden_size, num_layers)
    
    # 提取特征
    with torch.no_grad():
        features = model(mel_tensor)
    
    print(f"提取的特征形状: {features.shape}")

    # 梅尔谱图形状: (128, 864)
    # 提取的特征形状: torch.Size([1, 864, 128])    