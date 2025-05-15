


from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import librosa

def extract_features_wav2vec2(file_path):
    # 加载模型和特征提取器
    # model_name = "facebook/wav2vec2-base-960h"
    model_name = "/home/fu/Desktop/ubuntu_data/nlp/raw_data/wav2vec2-base-960h"

    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)

    # 加载音频文件
    speech_array, sampling_rate = librosa.load(file_path, sr=16000)  # 确保采样率为16000

    # 预处理音频
    inputs = processor(speech_array, sampling_rate=sampling_rate, return_tensors="pt", padding=True)

    # 提取特征
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # 获取隐藏状态（特征）
    hidden_states = outputs.hidden_states
    # 返回最后一层的特征
    return hidden_states[-1]

if __name__ == "__main__":
    # file_path = '/home/fu/Desktop/ubuntu_data/nlp/demo_51_qiangnao_voice/dda/material/wav/example.wav'
    wav_path = 'alert.wav'

    features = extract_features_wav2vec2(wav_path)
    print("提取的特征形状:", features.shape)

# You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
# 提取的特征形状: torch.Size([1, 1381, 768])