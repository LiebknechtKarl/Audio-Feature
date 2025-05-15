import librosa
import librosa.display
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt

import torchvision
import torch.nn.functional as F
from dataset_splitting import DatasetSegmentation
from time_frequency import  Audio_Transformation
from transformers import Wav2Vec2Model, Wav2Vec2Config


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


'''
2 Models Available:
   - AlexNet_Model : AlexNet model from pyTorch (CNN features layer + FC classifier layer)
'''

class AlexNet_Model (nn.Module):
    """
    Reference:
    https://pytorch.org/docs/stable/torchvision/models.html#id1

    AlexNet model from torchvision package. The model architecture is slightly
    different from the original model.
    See: AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.


    Parameters
    ----------
    num_classes : int
    in_ch   : int
        The number of input channel.
        Default AlexNet input channels is 3. Set this parameters for different
            numbers of input channels.
    pretrained  : bool
        To initialize the weight of AlexNet.
        Set to 'True' for AlexNet pre-trained weights.

    Input
    -----
    Input dimension (N,C,H,W)

    N   : batch size
    C   : channels
    H   : Height
    W   : Width

    Output
    ------
    logits (before Softmax)

    """


    def __init__(self,num_classes=4, in_ch=3, pretrained=True):
        super(AlexNet_Model , self).__init__()

        model = torchvision.models.alexnet(pretrained=pretrained)
        self.features = model.features
        self.avgpool  = model.avgpool
        self.classifier = model.classifier

        if in_ch != 3:
            self.features[0] = nn.Conv2d(in_ch, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
            init_layer(self.features[0])

        self.classifier[6] = nn.Linear(4096, num_classes)

        self._init_weights(pretrained=pretrained)
        
        print('\n<< SER AlexNet Finetuning model initialized >>\n')

    def forward(self, x):

        x = self.features(x)
        x = self.avgpool(x)
        x_ = torch.flatten(x, 1)
        out = self.classifier(x_)

        return x, out

    def _init_weights(self, pretrained=True):

        init_layer(self.classifier[6])

        if pretrained == False:
            init_layer(self.features[0])
            init_layer(self.features[3])
            init_layer(self.features[6])
            init_layer(self.features[8])
            init_layer(self.features[10])
            init_layer(self.classifier[1])
            init_layer(self.classifier[4])


# __all__ = ['BiLSTM_Model']
class BiLSTM_Model(nn.Module):
    # def __init__(self):
    def __init__(self, feature_channel = 40, feature_sequence_length= 300, hidden_size = 256, num_layers=2):

        super(BiLSTM_Model, self).__init__()
        
        # LSTM for MFCC        
        # self.lstm_mfcc = nn.LSTM(input_size=40, hidden_size=256, num_layers=2, batch_first=True, dropout=0.5,bidirectional = True) # bidirectional = True
        self.lstm_mfcc = nn.LSTM(input_size=feature_channel, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.5,bidirectional = True) # bidirectional = True

        self.post_mfcc_dropout = nn.Dropout(p=0.1)
        # self.post_mfcc_layer = nn.Linear(153600, 128) # 40 for attention and 8064 for conv, 32768 for cnn-lstm, 38400 for lstm
        self.post_mfcc_layer = nn.Linear(feature_sequence_length * hidden_size * num_layers, 128) # 40 for attention and 8064 for conv, 32768 for cnn-lstm, 38400 for lstm

    def forward(self ,  audio_mfcc):      
        
        # audio_spec: [batch, 3, 256, 384]
        # audio_mfcc: [batch, 300, 40]
        # audio_wav: [32, 48000]
        
        audio_mfcc = F.normalize(audio_mfcc, p=2, dim=2)
        
        # audio -- MFCC with BiLSTM
        audio_mfcc, _ = self.lstm_mfcc(audio_mfcc) # [batch, 300, 512]  
        
        #+ audio_mfcc = self.att(audio_mfcc)
        audio_mfcc_ = torch.flatten(audio_mfcc, 1) # [batch, 153600]  
        audio_mfcc_att_d = self.post_mfcc_dropout(audio_mfcc_) # [batch, 153600]  
        audio_mfcc_p = F.relu(self.post_mfcc_layer(audio_mfcc_att_d), inplace=False) # [batch, 128]  
        
        return audio_mfcc_p


class W2V2_Model(nn.Module):
    # def __init__(self):
    def __init__(self):

        super(W2V2_Model, self).__init__()
        
        # 定制模型配置
        self.config = Wav2Vec2Config(
            vocab_size=32112,
            num_mel_bins=64,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            )
        # 初始化模型
        self.model = Wav2Vec2Model(self.config)

    def forward(self ,  audio_sequence):   
        feature = self.model(audio_sequence)

        return feature





if __name__ == "__main__":

    # from your_module import AlexNet_Model  # 假设你把模型放在 your_module.py 中

    # 1. 创建模型实例
    #    - num_classes: 分类数，例如 4
    #    - in_ch: 输入通道数，例如特征提取时用的频谱图可能只有 1 通道
    model_alexnet = AlexNet_Model(num_classes=4, in_ch=3, pretrained=False)

    # 2. 切换到 eval 模式（如果只是做推理/示例）
    model_alexnet.eval()

    # 3. 构造一个随机输入张量
    #    - batch_size=2
    #    - in_ch=3
    #    - 高度和宽度都是 224，与 AlexNet 默认输入一致
    x = torch.randn(2, 3, 224, 224)

    # 4. 前向计算
    with torch.no_grad():
        features, logits = model_alexnet(x)

    # 5. 查看输出形状
    print("model_alexnet Features shape:", features.shape)  # e.g. torch.Size([2, 256, 6, 6]) 取决于中间层输出
    print("model_alexnet Logits shape:  ", logits.shape)   # torch.Size([2, 4])



    ############################ bilstm


    # 1. 创建模型实例
    #    - in_ch: 输入通道数，例如特征提取时用的频谱图可能只有 1 通道
    model_bilstm = BiLSTM_Model()

    # 2. 切换到 eval 模式（如果只是做推理/示例）
    model_bilstm.eval()

    # 3. 构造一个随机输入张量
    #    - batch_size=2
    #    - in_ch=3
    #    - 高度和宽度都是 224，与 AlexNet 默认输入一致
    # x = torch.randn(2, 3, 224, 224)
    x = torch.randn(8, 300, 40)


    # 4. 前向计算
    with torch.no_grad():
        features = model_bilstm(x)

    # 5. 查看输出形状
    print("model_bilstm Features shape:", features.shape)  # e.g. torch.Size([2, 256, 6, 6]) 取决于中间层输出




    ############################ wav2vec

    model_w2v = W2V2_Model()


    # 假设有一个音频输入（波形数据）
    # 这里使用随机数据作为示例
    audio_input = torch.randn(1, 100000)  # 假设音频长度为 100000 采样点

    # 模型前向传播
    with torch.no_grad():
        outputs = model_w2v(audio_input)

    # 获取特征
    last_hidden_states = outputs.last_hidden_state
    # print(last_hidden_states.shape)  # 输出特征的形状
    print("model_w2v Features shape:", last_hidden_states.shape)  

    # ########################### ------------------------------------下面的是 加载实际数据------------------------------------------------------

    # audio_dir='./CMLRdataset/audio_normal'
    # video_dir='./CMLRdataset/video'

    # DataSegModel = DatasetSegmentation(audio_dir = audio_dir , video_dir = video_dir)
    # train_loader,test_loader = DataSegModel.segmentation()

    # sr_wav = 16000
    # Audio_model = Audio_Transformation(sr = sr_wav)



    # for audio, video, name in train_loader:
    #     print("Audio shape:", audio.shape)    # (B, 1, T)
    #     print("Video shape:", video.shape)    # (B, T, C, H, W)
    #     print("Sample names:", name)
    #     data_mfcc = Audio_model.extract_mfcc(audio)

    #     print(data_mfcc)

    #     features, logits  = model(data_mfcc)
    #     # data_mfcc.shape
    #     # (1567, 13)
    #     # 
    # ############################ ----------------------------------------------------------------





