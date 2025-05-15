

import librosa
import librosa.display
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import python_speech_features as psf
import parselmouth

class Audio_Transformation():
    def __init__(self,sr=16000 ,
                 
                 # ------spectrogram
                 n_fft = 512 ,
                 hop_length = 160 ,
                 win_length = 400 ,
                 
                 # ------fbank
                 winlen=0.025 ,
                 winstep=0.01 ,
                 nfilt=40 ,

                 # ------mfcc
                 numcep=13, 

                 # ------PLP (via parselmouth)  ,
                 time_step=0.01                 
                 ):

        self.sr = sr

        # ------spectrogram
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        # ------fbank
        self.winlen=winlen
        self.winstep=winstep
        self.nfilt=nfilt

        # ------mfcc
        self.numcep=numcep

        # ------PLP (via parselmouth) 
        self.time_step=time_step

    def extract_spectrogram(self, audio_sequence):
        stft = librosa.stft(audio_sequence, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)
        spectrogram = np.abs(stft)
        return spectrogram
    
    def extract_fbank(self, audio_sequence):
        fbank_feat = psf.logfbank(audio_sequence, samplerate=self.sr, winlen=self.winlen, winstep=self.winstep, nfilt=self.nfilt)
        return fbank_feat
    
    def extract_mfcc(self, audio_sequence):
        # mfcc_feat = psf.logfbank(audio_sequence, samplerate=self.sr, winlen=self.winlen, winstep=self.winstep, nfilt=self.nfilt)
        mfcc_feat = psf.mfcc(audio_sequence, samplerate=self.sr, numcep=13, winlen=0.025, winstep=0.01, nfilt=40)

        return mfcc_feat
    
    def extract_plp_parselmouth(self, audio_sequence ):
        snd = parselmouth.Sound(audio_sequence, sampling_frequency=self.sr)
        formant = snd.to_formant_burg(self.time_step)
        num_frames = formant.get_number_of_frames()
        plp_feats = []
        for i in range(num_frames):
            t = formant.get_time_from_frame_number(i + 1)
            values = []
            for formant_index in range(1, 6):  # 取前5个共振峰
                try:
                    f = formant.get_value_at_time(formant_index, t)
                    values.append(f if f is not None else 0)
                except:
                    values.append(0)
            plp_feats.append(values)
        return np.array(plp_feats)


    



# 主程序
if __name__ == "__main__":
    # 替换为你的 WAV 文件路径
    # wav_path = '/home/fu/Desktop/ubuntu_data/nlp/demo_51_qiangnao_voice/dda/material/wav/example.wav'
    # wav_path = '/home/fu/Desktop/ubuntu_data/nlp/demo_55_audio_future/3_feature/alert.wav'
    wav_path = './CMLRdataset/audio_normal/section_1_000_80_002_91.wav'
    y, sr_wav = librosa.load(wav_path, sr=16000, mono=True)

    Audio_model = Audio_Transformation(sr = sr_wav)

    data_spectrogram = Audio_model.extract_spectrogram(y)
    data_fbank = Audio_model.extract_fbank(y)
    data_mfcc = Audio_model.extract_mfcc(y)
    data_plp_parselmouth = Audio_model.extract_plp_parselmouth(y)

    print('data_spectrogram 形状', data_spectrogram.shape)
    print('data_fbank 形状', data_fbank.shape)
    print('data_mfcc 形状', data_mfcc.shape)
    print('data_plp_parselmouth 形状', data_plp_parselmouth.shape)




    
    # # 可视化梅尔谱图
    # plt.figure(figsize=(10, 4))
    # librosa.display.specshow(data_spectrogram, sr=16000, x_axis='time', y_axis='mel')
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Mel-Spectrogram')
    # plt.tight_layout()
    # plt.show()


    # 设置画布
    plt.figure(figsize=(16, 12))

    # === 1. Waveform ===
    plt.subplot(5, 1, 1)
    librosa.display.waveshow(y, sr=sr_wav)
    plt.title("Waveform")

    # === 2. Spectrogram ===
    plt.subplot(5, 1, 2)
    librosa.display.specshow(librosa.amplitude_to_db(data_spectrogram, ref=np.max),
                            sr=sr_wav, hop_length=Audio_model.hop_length,
                            x_axis='time', y_axis='linear')
    plt.title("Spectrogram (dB)")
    plt.colorbar(format="%+2.0f dB")

    # === 3. Fbank ===
    plt.subplot(5, 1, 3)
    plt.imshow(data_fbank.T, aspect='auto', origin='lower')
    plt.title("Filter Bank Features (Fbank)")
    plt.xlabel("Frame Index")
    plt.ylabel("Filter Index")
    plt.colorbar()

    # === 4. MFCC ===
    plt.subplot(5, 1, 4)
    plt.imshow(data_mfcc.T, aspect='auto', origin='lower')
    plt.title("MFCC Features")
    plt.xlabel("Frame Index")
    plt.ylabel("MFCC Coefficients")
    plt.colorbar()

    # === 5. PLP ===
    plt.subplot(5, 1, 5)
    plt.imshow(data_plp_parselmouth.T, aspect='auto', origin='lower')
    plt.title("PLP (Formant-based via Parselmouth)")
    plt.xlabel("Frame Index")
    plt.ylabel("Formant Index")
    plt.colorbar()

    plt.tight_layout()
    plt.show()






