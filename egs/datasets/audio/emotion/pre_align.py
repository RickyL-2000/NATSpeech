import os

from data_gen.tts.base_preprocess import BasePreprocessor
import glob
import re
from utils.hparams import hparams, set_hparams

class EmoPreAlign(BasePreprocessor):

    def meta_data(self):
        if hparams['preprocess_args']['txt_processor'] == 'zh':
            spks = ['0002', '0001', '0003', '0004', '0005', '0006', '0007', '0008', '0009', '0010']
        elif hparams['preprocess_args']['txt_processor'] == 'en':
            spks = ['0012', '0011', '0013', '0014', '0015', '0016', '0017', '0018', '0019', '0020']
        emotion_name = {"Neutral": "Neutral", "Angry": "Angry", "Happy": "Happy", "Sad": "Sad", "Surprise": "Surprise",
                        "中立": "Neutral", "生气": "Angry", "快乐": "Happy", "伤心": "Sad", "惊喜": "Surprise"}
        pattern = re.compile('[\t\n ]+')
        for spk in spks:
            for line in open(f"{self.raw_data_dir}/{spk}/{spk}.txt", 'r'):  # 打开文件
                line = re.sub(pattern, ' ', line)
                if line == ' ': continue
                split_ = line.split(' ')
                txt = ' '.join(split_[1: -2])
                item_name = split_[0]
                emotion = emotion_name[split_[-2]]
                wav_fn = f'{self.raw_data_dir}/{spk}/{emotion}/{item_name}.wav'
                # yield item_name, wav_fn, txt, spk, emotion
                yield {'item_name': item_name, 'wav_fn': wav_fn, 'txt': txt, 'spk_name': spk, 'others': None or emotion}


if __name__ == "__main__":
    set_hparams()
    EmoPreAlign().process()
    #
    # set_hparams('modules/CL-GenerSpeech/config/cl-generspeech.yaml')
    # preprocessor = EmoPreAlign()
    # preprocessor.process()
