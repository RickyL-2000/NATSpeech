import glob
import os
import json
from collections import OrderedDict
import copy
import subprocess

from tqdm import tqdm
import miditoolkit
import textgrid
import librosa

from data_gen.tts.base_preprocess import BasePreprocessor
from utils.audio.io import save_wav

ALL_PHONE = ['a', 'ai', 'an', 'ang', 'ao', 'b', 'c', 'ch', 'd', 'e', 'ei', 'en', 'eng', 'er', 'f', 'g', 'h', 'i', 'ia', 'ian', 'iang', 'iao', 'ie', 'in', 'ing', 'iong', 'iou', 'j', 'k', 'l', 'm', 'n', 'o', 'ong', 'ou', 'p', 'q', 'r', 's', 'sh', 't', 'u', 'ua', 'uai', 'uan', 'uang', 'uei', 'uen', 'uo', 'v', 'van', 've', 'vn', 'x', 'z', 'zh']
ALL_SHENGMU = ['b', 'c', 'ch', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 'sh', 't', 'x', 'z', 'zh']
ALL_YUNMU = ['a', 'ai', 'an', 'ang', 'ao',  'e', 'ei', 'en', 'eng', 'er',  'i', 'ia', 'ian', 'iang', 'iao',
             'ie', 'in', 'ing', 'iong', 'iou', 'o', 'ong', 'ou', 'u', 'ua', 'uai', 'uan', 'uang', 'uei',
             'uen', 'uo', 'v', 'van', 've', 'vn']

class M4Preprocess(BasePreprocessor):
    def meta_data(self):
        wav_fns = sorted(
            glob.glob(f'{self.processed_dir}/wav/*/*.wav')
        )
        for wav_fn in wav_fns:
            item_name = os.path.basename(wav_fn)[:-4]
            singer, song_name, sen_id = item_name.split("#")
            item_basename = singer + '#' + song_name
            tg_fn = f'{self.processed_dir}/textgrid/{item_basename}/{item_name}.TextGrid'
            midi_fn = f'{self.processed_dir}/midi/{item_basename}/{item_name}.mid'
            yield item_name, wav_fn, singer, tg_fn, midi_fn

    def process(self):
        gen = self.meta_data()
        phs_meta = {}
        #新建文件
        meta_out = []
        #统计时长
        sen_dur_list = []
        for item in tqdm(gen):
            (item_name, wav_fn, singer, tg_fn, midi_fn) = item

            txt_list = []
            ph_list = []
            # 这里暂时做简单处理，一个word对应一个note
            # word级别的note对齐
            notes_pitch_list = []
            # 这里每个ph note的dur就是word的dur
            notes_dur_list = []
            ph_dur_list = []
            # word boundary
            wbd_list = []
            # 读取TG
            with open(tg_fn, "r") as f:
                tg = f.readlines()
            tg = TextGrid(tg)
            tg = json.loads(tg.toJson())

            tg_align_word = [x for x in tg['tiers'][0]['items']]
            tg_align_ph = [x for x in tg['tiers'][1]['items']]
            tg_align_slur = [int(x['text']) for x in tg['tiers'][2]['items']]
            slur_list = tg_align_slur
            sen_dur_list.append(float(tg_align_word[-1]['xmax']))
            if float(tg_align_word[-1]['xmax']) > 20:
                print(item_name)
            # 读取midi,处理后一字一note(非静音区)
            mf = miditoolkit.MidiFile(midi_fn)
            instru = mf.instruments[0]
            note_list = instru.notes

            # 保证note数和非sil的word数相等
            word_list = [xx['text'] for xx in tg_align_word]
            assert len(note_list) == word_count(word_list)

            word_id = 0
            ph_id = 0
            last_note_end = 0.0
            for note_id, note in enumerate(note_list):
                note_start = round(note.start / 440, 2)
                note_end = round(note.end / 440, 2)
                # midi中的休止符为两个音符间的空隙
                if abs(last_note_end - note_start) >= 1e-5:
                    note_dur = round(note_start - last_note_end, 2)
                    assert note_dur > 0
                    # 合并小空隙
                    if note_dur < 0.025:
                        note_start = last_note_end
                    else:
                        notes_dur_list.append(note_dur)
                        notes_pitch_list.append(0)

                note_pitch = note.pitch
                note_dur = round(note_end - note_start, 2)
                notes_dur_list.append(note_dur)
                notes_pitch_list.append(note_pitch)
                last_note_end = note_end

                # 最后一个note,用休止符补齐
                if note_id == len(note_list) - 1:
                    if note_end <= float(tg_align_word[-1]['xmax']) - 0.02:
                        note_dur = round(float(tg_align_word[-1]['xmax']) - last_note_end, 2)
                        notes_dur_list.append(note_dur)
                        notes_pitch_list.append(0)
                        note_end = round(float(tg_align_word[-1]['xmax']), 2)
                    # 空隙小用最后一个音符补充
                    elif note_end <= float(tg_align_word[-1]['xmax']):
                        notes_dur_list[-1] = round(notes_dur_list[-1] + float(tg_align_word[-1]['xmax']) - note_end, 2)
                        note_end = round(float(tg_align_word[-1]['xmax']), 2)
                    # 保证phoneme和note结尾相同
                    assert abs(float(tg_align_word[-1]['xmax']) - note_end) <= 1e-5
            while word_id < len(tg_align_word):
                word = tg_align_word[word_id]
                word['xmin'] = float(word['xmin'])
                word['xmax'] = float(word['xmax'])
                if is_word(word['text']):
                    txt_list.append(word['text'])
                else:
                    pass

                while ph_id < len(tg_align_ph):
                    # ph部分
                    ph = tg_align_ph[ph_id]
                    ph_list.append(ph['text'])
                    ph['xmax'] = float(ph['xmax'])
                    ph['xmin'] = float(ph['xmin'])
                    ph_dur = round(ph['xmax'] - ph['xmin'], 2)
                    ph_dur_list.append(ph_dur)
                    wbd_list.append(1 if ph['text'] in ALL_YUNMU+['SP', 'AP'] else 0)

                    # 判断相等，到达word边界
                    if abs(ph['xmax'] - word['xmax']) <= (1e-5):
                        ph_id += 1
                        # 最后一个为字边界
                        wbd_list[-1] = 1
                        break
                    ph_id += 1
                word_id += 1


            # SL也是word，但不加入txt中
            txt = "".join(t for t in txt_list if t != 'SL')


            for ii in ph_list:
                if ii in phs_meta:
                    phs_meta[ii] += 1
                else:
                    phs_meta[ii] = 1
            # 'iu': 'iou'
            # {'iu': 25, 'vou': 3, 'vun': 2, 'ui': 55, 'un': 19, 'w': 12, 'uam': 1, 'hai': 3, 'y': 12, 'ji': 1, 'ue': 2, 'de': 1, 'ven': 1, 'iue': 1, 'io': 21, 'hang': 1, 'qi': 1, 'iv': 1, 'vuan': 1, 'h en': 1}
            #if 'iuan' in ph_list or 'iu' in ph_list:
            #    print(txt, ph_list)
            #    print(item_name)
            #if '翁' in txt or '瓮' in txt or '嗡' in txt:
            #    print(item_name)
            #    print(txt, ph_list)
            res = {'item_name': item_name, 'txt': txt, 'phs': ph_list, 'notes_pitch': notes_pitch_list,
                   'notes_dur': notes_dur_list, 'ph_dur': ph_dur_list, 'wav_fn': wav_fn, 'wbd': wbd_list,
                   'is_slur': slur_list, 'singer': singer}

            meta_out.append(res)

        print(phs_meta)
        ph_l = set()
        for ph in phs_meta.keys():
            ph_l.add(ph)
        ph_l = list(ph_l)
        ph_l.sort()
        print(ph_l)
        plot_stats(sen_dur_list)
        meta_path = f'{self.processed_dir}/meta.json'
        json.dump(meta_out, open(meta_path, 'w'), ensure_ascii=False, indent=4)

class TextGrid(object):
    def __init__(self, text):
        text = remove_empty_lines(text)
        self.text = text
        self.line_count = 0
        self._get_type()
        self._get_time_intval()
        self._get_size()
        self.tier_list = []
        self._get_item_list()

    def _extract_pattern(self, pattern, inc):
        """
        Parameters
        ----------
        pattern : regex to extract pattern
        inc : increment of line count after extraction
        Returns
        -------
        group : extracted info
        """
        try:
            group = re.match(pattern, self.text[self.line_count]).group(1)
            self.line_count += inc
        except AttributeError:
            raise ValueError("File format error at line %d:%s" % (self.line_count, self.text[self.line_count]))
        return group

    def _get_type(self):
        self.file_type = self._extract_pattern(r"File type = \"(.*)\"", 2)

    def _get_time_intval(self):
        self.xmin = self._extract_pattern(r"xmin = (.*)", 1)
        self.xmax = self._extract_pattern(r"xmax = (.*)", 2)

    def _get_size(self):
        self.size = int(self._extract_pattern(r"size = (.*)", 2))

    def _get_item_list(self):
        """Only supports IntervalTier currently"""
        for itemIdx in range(1, self.size + 1):
            tier = OrderedDict()
            item_list = []
            tier_idx = self._extract_pattern(r"item \[(.*)\]:", 1)
            tier_class = self._extract_pattern(r"class = \"(.*)\"", 1)
            if tier_class != "IntervalTier":
                raise NotImplementedError("Only IntervalTier class is supported currently")
            tier_name = self._extract_pattern(r"name = \"(.*)\"", 1)
            tier_xmin = self._extract_pattern(r"xmin = (.*)", 1)
            tier_xmax = self._extract_pattern(r"xmax = (.*)", 1)
            tier_size = self._extract_pattern(r"intervals: size = (.*)", 1)
            for i in range(int(tier_size)):
                item = OrderedDict()
                item["idx"] = self._extract_pattern(r"intervals \[(.*)\]", 1)
                item["xmin"] = self._extract_pattern(r"xmin = (.*)", 1)
                item["xmax"] = self._extract_pattern(r"xmax = (.*)", 1)
                item["text"] = self._extract_pattern(r"text = \"(.*)\"", 1)
                item_list.append(item)
            tier["idx"] = tier_idx
            tier["class"] = tier_class
            tier["name"] = tier_name
            tier["xmin"] = tier_xmin
            tier["xmax"] = tier_xmax
            tier["size"] = tier_size
            tier["items"] = item_list
            self.tier_list.append(tier)

    def toJson(self):
        _json = OrderedDict()
        _json["file_type"] = self.file_type
        _json["xmin"] = self.xmin
        _json["xmax"] = self.xmax
        _json["size"] = self.size
        _json["tiers"] = self.tier_list
        return json.dumps(_json, ensure_ascii=False, indent=2)

def remove_empty_lines(text):
    """remove empty lines"""
    assert (len(text) > 0)
    assert (isinstance(text, list))
    text = [t.strip() for t in text]
    if "" in text:
        text.remove("")
    return text

def is_word(word):
    # 'SL'算一个word
    if word != 'SP' and word != 'AP':
        return True
    else:
        return False


def word_count(word_list):
    count = 0
    for word in word_list:
        if word != 'SP' and word != 'AP':
            count += 1
    return count


def plot_stats(dur_list):
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    # 设置分段
    bins = np.arange(0, 30, 0.5)
    # 按分段离散化数据
    segments = pd.cut(dur_list, bins, right=False)
    print(segments)
    # 统计各分段人数
    counts = pd.value_counts(segments, sort=False)
    print(counts.index.astype(str))
    print(counts)
    # 绘制柱状图
    b = plt.bar(counts.index.astype(str), counts)
    # 添加数据标签
    #plt.bar_label(b, counts)
    plt.show()


class SegLong:
    def __init__(self):
        # midi  textgrid  wav
        self.raw_dir_long = 'data/raw/lasinger-long'
        self.processed_dir_short = 'data/processed/lasinger-short3'

    def meta_data(self):
        wav_fns = sorted(
            glob.glob(f'{self.raw_dir_long}/wav/*.wav')
        )
        for wav_fn in wav_fns:
            item_name = os.path.basename(wav_fn)[:-4]
            singer, song_name = item_name.split("#")
            tg_fn = f'{self.raw_dir_long}/textgrid/{item_name}.TextGrid'
            midi_fn = f'{self.raw_dir_long}/midi/{item_name}.mid'

            yield item_name, wav_fn, singer, tg_fn, midi_fn

    # 用于process中tg的保存(分割)
    def sav_tg(self, item_name, sen_id, tg_align_word_seg, tg_align_ph_seg, text2midi_list):
        text2midi_list = copy.deepcopy(text2midi_list)
        dur_bias = round(float(tg_align_word_seg[0]['xmin']), 2)
        tg_align_word_seg = copy.deepcopy(tg_align_word_seg)
        tg_align_ph_seg = copy.deepcopy(tg_align_ph_seg)
        tg = textgrid.TextGrid(minTime=0., maxTime=0.)
        tier_word = textgrid.IntervalTier(name="word", minTime=0., maxTime=0.)  # 添加一层,命名为word层
        tier_phone = textgrid.IntervalTier(name="phone", minTime=0., maxTime=0.)  # 添加一层,命名为phone音素层
        tier_slur = textgrid.IntervalTier(name="slur", minTime=0., maxTime=0.)  # 添加一层,命名为slur层
        # word和ph对齐
        ph_id = 0
        word2ph_align = []
        for word_id, word in enumerate(tg_align_word_seg):
            ph_table = []
            while ph_id < len(tg_align_ph_seg):
                ph = tg_align_ph_seg[ph_id]
                ph_table.append(ph)
                ph['xmax'] = float(ph['xmax'])
                word['xmax'] = float(word['xmax'])
                # 判断相等，到达word边界
                if abs(ph['xmax'] - word['xmax']) <= (1e-5):
                    ph_id += 1
                    break
                ph_id += 1
            word2ph_align.append(ph_table)
        #print(word2ph_align)

        for idx, word in enumerate(tg_align_word_seg):
            word['xmax'] = float(word['xmax'])
            word['xmin'] = float(word['xmin'])
            # 保留两位小数
            # 这里注意如果maxTime和最后一个的xmax不一致会自动补空白
            min_T = round(word['xmin'] - dur_bias, 2)
            max_T = round(word['xmax'] - dur_bias, 2)

            # 这里注意整句标注的text可能带有多余的空格，坑死了
            word['text'] = word['text'].strip()
            phs = word2ph_align[idx]

            # 静音or一个note的情况，不需要切滑音
            if text2midi_list[idx]['word'] == 'SP' or text2midi_list[idx]['word'] == 'AP' or len(text2midi_list[idx]['notes']) <= 1:
                interval = textgrid.Interval(minTime=min_T, maxTime=max_T, mark=word['text'])
                tier_word.addInterval(interval)
                min_T_ph = min_T
                for ph in phs:
                    ph['text'] = ph['text'].strip()
                    max_T_ph = round(float(ph['xmax']) - dur_bias, 2)
                    interval = textgrid.Interval(minTime=min_T_ph, maxTime=max_T_ph, mark=ph['text'])
                    tier_phone.addInterval(interval)
                    interval = textgrid.Interval(minTime=min_T_ph, maxTime=max_T_ph, mark='0')
                    tier_slur.addInterval(interval)
                    min_T_ph = max_T_ph
            # 多note情况
            # maxTime按照note.end来，最后一个字要按照字的边界
            else:
                note_min_T = min_T
                notes = text2midi_list[idx]['notes']
                min_T_ph = min_T
                # 有两个ph
                if len(phs) > 1:
                    ph = phs[0]
                    max_T_ph = round(float(ph['xmax']) - dur_bias, 2)
                    #if round(notes[0].end / 440, 2) <= round(float(phs[0]['xmax']), 2):
                    #    print('note error', notes[0].end / 440, phs[0]['xmax'])
                        #print(item_name, )
                        #return
                        #exit()
                    interval = textgrid.Interval(minTime=min_T_ph, maxTime=max_T_ph, mark=ph['text'])
                    tier_phone.addInterval(interval)
                    interval = textgrid.Interval(minTime=min_T_ph, maxTime=max_T_ph, mark='0')
                    tier_slur.addInterval(interval)
                    min_T_ph = max_T_ph
                    ph = phs[1]
                # 只有一个ph
                else:
                    ph = phs[0]

                for note_idx, note in enumerate(notes):
                    # 最后一个音符，按照字边界分割
                    note_end = round(note.end / 440 - dur_bias, 2)
                    note_max_T = min(note_end, max_T)
                    if note_idx == 0:
                        # 有声母并且第一个音符<=声母的时间，第一个音符完全分配给声母
                        if len(phs) > 1 and round(notes[0].end / 440, 2) <= round(float(phs[0]['xmax']) + 0.02, 2):
                            note_max_T = round(float(phs[0]['xmax']) - dur_bias, 2)
                            note_min_T = note_max_T
                            continue
                            #print('note error', notes[0].end / 440, phs[0]['xmax'])
                        else:
                            interval = textgrid.Interval(minTime=note_min_T, maxTime=note_max_T, mark=word['text'])
                            interval_ph = textgrid.Interval(minTime=min_T_ph, maxTime=note_max_T, mark=ph['text'])
                            interval_slur = textgrid.Interval(minTime=min_T_ph, maxTime=note_max_T, mark='0')
                    elif note_idx == len(notes) - 1:
                        interval = textgrid.Interval(minTime=note_min_T, maxTime=max_T, mark='SL')
                        interval_ph = textgrid.Interval(minTime=note_min_T, maxTime=max_T, mark=ph['text'])
                        interval_slur = textgrid.Interval(minTime=note_min_T, maxTime=max_T, mark='1')
                    else:
                        interval = textgrid.Interval(minTime=note_min_T, maxTime=note_max_T, mark='SL')
                        interval_ph = textgrid.Interval(minTime=note_min_T, maxTime=note_max_T, mark=ph['text'])
                        interval_slur = textgrid.Interval(minTime=note_min_T, maxTime=note_max_T, mark='1')
                    note_min_T = note_max_T
                    tier_word.addInterval(interval)
                    tier_phone.addInterval(interval_ph)
                    tier_slur.addInterval(interval_slur)

        sum_sec = round(float(tg_align_word_seg[-1]['xmax']) - dur_bias, 2)
        # 添加到tg对象中
        tg.tiers.append(tier_word)
        tg.tiers.append(tier_phone)
        tg.tiers.append(tier_slur)

        tg.maxTime = tier_slur.maxTime = tier_word.maxTime = tier_phone.maxTime = sum_sec

        seg_tg_path = item_name + '#' + str(sen_id)
        tg.write(f'{self.processed_dir_short}/textgrid/{item_name}/{seg_tg_path}.TextGrid')

    # 用于process中midi的分割
    def seg_midi(self, item_name, ori_mf, sen_id, midi_list, text2midi_list, midi_bias, midi_max):
        text2midi_list = copy.deepcopy(text2midi_list)
        midi_list = copy.deepcopy(midi_list)
        # midi修改时间
        #midi_bias = midi_list[0].start

        for idx, midi in enumerate(midi_list):
            midi.start = max(midi.start - midi_bias, 0)
            midi.end = min(midi.end - midi_bias, midi_max)
            midi_list[idx] = midi

        #print(midi_list)
        # 不修改原对象
        out_mf = copy.deepcopy(ori_mf)
        out_instru = out_mf.instruments[0]
        # 换midi列表
        out_notes = midi_list
        out_instru.notes = out_notes
        out_mf.instruments[0] = out_instru

        seg_midi_path = item_name + '#' + str(sen_id)
        midi_out_path = f'{self.processed_dir_short}/midi/{item_name}/{seg_midi_path}.mid'
        out_mf.dump(midi_out_path)
        #import numpy as np
        #np_out_path = f'{self.processed_dir_short}/midi/{item_name}/{seg_midi_path}.npy'
        #np.save(np_out_path, text2midi_list)
    # 用于process中wav的分割
    def seg_wav(self, item_name, sen_id, xmin, xmax, sr):
        seg_wav_path = item_name + '#' + str(sen_id)
        min_frame = int(sr * xmin)
        max_frame = int(sr * xmax)
        wav_out = self.wav_input[min_frame:max_frame]
        save_wav(wav_out, f'{self.processed_dir_short}/wav/{item_name}/{seg_wav_path}.wav', sr)

    '''
    task1:根据breathe分割TG √
    task2:重写sentence-level的TG √
    task3:分割wav √
    task4:分割midi
    '''
    def process(self):
        gen = self.meta_data()

        #重建文件夹
        subprocess.check_call(f'rm -rf {self.processed_dir_short}', shell=True)
        os.makedirs(f"{self.processed_dir_short}", exist_ok=True)
        os.makedirs(f"{self.processed_dir_short}/wav", exist_ok=True)
        os.makedirs(f"{self.processed_dir_short}/midi", exist_ok=True)
        os.makedirs(f"{self.processed_dir_short}/textgrid", exist_ok=True)

        for item in tqdm(gen):
            (item_name, wav_fn, singer, tg_fn, midi_fn) = item

            os.makedirs(f"{self.processed_dir_short}/wav/{item_name}", exist_ok=True)
            os.makedirs(f"{self.processed_dir_short}/midi/{item_name}", exist_ok=True)
            os.makedirs(f"{self.processed_dir_short}/textgrid/{item_name}", exist_ok=True)

            # sr=None表示以原采样率读取，这里分割wav不损失音频
            self.wav_input, sr = librosa.core.load(wav_fn, sr=None)
            # 读取TG
            with open(tg_fn, "r") as f:
                tg = f.readlines()
            tg = TextGrid(tg)
            tg = json.loads(tg.toJson())

            tg_align_word = [x for x in tg['tiers'][0]['items']]
            tg_align_ph = [x for x in tg['tiers'][1]['items']]
            # 副本，用于修改word后存储，里面的''和breathe被更改为SP和AP
            tg_align_word_copy = copy.deepcopy(tg_align_word)
            tg_align_ph_copy = copy.deepcopy(tg_align_ph)
            # 读取midi
            mf = miditoolkit.MidiFile(midi_fn)
            instru = mf.instruments[0]
            midi_align_list = align_midi2text(instru.notes, tg_align_word)

            # 句子word字符列表
            words_list = []
            phs_list = []
            # 这里是纯midi的列表，用于分割midi
            midi_list = []
            # 这里是text2midi，用于后续的对齐
            text2midi_list = []
            word_id = 0
            ph_id = 0
            # 记录上一个句子的边界（秒），用于音频分割
            last_sen_bound = 0
            # 句子边界的idx
            sen_word_lower_bound = 0
            sen_ph_lower_bound = 0
            # 记录sentence的id
            sen_id = 0
            # 记录空字符（breathe，‘’）的时间
            sil_time = 0
            max_dur = 0
            while word_id <= len(tg_align_word):
                # 一句话结束，这里以breathe为句子边界
                # 或者word_id已经到最后
                if word_id < len(tg_align_word):
                    word = tg_align_word[word_id]
                    word['xmin'] = float(word['xmin'])
                    word['xmax'] = float(word['xmax'])
                # 这里要先合并非字区域，非字>0.03作为分割
                # 做句首句尾处理
                # 做noise处理
                # sil_time控制句子最小长度
                # 可以加sen_dur继续控制时长
                if word_id - 1 >= 0:
                    last_word_in_sentence = tg_align_word[word_id - 1]
                    sen_dur = float(last_word_in_sentence['xmax']) - last_sen_bound
                else:
                    sen_dur = 0
                if word_id >= len(tg_align_word) or \
                        ((sil_time >= 0.4 or sen_dur >= 5.0) and self.is_word(word['text']) and (tg_align_word[word_id-1]['text'] == '' or tg_align_word[word_id-1]['text'] == 'breathe')) or \
                        (sen_dur >= 12) or \
                        word['text'] == 'noise':
                    #if not self.is_word(word['text']) and word_id < len(tg_align_word):
                        #print(1)
                        #print(1)
                    # 统计其中非AP、SP的数目
                    if self.word_count(words_list) > 0:
                        # 句子最后一个字符
                        last_word_in_sentence = tg_align_word[word_id - 1]
                        last_word_in_sentence['xmin'] = float(last_word_in_sentence['xmin'])
                        last_word_in_sentence['xmax'] = float(last_word_in_sentence['xmax'])
                        # 句子组合
                        # words_list中没有符号，phs中有|
                        sentence = " ".join(words_list)
                        words_list = []
                        # 去掉最后一个分界符
                        if phs_list[-1] == '|':
                            phs_list.pop()
                        phs = " ".join(phs_list)
                        phs_list = []
                        #print(sentence, phs)
                        # 保存TG
                        # 当前的word不保存，到word_id - 1
                        tg_align_word_seg = tg_align_word_copy[sen_word_lower_bound : word_id]
                        tg_align_ph_seg = tg_align_ph_copy[sen_ph_lower_bound : ph_id]
                        for item in tg_align_word_seg:
                            if item['text'] == '':
                                print(1)
                                print(1)
                        self.sav_tg(item_name, sen_id, tg_align_word_seg, tg_align_ph_seg, text2midi_list)
                        '''
                        # 修复midi
                        for xx in text2midi_list:
                            #print(xx)
                            notes_ = xx['notes']
                            if len(notes_) > 1:
                                notes_idx_ = xx['notes_idx']
                                for idx_, note_ in enumerate(notes_):
                                    if idx_ < len(notes_) - 1:
                                        next_note_ = notes_[idx_+1]
                                        if note_.pitch == next_note_.pitch and note_.end == next_note_.start:
                                            fix_midi.concat_note(item_name, notes_idx_[idx_])
                                            #print(text2midi_list)
                                            print(item_name, notes_idx_[idx_])
                                            return file_id
                                            #print(note_, next_note_)
                                            #print(notes_idx_[idx_])
                        '''
                        # 切割音频
                        self.seg_wav(item_name, sen_id, last_sen_bound, last_word_in_sentence['xmax'], sr)
                        sen_dur = last_word_in_sentence['xmax'] - last_sen_bound
                        max_dur = max(max_dur, sen_dur)
                        # 切割midi
                        # 限制midi的范围
                        midi_bias = round(float(tg_align_word_seg[0]['xmin']) * 440)
                        midi_max = round(float(tg_align_word_seg[-1]['xmax']) * 440)
                        self.seg_midi(item_name, mf, sen_id, midi_list, text2midi_list, midi_bias, midi_max)
                        #print(text2midi_list)
                        if len(text2midi_list) != len(tg_align_word_seg):
                            print(1)
                            print(1)
                        # 清空midi列表
                        midi_list = []
                        text2midi_list = []
                        sen_id += 1

                        last_sen_bound = float(last_word_in_sentence['xmax'])

                        sen_word_lower_bound = word_id
                        sen_ph_lower_bound = ph_id

                        # 静音时间清空
                        sil_time = 0

                        # 结束循环
                        if word_id >= len(tg_align_word):
                            word_id += 1
                            ph_id += 1
                    else:
                        if self.is_word(word['text']):
                            sil_time = 0
                            last_sen_bound = float(word['xmin'])
                            sen_word_lower_bound = word_id
                            sen_ph_lower_bound = ph_id
                        else:
                            sil_time = 0
                            last_sen_bound = float(word['xmax'])
                            word_id += 1
                            ph_id += 1
                            sen_word_lower_bound = word_id
                            sen_ph_lower_bound = ph_id
                        words_list = []
                        phs_list = []
                        text2midi_list = []
                        # noise在这里不作处理
                        # pass


                # 句子中的空字符，在phone中添加sil标记
                # todo：合并空字符
                elif word['text'] == '':
                    words_list.append('SP')
                    phs_list.append('SP')
                    phs_list.append('|')
                    sil_time += word['xmax'] - word['xmin']
                    text2midi_list.append({'word': 'SP', 'notes': []})
                    # 更改text，切割后的textgrid用AP、SP表示
                    tg_align_word_copy[word_id]['text'] = 'SP'
                    tg_align_ph_copy[ph_id]['text'] = 'SP'
                    word_id += 1
                    ph_id += 1
                elif word['text'] == 'breathe':
                    words_list.append('AP')
                    phs_list.append('AP')
                    phs_list.append('|')
                    sil_time += word['xmax'] - word['xmin']
                    text2midi_list.append({'word': 'AP', 'notes': []})
                    tg_align_word_copy[word_id]['text'] = 'AP'
                    tg_align_ph_copy[ph_id]['text'] = 'AP'
                    word_id += 1
                    ph_id += 1

                # 其他正常字符，作为单个字
                else:
                    words_list.append(word['text'])
                    word_max = word['xmax']
                    # 列表合并
                    midi_list = midi_list + midi_align_list[word_id]['notes']
                    text2midi_list.append(copy.deepcopy(midi_align_list[word_id]))
                    while ph_id < len(tg_align_ph):
                        ph = tg_align_ph[ph_id]
                        phs_list.append(ph['text'])
                        ph['xmax'] = float(ph['xmax'])
                        # 判断相等，到达word边界
                        if abs(ph['xmax'] - word['xmax']) <= (1e-5):
                            phs_list.append('|')
                            ph_id += 1
                            break
                        ph_id += 1
                    word_id += 1
                    # 静音时间清空
                    sil_time = 0

            #print(max_dur)
