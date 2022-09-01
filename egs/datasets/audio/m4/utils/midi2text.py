import librosa
from data_gen.tts.data_gen_utils import TextGrid
import glob
import os
import json
from tqdm import tqdm
import subprocess
from utils import audio
import textgrid
import miditoolkit
from fix_midi import  fix_tool

# 如果note在word左侧，return -1
# 在note内return 0
# 在note右侧return 1
# 这个地方eps要调，标注的不标准
def note_in_word(note, word, eps=0.00):
    word['xmin'] = float(word['xmin'])
    word['xmax'] = float(word['xmax'])
    note_start = note.start / 220 * 0.5
    note_end = note.end / 220 * 0.5

    if note_end <= word['xmin'] + eps:
        return -1
    elif note_start >= word['xmax'] - eps:
        return 1
    else:
        note_in_word_max = min(note_end, word['xmax'])
        note_in_word_min = max(note_start, word['xmin'])
        word_len = note_end - note_start#min(word['xmax'] - word['xmin'], note_end - note_start)
        note_in_word_ratio = (note_in_word_max - note_in_word_min) / word_len
        if note_in_word_ratio > 0.7 or (note_in_word_ratio > 0.5 and note_in_word_max < word['xmax']) or (note_in_word_ratio > 0.5 and note_in_word_min > word['xmin']):
            return 0
        else:
            return 2



def is_word(word):
    if word != '' and word != 'breathe' and word != 'noise':
        return True
    else:
        return False

def align_midi2text(notes, text_items):
    text_idx = 0
    align_text2midi = []
    note_list = []
    note_idx_list = []
    for idx, note in enumerate(notes):
        while True:
            word = text_items[text_idx]
            flg = note_in_word(note, word)

            note_start = note.start / 220 * 0.5
            note_end = note.end / 220 * 0.5

            # 下一个是语气词
            if text_idx < len(text_items) - 1 and not is_word(text_items[text_idx + 1]['text']) and note_start < word['xmax'] - 0.05:
                flg = 0

            # 特殊情况，前一个word是breathe或者''，则note一定在word里
            if text_idx > 0 and len(note_list) == 0 and not is_word(text_items[text_idx - 1]['text']) and (note_start < word['xmax'] and note_end > word['xmin']):
                flg = 0

            # 语气词无note
            if not is_word(text_items[text_idx]['text']):
                flg = 3

            if flg == 0:
                note_list.append(note)
                note_idx_list.append(idx)
                # 最后一个音符了
                if idx == len(notes) - 1:
                    align_text2midi.append({'word': word['text'], 'notes': note_list, 'notes_idx': note_idx_list})
                    #print({'word': word['text'], 'notes': note_list})
                break
            else:
                # debug用的，word是text但是没有音符表示有问题
                if is_word(word['text']) and len(note_list) == 0:
                    print(word)
                    print(note)
                    print('note_min:', note.start / 220 * 0.5)
                    print('note_max:', note.end / 220 * 0.5)
                    print('note_idx', idx)
                    print('item_name', item_name)
                    print(align_text2midi[-20:])
                    print('next_word', text_items[text_idx + 1])
                    print('next_word2', text_items[text_idx + 2])
                    print('next_note', notes[idx + 1])
                    print('words', text_items[text_idx - 20:text_idx + 2])
                    if abs(note.start / 440 - word['xmin']) < 0.03:
                        #fix_tool(item_name, idx, 440 * word['xmax'])
                        print('fix done1')
                    elif abs(note.end / 440 - word['xmax']) < 0.03:
                        #fix_tool(item_name, idx - 1, 440 * word['xmin'])
                        print('fix done2')
                    #elif note_start > word['xmin'] and note_start < word['xmax'] and note_start < (word['xmin'] + word['xmax'])/2:
                    #    fix_tool(item_name, idx - 1, 440 * word['xmin'])


                    print(1)
                    print(1)
                    exit()
                if not is_word(word['text']) and len(note_list) != 0:
                    print(word)
                    print(note_list)
                    exit()

                align_text2midi.append({'word': word['text'], 'notes': note_list, 'notes_idx': note_idx_list})
                #print({'word': word['text'], 'notes': note_list})
                note_list = []
                note_idx_list = []
                text_idx += 1

    # 后处理，note用完可能还有word
    align_idx = len(align_text2midi)

    while align_idx < len(text_items):
        if is_word(text_items[align_idx]['text']):
            print(align_text2midi[-3:])
            print(text_items[-3:])
            print(len(notes))
            print(item_name)
            print(notes[-1])
            print(1)
        else:
            align_text2midi.append({'word': text_items[align_idx]['text'], 'notes': [], 'notes_idx': []})

        align_idx += 1

    #print(notes[-1])
    if len(align_text2midi) != len(text_items):
        pass
        #print(len(text_items), len(align_text2midi))
    return align_text2midi

if __name__ == '__main__':
    wav_fns = sorted(glob.glob('data/raw/lasinger-long/wav/*.wav'))
    item_names = []
    for wav_fn in wav_fns:
        item_name = os.path.basename(wav_fn)[:-4]
        item_names.append(item_name)
    import numpy as np
    #item_name = '声乐1号女#你就不要想起我'
    x_ = np.array(range(0, 30))
    y_ = np.array(range(0, 30)) * 0
    for item_name in tqdm(item_names):
        tg_fn = 'data/raw/lasinger-long/' + 'textgrid/' + item_name + '.TextGrid'
        # 读取TG
        with open(tg_fn, "r") as f:
            tg = f.readlines()
        tg = TextGrid(tg)
        tg = json.loads(tg.toJson())
        tg_align_word = [x for x in tg['tiers'][0]['items']]

        midi_path = 'data/raw/lasinger-long/' + 'midi/' + item_name + '.mid'
        mf = miditoolkit.MidiFile(midi_path)
        instru = mf.instruments[0]
        align_list = align_midi2text(instru.notes, tg_align_word)
        #print(align_list)
        # 统计部分

        #print(align_list)
        for item in align_list:
            i = len(item['notes'])
            if i > 10:
                print(item_name, item)
            y_[i] += 1
    print(y_)
