import os
import re
from collections import OrderedDict
import numpy as np
import json
import shutil

def make_score_file():
    dir_path = os.path.dirname(os.path.realpath(__file__))

    seq_file_name = 'live_video_quality_seqs.txt'
    score_file_name = 'live_video_quality_data.txt'

    seqs = np.genfromtxt(os.path.join(dir_path, seq_file_name), dtype='str')
    scores = np.genfromtxt(os.path.join(dir_path, score_file_name), dtype='float')

    ret = OrderedDict()
    for seq, mos in zip(seqs, scores):
        seq_info = re.split('\.|_', seq)
        # print(seq_info)
        seq_name = seq_info[0][0:2] + '{:02d}'.format(int(seq_info[0][2:])) + '_' + seq_info[1]
        # print(seq_name)
        ret[seq_name] = '{:.4f}'.format(100.0 - mos[0])

    mos_file = 'LIVE_VQA_MOS.json'
    with open(mos_file, 'w') as f:
        json.dump(ret, f, indent=4, sort_keys=True)
    shutil.copy(mos_file, '../../mos')
    os.remove(mos_file)
    print('ok, unify {}'.format(mos_file))
    print('ok, {f} moved to BQAM/mos/{f}'.format(f=mos_file))


if __name__ == "__main__":
    make_score_file()
