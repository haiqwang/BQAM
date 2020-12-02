import os
import re
from collections import OrderedDict
import numpy as np
import json
import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', default='', type=str, choices=['psnr', 'ssim', 'ms-ssim', 'vif', 'adm', 'vmaf'], help='metric')
    parser.add_argument('--log', default='', type=str, help='directory of score files')
    parser.add_argument('--save', default='', type=str, help='file name of unified obj score')
    args = parser.parse_args()

    return args


def load_per_frame_score(dir_score_files, metric):
    '''FR metrics calculated with VMAF package
    PSNR, SSIM, MSSSIM, VIF, ADM, VMAF
    '''
    # metric = lowercase(metric)
    if metric in ['msssim', 'ms-ssim', 'ms_ssim']:
        metric = 'ms_ssim'
    elif metric == 'adm':
        metric = 'adm2'
    elif metric == 'vif':
        metric = 'vif_scale0'

    per_seq_score = OrderedDict()
    for dirpath, dirs, files in os.walk(dir_score_files, topdown=False):
        for file in files:
            if file.startswith('.') or not file.endswith('.json'):
                continue
            path = os.path.join(dirpath, file)
            with open(path) as f:
                data = json.load(f)
            frames = data['frames']
            cnt = 0
            ttl = 0
            for frame in frames:
                if metric in frame['metrics'].keys():
                    ttl += frame['metrics'][metric]
                    cnt += 1
            vid_name = re.split('\.', file)[0]
            per_seq_score[vid_name] = '{:.4f}'.format(float(ttl)/cnt)
    return per_seq_score



def main(args):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_score_files = os.path.join(dir_path, args.log)
    obj_file_path = os.path.join(dir_path, args.save)
    ret = load_per_frame_score(dir_score_files, args.metric)
    with open(obj_file_path, 'w') as f:
        json.dump(ret, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    args = parse_opts()
    main(args)
    print('Done')
