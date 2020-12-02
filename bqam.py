import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau
from src.cubic_4_fitting import cubic_4_fitting
from src.logistic_4_fitting import logistic_4_fitting
from src.logistic_5_fitting import logistic_5_fitting
from src.cubic_4_fitting_no_constraint import cubic_4_fitting_no_constraint
from src.logistic_5_fitting_no_constraint import logistic_5_fitting_no_constraint


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='', type=str,choices=['LIVE_VQA', 'CSIQ_VQA', 'KONVID-1K'], help='image/video dataset to benchmark')
    parser.add_argument('--mos', default='', type=str, required=True, help='subjective score file (.json)')
    parser.add_argument('--prediction', default='', type=str, required=True, help='objective score file (.json)')
    parser.add_argument('--metric', default='', type=str, required=True, help='metric to benchmark')
    parser.add_argument('--function', default='', type=str, choices=['c4', 'l4', 'l5', 'c4_no', 'l5_no'], help='non-linear regression function')
    parser.add_argument('--save', action='store_true', help='save figure')
    args = parser.parse_args()

    supported_metrics = {}
    supported_metrics['LIVE_VQA'] = ['PSNR', 'SSIM', 'MS_SSIM', 'VIF', 'ADM', 'VMAF']
    supported_metrics['CSIQ_VQA'] = ['PSNR', 'SSIM', 'MS_SSIM', 'VIF', 'ADM', 'VMAF']
    supported_metrics['KONVID-1K'] = ['BRISQUE', 'CORNIA']

    if args.metric not in supported_metrics[args.dataset]:
        parser.error('{m} is unavailable for {d}'.format(m=args.metric.upper(), d=args.dataset.upper()))

    return args

def main(args):
    dir_path = os.path.dirname(os.path.realpath(__file__))

    mos_dir = os.path.join(dir_path, 'mos')
    os.makedirs(mos_dir, exist_ok=True)
    mos_path = os.path.join(mos_dir, args.mos)

    pred_dir = os.path.join(dir_path, *['prediction', args.dataset.upper()])
    os.makedirs(pred_dir, exist_ok=True)
    pred_path = os.path.join(pred_dir, args.prediction)

    fig_dir = os.path.join(dir_path, *['output', args.dataset.upper()])
    os.makedirs(fig_dir, exist_ok=True)
    fig_name = args.dataset.upper() + '_' + args.metric.upper() + '_' + args.function + '.png'
    fig_path = os.path.join(fig_dir, fig_name)


    with open(mos_path) as f:
        mos_dict = json.load(f)
    with open(pred_path) as f:
        pred_dict = json.load(f)

    mos, pred = [], []
    # assume a metric could process all contents
    for seq in mos_dict.keys():
        mos.append(float(mos_dict[seq]))
        pred.append(float(pred_dict[seq]))

    # stats w/o regression
    mos, pred = np.asarray(mos), np.asarray(pred)
    plcc, _ = pearsonr(pred, mos)
    srocc, _ = spearmanr(pred, mos)
    krocc, _ = kendalltau(pred, mos)
    rmse = np.sqrt(np.mean((pred - mos)**2))
    print('stats w/o curve fitting...')
    print('plcc: \t {:.4f}'.format(plcc))
    print('srocc: \t {:.4f}'.format(srocc))
    print('krocc: \t {:.4f}'.format(krocc))
    print('rmse: \t {:.4f}\n'.format(rmse))

    # flip +/- if MOS VS. DMOS occurs
    # if srocc < 0:
    #     max_val, min_val = np.max(pred), np.min(pred)
    #     pred = [max_val + min_val - x for x in pred]

    if args.function == 'c4':
        x_axis, curve, _pred = cubic_4_fitting(pred, mos)
    elif args.function == 'l4':
        x_axis, curve, _pred = logistic_4_fitting(pred, mos)
    elif args.function == 'l5':
        x_axis, curve, _pred = logistic_5_fitting(pred, mos)
    elif args.function == 'l5_no':
        x_axis, curve, _pred = logistic_5_fitting_no_constraint(pred, mos)
    elif args.function == 'c4_no':
        x_axis, curve, _pred = cubic_4_fitting_no_constraint(pred, mos)
    else:
        raise NotImplementedError('{} not supported'.format(args.function))

    # # stats after curving fitting
    plcc = pearsonr(_pred, mos)[0]
    srocc = spearmanr(_pred, mos)[0]
    krocc, _ = kendalltau(pred, mos)
    rmse = np.sqrt(np.mean((_pred - mos)**2))
    print('func: \t {}'.format(args.function))
    print('plcc: \t {:.4f}'.format(plcc))
    print('srocc: \t {:.4f}'.format(srocc))
    print('krocc: \t {:.4f}'.format(krocc))
    print('rmse: \t {:.4f}\n'.format(rmse))

    plt.plot(pred, mos, 'go', x_axis, curve, 'k--', linewidth=3)
    plt.xlabel('{}'.format(args.metric.upper()))
    plt.ylabel('MOS')

    if args.save:
        plt.savefig(fig_path)
    else:
        plt.show()


if __name__ == '__main__':
    args = parse_opts()
    print(args)
    main(args)
