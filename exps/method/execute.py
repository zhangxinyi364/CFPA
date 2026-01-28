#!/usr/bin/env python
import sys
# sys.path.append('/Point_otk')

import argparse
import os
import os.path as osp
import warnings
import time
from datetime import datetime
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
from plyfile import PlyData, PlyElement
from models.func import qrot  # 引入四元数旋转函数


import matplotlib.pyplot as plt
plt.switch_backend('agg')
warnings.filterwarnings('ignore')

from scripts import d_utils as d_utils
from scripts import z_utils as z_utils

from models.cfpa_model import hierarchical_trans

from datasets.partnet import PartNetPartDataset
from exps.method.cfpa_train import train_one_epoch
from exps.method.cfpa_eval import eval_func


parser = argparse.ArgumentParser(description='3D part assembly.')

# * Dataset.
parser.add_argument('--data_dir', type=str, default='../../prep_data', help='data directory')
parser.add_argument('--category', type=str, default='Chair',
                    choices=['Chair', 'Table', 'Lamp'], help='model def file')
parser.add_argument('--train_data_fn', type=str, default='Chair.train.npy',
                    choices=['Chair.train.npy', 'Table.train.npy', 'Lamp.train.npy'],
                    help='training data file that index all data tuples')
parser.add_argument('--val_data_fn', type=str, default='Chair.val.npy',
                    choices=['Chair.val.npy', 'Table.val.npy', 'Lamp.val.npy', 'Chair.test.npy', 'Table.test.npy', 'Lamp.test.npy', 'Chair.train.npy', 'Lamp.train.npy'],
                    help='validation data file that index all data tuples')
parser.add_argument('--test_data_fn', type=str, default='Chair.test.npy',
                    choices=['Chair.val.npy', 'Table.val.npy', 'Lamp.val.npy', 'Chair.test.npy', 'Table.test.npy', 'Lamp.test.npy'],
                    help='validation data file that index all data tuples')
parser.add_argument('--level', type=str, default='3', help='level of dataset')

# * Training.

parser.add_argument('--num_super_parts', type=int, default=16,
                   help='number of super parts to group into')
parser.add_argument('--super_part_weight', type=float, default=0.1,
                   help='weight for super part supervision')
parser.add_argument('--use_geo_constraint', type=int, default=1,
                   help='whether to use geometric constraints in OT')


parser.add_argument('--eval-only', default=0, type=int)
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', default=1000, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--lr_drop', default=200, type=int)
parser.add_argument('--warmup-epochs', default=10, type=int, metavar='N',
                    help='number of warmup epochs')
parser.add_argument('--type-sched', default='step', type=str,
                    choices=['step', 'cosine'], help='The type of learning rate update.')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 4096), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=1.5e-4, type=float,
                    metavar='LR', help='initial (base) learning rate')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-6)',
                    dest='weight_decay')
parser.add_argument('--output-dir', default="checkpoints/", type=str)
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--save-freq', default=10, type=int,
                    metavar='N', help='save frequency (default: 10)')
# parser.add_argument('--resume', default="/root/autodl-tmp/zxy666/exps/method/checkpoints/train/Chair/20250402_230903/checkpoint_0380.pth.tar", 
#                     type=str, metavar='PATH',help='path to latest checkpoint (default: none)')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--num-hidden', type=int, default=32, help='hidden size')

# hyper-parameter for model arch
parser.add_argument("--online", action='store_true')
parser.add_argument("--layernorm", action='store_true')
parser.add_argument("--remain-k1", action='store_true')
parser.add_argument("--diffQ", action='store_true')
parser.add_argument("--residual", type=str, default='cat',
                    choices=['None', 'cat', 'sum'])
parser.add_argument("--kernel_method", type=str,
                    default='elu', choices=['relu', 'elu'])
parser.add_argument("--deepset-layers", type=int, default=2)
parser.add_argument("--pos-enc-rw-dim", type=int, default=8)
parser.add_argument("--pos-enc-lap-dim", type=int, default=0)
parser.add_argument("--n-patches", type=int, default=8)
parser.add_argument("--prop-w-norm-on-coarsened", action='store_true')
parser.add_argument("--pos-enc-patch-rw-dim", type=int, default=0)
parser.add_argument("--pos-enc-patch-num-diff", type=int, default=-1)
parser.add_argument("--attention-based-readout", action='store_true')
parser.add_argument("--convex-linear-combination", action='store_true')
parser.add_argument('--num-features', type=int, default=256)
parser.add_argument('--num_classes', type=int, default=7) 

# input transformation setting
parser.add_argument("--use-gnn", action='store_true')
parser.add_argument("--conv", type=str, default='GIN', choices=['GCN', 'GIN'])
parser.add_argument("--num-convs", type=int, default=1)

# * Eval.
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-eb', '--eval-batch-size', default=64, type=int, metavar='N')
parser.add_argument('--eval-epochs', default=0, type=int, metavar='N', help='epoch start for evaluation.')

parser.add_argument('--compute-mean-metrics', type=int, default=0,
                    help='Whether to compute mPA and mCA during evaluation (0: only threshold=0.01, 1: compute mean metrics)')

parser.add_argument('--ca-threshold', type=float, default=45.2,
                   help='Threshold for saving model based on connectivity accuracy')

# * Loss weights.
parser.add_argument('--loss_weight_trans_l2', type=float, default=1.0, help='loss weight')
parser.add_argument('--loss_weight_rot_l2', type=float, default=0.0, help='loss weight')
parser.add_argument('--loss_weight_trans_cd', type=float,default=1.0, help='loss weight')
parser.add_argument('--loss_weight_rot_cd', type=float, default=10.0, help='loss weight')
parser.add_argument('--loss_weight_shape_cd', type=float, default=1.0, help='loss weight')
parser.add_argument('--loss_weight_part_cd', type=float, default=1.0, help='loss weight')
parser.add_argument('--ot_weight', type=float, default=0.1,help='weight for OT loss')
parser.add_argument('--loss_weight_cate', type=float, default=0.0, help='loss weight')
parser.add_argument('--loss_weight_decode', type=float, default=1.0, help='loss weight')
parser.add_argument('--kmeans_weight', type=float, default=0.1, 
                    help='weight for kmeans loss')
parser.add_argument('--som_weight', type=float, default=0.1, 
                    help='weight for som loss')
parser.add_argument('--hierarchical_weight', type=float, default=0.1, 
                    help='weight for hierarchical clustering loss')



# * Network settings.
parser.add_argument('--model', default='trans_assembly_v1',
                    type=str, help='model name.')
parser.add_argument('--model_version', default='original', type=str, help='model version.')
parser.add_argument('--backbone', default='pointnet_cls',
                    type=str, help='the backbone to extract init features.')
parser.add_argument('--feat_dim', type=int, default=256)
parser.add_argument('--hidden_gru', type=int, default=128)
parser.add_argument('--max_num_part', type=int, default=20)
parser.add_argument('--num_mlp', type=int, default=2)
parser.add_argument('--base-cat', default=0, type=int, help='Whether to cat base feat when prediction.')
parser.add_argument('--pose-cat', default=1, type=int, help='Whether to cat pose when prediction.')
parser.add_argument('--pose-cat-in-encoder', default=1, type=int, help='Whether to cat pose in trans-encoder.')
parser.add_argument('--shared-pred', default=1, type=int, help='Whether to apply shared mlp & predictor.')
parser.add_argument('--pred-detach', default=1, type=int, help='Whether to detach pose when cat pose.')
parser.add_argument('--train-mon', default=5, type=int, help='MoN iterations in training.')
parser.add_argument('--eval-mon', default=10, type=int, help='MoN iterations in inference.')
parser.add_argument('--noise-cat', default=1, type=int, help='Whether to cat noise when prediction.')
parser.add_argument('--noise-cat-in-encoder', default=1, type=int, help='Whether to cat noise when in trans-encoder.')
parser.add_argument('--noise-dim', default=64, type=int, help='The dim of random noise.')
parser.add_argument('--ins-cat', default=0, type=int, help='Whether to cat ins part id when prediction.')
parser.add_argument('--ins-cat-in-encoder', default=0, type=int, help='Whether to cat ins part id when in trans-encoder.')
parser.add_argument('--ins-cat-inter-only', default=0, type=int, help='Whether to only cat ins inter pos when in trans-encoder.')
parser.add_argument('--ins-cat-intra-only', default=0, type=int, help='Whether to only cat ins intra pos when in trans-encoder.')
parser.add_argument('--ins-version', default='v2', type=str, help='the version of instance pos.')
parser.add_argument('--type-eval', default='encoder', type=str, choices=['encoder', 'wip', 'decoder'], help='eval type.')
parser.add_argument('--worst-mon', default=0, type=int, help='Whether to apply worst mon during encoder inference.')


parser.add_argument('--use_cluster_node_interaction', type=int, default=1,
                    help='whether to use cluster-node interaction')
parser.add_argument('--interaction_hidden_dim', type=int, default=256,
                    help='hidden dimension for interaction module')
parser.add_argument("--save-vis", type=int, default=0, help="Set to 1 to save visualization")
parser.add_argument("--save-dir", type=str, default="./results/vis", help="Directory to save visualization files")



# for decoder.
parser.add_argument('--decode-on', default=0, type=int, help='Whether to use transformer decoder.')
parser.add_argument('--num-pos', default=1, type=int, help='Number of positive query in decoder.')
parser.add_argument('--rand-pos', default=0, type=int, help='Whether to apply random positive number (1 - num-pos)')
parser.add_argument('--pose-cat-in-decoder-pred', default=1, type=int, help='Whether to cat pose when in decoder-pred.')
parser.add_argument('--pose-cat-in-decoder-trans', default=1, type=int, help='Whether to cat pose when in decoder-trans.')
parser.add_argument('--noise-cat-in-decoder-pred', default=1, type=int, help='Whether to cat noise when in decoder-pred.')
parser.add_argument('--noise-cat-in-decoder-trans', default=1, type=int, help='Whether to cat noise when in decoder-trans.')
parser.add_argument('--memory-detach', default=0, type=int, help='Whether to detach memory feats.')
parser.add_argument('--feat-in-detach', default=0, type=int, help='Whether to detach in-decoder feats.')
parser.add_argument('--cate-on', default=0, type=int, help='Whether to apply cate pred during decoder.')
parser.add_argument('--encode-freeze', default=0, type=int, help='Whether to freeze backbone/encoder during training.')
parser.add_argument('--ins-cat-in-decoder', default=1, type=int, help='Whether to cat ins part id in trans-decoder.')
parser.add_argument('--pose-cat-in-memory', default=0, type=int, help='Whether to cat pose in memory.')
parser.add_argument('--noise-cat-in-memory', default=0, type=int, help='Whether to cat noise in memory.')

# for filter.
parser.add_argument('--filter-on', default=0, type=int, help='Whether to apply filter augmentation.')
parser.add_argument('--filter-thresh', type=float, default=0.2, help='filter thresh')
parser.add_argument('--num-filter', default=1, type=int, help='Number of filter in training.')

# vis.
parser.add_argument('--gt-vis', default=0, type=int, help='Whether to apply gt visualization.')
parser.add_argument('--gt-vis-dir', default='./checkpoints/gt_vis/', type=str, help='relative path to root dir.')
parser.add_argument('--pred-encoder-vis', default=0, type=int, help='Whether to apply encoder pred visualization.')
parser.add_argument('--pred-encoder-vis-dir', default='./checkpoints/encoder_pred/', type=str, help='relative path to root dir.')
parser.add_argument('--num-pred-vis', default=3, type=int, help='Number of pred visualization.')

# * Transformer.
parser.add_argument('--enc_layers', default=6, type=int,
                    help="Number of encoding layers in the transformer")
parser.add_argument('--dec_layers', default=6, type=int,
                    help="Number of decoding layers in the transformer")
parser.add_argument('--dim_feedforward', default=2048, type=int,
                    help="Intermediate size of the feedforward layers in the transformer blocks")
parser.add_argument('--hidden_dim', default=256, type=int,
                    help="Size of the embeddings (dimension of the transformer)")
parser.add_argument('--dropout', default=0.1, type=float,
                    help="Dropout applied in the transformer")
parser.add_argument('--nheads', default=8, type=int,
                    help="Number of attention heads inside the transformer's attentions")
parser.add_argument('--num_queries', default=10, type=int,
                    help="Number of query slots")

parser.add_argument('--pre_norm', default=2, type=int)
parser.add_argument('--offset_attention', default=2, type=int)
parser.add_argument('--bi_pn', default=0, type=int)



parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='env://', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')




    
def main():

    # =================== Parameters Setting ===================
    args = parser.parse_args()
    d_utils.init_distributed_mode(args)
    z_utils.func_init(args)

    assert args.category in ["Chair", "Table", "Lamp"]
    if args.train_mon > 1 or args.eval_mon > 1:
        assert args.noise_cat or args.noise_cat_in_encoder, print("Noise issue 1...")
    else:
        assert (not args.noise_cat) and (not args.noise_cat_in_encoder), print("Noise issue 2...")

    # =================== Network ===================
    args.lr = args.lr * args.batch_size / 64

    # create model.
    model = hierarchical_trans(args)
    model = z_utils.device_func(model, args)

    # optimizer.
    if args.encode_freeze:
        assert args.decode_on, print("Decoder needed in this version.")
        if not args.eval_only:
            assert os.path.isfile(args.resume), print("Pre-train model needed in this version.")
        parameters = list()
        for name, param in model.named_parameters():
            if "decoder" in name:
                parameters.append(param)
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.type_sched == "step":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop, gamma=0.85)

    # automatic mixed-precision.
    scaler = torch.cuda.amp.GradScaler()

    # output dir.
    base_dir = osp.join(osp.abspath(osp.dirname(__file__)), args.output_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = osp.join(base_dir, timestamp)
    if not osp.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    summary_writer = SummaryWriter(log_dir=output_dir) if args.rank == 0 else None
    log_writer = open(osp.join(output_dir, "train_log.txt"), "w")




    log_writer.write(str(args) + "\n")
    log_writer.flush()

    # optionally resume from a checkpoint.
    if args.resume and not args.eval_only:
        model = z_utils.load_model(model, args, resume_on=True)

    cudnn.benchmark = True

    # =================== Data Preparation ===================
    if not args.eval_only:
        train_dataset = PartNetPartDataset(args.data_dir, args.train_data_fn, args.category,
                                           level=args.level, max_num_part=args.max_num_part)
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler,
            drop_last=True, collate_fn=z_utils.collate_feats_with_none)

    val_dataset = PartNetPartDataset(args.data_dir, args.val_data_fn, args.category,
                                     level=args.level, max_num_part=args.max_num_part)
    val_sampler = None

    test_dataset = PartNetPartDataset(args.data_dir, args.test_data_fn, args.category,
                                     level=args.level, max_num_part=args.max_num_part)
    test_sampler = None

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler,
        drop_last=False, collate_fn=z_utils.collate_feats_with_none)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=test_sampler,
        drop_last=False, collate_fn=z_utils.collate_feats_with_none)

    # ========================================================
    # =================== Eval ===============================

    print("Begin Evaluation...")
    if args.eval_only:
        # load checkpoint.
        model = z_utils.load_model(model, args)
        # eval model.
        part_acc = eval_func(val_loader, model, log_writer, args)
        # mPA, mCA = eval_func(val_loader, model, log_writer, args)
        # print(f"Evaluation completed.")
        # print(f"Mean Part Accuracy (mPA): {mPA:.2f}%")
        # print(f"Mean Connectivity Accuracy (mCA): {mCA:.2f}%")
        return


    # ========================================================
    # =================== Training ===========================
    countdown = 0.
    best_acc = 0.
    for epoch in range(args.start_epoch, args.epochs):
        print('Begin Epoch... {}'.format(epoch))
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch.
        print("Begin Training...")
        countdown = train_one_epoch(train_loader, model, optimizer, scaler, summary_writer,
                                    log_writer, epoch, countdown, args)
        if args.type_sched == "step":
            lr_scheduler.step()

        print("Begin Evaluation...")
        if (epoch + 1) % args.save_freq == 0:
            print("Save Epoch {}".format(epoch))
            z_utils.save_model(epoch, scaler, output_dir, model, optimizer, 0.1, 0.1, args)

        # if (epoch + 1) >= 580 and (epoch + 1) % 2 == 0 :
        #     part_acc = eval_func(val_loader, model, log_writer, args)
        # else:
        #     part_acc = 0.1
        
        if (epoch + 1) >= 580 and (epoch + 1) % 2 == 0 :
            part_acc, contact_acc = eval_func(val_loader, model, log_writer, args)
            z_utils.save_model(epoch, scaler, output_dir, model, optimizer, part_acc, best_acc, args, contact_acc=contact_acc)
        else:
            part_acc = 0.1
            contact_acc = 0.1
        
        best_acc = part_acc if part_acc > best_acc else best_acc
        print("==========================================================")
        print("Best Part Accuracy: {}".format(best_acc))
        print("==========================================================")
        res_best = "\n==========================================================" + "\n" + \
                    "Best Part Accuracy: {}".format(best_acc) + "\n" + \
                    "==========================================================\n"
        log_writer.write(res_best + "\n")
        log_writer.flush()




        print('End Epoch... {}'.format(epoch))

    if args.rank == 0:
        summary_writer.close()
        log_writer.close()


if __name__ == '__main__':
    main()
