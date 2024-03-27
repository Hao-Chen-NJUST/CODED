import argparse
import os
import torch
import numpy as np


class TrainOptions():
    def __init__(self):
        self.args = None
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--exp_name', type=str, default="Exp0-r50", help='the name of the experiment')
        self.parser.add_argument('--img_size', type=int, default=256, help='the size of input images')
        self.parser.add_argument('--crop_size', type=int, default=224, help='the size of cropped images')
        self.parser.add_argument('--activation', type=str, default='gelu', help='activation type for transformer')

        self.parser.add_argument('--lr', default=4e-4, type=float)
        self.parser.add_argument('--lr_scale', default=1.0, type=float)
        self.parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
        self.parser.add_argument('--lr_backbone', default=4e-5, type=float)
        self.parser.add_argument('--lr_linear_proj_names', default=['sampling_offsets'], type=str,
                                 nargs='+')
        self.parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
        self.parser.add_argument('--batch_size', default=4, type=int)
        self.parser.add_argument('--weight_decay', default=1e-4, type=float)
        self.parser.add_argument('--epochs', default=50, type=int)
        self.parser.add_argument('--lr_drop', default=40, type=int)
        self.parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
        self.parser.add_argument('--clip_max_norm', default=0.1, type=float,
                                 help='gradient clipping max norm')
        self.parser.add_argument('--sgd', action='store_true')
        # Model parameters
        self.parser.add_argument('--dilation', action='store_true',
                                 help="If true, we replace stride with dilation in the last convolutional block (DC5)")
        self.parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                                 help="Type of positional embedding to use on top of the image features")
        self.parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                                 help="position / size * scale")
        self.parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')
        # * Transformer
        self.parser.add_argument('--enc_layers', default=6, type=int,
                                 help="Number of encoding layers in the transformer")
        self.parser.add_argument('--dec_layers', default=6, type=int,
                                 help="Number of decoding layers in the transformer")
        self.parser.add_argument('--dim_feedforward', default=1024, type=int,
                                 help="Intermediate size of the feedforward layers in the transformer blocks")
        self.parser.add_argument('--hidden_dim', default=256, type=int,
                                 help="Size of the embeddings (dimension of the transformer)")
        self.parser.add_argument('--dropout', default=0.1, type=float,
                                 help="Dropout applied in the transformer")
        self.parser.add_argument('--nheads', default=8, type=int,
                                 help="Number of attention heads inside the transformer's attentions")
        self.parser.add_argument('--pre_norm', action='store_true')

        # dataset parameters
        self.parser.add_argument('--output_dir', default='./results/',
                                 help='path where to save, empty for no saving')
        self.parser.add_argument('--device', default='cuda:0',
                                 help='device to use for training / testing')
        self.parser.add_argument('--seed', default=0, type=int)
        self.parser.add_argument('--resume', default='', help='resume from checkpoint')
        self.parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                                 help='start epoch')
        self.parser.add_argument('--num_workers', default=4, type=int)
        self.parser.add_argument('--eval', action='store_true')

        # PULTAD
        self.parser.add_argument('--featdim', default=1024, type=int)
        self.parser.add_argument('--pretrain', default='', help='initialized from the pre-training model')
        self.parser.add_argument('--backbone', default='resnet50', type=str,
                                 help="Name of the convolutional backbone to use")
        self.parser.add_argument('--data_root', default='./Data/CODED', type=str)
        self.parser.add_argument('--dataset_name', type=str, default="1", help='category name of the dataset')
        self.parser.add_argument('--alpha', default=0.9, type=float, help='the update weight of batch proto in the '
                                                                          'new proto memory')
        self.parser.add_argument('--loss_type', type=int, default=2, help='loss_type in [0, 1, 2], 0: CE, 1: CE+KL，'
                                                                          '2: CE(R+P)+KL')
        self.parser.add_argument('--hyp_c', default=0.7, type=float, help="the curvature of Poincare hyperbolic space")
        self.parser.add_argument('--clip_r', default=2.0, type=float, help="feature clipping radius")
        self.parser.add_argument('--k', default=0.3, type=float, help="threshold of pseudo labels")
        self.parser.add_argument('--k_learnable', default=True, type=bool, help="k is learnable or not")

    def parse(self):
        if not self.initialized:
            self.initialize()
        args = self.parser.parse_args()

        os.makedirs('%s-%s/%s' % (args.exp_name, args.dataset_name, args.output_dir), exist_ok=True)

        self.args = args
        return self.args
