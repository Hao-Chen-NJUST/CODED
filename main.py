import numpy as np
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import build_model
from util import save_on_master, get_total_grad_norm, NestedTensor
from datasets import Get_dataloader
from options import TrainOptions
import random
from itertools import cycle
from tqdm import *
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

import hyptorch.pmath as pmath

import warnings

warnings.filterwarnings("ignore")

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')


def main():
    args = TrainOptions().parse()

    with open("./%s-%s/args.log" % (args.exp_name, args.dataset_name), "a") as args_log:
        for k, v in sorted(vars(args).items()):
            print('%s: %s ' % (str(k), str(v)))
            args_log.write('%s: %s \n' % (str(k), str(v)))

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

    model = build_model(args)
    model.to(device)
    criterions = [nn.CrossEntropyLoss(),
                  nn.KLDivLoss(reduction='batchmean'),
                  ]

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    train_dataloader, train_unlabelled_dataloader, test_dataloader = Get_dataloader(args)
    
    dataloader_len = max(len(train_dataloader), len(train_unlabelled_dataloader))
    if train_dataloader.__len__() > train_unlabelled_dataloader.__len__():
        train_unlabelled_dataloader = cycle(train_unlabelled_dataloader)
    elif train_dataloader.__len__() < train_unlabelled_dataloader.__len__():
        train_dataloader = cycle(train_dataloader)

    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, args.lr_backbone_names) and not
                 match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_scale,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if
                       match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone * args.lr_scale,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if
                       match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult * args.lr_scale,
        }
    ]
    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    output_dir = '%s-%s/%s' % (args.exp_name, args.dataset_name, args.output_dir)
    best_auroc = 0.0
    proto_memory = torch.zeros([1, args.hidden_dim], device=device, requires_grad=False)

    if args.pretrain:
        print('Initialized from the pre-training model')
        checkpoint = torch.load(args.pretrain, map_location='cpu')
        state_dict = checkpoint['model']
        msg = model_without_ddp.load_state_dict(state_dict, strict=False)
        print(msg)
        args.start_epoch = checkpoint['epoch'] + 1
        best_auroc = checkpoint['best_auroc']
        proto_memory = checkpoint['proto_memory'].to(device).detach()

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            print(optimizer.param_groups)
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                print(
                    'Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override '
                    'lr_drop in resumed lr_scheduler.')
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1
            best_auroc = checkpoint['best_auroc']
            proto_memory = checkpoint['proto_memory'].to(device).detach()
        if args.eval:
            img_auroc = evaluation(args, model, test_dataloader, device, proto_memory)
            print('Image AUROC:%f' % img_auroc)
            return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        avg_loss = 0
        total = 0
        model.train()

        avg_loss, total, proto_memory = train_with_un(args, model, optimizer, train_dataloader,
                                                      train_unlabelled_dataloader, criterions, epoch, avg_loss,
                                                      total, device, dataloader_len, proto_memory)

        lr_scheduler.step()

        print("Start evaluation on test set!")
        test_start_time = time.time()
        img_auroc, ap = evaluation(args, model, test_dataloader, device, proto_memory)
        test_time = time.time() - test_start_time
        test_time_str = str(datetime.timedelta(seconds=int(test_time)))

        if args.output_dir:
            if img_auroc > best_auroc:
                best_auroc = img_auroc
                checkpoint_path = output_dir + '/best_auroc.pth'
                save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'best_auroc': best_auroc,
                    'proto_memory': proto_memory,
                }, checkpoint_path)

            checkpoint_paths = [output_dir + '/checkpoint.pth']

            for checkpoint_path in checkpoint_paths:
                save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'best_auroc': best_auroc,
                    'proto_memory': proto_memory,
                }, checkpoint_path)

        print('[Epoch%d]-[Loss:%f]-[image_AUROC:%f]-[best_AUROC:%f]' %
              (epoch + 1, avg_loss / total, img_auroc, best_auroc))

        print('Test time {}'.format(test_time_str))

        with open("./%s-%s/args.log" % (args.exp_name, args.dataset_name), "a") as train_log:
            train_log.write("\r[Epoch%d]-[Loss:%f]-[image_AUROC:%f]-[best_AUROC:%f]" %
                            (epoch + 1, avg_loss / total, img_auroc, best_auroc))
            train_log.write('Test time {}'.format(test_time_str))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    with open("./%s-%s/args.log" % (args.exp_name, args.dataset_name), "a") as train_log:
        train_log.write('Training time {}'.format(total_time_str))


def train_with_un(args, model, optimizer, train_dataloader, train_unlabelled_dataloader, criterion, epoch, avg_loss,
                  total, device, dataloader_len, proto_memory):
    for i, data in enumerate(zip(train_dataloader, train_unlabelled_dataloader)):
        inputs = NestedTensor(torch.cat((data[0][0].tensors, data[1][0].tensors), 0),
                              torch.cat((data[0][0].mask, data[1][0].mask), 0))
        label_pos = torch.as_tensor(data[0][1])
        inputs = inputs.to(device)
        label_pos = label_pos.to(device)
        logits, proto_memory, hs, resnet_fea, p_label_un = model(inputs, proto_memory, args.alpha)

        if logits.size().__len__() == 2:
            assert logits.size(0) == args.batch_size * 2
            # if args.loss_type == 0:
            #     losses = criterion[0](logits[:args.batch_size], label_pos)
            # elif args.loss_type == 1:
            #     losses = criterion[0](logits[:args.batch_size], label_pos) + \
            #              criterion[1](F.log_softmax(hs[args.batch_size:]), F.softmax(resnet_fea[args.batch_size:]))
            # elif args.loss_type == 2:
            #     losses = criterion[0](logits, torch.cat([label_pos, p_label_un])) + \
            #              criterion[1](F.log_softmax(hs[args.batch_size:]), F.softmax(resnet_fea[args.batch_size:]))
            if p_label_un is not None:
                losses = criterion[0](logits, torch.cat([label_pos, p_label_un])) + \
                         criterion[1](F.log_softmax(hs[args.batch_size:]), F.softmax(resnet_fea[args.batch_size:]))
            else:
                losses = criterion[0](logits[:args.batch_size], label_pos) + \
                         criterion[1](F.log_softmax(hs[args.batch_size:]), F.softmax(resnet_fea[args.batch_size:]))

        optimizer.zero_grad()
        losses.backward()

        if args.clip_max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
        else:
            grad_total_norm = get_total_grad_norm(model.parameters(), 0)
        optimizer.step()
        avg_loss += losses.item() * inputs.tensors.size(0)
        total += inputs.tensors.size(0)
        print(("\r[Epoch%d/%d]-[Batch%d/%d]-[Loss:%f]-[Grad Norm:%f]" %
               (epoch + 1, args.epochs, i + 1, dataloader_len, avg_loss / total, grad_total_norm.item())))
    return avg_loss, total, proto_memory

def evaluation(args, model, test_dataloader, device, proto_memory):
    model.eval()
    gt_list = []
    pred_list = []
    for i, (batch, gt, filename) in enumerate(tqdm(test_dataloader)):
        with torch.no_grad():
            inputs = batch.to(device)
            outputs, _, hs, _ = model(inputs, proto_memory, args.unlabelled)
            gt_list.append(torch.as_tensor(gt))
            pred_list.append(outputs.cpu())

    # calculate image-level ROC AUC score
    pred_list = torch.cat(pred_list, dim=0)
    gt_list = torch.cat(gt_list, dim=0)
    pred_logits = pred_list.cpu().max(dim=1)[0]
    gt_list = gt_list.numpy()
    fpr, tpr, _ = roc_curve(gt_list, pred_logits)
    img_roc_auc = roc_auc_score(gt_list, pred_logits)
    return img_roc_auc, ap


if __name__ == '__main__':
    main()
