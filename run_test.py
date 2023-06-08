import os

if __name__ == '__main__':
    # CODED dataset
    # test

    cmd = 'python main.py --exp_name Exp0-r50 --dataset_name 0 --pretrain ./Exp0-r50-0/results/best_auroc.pth --eval'
    os.system(cmd)
    cmd = 'python main.py --exp_name Exp0-r50 --dataset_name 1 --pretrain ./Exp0-r50-1/results/best_auroc.pth --eval'
    os.system(cmd)
    cmd = 'python main.py --exp_name Exp0-r50 --dataset_name 2 --pretrain ./Exp0-r50-2/results/best_auroc.pth --eval''
    os.system(cmd)
    cmd = 'python main.py --exp_name Exp0-r50 --dataset_name 3 --pretrain ./Exp0-r50-3/results/best_auroc.pth --eval''
    os.system(cmd)
    cmd = 'python main.py --exp_name Exp0-r50 --dataset_name 4 --pretrain ./Exp0-r50-4/results/best_auroc.pth --eval''
    os.system(cmd)
    cmd = 'python main.py --exp_name Exp0-r50 --dataset_name 5 --pretrain ./Exp0-r50-5/results/best_auroc.pth --eval''
    os.system(cmd)
