import os

if __name__ == '__main__':
    # CODED dataset
    # test

    exp_num = 0
    device = 'cuda:0'
    alpha = 0.9
    num_feature_levels = 4
    seed = random.randint(1, 5000000)
    hyp_c = 0.7
    clip_r = 2.0
    k = 0.3
    k_learnable = True

    for dataset_name in range(0, 6):
        pretrain = './Exp{}-Hy-r50-{}/results/checkpoint.pth'.format(exp_num, dataset_name)

        cmd = ('python main.py --exp_name Exp{}-Hy-r50 --dataset_name 0 --device {} --hyp_c {} --clip_r {} --k {}'
               ' --k_learnable {} --seed {} --pretrain {} --eval'.format(exp_num, device, loss_type,
                                                                         hyp_c, clip_r, k,
                                                                         k_learnable,
                                                                         seed, pretrain))
        os.system(cmd)
