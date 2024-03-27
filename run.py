import os

if __name__ == '__main__':
    # CODED dataset
    # train

    exp_num = 0
    device = 'cuda:0'
    alpha = 0.9
    num_feature_levels = 4
    seed = random.randint(1, 5000000)
    hyp_c = 0.7
    clip_r = 2.0
    k = 0.3
    k_learnable = True
    pretrain = './Exp{}-Hy-r50-0/results/checkpoint.pth'.format(exp_num)

    # cmd = ('python main.py --exp_name Exp{}-Hy-r50 --dataset_name 0 --device {} --loss_type {} --hyp_c {} --clip_r {}'
    #        ' --k {} --k_learnable {} --seed {} --pretrain {}'.format(exp_num, device, loss_type,
    #                                                                  hyp_c, clip_r, k,
    #                                                                  k_learnable,
    #                                                                  seed, pretrain))
    # os.system(cmd)
    for dataset_name in range(0, 6):
        cmd = ('python main.py --exp_name Exp{}-Hy-r50 --dataset_name {} --device {}  --alpha {} --loss_type {} '
               '--hyp_c {} --clip_r {} --k {} --k_learnable {} --seed {} '
               '--num_feature_levels {}'.format(exp_num, dataset_name, device, alpha, loss_type, hyp_c, clip_r,
                                                k, k_learnable, seed, num_feature_levels))
        os.system(cmd)
