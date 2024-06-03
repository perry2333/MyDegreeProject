def POT_calculate(m = 0.8, nb_dummies = 10):
    import numpy as np  # always need it
    import ot  # ot
    import time
    import os
    os.environ['CXX'] = 'g++-8'
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    import scipy as sp
    import matplotlib.pylab as pl
    import matplotlib.pyplot as plt
    from scipy.io import savemat
    from scipy.io import loadmat
    from scipy.sparse import random
    from mpl_toolkits.mplot3d import Axes3D
    
    # 读取 .mat 文件
    Pointcld_1 = loadmat('PointCloud_pure_square1.mat');
    Pointcld_2 = loadmat('PointCloud_pure_square2.mat');
    ptcld_1 = Pointcld_1['PointCloud'];
    ptcld_2 = Pointcld_2['PointCloud'];
    
    xs = abs(ptcld_1[:,3]);#是根据能量大小进行传输还是根据幅值大小进行传输，能量就在后面**2
    xt = abs(ptcld_2[:,3]);
    cld1_pos = ptcld_1[:,0:3];
    cld2_pos = ptcld_2[:,0:3];

    total = np.sum(xs)
    normalized_xs = xs/total
    total = np.sum(xt)
    normalized_xt = xt/total

    M = ot.dist(cld1_pos,cld2_pos)#使用欧几里得距离计算代价矩阵

    gamma, log_emd= ot.partial.partial_wasserstein(normalized_xs, normalized_xt, M, m, nb_dummies, log = True)#进行部分最优传输计算

    # 准备要保存的数据
    data_to_save = {
        'cost': log_emd['cost'],
        'u': log_emd['u'],
        'v': log_emd['v']
    }

    # 保存数据到 .mat 文件
    savemat('log_emd.mat', {'log_emd': data_to_save})
    savemat('T.mat',{'T': gamma})