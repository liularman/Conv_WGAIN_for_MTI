import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from utils import normalization, renormalization, precess_conv_data, rounding
from model.Conv_WGAIN import Conv_GAIN_G, Conv_GAIN_D


def train(data_x, data_m, parameters, args=None):

    batch_size = 128
    window_size = 8
    alpha = 50
    hidden_channel = 8
    hint_rate = 0.9
    iterations = parameters['iterations']
    n_critic = parameters['n_critic']
    device = parameters['device']

    no, dim = data_m.shape

    # Preprocessing
    norm_data, norm_parameters = normalization(data_x)
    norm_data = np.nan_to_num(norm_data, 0)
    # Missing values are preprocessed with Gaussian noise
    Z_mb = np.random.uniform(0, 0.1, size = [no, dim])
    norm_data = norm_data * data_m + Z_mb * (1-data_m)

    # Build an imputed feature map
    data = precess_conv_data(norm_data, data_m, window_size)

    data = torch.Tensor(data)
    dataset = TensorDataset(
        data,
        torch.Tensor(norm_data),
        torch.Tensor(data_m)
        )
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size)

    kn1 = (5, dim)
    kn2 = (3, 1)
    pool_size = (2, 1)

    G_tor = Conv_GAIN_G(kn1, kn2, hidden_channel, pool_size, dim).to(device)
    D_tor = Conv_GAIN_D(dim, dim).to(device)

    # 优化器
    G_solver = torch.optim.RMSprop(G_tor.parameters(), lr=0.0001)
    D_solver = torch.optim.RMSprop(D_tor.parameters(), lr=0.0001)

    for it in range(iterations):
            
        for ii, (X_g, X_mb, M_mb) in enumerate(data_loader):
            X_g = X_g.to(device)
            X_mb = X_mb.to(device)
            M_mb = M_mb.to(device)

            no_size = X_mb.shape[0]
            
            # clip param for D
            for parm in D_tor.parameters():
                parm.data.clamp_(-0.01, 0.01)

            G_sample = G_tor(X_g)
            Hat_X = X_mb * M_mb + G_sample * (1-M_mb)

            H_mb_temp = torch.tensor(1*(np.random.uniform(0., 1., size=[no_size, dim]) < hint_rate), dtype=torch.float32, device=device)
            H_mb = M_mb * H_mb_temp + 0.5*(1-H_mb_temp)

            D_prob = D_tor(Hat_X, H_mb)

            # ***优化D
            D_solver.zero_grad()
            D_loss = torch.mean((1-M_mb) * (D_prob + 1e-8) - M_mb * (D_prob + 1e-8))
            D_loss.backward(retain_graph=True)
            
            
            if (ii+1)%n_critic == 0:
                # ***优化G
                G_solver.zero_grad()
                # 计算G损失
                G_loss_temp = -torch.mean((1-M_mb) * (D_prob + 1e-8))
                MSE_loss = torch.mean((M_mb * X_mb - M_mb * G_sample)**2) / torch.mean(M_mb)
                G_loss = G_loss_temp + alpha * MSE_loss
                G_loss.backward()
                D_solver.step()
                G_solver.step()
            else:
                D_solver.step()

        # 更新当前进度信息
        if args is not None:
            # 计算当前进度
            progress = (it+1)*100//iterations
            args.set_progress(progress)


    imputed_data = G_tor(data.to(device))
    imputed_data = data_m * norm_data + (1-data_m) * imputed_data.detach().cpu().numpy()
    # 归一化转换
    imputed_data = renormalization(imputed_data, norm_parameters)  
    # 将离散值进行估计
    imputed_data = rounding(imputed_data, data_x)

    return imputed_data
