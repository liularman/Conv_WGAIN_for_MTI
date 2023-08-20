from utils import data_loader
from train_model import train
from pandas import DataFrame
import numpy as np
      

def impute_data(device, n_critic, iterations, data_path, save_path, args=None):
    parameters = {'device': device,
                'n_critic': n_critic,
                'iterations': iterations}

    X_data, M_data, col_list, times = data_loader(data_path)
    imputed_data = train(X_data, M_data, parameters, args)
    new_data = np.concatenate((times.reshape(-1, 1), imputed_data), 1)

    DataFrame(new_data, columns=col_list).to_csv(save_path, index=False)

if __name__ == '__main__':
    device = 'cuda'
    n_critic=1     # 1~5
    iterations = 1000
    data_path = 'data/raw_data.csv'
    save_path = 'data/imputed_data.csv'
    impute_data(device, n_critic, iterations, data_path, save_path)