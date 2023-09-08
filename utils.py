from pandas import read_csv
import numpy as np

def data_loader(data_path):
    '''
    return X_data, M_data, col_list, times
    '''
    df = read_csv(data_path)
    col_list = df.columns.to_list()
    times = df.iloc[:,0].values

    X_data = df.iloc[:,1:].values
    M_data = np.ones_like(X_data)
    M_data[np.isnan(X_data)] = 0

    return X_data, M_data, col_list, times

def precess_conv_data(data_x, data_m, ws):
    '''
    create Conv-GAIN train dataset
    input:
        data_x:[no, dim]
        data_m:[no, dim]
        ws: size of rolling windows
    out:
        feature image:[no, 2, ws*2+1, dim]
    '''
    no, dim = data_x.shape
    no, dim = data_x.shape
    data = np.ones((no, 2, ws*2+1, dim))
    for i in range(no):
        if(i<ws):
            data[i, 0, :ws-i, :] = -1
            data[i, 0, ws-i:, :] = data_x[:i+ws+1]
            data[i, 1, :ws-i, :] = -1
            data[i, 1, ws-i:, :] = data_m[:i+ws+1]
            
        elif(i>=no-ws):
            data[i, 0, :no-i+ws, :] = data_x[i-ws:]
            data[i, 0, no-i+ws:, :] = -1
            data[i, 1, :no-i+ws, :] = data_m[i-ws:]
            data[i, 1, no-i+ws:, :] = -1

        else:
            data[i, 0, :, :] = data_x[i-ws:i+ws+1]
            data[i, 1, :, :] = data_m[i-ws:i+ws+1]

    return data

def precess_bi_data(data_x, ws):
    '''
    use missed_data to creat supervise data by rooling window
    input:
        data: (raw_data, missed_data, m_data)
    return :
        x: (no-window_size, window_size, dim)
        y: (no-window_size, 2, dim) 2 is raw_data + m_data
    '''
    no, dim = data_x.shape

    data = np.ones((no, ws*2+1, dim))
    for i in range(no):
        if(i<ws):
            data[i, :ws-i, :] = -1
            data[i, ws-i:, :] = data_x[:i+ws+1]
        elif i>=no-ws:
            data[i, :no-i+ws, :] = data_x[i-ws:]
            data[i, i-no:, :] = -1
        else:
            data[i, :, :] = data_x[i-ws:i+ws+1]

    return data

def normalization (data):
    '''Normalize data in [0, 1] range.
    
    Args:
        - data: original data
    
    Returns:
        - norm_data: normalized data
        - norm_parameters: min_val, max_val for each feature for renormalization
    '''

    # Parameters
    _, dim = data.shape
    norm_data = data.copy()

    min_val = np.zeros(dim)
    max_val = np.zeros(dim)
    # Normalize for each dimension
    for i in range(dim):
        min_val[i] = np.nanmin(norm_data[:,i])
        norm_data[:,i] = norm_data[:,i] - np.nanmin(norm_data[:,i])
        max_val[i] = np.nanmax(norm_data[:,i])
        norm_data[:,i] = norm_data[:,i] / (max_val[i] + 1e-6)   
    # Return norm_parameters for renormalization
    norm_parameters = {'min_val': min_val,
                    'max_val': max_val}  
        
    return norm_data, norm_parameters

def renormalization (norm_data, norm_parameters):
    '''Renormalize data from [0, 1] range to the original range.
    
    Args:
        - norm_data: normalized data
        - norm_parameters: min_val, max_val for each feature for renormalization
    
    Returns:
        - renorm_data: renormalized original data
    '''
    
    min_val = norm_parameters['min_val']
    max_val = norm_parameters['max_val']

    _, dim = norm_data.shape
    renorm_data = norm_data.copy()
        
    for i in range(dim):
        renorm_data[:,i] = renorm_data[:,i] * (max_val[i] + 1e-6)   
        renorm_data[:,i] = renorm_data[:,i] + min_val[i]
        
    return renorm_data

def rounding (imputed_data, data_x):
    '''Round imputed data for categorical variables.
    
    Args:
        - imputed_data: imputed data
        - data_x: original data with missing values
        
    Returns:
        - rounded_data: rounded imputed data
    '''
    
    _, dim = data_x.shape
    rounded_data = imputed_data.copy()
    
    for i in range(dim):
        temp = data_x[~np.isnan(data_x[:, i]), i]
        # Only for the categorical variable
        if len(np.unique(temp)) < 20:
            rounded_data[:, i] = np.round(rounded_data[:, i])
        
    return rounded_data

def rmse_loss (ori_data, imputed_data, data_m):
    '''Compute RMSE loss between ori_data and imputed_data
    
    Args:
        - ori_data: original data without missing values
        - imputed_data: imputed data
        - data_m: indicator matrix for missingness
        
    Returns:
        - rmse: Root Mean Squared Error
    '''
    
    ori_data, norm_parameters = normalization(ori_data)
    imputed_data, _ = normalization(imputed_data, norm_parameters)

    nominator = np.sum(((1-data_m) * ori_data - (1-data_m) * imputed_data)**2)
    denominator = np.sum(1-data_m)
    
    rmse = np.sqrt(nominator/float(denominator))
    
    return rmse