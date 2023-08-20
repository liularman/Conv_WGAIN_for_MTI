from pandas import read_csv
import numpy as np
import pickle
import torch
import random
from datetime import datetime

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

def pkl_save(name, var):
    with open(name, 'wb') as f:
        pickle.dump(var, f)

def pkl_load(name):
    with open(name, 'rb') as f:
        return pickle.load(f)
    
def torch_pad_nan(arr, left=0, right=0, dim=0):
    if left > 0:
        padshape = list(arr.shape)
        padshape[dim] = left
        arr = torch.cat((torch.full(padshape, np.nan), arr), dim=dim)
    if right > 0:
        padshape = list(arr.shape)
        padshape[dim] = right
        arr = torch.cat((arr, torch.full(padshape, np.nan)), dim=dim)
    return arr
    
def pad_nan_to_target(array, target_length, axis=0, both_side=False):
    assert array.dtype in [np.float16, np.float32, np.float64]
    pad_size = target_length - array.shape[axis]
    if pad_size <= 0:
        return array
    npad = [(0, 0)] * array.ndim
    if both_side:
        npad[axis] = (pad_size // 2, pad_size - pad_size//2)
    else:
        npad[axis] = (0, pad_size)
    return np.pad(array, pad_width=npad, mode='constant', constant_values=np.nan)

def split_with_nan(x, sections, axis=0):
    assert x.dtype in [np.float16, np.float32, np.float64]
    arrs = np.array_split(x, sections, axis=axis)
    target_length = arrs[0].shape[axis]
    for i in range(len(arrs)):
        arrs[i] = pad_nan_to_target(arrs[i], target_length, axis=axis)
    return arrs

def take_per_row(A, indx, num_elem):
    all_indx = indx[:,None] + np.arange(num_elem)
    return A[torch.arange(all_indx.shape[0])[:,None], all_indx]

def centerize_vary_length_series(x):
    prefix_zeros = np.argmax(~np.isnan(x).all(axis=-1), axis=1)
    suffix_zeros = np.argmax(~np.isnan(x[:, ::-1]).all(axis=-1), axis=1)
    offset = (prefix_zeros + suffix_zeros) // 2 - prefix_zeros
    rows, column_indices = np.ogrid[:x.shape[0], :x.shape[1]]
    offset[offset < 0] += x.shape[1]
    column_indices = column_indices - offset[:, np.newaxis]
    return x[rows, column_indices]

def data_dropout(arr, p):
    B, T = arr.shape[0], arr.shape[1]
    mask = np.full(B*T, False, dtype=np.bool)
    ele_sel = np.random.choice(
        B*T,
        size=int(B*T*p),
        replace=False
    )
    mask[ele_sel] = True
    res = arr.copy()
    res[mask.reshape(B, T)] = np.nan
    return res

def name_with_datetime(prefix='default'):
    now = datetime.now()
    return prefix + '_' + now.strftime("%Y%m%d_%H%M%S")

def init_dl_program(
    device_name,
    seed=None,
    use_cudnn=True,
    deterministic=False,
    benchmark=False,
    use_tf32=False,
    max_threads=None
):
    import torch
    if max_threads is not None:
        torch.set_num_threads(max_threads)  # intraop
        if torch.get_num_interop_threads() != max_threads:
            torch.set_num_interop_threads(max_threads)  # interop
        try:
            import mkl
        except:
            pass
        else:
            mkl.set_num_threads(max_threads)
        
    if seed is not None:
        random.seed(seed)
        seed += 1
        np.random.seed(seed)
        seed += 1
        torch.manual_seed(seed)
        
    if isinstance(device_name, (str, int)):
        device_name = [device_name]
    
    devices = []
    for t in reversed(device_name):
        t_device = torch.device(t)
        devices.append(t_device)
        if t_device.type == 'cuda':
            assert torch.cuda.is_available()
            torch.cuda.set_device(t_device)
            if seed is not None:
                seed += 1
                torch.cuda.manual_seed(seed)
    devices.reverse()
    torch.backends.cudnn.enabled = use_cudnn
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
    
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = use_tf32
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
        
    return devices if len(devices) > 1 else devices[0]

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