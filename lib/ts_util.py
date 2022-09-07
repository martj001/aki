import pandas as pd
import numpy as np
import time

import torch
import torch.nn.functional as F

import lib.config as cfg

def get_ts_by_pid(df_train: pd.DataFrame, pid, model_config: dict):
    col_to_drop = model_config['col_to_drop']
    col_label = model_config['col_label']
    col_regres = model_config['col_regres']
    max_ts_length = model_config['max_ts_length']
    
    df_temp = df_train[df_train['patientunitstayid'] == pid]
    ts_length = df_temp.shape[0]

    if ts_length > max_ts_length:
        sample_space = range(ts_length - max_ts_length)
        tsid_start = np.random.choice(sample_space, 1)[0]
        tsid_end = tsid_start + max_ts_length
        
        df_temp = df_temp.iloc[tsid_start:tsid_end]
        
    df_data = df_temp.drop(col_to_drop, axis=1)
    df_label = df_temp[col_label[1:]] # remove gt_column
    df_regres = df_temp[col_regres]

    tensor_data = torch.tensor(df_data.astype(np.float32).values)
    tensor_data.requires_grad = True
    tensor_label = torch.tensor(df_label.astype(np.float32).values)
    tensor_regres = torch.tensor(df_regres.astype(np.float32).values)
    
    return tensor_data, tensor_label, tensor_regres


def build_fixed_length_ts(tensor_values: torch.tensor, max_ts_length: int):
    ts_length = tensor_values.shape[0]
    assert(ts_length <= max_ts_length)
    
    tensor_values = tensor_values.unsqueeze(1)

    n_ts_pad = (max_ts_length - ts_length)
    tensor_data = F.pad(input=tensor_values, pad=(0, 0, 0, 0, 0, n_ts_pad), mode='constant', value=0)
    
    return tensor_data


def get_ts_by_pid_test(
    df_data: pd.DataFrame, 
    df_train: pd.DataFrame, 
    pid: int, 
    model_config: dict
):
    col_to_drop = model_config['col_to_drop']
    col_label = model_config['col_label']
    col_regres = model_config['col_regres']
    max_ts_length = model_config['max_ts_length']
    
    df_temp = df_data[df_data['patientunitstayid'] == pid]
    ts_length = df_temp.shape[0]
    
    # Imputation + Normalize
    # Missing value impute with -1
    df_temp = df_temp.fillna(-1)

    # Normalization
    col_lab_min = list(map(lambda x: x+'_min', cfg.selected_lab))
    col_lab_max = list(map(lambda x: x+'_max', cfg.selected_lab))
    col_require_norm = col_lab_min + col_lab_max + col_regres

    for col in col_require_norm:
        v_mean = np.mean(df_train[col])
        v_std = np.std(df_train[col])

        df_temp[col] = (df_temp[col] - v_mean)/v_std

    if ts_length > max_ts_length:
        sample_space = range(ts_length - max_ts_length)
        tsid_start = np.random.choice(sample_space, 1)[0]
        tsid_end = tsid_start + max_ts_length
        
        df_temp = df_temp.iloc[tsid_start:tsid_end]
        
    df_data = df_temp.drop(col_to_drop, axis=1)
    df_label = df_temp[col_label[1:]] # remove gt_column
    df_regres = df_temp[col_regres]

    tensor_data = torch.tensor(df_data.astype(np.float32).values)
    tensor_data.requires_grad = True
    tensor_label = torch.tensor(df_label.astype(np.float32).values)
    tensor_regres = torch.tensor(df_regres.astype(np.float32).values)
    
    return tensor_data

