import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import torch

import lib.ts_util as ts_util

def get_model_pred(
    model, 
    model_config, 
    df_data: pd.DataFrame, 
    df_train: pd.DataFrame, 
    pid=-1,
    max_ts_length=128,
):
    pid_list = np.unique(df_train['patientunitstayid'])

    if pid == -1:
        pid = np.random.choice(pid_list, 1)[0]

    list_tensor_data = []
    
    assert(pid in pid_list)

    tensor_data = ts_util.get_ts_by_pid_test(df_data, df_train, pid, model_config)
    tensor_data = ts_util.build_fixed_length_ts(tensor_data, max_ts_length)
    list_tensor_data.append(tensor_data)

    tensor_data_agg = torch.cat(list_tensor_data, dim=1)

    tensor_data_agg = tensor_data_agg.cuda()

    model.zero_grad()
    model_out = model(tensor_data_agg)
    
    return pid, model_out.cpu().detach().squeeze().numpy()


def timelines(arr_label, y, color, legend):
    plt.hlines(y, -0.01, 0, color, lw=4, label = legend)
    for i in range(len(arr_label)):
        if arr_label[i]:
            xstart=i-0.5
            xstop=i+0.5
            if legend == 'Prediction':
                plt.hlines(y, xstart, xstop, color, lw=500, alpha= 0.2)
            elif legend == 'Ground truth':
                plt.hlines(y, xstart, xstop, color, lw=200, alpha= 0.2)
            else :
                plt.hlines(y, xstart, xstop, color, lw=4, alpha= 1)
                
                
                