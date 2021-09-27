import numpy as np
import matplotlib.pyplot as plt
import json


def _loss_plot(ax, title, e_train, l_train, e_valid, l_valid):
    ax.plot(e_train, l_train, '-b', label='training loss')
    ax.plot(e_valid, l_valid, '-r', label='validation loss')
    ax.set_xlabel('epoch')
    ax.legend()
    ax.grid(True)
    ax.set_xticks(list(range(0, e_train[-1], 20)))
    ax.set_title(title)


# parameters
# json path： plot loss curves
TRAIN_JSON_FILE_PTH = './saves/saves_debug/summary_train_debug.json'
VALID_JSON_FILE_PTH = './saves/saves_debug/summary_valid_debug.json'

if __name__ == '__main__':
    sav_train, sav_valid = None, None
    with open(f"{TRAIN_JSON_FILE_PTH}") as f:
        sav_train = json.load(f)
    with open(f"{VALID_JSON_FILE_PTH}") as f:
        sav_valid = json.load(f)
    
    # transform dictionary to list for pyplot
    train_losses_list = {'loss': [], 'KLD': [], 'AELoss': [], 'BIASLoss': [], 'BESTMse':[] }
    valid_losses_list = {'loss': [], 'KLD': [], 'AELoss': [], 'BIASLoss': [], 'BESTMse':[] }
    epoch_train = list(sav_train.keys())
    epoch_valid = list(sav_valid.keys())
    
    # accumlate dictionary
    for epoch in epoch_train:
        for m_key in train_losses_list:
            train_losses_list[m_key].append(sav_train[epoch][m_key])
            
    for epoch in epoch_valid:
        for m_key in valid_losses_list:
            valid_losses_list[m_key].append(sav_valid[epoch][m_key])
    
    # plot all sub losses
    fig, ax = plt.subplots(nrows=len(train_losses_list.keys()), ncols=1, figsize=(25,25))    
    set_types = [train_losses_list, valid_losses_list]
    for row, loss_type in enumerate(train_losses_list.keys()):
        _loss_plot(ax[row], loss_type, 
                  list(map(int, epoch_train)), train_losses_list[loss_type], 
                  list(map(int, epoch_valid)), valid_losses_list[loss_type])    
    
    # plot the selected loss especially if necessary
    for loss_type in train_losses_list.keys():
        fig, ax = plt.subplots()
        _loss_plot(ax, loss_type, 
                  list(map(int, epoch_train)), train_losses_list[loss_type], 
                  list(map(int, epoch_valid)), valid_losses_list[loss_type])