import pandas as pd 
import numpy as np
import public_timeseries_testing_util as optiver2023
from torch.nn.utils.rnn import pack_padded_sequence, pack_sequence, unpack_sequence, unpad_sequence
import torch
from tqdm.notebook import trange,tqdm
import torch.nn as nn 
import torch.optim as optim
import wandb
import torch_classes
from model_saver import model_saver_wandb as model_saver

def train_model(trading_df:torch_classes.TradingData, model:torch_classes.GRUNet, config:dict, optimizer, criterion):
    example_ct = 0
    epochs = 10000
    num_batches = len(trading_df.train_batches)-2
    reg_L1 = nn.L1Loss()
    model = model.to('cuda:0')
    trading_df.reset_hidden(config['hidden_size'])
    for epoch in trange(epochs):
        model.train()
        loss_list = []
        
        for i in range(0,384):
            # print(i)

            stocks = [trading_df.stocksDict[x] for x in trading_df.stock_batches[i]] #Stocks for the Day
            stock_data = trading_df.train_batches

            example_ct+=1

            X = trading_df.packed_x[i]
            Y = trading_df.packed_y[i].data

            hidden_in = torch.stack([x.hidden for x in stocks]).transpose(0,1)

            output,hidden = model(X,hidden_in)
            hidden = hidden.transpose(0,1)
            output  = torch.flatten(output)

            [setattr(obj, 'hidden', val.detach()) for obj, val in zip(stocks,hidden)]

            loss = criterion(output,Y)
            L1_loss = reg_L1(output,Y)

            loss.backward()
            loss_list.append((i,loss))
            optimizer.step()

            if i == 0:
                epoch_loss = loss
                epoch_reg_l1 = L1_loss
            else:
                if loss.isnan():
                    pass
                epoch_loss = loss+epoch_loss
                epoch_reg_l1 = L1_loss+epoch_reg_l1

            wandb.log({"loss_1": torch.mean(loss).item()})
            trading_df.detach_hidden()
        wandb.log({"epoch_loss": epoch_loss/384, "epoch_l1_loss": epoch_reg_l1/384, 'epoch':epoch})
        # print({"epoch_loss": epoch_loss/384, "epoch_l1_loss": epoch_reg_l1/384, 'epoch':epoch})
        validate_model(trading_df,model,criterion,epoch)
        if epoch%10==0:
            trading_df.create_hidden_states_dict_v2()
            model_saver(model,optimizer,epoch,0,0,trading_df.train_hidden_dict)
        trading_df.reset_hidden(hidden_size=config['hidden_size'])
        # print(epoch_loss)
        # print(loss_list)
              

@torch.no_grad()          
def validate_model(trading_df:torch_classes.TradingData,model:torch_classes.GRUNet,criterion,epoch,):
    model.eval()
    val_batches = trading_df.packed_val_x
    len_val = len(val_batches)
    reg_L1 = nn.L1Loss()
    loss_list = []

    # print(len_val)
    output_dict = {}
    output_dict['stock'] = []
    output_dict['day'] = []
    output_dict['id'] = []
    output_dict['target'] = []
    output_dict['pred'] = []

    for i in range(0,len_val-1):
        # print(i)
        stocks = [trading_df.stocksDict[x] for x in trading_df.val_stock_batches[i]] 
         
        X = trading_df.packed_val_x[i]
        Y = trading_df.packed_val_y[i].data

        hidden_in = torch.stack([x.hidden for x in stocks]).transpose(0,1)

        output,hidden = model(X,hidden_in)
        hidden = hidden.transpose(0,1)
        output  = torch.flatten(output)

        [setattr(obj, 'hidden_test', val.detach()) for obj, val in zip(stocks,hidden)]

        loss = criterion(output,Y)
        L1_loss = reg_L1(output,Y)

        if i == 0:
            epoch_loss = loss
            epoch_reg_l1 = L1_loss
        else:
            if loss.isnan():
                pass
            epoch_loss = loss+epoch_loss
            epoch_reg_l1 = L1_loss+epoch_reg_l1

        output_dict['stock'].append()
        output_dict['day'].append()
        output_dict['id'].append()
        output_dict['target'].append()
        output_dict['pred'].append()


        

    wandb.log({"val_epoch_loss": epoch_loss/len_val, "val_epoch_loss_l1": epoch_reg_l1/len_val,'epoch':epoch})
    # print({"epoch_loss": epoch_loss/384, "epoch_l1_loss": epoch_reg_l1/384, 'epoch':epoch})
