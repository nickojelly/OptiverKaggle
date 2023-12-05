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

def custom_MSE(x,y):
    return (((x-y)+1)**2).mean()

def train_model(trading_df:torch_classes.TradingData, model:torch_classes.GRUNet, config:dict, optimizer, criterion):
    example_ct = 0
    epochs = config['epochs']
    setup_loss = 1
    num_batches = len(trading_df.train_batches)-2
    reg_L1 = nn.L1Loss()
    model = model.to('cuda:0')
    trading_df.reset_hidden(config['hidden_size'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True,factor=0.5)
    for epoch in trange(epochs):
        model.train()
        loss_list = []
        setup_loss = 1
        
        for i in range(0,384):
            # print(i)

            stocks = [trading_df.stocksDict[x] for x in trading_df.stock_batches[i]] #Stocks for the Day
            stock_data = trading_df.train_batches

            example_ct+=1

            X = trading_df.packed_x[i]
            Y = trading_df.packed_y[i].data
            Y_price_ask = trading_df.packed_ask_price_daily[i].data
            Y_price_bid = trading_df.packed_bid_price_daily[i].data
            Y_price_wap = trading_df.packed_wap_price_daily[i].data

            hidden_in = torch.stack([x.hidden for x in stocks]).transpose(0,1)

            output_ask,output_bid,output_wap,hidden,_ = model(X,hidden_in)
            # output_wap,hidden,_ = model(X,hidden_in)
            hidden = hidden.transpose(0,1)
            output_ask  = torch.flatten(output_ask)
            output_bid  = torch.flatten(output_bid)
            output_wap  = torch.flatten(output_wap)

            [setattr(obj, 'hidden', val.detach()) for obj, val in zip(stocks,hidden)]
            
            # print(f"{output.shape=}")
            
            loss_ask = criterion(output_ask,Y_price_ask)
            loss_bid = criterion(output_bid,Y_price_bid)
            loss_wap = criterion(output_wap,Y_price_wap)

            loss = loss_ask + loss_bid + loss_wap
            # loss = loss_ask

            L1_loss_ask = reg_L1(output_ask,Y_price_ask).detach()
            L1_loss_bid = reg_L1(output_bid,Y_price_bid).detach()
            L1_loss_wap = reg_L1(output_wap,Y_price_wap).detach()

            L1_loss = (L1_loss_ask +L1_loss_bid + L1_loss_wap)/3
            # L1_loss = L1_loss_wap

            wandb.log({"loss_ask": torch.mean(loss_ask).item(), })
                    #    "loss_bid": torch.mean(loss_bid).item()})

            # wandb.log({"loss_wap": torch.mean(loss_wap).item()})

            # loss.backward()
            # # optimizer.step()

            if setup_loss:
                epoch_loss = loss
                epoch_reg_l1 = L1_loss
                setup_loss= 0
                loss_count = 1
            else:
                if loss.isnan():
                    pass
                
                epoch_loss = loss+epoch_loss
                loss_count+=1
                epoch_reg_l1 = L1_loss+epoch_reg_l1
                if i%100==0:
                    if i==0:
                        pass
                    else:
                        wandb.log({"epoch_loss": epoch_loss/loss_count})
                        epoch_loss.backward()
                        optimizer.step() 
                    setup_loss=1

        if not setup_loss:
            wandb.log({"epoch_loss": epoch_loss/loss_count})
            epoch_loss.backward()
            optimizer.step()

        trading_df.detach_hidden()
        wandb.log({"loss_1": torch.mean(loss).item()})
        
        wandb.log({"epoch_l1_loss": epoch_reg_l1/384, 'epoch':epoch, 'output_sd':torch.std(output_bid)})
        val_loss = validate_model(trading_df,model,criterion,epoch)
        # scheduler.step(val_loss)
        if epoch%20==0:
            trading_df.create_hidden_states_dict_v2()
            model_saver(model,optimizer,epoch,0,0,trading_df.train_hidden_dict)
        trading_df.reset_hidden(hidden_size=config['hidden_size'], num_layers=config['num_layers'])
              

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
    output_dict['time'] = []
    # output_dict['id'] = []
    output_dict['target'] = []
    output_dict['ask_target'] = []
    output_dict['ask_pred'] = []
    output_dict['bid_target'] = []
    output_dict['bid_pred'] = []
    output_dict['wap_target'] = []
    output_dict['wap_pred'] = []
    output_dict['actual_wap'] = []

    time_periods = len(trading_df.stocksDict[0].data_daily[0])
    

    # print(f"{time_periods=}\n{len_val=}")

    # print(time_periods)

    for i in range(0,len_val-1):
        # print(i)
        stocks = [trading_df.stocksDict[x] for x in trading_df.val_stock_batches[i]] 
        stock_ids = trading_df.val_stock_batches[i]
        # train_dog_input = trading_df.batches['train_dog_input'][i]
        time_ids = list(range(0,time_periods))
         
        X = trading_df.packed_val_x[i]
        Y =  trading_df.packed_val_y[i].data
        Y_price_ask = trading_df.packed_val_ask_price_daily[i].data
        Y_price_bid = trading_df.packed_val_bid_price_daily[i].data
        Y_price_wap = trading_df.packed_val_wap_price_daily[i].data
        Y_actual_wap = trading_df.packed_val_actual_wap[i].data

        hidden_in = torch.stack([x.hidden for x in stocks]).transpose(0,1)

        output_ask,output_bid,output_wap,hidden,relu = model(X,hidden_in)
        # output_wap,hidden,relu = model(X,hidden_in)
        hidden = hidden.transpose(0,1)
        output_ask  = torch.flatten(output_ask)
        output_bid  = torch.flatten(output_bid)
        output_wap  = torch.flatten(output_wap)


        [setattr(obj, 'hidden_test', val.detach()) for obj, val in zip(stocks,hidden)]

        # for i,dog in enumerate(train_dog_input):
        #     [setattr(obj, 'hidden_out', val) for obj, val in zip(dog,output[i])]


        loss_ask = criterion(output_ask,Y_price_ask)
        loss_bid = criterion(output_bid,Y_price_bid)
        loss_wap = criterion(output_wap,Y_price_wap)

        loss = loss_ask + loss_bid + loss_wap
        # loss = loss_bid

        L1_loss_ask = reg_L1(output_ask,Y_price_ask).detach()
        L1_loss_bid = reg_L1(output_bid,Y_price_bid).detach()
        L1_loss_wap = reg_L1(output_wap,Y_price_wap).detach()

        L1_loss = (L1_loss_ask +L1_loss_bid + L1_loss_wap)/3
        # L1_loss = L1_loss_wap

        if i == 0:
            epoch_loss = loss
            epoch_reg_l1 = L1_loss
            L1_loss_ask_epoch = L1_loss_ask 
            L1_loss_bid_epoch = L1_loss_bid 
            L1_loss_wap_epoch = L1_loss_wap 
        else:
            if loss.isnan():
                pass
            epoch_loss = loss+epoch_loss
            epoch_reg_l1 = L1_loss+epoch_reg_l1
            L1_loss_ask_epoch += L1_loss_ask 
            L1_loss_bid_epoch += L1_loss_bid 
            L1_loss_wap_epoch += L1_loss_wap 

        output_dict['stock'].append(stock_ids*time_periods)
        output_dict['day'].append([i+384]*time_periods*len(stock_ids))
        output_dict['time'].extend([[x]*len(stock_ids) for x in time_ids])
        output_dict['target'].append(Y.flatten().tolist())
        output_dict['ask_target'].append(Y_price_ask.flatten().tolist())
        output_dict['ask_pred'].append(output_ask.flatten().tolist())
        output_dict['bid_target'].append(Y_price_bid.flatten().tolist())
        output_dict['bid_pred'].append(output_bid.flatten().tolist())
        output_dict['wap_target'].append(Y_price_wap.flatten().tolist())
        output_dict['wap_pred'].append(output_wap.flatten().tolist())
        output_dict['actual_wap'].append(Y_actual_wap.flatten().tolist())
        # print(len(stock_ids*time_periods))
        # print(len(output_wap.flatten().tolist()))

    for k,value in output_dict.items():
        # print(k)
        # print(value)
        output_dict[k] = [item for sublist in value for item in sublist]
        # print(len(output_dict[k]))

    wandb.log({"val_epoch_loss": epoch_loss/len_val,
               "val_loss":torch.mean(loss).item() , 
               "val_epoch_loss_l1": epoch_reg_l1/len_val,
               'epoch':epoch, 
               'relu_sum':relu.sum(),
               'L1_loss_ask_epoch':L1_loss_ask_epoch/len_val,
                'L1_loss_bid_epoch':L1_loss_bid_epoch/len_val,
                'L1_loss_wap_epoch':L1_loss_wap_epoch/len_val,})
    # print({"epoch_loss": epoch_loss/384, "epoch_l1_loss": epoch_reg_l1/384, 'epoch':epoch})

    if epoch % 10 == 0:
        log_dict = pd.DataFrame(data=output_dict)
        # log_dict.to_feather(f'validation_outputs/{wandb.run.name}_validation_output_{epoch}.fth')
        # print(log_dict.head(10))
        log_dict = log_dict.head(200*49)
        log_dict = wandb.Table(data=log_dict)
        wandb.log({'data_table':log_dict})
        pass

    return epoch_reg_l1/len_val