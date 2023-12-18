import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from torch.nn.utils.rnn import pack_padded_sequence, pack_sequence, unpack_sequence, unpad_sequence
import numpy as np
import gc
import multiprocessing

class TradingDays():
    def __init__(self) -> None:
        pass

class Stock():
    def __init__(self, stock_id:int,hidden_size=64,num_layers=2) -> None:
        self.stock_id = stock_id
        self.data_daily = {}
        self.data_volumes = {}
        self.target_daily = {}
        self.target_daily_ohe = {}
        self.wap_daily_ohe = {}
        self.bid_size_daily = {}
        self.bid_price_daily = {}
        self.ask_size_daily = {}
        self.ask_price_daily = {}
        self.wap_price_daily = {}
        self.actual_wap = {}
        self.hidden = torch.zeros(1,num_layers,hidden_size)
        self.hidden_test = torch.zeros(1,num_layers,hidden_size)
        self.hidden_all = torch.zeros(30,hidden_size).to('cuda:0')
        self.hidden_out = torch.zeros(49,hidden_size).to('cuda:0')


class TradingData():
    def __init__(self,train_data=None,means=None) -> None:
        self.mode = 'train'
        self.stocksDict = {}
        self.daily_variance = {}
        self.daysDict = {}
        self.means_df = means
        self.wap_daily_weights = {}
        self.target_daily_weights = {}
        if isinstance(train_data,pd.DataFrame):
            # self.data = train_data
            self.add_data(train_data)

    def add_stocks(self, stock_id):
        self.stocksDict[stock_id] = Stock(stock_id)

    def add_data(self,data:pd.DataFrame):

        data_grouped_stock_id = data.groupby('stock_id')
        data['volume'] = data['ask_size']+data['bid_size']

        data_grouped_daily = data.groupby('date_id',sort=False)

        for date_id, data_daily in tqdm(data_grouped_daily):
            stocks = data_daily.stock_id.unique().tolist()
            stocks.sort()
            self.daysDict[date_id] = stocks
            self.daily_variance[date_id] = data_daily['wap_price_t-60'].std()
            weights = (data_daily["wap_category"].value_counts(sort=False).reset_index().sort_values("wap_category"))
            weights["norm_count"] = 1 - (weights["count"] / weights["count"].sum())
            self.wap_daily_weights[date_id] = torch.tensor(weights["norm_count"].to_numpy(), device="cuda:0")
            weights = (data_daily["target_category"].value_counts(sort=False).reset_index().sort_values("target_category"))
            weights["norm_count"] = 1 - (weights["count"] / weights["count"].sum())
            self.target_daily_weights[date_id] = torch.tensor(weights["norm_count"].to_numpy(), device="cuda:0")
            
            
            
        for stock in data.stock_id.unique():
            self.stocksDict[stock] = Stock(stock)

        tensor_columns = ['volume','target','wap_target_OHE','target_OHE','bid_price_t-60','ask_price_t-60','wap_price_t-60','wap']

        # for col in tqdm(tensor_columns):
        #     data[col] = [torch.tensor(x, requires_grad=False) for x in data['volume'].tolist()]

        data_grouped_stock_id = data.groupby(['stock_id','date_id'],sort=False)
        for (stock_id,day),stock_daily_data in tqdm(data_grouped_stock_id):

            if stock_daily_data['target'].isna().sum():
                print(f'Missing Targets for {day=},for {stock_id=}, Excluding')
                data.drop(stock_daily_data.index, inplace=True)
                continue

            self.stocksDict[stock_id].data_daily[day] =         torch.stack([torch.tensor(x[0]) for x in stock_daily_data['stats'].to_numpy()]).to('cuda:0')
            self.stocksDict[stock_id].data_volumes[day] =       torch.tensor(stock_daily_data['volume'].to_numpy(),requires_grad=False,device='cuda:0')
            self.stocksDict[stock_id].target_daily[day] =       torch.tensor(stock_daily_data['target'].to_numpy(),requires_grad=False,device='cuda:0')
            self.stocksDict[stock_id].wap_daily_ohe[day] =      torch.tensor(stock_daily_data['wap_target_OHE'].to_list(),requires_grad=False,device='cuda:0')
            self.stocksDict[stock_id].target_daily_ohe[day] =   torch.tensor(stock_daily_data['target_OHE'].to_list(),requires_grad=False,device='cuda:0')
            self.stocksDict[stock_id].bid_price_daily[day] =    torch.tensor(stock_daily_data['bid_price_t-60'].to_numpy(),requires_grad=False,device='cuda:0')
            self.stocksDict[stock_id].ask_price_daily[day] =    torch.tensor(stock_daily_data['ask_price_t-60'].to_numpy(),requires_grad=False,device='cuda:0')
            self.stocksDict[stock_id].wap_price_daily[day] =    torch.tensor(stock_daily_data['wap_price_t-60'].to_numpy(),requires_grad=False,device='cuda:0')
            self.stocksDict[stock_id].actual_wap[day] =         torch.tensor(stock_daily_data['wap'].to_numpy(),requires_grad=False,device='cuda:0')
            # print(self.stocksDict[stock_id].actual_wap[day] )
            # asdf





    def generate_batches(self, validation_split=0.20):
        len_data = len(self.daysDict)
        len_validation = int(len(self.daysDict)*validation_split)
        len_train = len_data-len_validation
        print(f"Length of train: {len_train}, Length of test {len_validation}")

        train_range = range(0,len_train)
        val_range = range(len_train+1,len_data)

        self.train_batches = []
        self.train_class_batches = []
        self.train_wap_ohe_batches = []
        self.train_target_ohe_batches = []
        self.stock_batches = []
        self.train_bid_size_daily  = []
        self.train_bid_price_daily = []
        self.train_ask_size_daily  = []
        self.train_ask_price_daily = []
        self.train_wap_price_daily = []


        for i in tqdm(train_range):
            self.stock_batches.append(self.daysDict[i])
            
            train_data = torch.stack([self.stocksDict[x].data_daily[i] for x in self.daysDict[i]]).to('cuda:0')
            train_classes = torch.stack([self.stocksDict[x].target_daily[i] for x in self.daysDict[i]]).to('cuda:0')
            train_wap_ohe = torch.stack([self.stocksDict[x].wap_daily_ohe[i] for x in self.daysDict[i]]).to('cuda:0')
            train_target_ohe = torch.stack([self.stocksDict[x].target_daily_ohe[i] for x in self.daysDict[i]]).to('cuda:0')


            bid_price_daily = torch.stack([self.stocksDict[x].bid_price_daily[i] for x in self.daysDict[i]]).to('cuda:0')
            ask_price_daily = torch.stack([self.stocksDict[x].ask_price_daily[i] for x in self.daysDict[i]]).to('cuda:0')

            wap_price_daily = torch.stack([self.stocksDict[x].wap_price_daily[i] for x in self.daysDict[i]]).to('cuda:0')

            self.train_batches.append(train_data)
            self.train_class_batches.append(train_classes)
            self.train_wap_ohe_batches.append(train_wap_ohe)
            self.train_target_ohe_batches.append(train_target_ohe)
            self.train_bid_price_daily.append(bid_price_daily) 
            self.train_ask_price_daily.append(ask_price_daily)
            self.train_wap_price_daily.append(wap_price_daily)  

        self.val_batches = []
        self.val_class_batches = []
        self.val_wap_ohe_batches = []
        self.val_target_ohe_batches = []
        self.val_stock_batches = []
        self.val_bid_size_daily  = []
        self.val_bid_price_daily = []
        self.val_ask_size_daily  = []
        self.val_ask_price_daily = []
        self.val_wap_price_daily = []
        self.val_actual_wap = []
        for i in tqdm(val_range):
            self.val_stock_batches.append(self.daysDict[i])
            train_data = torch.stack([self.stocksDict[x].data_daily[i] for x in self.daysDict[i]]).to('cuda:0')
            train_classes = torch.stack([self.stocksDict[x].target_daily[i] for x in self.daysDict[i]]).to('cuda:0')
            train_wap_ohe = torch.stack([self.stocksDict[x].wap_daily_ohe[i] for x in self.daysDict[i]]).to('cuda:0')
            train_target_ohe = torch.stack([self.stocksDict[x].target_daily_ohe[i] for x in self.daysDict[i]]).to('cuda:0')
            self.val_batches.append(train_data)
            self.val_class_batches.append(train_classes)
            self.val_wap_ohe_batches.append(train_wap_ohe)
            self.val_target_ohe_batches.append(train_target_ohe)

            bid_price_daily = torch.stack([self.stocksDict[x].bid_price_daily[i] for x in self.daysDict[i]]).to('cuda:0')
            ask_price_daily = torch.stack([self.stocksDict[x].ask_price_daily[i] for x in self.daysDict[i]]).to('cuda:0')
            wap_price_daily = torch.stack([self.stocksDict[x].wap_price_daily[i] for x in self.daysDict[i]]).to('cuda:0')
            actual_wap = torch.stack([self.stocksDict[x].actual_wap[i] for x in self.daysDict[i]]).to('cuda:0')

            self.val_bid_price_daily.append(bid_price_daily) 
            self.val_ask_price_daily.append(ask_price_daily)
            self.val_wap_price_daily.append(wap_price_daily) 
            self.val_actual_wap.append(actual_wap)

    def reset_hidden(self, hidden_size=32,num_layers=2, device='cuda:0'): 
        if hidden_size==None:
            hidden_size = self.hidden_size
        for stock in self.stocksDict.values():
            stock.hidden =      torch.zeros(num_layers,hidden_size).to(device)
            stock.hidden_test = torch.zeros(num_layers,hidden_size).to(device)
            stock.hidden_all =  torch.zeros(55,hidden_size).to(device)
            stock.hidden_out = torch.zeros(49,hidden_size).to('cuda:0')

    def detach_hidden(self, stocks_list=None):
        for stock in self.stocksDict.values():
            stock.hidden = stock.hidden.detach()
            stock.hidden_out = stock.hidden_out.detach()

    def detach_hidden_out(self, stocks_list=None):
        for stock in self.stocksDict.values():
            stock.hidden_out = stock.hidden_out.detach()

    def create_hidden_states_dict_v2(self):
        self.train_hidden_dict = {}
        for stock_id in self.stocksDict.keys():
            stock = self.stocksDict[stock_id]

            hidden = stock.hidden_test
            self.train_hidden_dict[stock_id] = hidden


    def fill_hidden_states_dict_v2(self, hidden_dict):
        filled, empty, null_stock = 0, 0, 0
        for stock in tqdm(self.stocksDict.values()):
            stock_id = stock.stock_id
            val = hidden_dict[stock_id]
            hidden = val
            stock.hidden_test = hidden
            stock.hidden_filled = 1

    def fill_hidden_states_for_test(self,hidden_dict):
        for stock,hidden in hidden_dict.items():
            self.stocksDict[stock] = Stock(stock)
            self.stocksDict[stock].hidden = hidden
            if stock <5 : 
                print(stock)
                print(self.stocksDict[stock].hidden)

class GRUNet(nn.Module):

    def __init__(self, input_size, hidden_size,hidden=None,output='raw', dropout=0.3, fc0_size=256,fc1_size=64,num_layers=1):
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(input_size,hidden_size,num_layers=num_layers, dropout=0.3)
        self.relu0 = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(input_size)

        self.relu0 = nn.ReLU()
        self.fc0 = nn.Linear(hidden_size,1)        
        self.rl1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear((hidden_size * 8)+70, fc0_size)
        self.rl2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        #regular
        self.fc2 = nn.Linear(fc0_size, fc1_size)
        self.rl3 = nn.ReLU()
        self.drop3 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(fc1_size, 8)
        self.hidden_size = hidden_size


    def forward(self, x,h=None, test=False):
        
        if test:
            x = x.float()
            x = self.batch_norm(x)
            x = x.view(1,-1,12)
            x,hidden = self.gru(x,h)
            x = self.relu0(x.data)
            # x = self.relu0(x)
            x = self.fc0(x)

        else:
            x = x.float()
            x = x._replace(data=self.batch_norm(x.data))
            
            x,hidden = self.gru(x,h)
            x = self.relu0(x.data)
            # x = self.relu0(x)
            x = self.fc0(x)

        return x,hidden
    
class GRUNetV2(nn.Module):

    def __init__(self, input_size, hidden_size,hidden=None,output='raw', dropout=0.5, fc0_size=256,fc1_size=64,num_layers=1):
        super(GRUNetV2, self).__init__()
        self.gru = nn.GRU(input_size,hidden_size,num_layers=num_layers, dropout=0.3,batch_first=True)
        self.relu = nn.ReLU()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(input_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.drop = nn.Dropout(dropout)
        self.drop_1 = nn.Dropout(dropout)
        

        self.fc0 = nn.Linear(input_size,input_size)
        self.fc1 = nn.Linear(hidden_size,hidden_size)        

        self.fc_ask_price = nn.Linear(hidden_size, 1)
        self.fc_bid_price = nn.Linear(hidden_size, 1)
        self.fc_wap_price = nn.Linear(hidden_size, 1)



        #regular
        self.fc2 = nn.Linear(fc0_size, fc1_size)
        self.fc3 = nn.Linear(fc1_size, 8)
        self.hidden_size = hidden_size

        #bid ask



    def forward(self, x,h=None, test=False):
        x = x.float()
        x = x.transpose(1,2)
        x = self.batch_norm(x)
        x = x.transpose(1,2)
        x = self.fc0(x)
        x = self.relu(x)
        
        x_h,hidden = self.gru(x,h.contiguous())
        x = self.layer_norm(x_h)
        x = self.relu1(x)
        x = self.drop(x)
        x = self.fc1(x)
        x = self.layer_norm(x)
        x_rl1 = self.relu2(x)
        x = self.drop_1(x_rl1)

        x_ask_price = self.fc_ask_price(x)
        x_bid_price = self.fc_bid_price(x)
        x_wap_price = self.fc_wap_price(x)
        

        return x_ask_price,x_bid_price,x_wap_price,hidden,x_rl1,x_h

class GRUNetV3(nn.Module):

    def __init__(self, input_size, hidden_size,hidden=None,output='raw', dropout=0.5, fc0_size=256,fc1_size=64,num_layers=1,target_size=7):
        super(GRUNetV3, self).__init__()
        self.gru = nn.GRU(hidden_size,hidden_size,num_layers=num_layers, dropout=0.3,batch_first=True)
        self.relu = nn.ReLU()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(input_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(fc0_size)
        self.drop = nn.Dropout(dropout)
        self.drop_1 = nn.Dropout(dropout)
        self.target_size = target_size
        
        #target
        self.fc0 = nn.Linear(input_size,hidden_size)
        self.fc1 = nn.Linear(hidden_size,fc0_size)        
        self.fc_final = nn.Linear(fc0_size, target_size)

        self.fc_target0 = nn.Linear(target_size,64)
        self.fc_target_relu = nn.ReLU()
        self.fc_target1 = nn.Linear(64,1)

        #wap 
        self.wap_fc0 = nn.Linear(input_size,hidden_size)
        self.wap_fc1 = nn.Linear(hidden_size,fc0_size)        
        self.wap_fc_final = nn.Linear(fc0_size, target_size)

        self.fc_wap0 = nn.Linear(target_size,64)
        self.fc_wap_relu = nn.ReLU()
        self.fc_wap1 = nn.Linear(64,1)



        # self.all_fc0 = nn.Linear(hidden_size*200,512)
        # self.all_fc0 = nn.Linear(hidden_size*200,2048)
        # self.all_fc1 = nn.Linear(2048,1024)
        # self.all_fc2 = nn.Linear(1024,200*target_size)


        #regular
        self.hidden_size = hidden_size

        #bid ask



    def forward(self, x,h=None, test=False, p2=False):
        if p2:
            x = x.float().detach().requires_grad_()
            # x = x.transpose(0,1).view(49,-1).shape
            # x = self.relu(x)
            # x = self.all_fc0(x)
            # x = self.relu(x)
            # x = self.drop(x)
            # x = self.all_fc1(x)
            # x = self.relu(x)
            # x = self.drop(x)
            # x = self.all_fc2(x)
            # x = x.view((-1,200,self.target_size))
            
            return x



        else:
            x = x.float()
            x = x.transpose(1,2)
            x = self.batch_norm(x)
            x = x.transpose(1,2)
            x_bn = x
            x = self.fc0(x_bn)
            x = self.relu(x)

            # x = self.fc01(x)
            # x = self.relu(x)
            
            x_h,hidden = self.gru(x,h.contiguous())

            x = self.layer_norm(x_h)
            x = self.relu1(x)
            x_d = self.drop(x)

            #single target classifier
            x = self.fc1(x_d)
            x = self.layer_norm2(x)
            x_rl1 = self.relu2(x)
            x = self.drop_1(x_rl1)
            x_target_ohe = self.fc_final(x)

            #single wap classifier
            x = self.wap_fc1(x_d)
            x = self.layer_norm2(x)
            x_rl1 = self.relu2(x)
            x = self.drop_1(x_rl1)
            x_wap_ohe = self.wap_fc_final(x)




            #wap regression
            # x_wap = torch.cat([x_bn.detach(),x_ohe],dim=2).detach()
            x_wap = x_wap_ohe.detach()
            x_wap = self.fc_wap0(x_wap)
            x_wap = self.fc_wap_relu(x_wap)
            x_wap = self.fc_wap1(x_wap)
            # x_wap = x

            #target regression
            # print(x_bn.shape,x_ohe.shape)
            # x_target = torch.cat([x_bn.detach(),x_ohe],dim=2).detach()
            x_target = x_target_ohe.detach()
            x_target = self.fc_target0(x_target)
            x_target = self.fc_target_relu(x_target)
            x_target = self.fc_target1(x_target)

        

        return x_target_ohe,x_wap_ohe,x_wap,x_target,hidden,x_rl1,x_h 
    
class GRUNetV4(nn.Module):

    def __init__(self, input_size, hidden_size,hidden=None,output='raw', dropout=0.5, fc0_size=256,fc1_size=64,num_layers=1,target_size=7):
        super(GRUNetV4, self).__init__()
        self.gru = nn.GRU(hidden_size,hidden_size,num_layers=num_layers, dropout=0.3,batch_first=True)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(input_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(fc0_size)
        self.drop = nn.Dropout(dropout)
        self.drop_1 = nn.Dropout(dropout)
        

        self.fc0 = nn.Linear(input_size,hidden_size)
        self.fc1 = nn.Linear(hidden_size,fc0_size)        
        self.fc_final = nn.Linear(fc0_size, target_size)

        self.fc_reg0 = nn.Linear(target_size*2,128)
        self.fc_reg1 = nn.Linear(128,64)
        self.fc_reg2 = nn.Linear(64,1)

        self.softmax = nn.Softmax(dim=2)

        #regular
        self.hidden_size = hidden_size

    def forward(self, x,h=None, test=False):
        x = x.float()
        x = x.transpose(1,2)
        x = self.batch_norm(x)
        x = x.transpose(1,2)
        x = self.fc0(x)
        x = self.relu(x)
        x_h,hidden = self.gru(x,h.contiguous())

        x = self.layer_norm(x_h)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc1(x)
        x = self.layer_norm2(x)
        x_rl1 = self.relu(x)
        x = self.drop_1(x_rl1)
        x_ohe = self.fc_final(x)

        x_reg = torch.cat([self.softmax(x_ohe.detach()),x_ohe],dim=2).detach()
        x_reg = self.fc_reg0(x_reg)
        x_reg = self.relu(x_reg)
        x_reg = self.fc_reg1(x_reg)
        x_reg = self.relu(x_reg)
        x_reg = self.fc_reg2(x_reg)

        return x_ohe,x_reg,hidden,x_rl1,x_h 
    
class GRUNetV5(nn.Module):

    def __init__(self, input_size, hidden_size,hidden=None,output='raw', dropout=0.5, fc0_size=256,fc1_size=64,num_layers=1):
        super(GRUNetV5, self).__init__()
        self.gru = nn.GRU(input_size,hidden_size,num_layers=num_layers, dropout=0.3,batch_first=True)
        self.gru_wap = nn.GRU(input_size+7,hidden_size,num_layers=num_layers, dropout=0.3,batch_first=True)
        self.relu = nn.ReLU()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(input_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.drop = nn.Dropout(dropout)
        self.drop_1 = nn.Dropout(dropout)
        

        self.fc0 = nn.Linear(input_size,input_size)
        self.fc1 = nn.Linear(hidden_size,hidden_size)        

        self.fc_final = nn.Linear(hidden_size, 7)
        self.fc_wap = nn.Linear(hidden_size,1)

        #regular
        self.hidden_size = hidden_size

        #bid ask



    def forward(self, x,h=None,h_wap=None, test=False):
        x = x.float()
        x = x.transpose(1,2)
        x_og = self.batch_norm(x)
        #p1

        x = x.transpose(1,2)
        x = self.fc0(x)
        x = self.relu(x)
        
        x_h,hidden = self.gru(x,h.contiguous())
        x = self.layer_norm(x_h)
        x = self.relu1(x)
        x = self.drop(x)
        x = self.fc1(x)
        x = self.layer_norm(x)
        x_rl1 = self.relu2(x)
        x = self.drop_1(x_rl1)

        x_ohe = self.fc_final(x)

        #wap

        x = torch.cat([x_og,x_ohe])
        x_h,hidden_wap = self.gru_wap(x,h_wap.contiguous())


        x_wap = self.fc_wap(x)

        

        return x_ohe,x_wap,hidden,x_rl1,x_h 