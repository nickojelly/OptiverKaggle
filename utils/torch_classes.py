import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from torch.nn.utils.rnn import pack_padded_sequence, pack_sequence, unpack_sequence, unpad_sequence
import numpy as np
import gc

class TradingDays():
    def __init__(self) -> None:
        pass

class Stock():
    def __init__(self, stock_id:int,hidden_size=64,num_layers=2) -> None:
        self.stock_id = stock_id
        self.data_daily = {}
        self.data_volumes = {}
        self.target_daily = {}
        self.bid_size_daily = {}
        self.bid_price_daily = {}
        self.ask_size_daily = {}
        self.ask_price_daily = {}
        self.wap_price_daily = {}
        self.actual_wap = {}
        self.hidden = torch.zeros(1,num_layers,hidden_size)
        self.hidden_test = torch.zeros(1,num_layers,hidden_size)
        self.hidden_all = torch.zeros(30,hidden_size).to('cuda:0')

class TradingData():
    def __init__(self,train_data=None) -> None:
        self.mode = 'train'
        self.stat_cols = ['seconds_in_bucket', 'imbalance_size',
       'imbalance_buy_sell_flag', 'reference_price', 'matched_size',  
    #    'far_price', 'near_price', 
       'bid_price', 'bid_size', 'ask_price', 'ask_size', 
        'wap', 'index_weight','wap_calc','initial_wap','wap_weighted', 'index_wap', 
        'index_wap_init', 'index_wap_move_to_init',
        'wap_prev_move','bid_price_prev_move','ask_price_prev_move',  
              
         'overall_medvol', 'first5min_medvol',
       'last5min_medvol', 'bid_plus_ask_sizes', 'imbalance_ratio', 'imb_s1',
       'imb_s2', 'ask_x_size', 'bid_x_size', 'ask_minus_bid',
       'bid_price_over_ask_price', 'reference_price_minus_far_price',
       'reference_price_times_far_price', 'reference_price_times_near_price',
       'reference_price_minus_ask_price', 'reference_price_times_ask_price',
       'reference_price_ask_price_imb', 'reference_price_minus_bid_price',
       'reference_price_times_bid_price', 'reference_price_bid_price_imb',
       'reference_price_minus_wap', 'reference_price_times_wap',
       'reference_price_wap_imb', 'far_price_minus_near_price',
       'far_price_times_near_price', 'far_price_minus_ask_price',
       'far_price_times_ask_price', 'far_price_minus_bid_price',
       'far_price_times_bid_price', 'far_price_times_wap', 'far_price_wap_imb',
       'near_price_minus_ask_price', 'near_price_times_ask_price',
       'near_price_ask_price_imb', 'near_price_minus_bid_price',
       'near_price_times_bid_price', 'near_price_bid_price_imb',
       'near_price_minus_wap', 'near_price_wap_imb',
       'ask_price_minus_bid_price', 'ask_price_times_bid_price',
       'ask_price_minus_wap', 'ask_price_times_wap', 'ask_price_wap_imb',
       'bid_price_minus_wap', 'bid_price_times_wap', 'bid_price_wap_imb',
       'reference_price_far_price_near_price_imb2',
       'reference_price_far_price_ask_price_imb2',
       'reference_price_far_price_bid_price_imb2',
       'reference_price_far_price_wap_imb2',
       'reference_price_near_price_ask_price_imb2',
       'reference_price_near_price_bid_price_imb2',
       'reference_price_near_price_wap_imb2',
       'reference_price_ask_price_bid_price_imb2',
       'reference_price_ask_price_wap_imb2',
       'reference_price_bid_price_wap_imb2',
       'far_price_near_price_ask_price_imb2',
       'far_price_near_price_bid_price_imb2', 'far_price_near_price_wap_imb2',
       'far_price_ask_price_bid_price_imb2', 'far_price_ask_price_wap_imb2',
       'far_price_bid_price_wap_imb2', 'near_price_ask_price_bid_price_imb2',
       'near_price_ask_price_wap_imb2', 'near_price_bid_price_wap_imb2',
       'ask_price_bid_price_wap_imb2',
       'pca_0', 'pca_1', 'pca_2', 'pca_3', 'pca_4', 'pca_5', 
       'pca_6', 'pca_7',
       'pca_8', 'pca_9']
        self.stocksDict = {}
        self.daily_variance = {}
        self.daysDict = {}
        if isinstance(train_data,pd.DataFrame):
            # self.data = train_data
            self.add_data(train_data)

    def add_stocks(self, stock_id):
        self.stocksDict[stock_id] = Stock(stock_id)

    def add_data(self,data:pd.DataFrame):
        # data['stats'] = np.split(np.nan_to_num(data[self.stat_cols].to_numpy(),nan=-1),indices_or_sections=len(data))
        # data['stats'] = pd.Series(data[self.stat_cols].fillna(-1).values.tolist())
        data_grouped_stock_id = data.groupby('stock_id')
        data['volume'] = data['ask_size']+data['bid_size']
        for stock_id,stock_data in tqdm(data_grouped_stock_id):

            self.stocksDict[stock_id] = Stock(stock_id)
            stock_daily_group = stock_data.groupby('date_id')


            for day, stock_daily_data in stock_daily_group:
                if stock_daily_data['target'].isna().sum():
                    print(f'Missing Targets for {day=},for {stock_id=}, Excluding')
                    data.drop(stock_daily_data.index, inplace=True)
                    continue
                self.stocksDict[stock_id].data_daily[day] = torch.stack([torch.tensor(x[0]) for x in stock_daily_data['stats'].tolist()]).to('cuda:0')
                self.stocksDict[stock_id].data_volumes[day] = torch.stack([torch.tensor(x, requires_grad=False) for x in stock_daily_data['volume'].tolist()]).to('cuda:0')
                self.stocksDict[stock_id].target_daily[day] = torch.stack([torch.tensor(x, requires_grad=False) for x in stock_daily_data['target'].tolist()]).to('cuda:0')
                
                # self.stocksDict[stock_id].bid_size_daily[day]  = [torch.tensor(x) for x in stock_daily_data['bid_size_t-60'].tolist()]
                self.stocksDict[stock_id].bid_price_daily[day] = torch.stack([torch.tensor(x, requires_grad=False) for x in stock_daily_data['bid_price_t-60'].tolist()]).to('cuda:0')
                # self.stocksDict[stock_id].ask_size_daily[day]  = [torch.tensor(x) for x in stock_daily_data['ask_size_t-60'].tolist()]
                self.stocksDict[stock_id].ask_price_daily[day] = torch.stack([torch.tensor(x, requires_grad=False) for x in stock_daily_data['ask_price_t-60'].tolist()]).to('cuda:0')

                self.stocksDict[stock_id].wap_price_daily[day] = torch.stack([torch.tensor(x, requires_grad=False) for x in stock_daily_data['wap_price_t-60'].tolist()]).to('cuda:0')
                self.stocksDict[stock_id].actual_wap[day] = torch.stack([torch.tensor(x, requires_grad=False) for x in stock_daily_data['wap'].tolist()]).to('cuda:0')


        data_grouped_daily = data.groupby('date_id')

        for date_id, data_daily in data_grouped_daily:
            stocks = data_daily.stock_id.unique().tolist()
            self.daysDict[date_id] = stocks
            self.daily_variance[date_id] = data_daily['wap_price_t-60'].std()

        gc.collect()

    def generate_batches(self, validation_split=0.20):
        len_data = len(self.daysDict)
        len_validation = int(len(self.daysDict)*validation_split)
        len_train = len_data-len_validation
        print(f"Length of train: {len_train}, Length of test {len_validation}")

        train_range = range(0,len_train)
        val_range = range(len_train+1,len_data)

        self.train_batches = []
        self.train_class_batches = []
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

            # bid_size_daily = [self.stocksDict[x].bid_size_daily[i] for x in self.daysDict[i]]
            bid_price_daily = torch.stack([self.stocksDict[x].bid_price_daily[i] for x in self.daysDict[i]]).to('cuda:0')
            # ask_size_daily = [self.stocksDict[x].ask_size_daily[i] for x in self.daysDict[i]]
            ask_price_daily = torch.stack([self.stocksDict[x].ask_price_daily[i] for x in self.daysDict[i]]).to('cuda:0')

            wap_price_daily = torch.stack([self.stocksDict[x].wap_price_daily[i] for x in self.daysDict[i]]).to('cuda:0')

            self.train_batches.append(train_data)
            self.train_class_batches.append(train_classes)

            # self.train_bid_size_daily.append(bid_size_daily) 
            self.train_bid_price_daily.append(bid_price_daily) 
            # self.train_ask_size_daily.append(ask_size_daily) 
            self.train_ask_price_daily.append(ask_price_daily)

            self.train_wap_price_daily.append(wap_price_daily)  

        # self.packed_x = [pack_sequence([torch.stack(n,0)for n in [[z for z in inner] for inner in x]], enforce_sorted=False).to(device='cuda:0') for x in self.train_batches if x]
        # self.packed_y = [pack_sequence([torch.stack(n,0)for n in [[z for z in inner] for inner in x]], enforce_sorted=False).to(device='cuda:0') for x in self.train_class_batches if x]

        # self.packed_bid_size_daily  = [pack_sequence([torch.stack(n,0)for n in [[z for z in inner] for inner in x]], enforce_sorted=False).to(device='cuda:0') for x in self.train_bid_size_daily  if x]
        # self.packed_bid_price_daily = [pack_sequence([torch.stack(n,0)for n in [[z for z in inner] for inner in x]], enforce_sorted=False).to(device='cuda:0') for x in self.train_bid_price_daily if x]
        # self.packed_ask_size_daily  = [pack_sequence([torch.stack(n,0)for n in [[z for z in inner] for inner in x]], enforce_sorted=False).to(device='cuda:0') for x in self.train_ask_size_daily  if x]
        # self.packed_ask_price_daily = [pack_sequence([torch.stack(n,0)for n in [[z for z in inner] for inner in x]], enforce_sorted=False).to(device='cuda:0') for x in self.train_ask_price_daily if x]
        # self.packed_wap_price_daily = [pack_sequence([torch.stack(n,0)for n in [[z for z in inner] for inner in x]], enforce_sorted=False).to(device='cuda:0') for x in self.train_wap_price_daily if x]

        self.val_batches = []
        self.val_class_batches = []
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
            self.val_batches.append(train_data)
            self.val_class_batches.append(train_classes)

            # bid_size_daily = [self.stocksDict[x].bid_size_daily[i] for x in self.daysDict[i]]
            bid_price_daily = torch.stack([self.stocksDict[x].bid_price_daily[i] for x in self.daysDict[i]]).to('cuda:0')
            # ask_size_daily = [self.stocksDict[x].ask_size_daily[i] for x in self.daysDict[i]]
            ask_price_daily = torch.stack([self.stocksDict[x].ask_price_daily[i] for x in self.daysDict[i]]).to('cuda:0')
            wap_price_daily = torch.stack([self.stocksDict[x].wap_price_daily[i] for x in self.daysDict[i]]).to('cuda:0')
            actual_wap = torch.stack([self.stocksDict[x].actual_wap[i] for x in self.daysDict[i]]).to('cuda:0')

            # self.val_bid_size_daily.append(bid_size_daily) 
            self.val_bid_price_daily.append(bid_price_daily) 
            # self.val_ask_size_daily.append(ask_size_daily) 
            self.val_ask_price_daily.append(ask_price_daily)
            self.val_wap_price_daily.append(wap_price_daily) 
            self.val_actual_wap.append(actual_wap)

        # self.packed_val_x = [pack_sequence([torch.stack(n,0)for n in [[z for z in inner] for inner in x]], enforce_sorted=False).to(device='cuda:0') for x in self.val_batches if x]
        # self.packed_val_y = [pack_sequence([torch.stack(n,0)for n in [[z for z in inner] for inner in x]], enforce_sorted=False).to(device='cuda:0') for x in self.val_class_batches if x]
        # self.packed_val_bid_size_daily  = [pack_sequence([torch.stack(n,0)for n in [[z for z in inner] for inner in x]], enforce_sorted=False).to(device='cuda:0') for x in self.val_bid_size_daily  if x]
        # self.packed_val_bid_price_daily = [pack_sequence([torch.stack(n,0)for n in [[z for z in inner] for inner in x]], enforce_sorted=False).to(device='cuda:0') for x in self.val_bid_price_daily if x]
        # self.packed_val_ask_size_daily  = [pack_sequence([torch.stack(n,0)for n in [[z for z in inner] for inner in x]], enforce_sorted=False).to(device='cuda:0') for x in self.val_ask_size_daily  if x]
        # self.packed_val_ask_price_daily = [pack_sequence([torch.stack(n,0)for n in [[z for z in inner] for inner in x]], enforce_sorted=False).to(device='cuda:0') for x in self.val_ask_price_daily if x]
        # self.packed_val_wap_price_daily = [pack_sequence([torch.stack(n,0)for n in [[z for z in inner] for inner in x]], enforce_sorted=False).to(device='cuda:0') for x in self.val_wap_price_daily if x]
        # self.packed_val_actual_wap = [pack_sequence([torch.stack(n,0)for n in [[z for z in inner] for inner in x]], enforce_sorted=False).to(device='cuda:0') for x in self.val_actual_wap if x]


    def reset_hidden(self, hidden_size=32,num_layers=2, device='cuda:0'): 
        if hidden_size==None:
            hidden_size = self.hidden_size
        for stock in self.stocksDict.values():
            stock.hidden =      torch.zeros(num_layers,hidden_size).to(device)
            stock.hidden_test = torch.zeros(num_layers,hidden_size).to(device)
            stock.hidden_all =  torch.zeros(55,hidden_size).to(device)

    def detach_hidden(self, stocks_list=None):
        for stock in self.stocksDict.values():
            stock.hidden = stock.hidden.detach()

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
        

        self.fc0 = nn.Linear(input_size,hidden_size)
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
        # x = x._replace(data=self.fc0(x.data))
        # x = x._replace(data=self.relu(x.data))
        
        x_h,hidden = self.gru(x,h.contiguous())
        # x = self.layer_norm(x.data)
        x = self.relu1(x_h)
        x = self.drop(x)
        x = self.fc1(x)
        # x = self.layer_norm(x.data)
        x_rl1 = self.relu2(x)
        x = self.drop_1(x_rl1)

        x_ask_price = self.fc_ask_price(x)
        x_bid_price = self.fc_bid_price(x)
        x_wap_price = self.fc_wap_price(x)
        

        return x_ask_price,x_bid_price,x_wap_price,hidden,x_rl1,x_h 