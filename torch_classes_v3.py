import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from torch.nn.utils.rnn import pack_padded_sequence, pack_sequence, unpack_sequence, unpad_sequence

class TradingDays():
    def __init__(self) -> None:
        pass

class Stock():
    def __init__(self, stock_id:int) -> None:
        self.stock_id = stock_id
        self.data_daily = {}
        self.target_daily = {}
        self.hidden = torch.zeros(1,64)
        self.hidden_test = torch.zeros(1,64)

class TradingData():
    def __init__(self,train_data=None) -> None:
        self.mode = 'train'
        self.stat_cols = ['seconds_in_bucket','imbalance_size','imbalance_buy_sell_flag','reference_price','matched_size','far_price','near_price','bid_price','bid_size','ask_price','ask_size','wap']
        self.stocksDict = {}
        self.daysDict = {}
        if isinstance(train_data,pd.DataFrame):
            self.data = train_data
            self.add_data(train_data)

    def add_stocks(self, stock_id):
        self.stocksDict[stock_id] = Stock(stock_id)

    def add_data(self,data:pd.DataFrame):
        data['stats'] = pd.Series(data[self.stat_cols].fillna(-1).values.tolist())
        data_grouped_stock_id = data.groupby('stock_id')
        for stock_id,stock_data in tqdm(data_grouped_stock_id):

            self.stocksDict[stock_id] = Stock(stock_id)
            stock_daily_group = stock_data.groupby('date_id')


            for day, stock_daily_data in stock_daily_group:
                if stock_daily_data['target'].isna().sum():
                    print(f'Missing Targets for {day=},for {stock_id=}, Excluding')
                    data.drop(stock_daily_data.index, inplace=True)
                    continue
                self.stocksDict[stock_id].data_daily[day] = [torch.tensor(x) for x in stock_daily_data['stats'].tolist()]
                self.stocksDict[stock_id].target_daily[day] = [torch.tensor(x) for x in stock_daily_data['target'].tolist()]

        data_grouped_daily = data.groupby('date_id')

        for date_id, data_daily in data_grouped_daily:
            stocks = data_daily.stock_id.unique().tolist()
            self.daysDict[date_id] = stocks

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
        for i in tqdm(train_range):
            self.stock_batches.append(self.daysDict[i])
            
            train_data = [self.stocksDict[x].data_daily[i] for x in self.daysDict[i]]
            train_classes = [self.stocksDict[x].target_daily[i] for x in self.daysDict[i]]
            self.train_batches.append(train_data)
            self.train_class_batches.append(train_classes)

        self.packed_x = [pack_sequence([torch.stack(n,0)for n in [[z for z in inner] for inner in x]], enforce_sorted=False).to(device='cuda:0') for x in self.train_batches if x]
        self.packed_y = [pack_sequence([torch.stack(n,0)for n in [[z for z in inner] for inner in x]], enforce_sorted=False).to(device='cuda:0') for x in self.train_class_batches if x]

        self.val_batches = []
        self.val_class_batches = []
        self.val_stock_batches = []
        for i in tqdm(val_range):
            self.val_stock_batches.append(self.daysDict[i])
            train_data = [self.stocksDict[x].data_daily[i] for x in self.daysDict[i]]
            train_classes = [self.stocksDict[x].target_daily[i] for x in self.daysDict[i]]
            self.val_batches.append(train_data)
            self.val_class_batches.append(train_classes)

        self.packed_val_x = [pack_sequence([torch.stack(n,0)for n in [[z for z in inner] for inner in x]], enforce_sorted=False).to(device='cuda:0') for x in self.val_batches if x]
        self.packed_val_y = [pack_sequence([torch.stack(n,0)for n in [[z for z in inner] for inner in x]], enforce_sorted=False).to(device='cuda:0') for x in self.val_class_batches if x]

    def reset_hidden(self, hidden_size=32, device='cuda:0'): 
        if hidden_size==None:
            hidden_size = self.hidden_size
        for stock in self.stocksDict.values():
            stock.hidden = torch.zeros(1,hidden_size).to(device)
            stock.hidden_test = torch.zeros(1,hidden_size).to(device)

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

    def __init__(self, input_size, hidden_size,hidden=None,output='raw', dropout=0.3, fc0_size=256,fc1_size=64,num_layers=1):
        super(GRUNetV2, self).__init__()
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
            x = self.batch_norm(x.data)
            
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