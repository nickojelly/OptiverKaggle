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
import torch_classes_v2
from model_saver import model_saver_wandb as model_saver
import training_testing
from itertools import combinations
import gc
from sklearn.decomposition import PCA
import lightgbm as lgb
import time

trading_data = None

def generate_index(df_in, df_g, rolling_window=10, factor=''):
    df = df_in.copy()
    df[f'index_wap'] = df_g['wap_weighted'].transform('mean')
    return(df)

def generate_index_2(df_in, df_g, rolling_window=10, factor=''):
    df = df_in.copy()
    df[f'index_wap_t-60'] = df_g['index_wap'].shift(6)
    df[f'index_wap_init'] = df_g['index_wap'].transform('first')
    return(df)

def generate_index_3(df_in, df_g, rolling_window=10, factor=''):
    df = df_in.copy()
    df[f'index_wap_t-60'] = df_g['index_wap_move_to_init'].shift(6)
    return(df)

def generate_prev_race(df_in, df_g, rolling_window=10, factor=''):
    df = df_in.copy()
    original_cols = df_in.columns
    df[f'wap_t-60'] = df_g['wap'].shift(6)
    df[f'target_t-60'] = df_g['target'].shift(6)
    df[f'initial_wap'] = df_g['wap_calc'].transform('first')
    df[f'initial_bid_size'] = df_g['bid_size'].transform('first')
    df[f'initial_ask_size'] = df_g['ask_size'].transform('first')
    cols = ['bid_price','ask_price','bid_size','ask_size']
    for i in cols:
        df[f'{i}_t-60'] = df_g[i].shift(-6)

    return(df)

def feat_eng(df):
    
    cols = [c for c in df.columns if c not in ['row_id', 'time_id']]
    df = df[cols]
    df = df.merge(median_vol, how = "left", left_on = "stock_id", right_index = True)
    
    df['bid_plus_ask_sizes'] = df['bid_size'] + train['ask_size']
#     df['median_size'] = df['stock_id'].map(median_sizes.to_dict())
    df['std_size'] = df['stock_id'].map(std_sizes.to_dict())
#     df['high_volume'] = np.where(df['bid_plus_ask_sizes'] > df['median_size'], 1, 0) 
    df['imbalance_ratio'] = df['imbalance_size'] / df['matched_size']
    
    df['imb_s1'] = df.eval('(bid_size-ask_size)/(bid_size+ask_size)')
    df['imb_s2'] = df.eval('(imbalance_size-matched_size)/(matched_size+imbalance_size)')

    df['ask_x_size'] = df.eval('ask_size*ask_price')
    df['bid_x_size'] = df.eval('bid_size*bid_price')
        
    df['ask_minus_bid'] = df['ask_x_size'] - df['bid_x_size'] 
    
    df["bid_size_over_ask_size"] = df["bid_size"].div(df["ask_size"])
    df["bid_price_over_ask_price"] = df["bid_price"].div(df["ask_price"])
    
    prices = ['reference_price','far_price', 'near_price', 'ask_price', 'bid_price', 'wap']
    
    for c in combinations(prices, 2):
        
        df[f'{c[0]}_minus_{c[1]}'] = (df[f'{c[0]}'] - df[f'{c[1]}']).astype(np.float32)
        df[f'{c[0]}_times_{c[1]}'] = (df[f'{c[0]}'] * df[f'{c[1]}']).astype(np.float32)
        df[f'{c[0]}_{c[1]}_imb'] = df.eval(f'({c[0]}-{c[1]})/({c[0]}+{c[1]})')

    for c in combinations(prices, 3):
        
        max_ = df[list(c)].max(axis=1)
        min_ = df[list(c)].min(axis=1)
        mid_ = df[list(c)].sum(axis=1)-min_-max_

        df[f'{c[0]}_{c[1]}_{c[2]}_imb2'] = (max_-mid_)/(mid_-min_)
    
        
    df.drop(columns=[
        # 'date_id', 
        'reference_price_far_price_imb',
        'reference_price_minus_near_price',
        'reference_price_near_price_imb',
        'far_price_near_price_imb',
        'far_price_ask_price_imb',
        'far_price_bid_price_imb',
        'far_price_minus_wap',
        'std_size',
        'bid_size_over_ask_size',
        'ask_price_bid_price_imb',
        'near_price_times_wap'
    ], inplace=True)
        
    gc.collect()

    df.replace([np.inf, -np.inf], 0, inplace=True)
    
    return df

def model_pipeline(trading_df=trading_data, config=None):
    trading_df = trading_data
    with wandb.init(project="Optviver", config=config,save_code=True):
        wandb.define_metric("val_epoch_loss_l1", summary="min")
        wandb.define_metric("epoch_l1_loss", summary="min")
        config = wandb.config

        input_size = len(trading_df.stocksDict[0].data_daily[0][0])
        
        model = torch_classes.GRUNetV2(input_size,config['hidden_size'],num_layers=config['num_layers']).to('cuda:0')
        config = wandb.config
        optimizer = optim.RMSprop(model.parameters(), lr=config['learning_rate'])
        # optimizer = optim.Adadelta(model.parameters())
        print(model)
        trading_df.reset_hidden(hidden_size=config['hidden_size'],num_layers=config['num_layers'])
        criterion = nn.MSELoss()
        
        training_testing.train_model(trading_df,model,config,optimizer,criterion)

    return(model)



if __name__ == "__main__":

    train = pd.read_csv('train.csv')
    train.head()
    train.date_id.value_counts()

    weights = [
    0.004, 0.001, 0.002, 0.006, 0.004, 0.004, 0.002, 0.006, 0.006, 0.002, 0.002, 0.008,
    0.006, 0.002, 0.008, 0.006, 0.002, 0.006, 0.004, 0.002, 0.004, 0.001, 0.006, 0.004,
    0.002, 0.002, 0.004, 0.002, 0.004, 0.004, 0.001, 0.001, 0.002, 0.002, 0.006, 0.004,
    0.004, 0.004, 0.006, 0.002, 0.002, 0.04 , 0.002, 0.002, 0.004, 0.04 , 0.002, 0.001,
    0.006, 0.004, 0.004, 0.006, 0.001, 0.004, 0.004, 0.002, 0.006, 0.004, 0.006, 0.004,
    0.006, 0.004, 0.002, 0.001, 0.002, 0.004, 0.002, 0.008, 0.004, 0.004, 0.002, 0.004,
    0.006, 0.002, 0.004, 0.004, 0.002, 0.004, 0.004, 0.004, 0.001, 0.002, 0.002, 0.008,
    0.02 , 0.004, 0.006, 0.002, 0.02 , 0.002, 0.002, 0.006, 0.004, 0.002, 0.001, 0.02,
    0.006, 0.001, 0.002, 0.004, 0.001, 0.002, 0.006, 0.006, 0.004, 0.006, 0.001, 0.002,
    0.004, 0.006, 0.006, 0.001, 0.04 , 0.006, 0.002, 0.004, 0.002, 0.002, 0.006, 0.002,
    0.002, 0.004, 0.006, 0.006, 0.002, 0.002, 0.008, 0.006, 0.004, 0.002, 0.006, 0.002,
    0.004, 0.006, 0.002, 0.004, 0.001, 0.004, 0.002, 0.004, 0.008, 0.006, 0.008, 0.002,
    0.004, 0.002, 0.001, 0.004, 0.004, 0.004, 0.006, 0.008, 0.004, 0.001, 0.001, 0.002,
    0.006, 0.004, 0.001, 0.002, 0.006, 0.004, 0.006, 0.008, 0.002, 0.002, 0.004, 0.002,
    0.04 , 0.002, 0.002, 0.004, 0.002, 0.002, 0.006, 0.02 , 0.004, 0.002, 0.006, 0.02,
    0.001, 0.002, 0.006, 0.004, 0.006, 0.004, 0.004, 0.004, 0.004, 0.002, 0.004, 0.04,
    0.002, 0.008, 0.002, 0.004, 0.001, 0.004, 0.006, 0.004,
    ]
    
    weights_df = pd.DataFrame(data=list(zip(range(0,201),weights)),columns=['stock_id','index_weight'])
    train = train.merge(weights_df,on='stock_id')
    train['wap_calc'] = (train['bid_price']*train['ask_size']+train['ask_price']*train['bid_size'])/(train['ask_size']+train['bid_size'])

    train['wap_weighted'] = train['wap']*train['index_weight']
    train_g = train.groupby(['stock_id','date_id'])
    train = generate_prev_race(train,train_g)
    train['delta_wap'] = train['wap']/train['wap_t-60']

    train_g = train.groupby(['seconds_in_bucket','date_id'])
    train = generate_index(train,train_g)


    train['wap_move_to_init'] = train['wap_calc']/train['initial_wap']
    train_g = train.groupby(['date_id'])
    train = generate_index_2(train,train_g)

    train['index_wap_move_to_init'] = train['index_wap']/train['index_wap_init']
    train_g = train.groupby(['date_id'])
    train = generate_index_3(train,train_g)

    train['target_calc'] = -((train['wap_t-60']/train['wap'])-(train['index_wap_t-60']/train['index_wap_move_to_init']))*10000
    train['target_delta'] = train['target_t-60']-train['target_calc']

    median_vol = pd.read_csv("archive/MedianVolV2.csv")
    median_vol.index.name = "stock_id"
    median_vol = median_vol[['overall_medvol', "first5min_medvol", "last5min_medvol"]]
    median_sizes = train.groupby('stock_id')['bid_size'].median() + train.groupby('stock_id')['ask_size'].median()
    std_sizes = train.groupby('stock_id')['bid_size'].median() + train.groupby('stock_id')['ask_size'].median()
    
    train['bid_price_target'] = train['bid_price']-train['bid_price_t-60']
    train['bid_price_t-60'] = train['bid_price_target']*10_000

    y = train['target'].values
    X = feat_eng(train)
    prices = [c for c in train.columns if 'price' in c]
    pca_prices = PCA(n_components=1)
    X['pca_prices'] = pca_prices.fit_transform(X[prices].fillna(1))

    # X = train
    X = X.dropna(subset='bid_size_t-60').copy().reset_index()


    trading_data = torch_classes.TradingData(X)
    hidden_size = 64
    trading_data.generate_batches()

    for i,stocks in enumerate(trading_data.stocksDict.values()):
        if i==0:
            continue
        else:
            stocks.data_daily = []

    config_static = {'learning_rate':0.0001, 'hidden_size':256, 'num_layers':2, 'batch_norm':1}
    config = config_static

    sweep_config = {"method": "random"}

    metric = {"name": "val_epoch_loss", "goal": "minimize"}

    sweep_config["metric"] = metric


    parameters_dict = {
        "optimizer": {"values": ["adamW", 'adam', 'SGD', 'RMSprop']},
        "f0_layer_size": {"values": [128]},
        "f1_layer_size": {"values": [64]},
        "num_layers": {"values": [2,3,4,5]},
        'hidden_size':{'values':[128,256,512]},
        'learning_rate': {'values':[0.1,0.01,0.001,0.0005,0.0001,0.00005,0.00001]},
        'epochs':{'value':500}
        # 'batch_norm':{'values':[0,1,2]}
    }

    sweep_config["parameters"] = parameters_dict

    sweep_id = wandb.sweep(sweep_config, project="Optiver Sweeps")
    # CUDA_LAUNCH_BLOCKING=1
    wandb.agent(sweep_id, function=model_pipeline, count=100)