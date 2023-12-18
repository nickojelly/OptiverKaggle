
import pandas as pd 
import numpy as np
import utils.public_timeseries_testing_util as optiver2023
from torch.nn.utils.rnn import pack_padded_sequence, pack_sequence, unpack_sequence, unpad_sequence
import torch
from tqdm.notebook import trange,tqdm
import torch.nn as nn 
import torch.optim as optim
import wandb
import utils.torch_classes as torch_classes
from utils.model_saver import model_saver_wandb as model_saver
import utils.training_testing_double 
from itertools import combinations
from sklearn.decomposition import PCA
import importlib
import gc
from utils.conts import stat_col_full
# from utils.conts import lgbm_columns
import time
import lightgbm as lgb

lgbm_columns = ['stock_id',
 'seconds_in_bucket',
 'imbalance_size',
 'imbalance_buy_sell_flag',
 'reference_price',
 'matched_size',
 'far_price',
 'near_price',
 'bid_price',
 'bid_size',
 'ask_price',
 'ask_size',
 'wap',
 'index_weight',
 'wap_calc',
 'wap_weighted',
 'initial_wap',
 'initial_bid_size',
 'initial_ask_size',
 'bid_price_t10',
 'ask_price_t10',
 'bid_size_t10',
 'ask_size_t10',
 'wap_t10',
 'index_wap',
 'wap_move_to_init',
 'index_wap_init',
 'index_wap_move_to_init',
 'wap_prev_move',
 'bid_price_prev_move',
 'ask_price_prev_move',
 'overall_medvol',
 'first5min_medvol',
 'last5min_medvol',
 'bid_plus_ask_sizes',
 'imbalance_ratio',
 'imb_s1',
 'imb_s2',
 'ask_x_size',
 'bid_x_size',
 'ask_minus_bid',
 'bid_price_over_ask_price',
 'reference_price_minus_far_price',
 'reference_price_times_far_price',
 'reference_price_times_near_price',
 'reference_price_minus_ask_price',
 'reference_price_times_ask_price',
 'reference_price_ask_price_imb',
 'reference_price_minus_bid_price',
 'reference_price_times_bid_price',
 'reference_price_bid_price_imb',
 'reference_price_minus_wap',
 'reference_price_times_wap',
 'reference_price_wap_imb',
 'far_price_minus_near_price',
 'far_price_times_near_price',
 'far_price_minus_ask_price',
 'far_price_times_ask_price',
 'far_price_minus_bid_price',
 'far_price_times_bid_price',
 'far_price_times_wap',
 'far_price_wap_imb',
 'near_price_minus_ask_price',
 'near_price_times_ask_price',
 'near_price_ask_price_imb',
 'near_price_minus_bid_price',
 'near_price_times_bid_price',
 'near_price_bid_price_imb',
 'near_price_minus_wap',
 'near_price_wap_imb',
 'ask_price_minus_bid_price',
 'ask_price_times_bid_price',
 'ask_price_minus_wap',
 'ask_price_times_wap',
 'ask_price_wap_imb',
 'bid_price_minus_wap',
 'bid_price_times_wap',
 'bid_price_wap_imb',
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
 'far_price_near_price_bid_price_imb2',
 'far_price_near_price_wap_imb2',
 'far_price_ask_price_bid_price_imb2',
 'far_price_ask_price_wap_imb2',
 'far_price_bid_price_wap_imb2',
 'near_price_ask_price_bid_price_imb2',
 'near_price_ask_price_wap_imb2',
 'near_price_bid_price_wap_imb2',
 'ask_price_bid_price_wap_imb2',
 'pca_prices']

model = torch_classes.GRUNetV4_sub(89,512,num_layers=2,target_size=15)
model_loc = f"models/expert-blaze-390/expert-blaze-390_20.pt"
model_data = torch.load(model_loc,map_location=torch.device('cpu'))
print(model_data['model_state_dict'].keys())
model.load_state_dict(model_data['model_state_dict'], strict=True)

train = pd.read_csv('data/train.csv')
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

median_vol = pd.read_csv("archive/MedianVolV2.csv")
median_vol.index.name = "stock_id"
median_vol = median_vol[['overall_medvol', "first5min_medvol", "last5min_medvol"]]
median_sizes = train.groupby('stock_id')['bid_size'].median() + train.groupby('stock_id')['ask_size'].median()
std_sizes = train.groupby('stock_id')['bid_size'].median() + train.groupby('stock_id')['ask_size'].median() 

trading_data = torch_classes.TradingData()
trading_data.fill_hidden_states_for_test(model_data['db_train'])

trading_data.fill_hidden_states_for_test(model_data['db_train'])

lgbm = lgb.Booster(model_file="data/lgbm_model_new_t60.lgb")

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
        
    # gc.collect()

    df.replace([np.inf, -np.inf], 0, inplace=True)
    
    return df

def generate_prev_race(df_in, df_g, rolling_window=10, factor=''):
    df = df_in.copy()
    original_cols = df_in.columns
    df[f'initial_wap'] = df_g['wap_calc'].transform('first')
    df[f'initial_bid_size'] = df_g['bid_size'].transform('first')
    df[f'initial_ask_size'] = df_g['ask_size'].transform('first')
    cols = ["bid_price", "ask_price", "bid_size", "ask_size", "wap"]
    for i in cols:
        df[f"{i}_t10"] = df_g[i].shift(1)
    return(df)

def generate_index(df_in, df_g, rolling_window=10, factor=''):
    df = df_in.copy()
    df[f'index_wap'] = df_g['wap_weighted'].transform('mean')
    return(df)

def generate_index_2(df_in, df_g, rolling_window=10, factor=''):
    df = df_in.copy()
    df[f'index_wap_init'] = df_g['index_wap'].transform('first')
    return(df)

def variable_eng(train):

    train['wap_weighted'] = train['wap']*train['index_weight']
    train_g = train.groupby(['stock_id','date_id'])
    train = generate_prev_race(train,train_g)

    train_g = train.groupby(['seconds_in_bucket','date_id'])
    train = generate_index(train,train_g)


    train['wap_move_to_init'] = train['wap_calc']/train['initial_wap']
    train_g = train.groupby(['date_id'])
    train = generate_index_2(train,train_g)

    train['index_wap_move_to_init'] = train['index_wap']/train['index_wap_init']
    targets = ["wap", "bid_price", "ask_price"]
    for i in targets:
        train[f"{i}_prev_move"] = (train[f"{i}"] - train[f"{i}_t10"]).fillna(0) * 10000

    return train

y = train["target"].values
X = feat_eng(train)
prices = [
    c for c in X.columns if ("price" in c) and ("target" not in c) and ("60" not in c)
]
print(prices)
prices = [
    c for c in X.columns if ("price" in c) and ("target" not in c) and ("60" not in c)
]
# prices = [c for c in train.columns if 'price' in c]
pca_prices = PCA(n_components=1)
X["pca_prices"] = pca_prices.fit_transform(X[prices].fillna(1))

def model_forward_pass(model, new_x, hidden_in):
    output_wap_ohe, output_wap, hidden, relu, x_h = model(new_x, hidden_in)
    output_wap_ohe = output_wap_ohe.squeeze()
    hidden = hidden.transpose(0, 1)
    output_wap = output_wap.squeeze()
    return output_wap_ohe, output_wap, hidden, x_h , relu

def gen_preds(test, trading_data:torch_classes.TradingData, model:torch_classes.GRUNet):
    # for col in test.columns:
    #     print(col)
    test['stats']  = pd.Series(test[stat_col_full].fillna(-1).values.tolist())
    stock_ids = test.stock_id.unique().tolist()
    stocks = [trading_data.stocksDict[x] for x in stock_ids]
    hidden = torch.stack([trading_data.stocksDict[x].hidden for x in stock_ids]).transpose(0,1).squeeze()
    
    X = [torch.tensor(x) for x in test['stats'].tolist()]
    
    Xstacked = torch.stack(X).unsqueeze(1)

    # print(f"{Xstacked.shape=}")
    # print(f"{hidden.shape=},{Xstacked.shape=}")
    
    
    output_wap_ohe, output_wap, hidden, x_h , relu = model_forward_pass(model, Xstacked, hidden)

    # print(hidden.shape)
    # hidden = hidden.transpose(0,1)
    [setattr(obj, "hidden", val) for obj, val in zip(stocks, hidden)]
    
    output = output_wap.flatten().tolist()
    
    return output

env = optiver2023.make_env()
iter_test = env.iter_test()
trading_data.fill_hidden_states_for_test(model_data['db_train'])

counter = 0
for (test, revealed_targets, sample_prediction) in iter_test:
    test_in = test[['stock_id','seconds_in_bucket','date_id']].copy()
    if counter == 0:
        print(test.head(3))
        print(revealed_targets.head(3))
        print(sample_prediction.head(3))
        all_df = test
    else:
        # all_df = pd.concat([all_df,test])
        pass
    model = model.eval()
    # test = all_df
    print(counter)
    test['wap_calc'] = (test['bid_price']*test['ask_size']+test['ask_price']*test['bid_size'])/(test['ask_size']+test['bid_size'])
    # t1 = time.perf_counter()
    test = test.merge(weights_df,on='stock_id')
    # t2 = time.perf_counter()
    # print(f"1{t2-t1=}")
    # t1 = time.perf_counter()
    test = feat_eng(test)
    # t2 = time.perf_counter()
    # print(f"feat eng{t2-t1=}")
    # t1 = time.perf_counter()
    test['pca_prices'] = pca_prices.transform(test[prices].fillna(1))
    if counter ==0:        
        test = variable_eng(test)
    else:
        test = pd.concat([all_df,test])
        test = variable_eng(test)
    test = test.merge(test_in,on=['stock_id','seconds_in_bucket','date_id'],how='inner')
    x = test[[c for c in test.columns if ("target" not in c) and ("60" not in c)]].drop(columns=["date_id","stats"])
    lgbm_preds = lgbm.predict(x[lgbm_columns])
    test["lgbm_preds"] = lgbm_preds

    if counter==300:
        break
    
    
    # print(test.shape)
    preditcions = gen_preds(test,trading_data,model)
    # t2 = time.perf_counter()
    # print(f"model {t2-t1=}")
    # t1 = time.perf_counter()
    
    sample_prediction['pred'] = preditcions
    test_df = test
    # print(preditcions)
    # break
    env.predict(sample_prediction)
    counter += 1
