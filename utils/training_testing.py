import pandas as pd
import numpy as np
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pack_sequence,
    unpack_sequence,
    unpad_sequence,
)
import torch
from tqdm.notebook import trange, tqdm
import torch.nn as nn
import torch.optim as optim
import wandb
import utils.torch_classes as torch_classes
from utils.model_saver import model_saver_wandb as model_saver
import torch.nn.functional as F


def custom_MSE(x, y):
    return (((x - y) + 1) ** 2).mean()


def train_model(
    trading_df: torch_classes.TradingData,
    model: torch_classes.GRUNet,
    config: dict,
    optimizer,
    criterion,
):
    example_ct = 0
    epochs = config["epochs"]
    setup_loss = 1
    num_batches = len(trading_df.train_batches) - 2
    reg_L1 = nn.L1Loss()
    model = model.to("cuda:0")
    trading_df.reset_hidden(config["hidden_size"], config["num_layers"])
    
    mini_batches = config["mini_batches"]
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", verbose=True, factor=0.5
    )
    sft_max = nn.Softmax(dim=-1)

    for epoch in trange(epochs):
        model.train()
        loss_list = []
        setup_loss = 1

        for i in range(0, 384):
            # trading_df.reset_hidden(config["hidden_size"], config["num_layers"])

            stocks = [
                trading_df.stocksDict[x] for x in trading_df.stock_batches[i]
            ]  # Stocks for the Day
            stock_data = trading_df.train_batches

            example_ct += 1

            new_x = trading_df.train_batches[i]
            Y_price_ask = trading_df.train_ask_price_daily[i]
            Y_price_bid = trading_df.train_bid_price_daily[i]
            Y_price_wap = trading_df.train_wap_price_daily[i]
            Y_ohe = trading_df.train_class_ohe_batches[i]
            y = Y_ohe

            w = torch.tensor(trading_df.daily_variance[i], device="cuda:0")
            daily_weights = trading_df.daily_weights[i]

            criterion.weight = daily_weights

            hidden_in = torch.stack([x.hidden for x in stocks]).transpose(0, 1)

            # if i == 0:
            #     print(hidden_in.shape)
            #     print(hidden_in)

            output, output_wap, hidden, _, x_h = model(new_x, hidden_in)
            output = output.squeeze()
            hidden = hidden.transpose(0, 1)
            output_wap = output_wap.squeeze()

            [setattr(obj, "hidden", val.detach()) for obj, val in zip(stocks, hidden)]

            weight_dif = (
                abs(torch.argmax(output, dim=2) - torch.argmax(Y_ohe, dim=2)) + 1
            )

            # print(weight_dif.shape)
            # print(weight_dif)
            # print(output.flatten(end_dim=1).shape)
            # print(output.shape)
            # print(criterion(output.flatten(end_dim=1),Y_ohe.flatten(end_dim=1)).shape)
            loss = (
                criterion(output.flatten(end_dim=1), Y_ohe.flatten(end_dim=1))
            ).mean()
            # print(loss)

            L1_loss_wap = reg_L1(output_wap, Y_price_wap)
            loss = loss + L1_loss_wap
            loss.backward()
            if setup_loss:
                epoch_loss = loss
                # epoch_reg_l1 = L1_loss
                setup_loss = 0
                loss_count = 1
            else:
                if loss.isnan():
                    pass

                epoch_loss = loss + epoch_loss
                loss_count += 1
                # epoch_reg_l1 = L1_loss+epoch_reg_l1
                if i % mini_batches == 0:
                    if i == 0:
                        pass
                    else:
                        wandb.log({"epoch_loss": epoch_loss / loss_count})
                        # epoch_loss.backward()
                        optimizer.step()
                        trading_df.detach_hidden()

                        
                    setup_loss = 1

        if not setup_loss:
            wandb.log({"epoch_loss": epoch_loss / loss_count})
            # epoch_loss.backward()
            optimizer.step()

        trading_df.detach_hidden()
        wandb.log({"loss_1": torch.mean(loss).item()})

        wandb.log(
            {
                "epoch": epoch,
                #    "epoch_l1_loss": epoch_reg_l1/384,
                "output_sd": torch.std(output),
            }
        )
        validate_model(trading_df, model, criterion, epoch, config)
        # scheduler.step(val_loss)
        if epoch % 20 == 0:
            trading_df.create_hidden_states_dict_v2()
            model_saver(model, optimizer, epoch, 0, 0, trading_df.train_hidden_dict)
        trading_df.reset_hidden(
            hidden_size=config["hidden_size"], num_layers=config["num_layers"]
        )


@torch.no_grad()
def validate_model(
    trading_df: torch_classes.TradingData,
    model: torch_classes.GRUNet,
    criterion,
    epoch,
    config: dict,
):
    model.eval()

    val_batches = trading_df.val_batches
    len_val = len(val_batches)
    reg_L1 = nn.L1Loss()
    loss_list = []
    sft_max = nn.Softmax(dim=-1)

    reg_CEL = nn.CrossEntropyLoss(reduction="none")

    # print(len_val)
    output_dict = {}
    output_dict["stock"] = []
    output_dict["day"] = []
    output_dict["time"] = []
    # output_dict['id'] = []
    output_dict["target"] = []
    # output_dict['ask_target'] = []
    # output_dict['ask_pred'] = []
    # output_dict['bid_target'] = []
    # output_dict['bid_pred'] = []
    output_dict["wap_target"] = []
    output_dict["wap_pred"] = []
    output_dict["actual_wap"] = []
    output_dict["correct"] = []
    output_dict["actual"] = []
    output_dict["pred"] = []
    output_dict["loss"] = []
    output_dict["conf"] = []
    correct = 0

    means = trading_df.means_df

    time_periods = len(trading_df.stocksDict[0].data_daily[0])
    criterion = nn.CrossEntropyLoss()

    # print(f"{time_periods=}\n{len_val=}")

    # print(time_periods)

    for i in range(0, len_val - 1):
        # print(i)
        stocks = [trading_df.stocksDict[x] for x in trading_df.val_stock_batches[i]]
        stock_ids = trading_df.val_stock_batches[i]
        # train_dog_input = trading_df.batches['train_dog_input'][i]
        time_ids = list(range(0, time_periods))

        Y_actual_wap = trading_df.val_actual_wap[i]

        new_x = trading_df.val_batches[i]
        Y = trading_df.val_class_batches[i]
        Y_price_ask = trading_df.val_ask_price_daily[i]
        Y_price_bid = trading_df.val_bid_price_daily[i]
        Y_price_wap = trading_df.val_wap_price_daily[i]
        Y = Y_price_wap
        Y_ohe = trading_df.val_class_ohe_batches[i]
        y = Y_ohe

        hidden_in = torch.stack([x.hidden for x in stocks]).transpose(0, 1).contiguous()

        output, output_wap, hidden, relu, x_h = model(new_x, hidden_in)
        # output_wap,hidden,relu = model(X,hidden_in)
        hidden = hidden.transpose(0, 1)
        output_wap = output_wap.squeeze()
        # print(hidden_in)

        [setattr(obj, "hidden", val.detach()) for obj, val in zip(stocks, hidden)]

        loss_wap = reg_L1(output_wap, Y_price_wap)

        loss = criterion(output.flatten(end_dim=1), Y_ohe.flatten(end_dim=1))
        loss_list = reg_CEL(output.flatten(end_dim=1), Y_ohe.flatten(end_dim=1))

        _, actual = torch.max(y, 2)
        # onehot_win = F.one_hot(actual, num_classes=7)
        conf, predicted = torch.max(output, 2)

        # One hot wins
        label = torch.zeros_like(y).scatter_(
            1, torch.argmax(y, dim=1).unsqueeze(1), 1.0
        )
        pred_label = torch.zeros_like(output).scatter_(
            1, torch.argmax(output, dim=1).unsqueeze(1), 1.0
        )
        # correct_tensor = label*pred_label

        correct_l = predicted == actual
        softmax_preds = sft_max(output)

        if i == 0:
            epoch_loss = loss
            L1_loss_wap_epoch = loss_wap
        else:
            epoch_loss = loss + epoch_loss
            L1_loss_wap_epoch += loss_wap

        output_dict["stock"].append(stock_ids * time_periods)
        output_dict["day"].append([i + 384] * time_periods * len(stock_ids))
        output_dict["time"].extend([[x] * len(stock_ids) for x in time_ids])
        output_dict["target"].append(Y.flatten().tolist())
        output_dict["correct"].append(correct_l.flatten().tolist())
        output_dict["actual"].append(actual.flatten().tolist())
        output_dict["pred"].append(predicted.flatten().tolist())
        output_dict["wap_target"].append(Y_price_wap.flatten().tolist())
        output_dict["wap_pred"].append(output_wap.flatten().tolist())
        output_dict["actual_wap"].append(Y_actual_wap.flatten().tolist())
        output_dict["loss"].append(loss_list.flatten().tolist())
        output_dict["conf"].append(conf.flatten().tolist())

    for k, value in output_dict.items():
        # print(k)
        # print(value)
        output_dict[k] = [item for sublist in value for item in sublist]
        # print(len(output_dict[k]))

    # print({"epoch_loss": epoch_loss/384, "epoch_l1_loss": epoch_reg_l1/384, 'epoch':epoch})
    log_dict = pd.DataFrame(data=output_dict)
    log_dict = log_dict.merge(
        means, left_on="pred", right_on="original_index", how="inner"
    )
    log_dict = log_dict.drop(columns="wap_category")

    log_dict["loss_adj"] = abs(log_dict.mean_wap - log_dict.wap_target)
    # print(log_dict)

    wandb.log(
        {
            "val_epoch_loss": epoch_loss / len_val,
            "val_loss": torch.mean(loss).item(),
            "epoch": epoch,
            "relu_sum": relu.sum(),
            "L1_loss_wap_epoch": L1_loss_wap_epoch / len_val,
            "Accuracy": log_dict.correct.mean(),
            "wap_pred_loss": log_dict["loss_adj"].mean(),
            "losst_to_zero": abs(log_dict["wap_target"]).mean(),
        }
    )

    if epoch % 10 == 0:
        log_dict.sort_values(inplace=True, ascending=True, by=["day", "time", "stock"])
        cm = wandb.plot.confusion_matrix(
            y_true=log_dict["actual"],
            preds=log_dict["pred"],
            class_names=means.cat_name.tolist(),
        )
        log_dict.to_feather(
            f"archive/validation_outputs/{wandb.run.name}_validation_output_{epoch}.fth"
        )

        loss_dict = wandb.Table(
            data=log_dict.groupby("day")[["loss"]].mean().reset_index()
        )
        log_dict = log_dict.head(200 * 49)
        # print(log_dict.pred.value_counts())
        log_dict = wandb.Table(data=log_dict)
        # print(log_dict)
        try:
            wandb.log({"data_table": log_dict, "loss_table": loss_dict,"conf_mat": cm})
        except Exception as e:


            pass

    return epoch_loss / len_val
