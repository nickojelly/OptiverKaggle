import torch
import os
import wandb


def model_saver_linux(model, optimizer, epoch, loss, hidden_state_dict, train_state_dict, model_name=None):
    if not model_name:
        model_name = "nz_model"
    isExist = os.path.exists("models/")
    if isExist:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optim": optimizer.state_dict(),
                "loss": loss,
                "db": hidden_state_dict,
                "db_train": train_state_dict,
            },
            f"models/{model_name}.pt",
        )
    else:
        os.makedirs("models/")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optim": optimizer.state_dict(),
                "loss": loss,
                "db": hidden_state_dict,
                "db_train": train_state_dict,
            },
            f"models/{model_name}.pt",
        )


def model_saver_wandb(model_ohe,model_reg, optimizer, epoch, loss, hidden_state_dict, train_state_dict, model_name=None):
    model_name = wandb.run.name
    if not model_name:
        model_name = "test NZ GRU saver"
    isExist = os.path.exists(f"models/{model_name}/")
    if isExist:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict_ohe": model_ohe.state_dict(),
                "model_state_dict_reg": model_reg.state_dict(),
                "optim": optimizer.state_dict(),
                "loss": loss,
                "db": hidden_state_dict,
                "db_train": train_state_dict,
            },
            f"models/{model_name}/{model_name}_{epoch}.pt",
        )
    else:
        os.makedirs(f"models/{model_name}/")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model_ohe.state_dict(),
                "model_state_dict_reg": model_reg.state_dict(),
                "optim": optimizer.state_dict(),
                "loss": loss,
                "db": hidden_state_dict,
                "db_train": train_state_dict,
            },
            f"models/{model_name}/{model_name}_{epoch}.pt",
        )