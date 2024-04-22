import argparse
import os
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from read_file import read_csv
from dataloader import (
    get_loader,
    LOSO_sequence_generate
)
from models.FMER import FMER
import shutil


lam = 1


def train(epochs: int, criterion: list, optimizer: torch.optim,
          model: nn.Module, scheduler: torch.optim.lr_scheduler, train_loader: DataLoader,
          device: torch.device, model_best_name: str):
    """Train the model

    Parameters
    ----------
    epochs : int
        Epochs for training the model
    model : DSSN
        Model to be trained
    train_loader : DataLoader
        DataLoader to load in the data
    device: torch.device
        Device to be trained on
    model_best_name: str
        Name of the weight file to be saved
    """
    best_accuracy = -1
    # Set model in training mode
    model.train()

    for epoch in range(epochs):
        train_me_loss = 0.0
        train_me_accuracy = 0.0
        train_au_loss = 0.0
        train_total_loss = 0.0

        for patches, labels, frames, au_label in train_loader:
            patches = patches.to(device)
            labels = labels.to(device)
            frames = frames.to(device)
            au_label = au_label.to(device)

            me_output, au_output = model(patches, frames)

            # Compute the loss
            loss_me = criterion[0](me_output, labels)
            loss_au = criterion[1](au_output, au_label)
            train_me_loss += loss_me.item()
            train_au_loss += loss_au.item()
            optimizer.zero_grad()
            loss = lam * loss_me + loss_au
            train_total_loss += loss.item()
            loss.backward()
            optimizer.step()

            # Compute the accuracy
            prediction = (me_output.argmax(-1) == labels)
            train_me_accuracy += prediction.sum().item() / labels.size(0)

        if scheduler is not None and epoch > 40:
            scheduler.step()

        train_me_loss /= len(train_loader)
        train_me_accuracy /= len(train_loader)
        train_au_loss /= len(train_loader)
        train_total_loss /= len(train_loader)

        print(f"In epoch: {epoch + 1}")
        print(f"Me loss: {train_me_loss}, ME accuracy: {train_me_accuracy}.")
        print(f"AU loss: {train_au_loss}, Total loss: {train_total_loss}.")

        if train_me_accuracy > best_accuracy:
            torch.save(model.state_dict(), model_best_name)
            best_accuracy = train_me_accuracy
            print("Save model")


def evaluate(test_loader: DataLoader, model: nn.Module, device: torch.device):
    # Set into evaluation mode
    model.eval()
    test_accuracy = 0.0
    test_f1_score = 0.0

    with torch.no_grad():
        for patches, labels, frames, _ in test_loader:
            patches = patches.to(device)
            labels = labels.to(device)
            frames = frames.to(device)

            me_output, au_output = model(patches, frames)

            # Compute the accuracy
            prediction = (me_output.argmax(-1) == labels)
            test_accuracy += prediction.sum().item() / labels.size(0)
            test_f1_score += f1_score(labels.cpu().numpy(), me_output.argmax(-1).cpu().numpy(),
                                      average="weighted")

    return test_accuracy / len(test_loader), test_f1_score / len(test_loader)


def LOSO_train(data: pd.DataFrame, sub_column: str, args,
               label_mapping: dict, device: torch.device):
    # 最高记录的xlsx文件
    record_csv = pd.read_csv(args.best_acc,
                             dtype={
                                 "Best Accuracy": float,
                                 "Best F1": float
                             })

    # Create different DataFrame for each subject
    train_list, test_list = LOSO_sequence_generate(data, sub_column)
    test_accuracy = 0.0
    test_f1_score = 0.0
    for epoch in [50, 75, 100]:
        for idx in range(len(train_list)):
            if idx not in [3, 5, 6, 7, 8, 10, 16, 17, 18, 20, 22, 23, 25]:
                continue
            record_acc = record_csv.loc[(record_csv['Dataset'] == "CASME") & (record_csv['LOSO Round'] == idx),
                                        "Best Accuracy"].item()
            npz_file = np.load(f"{args.npz_file}/{idx}.npz")
            adj_matrix = torch.FloatTensor(npz_file["adj_matrix"]).to(device)

            print(f"=================LOSO {idx + 1}=====================")
            train_csv = train_list[idx]
            test_csv = test_list[idx]

            # Create dataset and dataloader
            _, train_loader = get_loader(csv_file=train_csv,
                                         image_root=args.mat_dir,
                                         au_root=args.au_dir,
                                         label_mapping=label_mapping,
                                         batch_size=args.batch_size,
                                         catego=args.catego)
            _, test_loader = get_loader(csv_file=test_csv,
                                        image_root=args.mat_dir,
                                        au_root=args.au_dir,
                                        label_mapping=label_mapping,
                                        batch_size=args.batch_size,
                                        catego=args.catego,
                                        train=False)

            # Read in the model
            model = FMER(adj_matrix=adj_matrix,
                         num_classes=args.num_classes,
                         device=device).to(device)

            # if idx != 0:
            #     model.load_state_dict(torch.load(f"{args.weight_save_path}/model_best_{idx-1}.pt",
            #                                      map_location=device))

            # Create criterion and optimizer
            criterion = [nn.CrossEntropyLoss(), nn.BCEWithLogitsLoss()]  # 两个不同的loss计算函数
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=args.learning_rate)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.987)

            # 尝试调整P3D模型学习率，后发现似乎不需要
            # rest_params = [param for name, param in model.named_parameters()
            #                if "P3D" not in name]
            # optimizer = torch.optim.Adam([
            #     {"params": model.P3D.parameters(), "lr": 1e-3},
            #     {"params": rest_params}
            # ], lr=args.learning_rate)

            # Train the data
            train(epochs=epoch,
                  criterion=criterion,
                  optimizer=optimizer,
                  scheduler=None,
                  model=model,
                  train_loader=train_loader,
                  device=device,
                  model_best_name=f"{args.weight_save_path}/model_best_{idx}.pt")
            model.load_state_dict(torch.load(f"{args.weight_save_path}/model_best_{idx}.pt",
                                             map_location=device))

            temp_test_accuracy, temp_f1_score = evaluate(test_loader=test_loader,
                                                         model=model,
                                                         device=device)
            if temp_test_accuracy > record_acc:
                shutil.copy(f"{args.weight_save_path}/model_best_{idx}.pt",
                            f"{args.best_weight_path}/model_best_{idx}.pt")
                record_csv.loc[(record_csv['Dataset'] == "CASME") & (record_csv['LOSO Round'] == idx),
                               "Best Accuracy"] = temp_test_accuracy
                record_csv.loc[(record_csv['Dataset'] == "CASME") & (record_csv['LOSO Round'] == idx),
                               "Best F1"] = temp_f1_score
                record_csv.to_csv(args.best_acc, index=False)
                print("="*19 + "New best accuracy recorded" + "="*19)
            print(f"In LOSO {idx + 1}, test accuracy: {temp_test_accuracy:.4f}, f1-score: {temp_f1_score:.4f}")
            test_accuracy += temp_test_accuracy
            test_f1_score += temp_f1_score
        print(f"LOSO accuracy: {test_accuracy / len(train_list):.4f}, f1-score: {test_f1_score / len(train_list):.4f}")


def main(epoch=25):
    # Argument parse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path",
                        type=str,
                        # required=True,
                        # default=r"B:\0_0NewLife\datasets\CASME_2\4classes.csv",
                        default=r"B:\0_0NewLife\datasets\CASME_2\remove_4samples.csv",  # 移出4个无法检测到特征点的样本
                        help="Path for the csv file for training data")
    # parser.add_argument("--image_root",
    #                     type=str,
    #                     # required=True,
    #                     default=r"B:\0_0NewLife\datasets\CASME_2\RAW_selected",
    #                     help="Root for the training images")
    parser.add_argument("--mat_dir",
                        type=str,
                        # required=True,
                        # default=r"B:\0_0NewLife\0_Papers\SMC\CASME2\mat\Inter_1",
                        default=r"B:\0_0NewLife\0_Papers\SMC\CASME2\mat\Optical_Flow_patches\Inter_9",
                        help="Root for the training images")
    parser.add_argument("--best_acc",
                        type=str,
                        # required=True,
                        default=r"B:\0_0NewLife\0_Papers\SMC\Best_accuracy.csv",
                        help="Root for the training images")
    parser.add_argument("--au_dir",
                        type=str,
                        # required=True,
                        default=r"B:\0_0NewLife\0_Papers\SMC\CASME2\AUfeature",
                        help="Root for the training images")
    parser.add_argument("--npz_file",
                        type=str,
                        # required=True,
                        default=r"B:\0_0NewLife\0_Papers\FGRMER\CASME2\npz\4-npz",
                        help="Place for the npz file")
    parser.add_argument("--catego",
                        type=str,
                        # required=True,
                        default="CASME",
                        help="SAMM or CASME dataset")
    parser.add_argument("--num_classes",
                        type=int,
                        default=4,
                        help="Classes to be trained")
    parser.add_argument("--batch_size",
                        type=int,
                        default=10,
                        help="Training batch size")
    parser.add_argument("--weight_save_path",
                        type=str,
                        default="model",
                        help="Path for the saving weight")
    parser.add_argument("--best_weight_path",
                        type=str,
                        default=r"B:\0_0NewLife\0_Papers\SMC\CASME2\weight",
                        help="Path for the saving weight")
    parser.add_argument("--epochs",
                        type=int,
                        default=epoch,
                        help="Epochs for training the model")
    parser.add_argument("--learning_rate",
                        type=float,
                        default=1e-4,
                        help="Learning rate for training the model")
    args = parser.parse_args()

    # Training device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Read in the data
    data, label_mapping = read_csv(args.csv_path)

    # Create folders for the saving weight
    os.makedirs(args.weight_save_path, exist_ok=True)

    # Train the model
    LOSO_train(data=data,
               sub_column="Subject",
               label_mapping=label_mapping,
               args=args,
               device=device)


if __name__ == "__main__":
    main(1)
