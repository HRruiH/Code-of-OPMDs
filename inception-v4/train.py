# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
import time

import torch
from torch import nn
from torch import optim
from torch.cuda import amp
from torch.optim import lr_scheduler
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import config
from dataset import CUDAPrefetcher, ImageDataset
from utils import accuracy, load_state_dict, make_directory, save_checkpoint, Summary, AverageMeter, ProgressMeter
import model

model_names = sorted(
    name for name in model.__dict__ if name.islower() and not name.startswith("__") and callable(model.__dict__[name]))


def main():
    # Initialize the number of training epochs
    start_epoch = 0

    # Initialize training network evaluation indicators
    best_acc1 = 0.0

    train_prefetcher, valid_prefetcher = load_dataset()
    print(f"Load `{config.model_arch_name}` datasets successfully.")

    inception_v3_model, ema_inception_v3_model = build_model()
    print(f"Build `{config.model_arch_name}` model successfully.")

    pixel_criterion = define_loss()
    print("Define all loss functions successfully.")

    optimizer = define_optimizer(inception_v3_model)
    print("Define all optimizer functions successfully.")

    scheduler = define_scheduler(optimizer)
    print("Define all optimizer scheduler functions successfully.")

    print("Check whether to load pretrained model weights...")
    if config.pretrained_model_weights_path:
        inception_v3_model, ema_inception_v3_model, start_epoch, best_acc1, optimizer, scheduler = load_state_dict(
            inception_v3_model,
            config.pretrained_model_weights_path,
            ema_inception_v3_model,
            start_epoch,
            best_acc1,
            optimizer,
            scheduler)
        print(f"Loaded `{config.pretrained_model_weights_path}` pretrained model weights successfully.")
    else:
        print("Pretrained model weights not found.")

    print("Check whether the pretrained model is restored...")
    if config.resume:
        inception_v3_model, ema_inception_v3_model, start_epoch, best_acc1, optimizer, scheduler = load_state_dict(
            inception_v3_model,
            config.pretrained_model_weights_path,
            ema_inception_v3_model,
            start_epoch,
            best_acc1,
            optimizer,
            scheduler,
            "resume")
        print("Loaded pretrained generator model weights.")
    else:
        print("Resume training model not found. Start training from scratch.")

    # Create a experiment results
    samples_dir = os.path.join("samples", config.exp_name)
    results_dir = os.path.join("results", config.exp_name)
    make_directory(samples_dir)
    make_directory(results_dir)

    # 定义最佳模型路径
    best_model_path = os.path.join(samples_dir, "best_model.pth.tar")

    # Create training process log file
    writer = SummaryWriter(os.path.join("samples", "logs", config.exp_name))

    # Initialize the gradient scaler
    scaler = amp.GradScaler()
    # 早停相关参数
    early_stopping_counter = 0
    early_stopping_patience = 10  # 容忍的轮数
    best_epoch = 0
    for epoch in range(start_epoch, config.epochs):
        train(inception_v3_model, ema_inception_v3_model, train_prefetcher, pixel_criterion, optimizer, epoch, scaler,
              writer)
        acc1 = validate(ema_inception_v3_model, valid_prefetcher, epoch, writer, "Valid")
        print("\n")

        # Update LR
        scheduler.step()
        # 早停逻辑
        if acc1 > best_acc1:
            best_epoch = epoch
            early_stopping_counter = 0  # 重置计数器
            print(f"Epoch {epoch + 1}: 验证准确率提升至 {acc1:.4f}，重置早停计数器")
        else:
            early_stopping_counter += 1
            print(f"早停计数器: {early_stopping_counter}/{early_stopping_patience}")
            if early_stopping_counter >= early_stopping_patience:
                print(f"早停触发！连续{early_stopping_patience}轮验证准确率未提升，最佳模型在第{best_epoch + 1}轮")
                break  # 终止训练循环

        # Automatically save the model with the highest index
        is_best = acc1 > best_acc1
        if is_best:
            print(f"Epoch {epoch + 1}: 验证准确率从 {best_acc1:.4f} 提升到 {acc1:.4f}")
            print(f"保存新的最佳模型到 {best_model_path}")

            # 如果已存在最佳模型，删除它
            if os.path.exists(best_model_path):
                os.remove(best_model_path)
                print(f"删除旧的最佳模型: {best_model_path}")

            # 保存新的最佳模型
            save_checkpoint({
                "epoch": epoch + 1,
                "best_acc1": acc1,
                "state_dict": inception_v3_model.state_dict(),
                "ema_state_dict": ema_inception_v3_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict()
            }, "best_model.pth.tar", samples_dir, results_dir, is_best=True, is_last=False)

            best_acc1 = acc1

    # 训练结束后，复制最佳模型到结果目录
    if os.path.exists(best_model_path):
        final_model_path = os.path.join(results_dir, "final_model.pth.tar")
        if os.path.exists(final_model_path):
            os.remove(final_model_path)
        os.system(f"cp {best_model_path} {final_model_path}")
        print(f"训练结束，最终模型已保存到: {final_model_path}")
    else:
        print("警告: 未找到最佳模型，可能验证准确率未提升过")


def load_dataset() -> [CUDAPrefetcher, CUDAPrefetcher]:
    # Load train, test and valid datasets
    train_dataset = ImageDataset(config.train_image_dir,
                                 config.image_size,
                                 config.model_mean_parameters,
                                 config.model_std_parameters,
                                 "Train")
    valid_dataset = ImageDataset(config.valid_image_dir,
                                 config.image_size,
                                 config.model_mean_parameters,
                                 config.model_std_parameters,
                                 "Valid")

    # Generator all dataloader
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=config.num_workers,
                                  pin_memory=True,
                                  drop_last=True,
                                  persistent_workers=True)
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=False,
                                  num_workers=config.num_workers,
                                  pin_memory=True,
                                  drop_last=False,
                                  persistent_workers=True)

    # Place all data on the preprocessing data loader
    train_prefetcher = CUDAPrefetcher(train_dataloader, config.device)
    valid_prefetcher = CUDAPrefetcher(valid_dataloader, config.device)

    return train_prefetcher, valid_prefetcher


def build_model() -> [nn.Module, nn.Module]:
    # 修复：移除不支持的 transform_input 参数
    inception_v3_model = model.__dict__[config.model_arch_name](
        num_classes=config.model_num_classes
    )
    inception_v3_model = inception_v3_model.to(device=config.device, memory_format=torch.channels_last)

    ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: (
                                                                                          1 - config.model_ema_decay) * averaged_model_parameter + config.model_ema_decay * model_parameter
    ema_inception_v3_model = AveragedModel(inception_v3_model, avg_fn=ema_avg)

    return inception_v3_model, ema_inception_v3_model


def define_loss() -> nn.CrossEntropyLoss:
    criterion = nn.CrossEntropyLoss(label_smoothing=config.loss_label_smoothing)
    criterion = criterion.to(device=config.device, memory_format=torch.channels_last)

    return criterion


def define_optimizer(model) -> optim.SGD:
    optimizer = optim.SGD(model.parameters(),
                          lr=config.model_lr,
                          momentum=config.model_momentum,
                          weight_decay=config.model_weight_decay)

    return optimizer


def define_scheduler(optimizer: optim.SGD) -> lr_scheduler.CosineAnnealingWarmRestarts:
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                         config.lr_scheduler_T_0,
                                                         config.lr_scheduler_T_mult,
                                                         config.lr_scheduler_eta_min)

    return scheduler


def train(
        model: nn.Module,
        ema_model: nn.Module,
        train_prefetcher: CUDAPrefetcher,
        criterion: nn.CrossEntropyLoss,
        optimizer: optim.Adam,
        epoch: int,
        scaler: amp.GradScaler,
        writer: SummaryWriter
) -> None:
    # Calculate how many batches of data are in each Epoch
    batches = len(train_prefetcher)
    # Print information of progress bar during training
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":6.6f")
    acc1 = AverageMeter("Acc@1", ":6.4f")
    progress = ProgressMeter(batches,
                             [batch_time, data_time, losses, acc1],
                             prefix=f"Epoch: [{epoch + 1}]")

    # Put the generative network model in training mode
    model.train()

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initialize the data loader and load the first batch of data
    train_prefetcher.reset()
    batch_data = train_prefetcher.next()

    # Get the initialization training time
    end = time.time()

    while batch_data is not None:
        # Calculate the time it takes to load a batch of data
        data_time.update(time.time() - end)

        # Transfer in-memory data to CUDA devices to speed up training
        images = batch_data["image"].to(device=config.device, memory_format=torch.channels_last, non_blocking=True)
        target = batch_data["target"].to(device=config.device, non_blocking=True)

        # Get batch size
        batch_size = images.size(0)

        # Initialize generator gradients
        model.zero_grad(set_to_none=True)

        # Mixed precision training
        with amp.autocast():
            output = model(images)
            loss = config.loss_weights * criterion(output, target)

        # Backpropagation
        scaler.scale(loss).backward()
        # update generator weights
        scaler.step(optimizer)
        scaler.update()

        # Update EMA
        ema_model.update_parameters(model)

        # measure accuracy and record loss
        top1 = accuracy(output, target, topk=(1,))
        losses.update(loss.item(), batch_size)
        acc1.update(top1[0].item(), batch_size)

        # Calculate the time it takes to fully train a batch of data
        batch_time.update(time.time() - end)
        end = time.time()

        # Write the data during training to the training log file
        if batch_index % config.train_print_frequency == 0:
            # Record loss during training and output to file
            writer.add_scalar("Train/Loss", loss.item(), batch_index + epoch * batches + 1)
            progress.display(batch_index + 1)

        # Preload the next batch of data
        batch_data = train_prefetcher.next()

        # Add 1 to the number of data batches to ensure that the terminal prints data normally
        batch_index += 1


def validate(
        ema_model: nn.Module,
        data_prefetcher: CUDAPrefetcher,
        epoch: int,
        writer: SummaryWriter,
        mode: str
) -> float:
    # Calculate how many batches of data are in each Epoch
    batches = len(data_prefetcher)
    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    acc1 = AverageMeter("Acc@1", ":6.4f", Summary.AVERAGE)
    progress = ProgressMeter(batches, [batch_time, acc1], prefix=f"{mode}: ")

    # Put the exponential moving average model in the verification mode
    ema_model.eval()

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initialize the data loader and load the first batch of data
    data_prefetcher.reset()
    batch_data = data_prefetcher.next()

    # Get the initialization test time
    end = time.time()

    with torch.no_grad():
        while batch_data is not None:
            # Transfer in-memory data to CUDA devices to speed up training
            images = batch_data["image"].to(device=config.device, memory_format=torch.channels_last, non_blocking=True)
            target = batch_data["target"].to(device=config.device, non_blocking=True)

            # Get batch size
            batch_size = images.size(0)

            # Inference
            output = ema_model(images)

            # measure accuracy and record loss
            top1 = accuracy(output, target, topk=(1,))
            acc1.update(top1[0].item(), batch_size)

            # Calculate the time it takes to fully train a batch of data
            batch_time.update(time.time() - end)
            end = time.time()

            # Write the data during training to the training log file
            if batch_index % config.valid_print_frequency == 0:
                progress.display(batch_index + 1)

            # Preload the next batch of data
            batch_data = data_prefetcher.next()

            # Add 1 to the number of data batches to ensure that the terminal prints data normally
            batch_index += 1

    # print metrics
    progress.display_summary()

    if mode == "Valid" or mode == "Test":
        writer.add_scalar(f"{mode}/Acc@1", acc1.avg, epoch + 1)
    else:
        raise ValueError("Unsupported mode, please use `Valid` or `Test`.")

    return acc1.avg


if __name__ == "__main__":
    main()