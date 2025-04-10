import os
import torch
from utils import AverageMeter
from tqdm import tqdm
import torchvision
from dataset import crop_bsds, pad_nyud
from torch.cuda.amp import autocast


def train_bsds(train_loader, model, opt, lr_schd, print_freq, max_epoch, epoch, save_dir, logger, device, loss):
    # 创建目录
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # 设置训练模式
    model.train()

    # 初始化batch_loss记录表
    # batch_loss_meter.sum为一个epoch中的所有batch_loss之和
    # batch_loss_meter.avg为一个epoch的loss的平均值，loss_sum/the number of batch
    batch_loss_meter = AverageMeter()

    progress_bar = tqdm(train_loader, desc="Epoch:{}".format(epoch + 1))
    for batch_index, data in enumerate(progress_bar):
        # 获取图片和标签
        images, labels = data["images"].to(device), data["labels"].to(device)
        # images, labels = crop_bsds(images), crop_bsds(labels)  # BSDS500
        # 每一轮迭代先清零梯度
        opt.zero_grad()
        preds_list = model(images)
        # batch_loss = sum([loss(preds, labels) for preds in preds_list])
        batch_loss = loss(preds_list, labels)
        # 再反向传播更新参数
        batch_loss.backward()
        opt.step()

        # 显示loss
        progress_bar.set_postfix(train_loss_step="{:4f}".format(batch_loss.item()))

        # 记录loss
        batch_loss_meter.update(batch_loss.item())
        if batch_index % print_freq == print_freq - 1:
            logger.info(
                (
                    "Training epoch:{}/{}, batch:{}/{} current iteration:{}, "
                    + "current batch batch_loss:{}, epoch average batch_loss:{}, learning rate list:{}."
                ).format(
                    epoch + 1,
                    max_epoch,
                    batch_index + 1,
                    len(train_loader),
                    lr_schd.last_epoch + 1,
                    batch_loss_meter.val,
                    batch_loss_meter.avg,
                    lr_schd.get_last_lr(),
                )
            )
            preds_list_and_edges = [preds_list] + [labels]  # 7张图片
            height, width = preds_list_and_edges[0].shape[2:]
            interm_image = torch.zeros((len(preds_list_and_edges), 1, height, width))
            for i in range(len(preds_list_and_edges)):
                # 只保存一个批次当中的第一张图片
                interm_image[i, 0, :, :] = preds_list_and_edges[i][0, 0, :, :]
            torchvision.utils.save_image(
                interm_image, os.path.join(save_dir, "batch-{}-1st-image.png".format(batch_index))
            )

    return batch_loss_meter.avg


def train_voc(train_loader, model, opt, lr_schd, print_freq, max_epoch, epoch, save_dir, logger, device, loss):
    # 创建目录
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # 设置训练模式
    model.train()

    # 初始化batch_loss记录表
    # batch_loss_meter.sum为一个epoch中的所有batch_loss之和
    # batch_loss_meter.avg为一个epoch的loss的平均值，loss_sum/the number of batch
    batch_loss_meter = AverageMeter()

    progress_bar = tqdm(train_loader, desc="Epoch:{}".format(epoch + 1))
    for batch_index, data in enumerate(progress_bar):
        # 获取图片和标签
        images, labels = data["images"].to(device), data["labels"].to(device)
        # 每一轮迭代先清零梯度
        opt.zero_grad()
        preds = model(images)
        batch_loss = loss(preds, labels)
        # 再反向传播更新参数
        batch_loss.backward()
        opt.step()

        # 显示loss
        progress_bar.set_postfix(train_loss_step="{:4f}".format(batch_loss.item()))

        # 记录loss
        batch_loss_meter.update(batch_loss.item())
        if batch_index % print_freq == print_freq - 1:
            logger.info(
                (
                    "Training epoch:{}/{}, batch:{}/{} current iteration:{}, "
                    + "current batch batch_loss:{}, epoch average batch_loss:{}, learning rate list:{}."
                ).format(
                    epoch + 1,
                    max_epoch,
                    batch_index,
                    len(train_loader),
                    lr_schd.last_epoch + 1,
                    batch_loss_meter.val,
                    batch_loss_meter.avg,
                    lr_schd.get_last_lr(),
                )
            )
            preds_list_and_edges = [preds] + [labels]  # 7张图片
            height, width = preds_list_and_edges[0].shape[2:]
            interm_image = torch.zeros((len(preds_list_and_edges), 1, height, width))
            for i in range(len(preds_list_and_edges)):
                # 只保存一个批次当中的第一张图片
                interm_image[i, 0, :, :] = preds_list_and_edges[i][0, 0, :, :]
            torchvision.utils.save_image(
                interm_image, os.path.join(save_dir, "batch-{}-1st-image.png".format(batch_index))
            )

    return batch_loss_meter.avg


def train_nyud(train_loader, model, opt, lr_schd, print_freq, max_epoch, epoch, save_dir, logger, device, loss):
    # 创建目录
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # 设置训练模式
    model.train()

    # 初始化batch_loss记录表
    # batch_loss_meter.sum为一个epoch中的所有batch_loss之和
    # batch_loss_meter.avg为一个epoch的loss的平均值，loss_sum/the number of batch
    batch_loss_meter = AverageMeter()

    progress_bar = tqdm(train_loader, desc="Epoch:{}".format(epoch + 1))
    for batch_index, data in enumerate(progress_bar):
        # 获取图片和标签
        images, labels = data["images"].to(device), data["labels"].to(device)
        # 每一轮迭代先清零梯度
        opt.zero_grad()
        preds = model(images)
        batch_loss = loss(preds, labels)
        # 再反向传播更新参数
        batch_loss.backward()
        opt.step()

        # 显示loss
        progress_bar.set_postfix(train_loss_step="{:4f}".format(batch_loss.item()))

        # 记录loss
        batch_loss_meter.update(batch_loss.item())
        if batch_index % print_freq == print_freq - 1:
            logger.info(
                (
                    "Training epoch:{}/{}, batch:{}/{} current iteration:{}, "
                    + "current batch batch_loss:{}, epoch average batch_loss:{}, learning rate list:{}."
                ).format(
                    epoch + 1,
                    max_epoch,
                    batch_index,
                    len(train_loader),
                    lr_schd.last_epoch + 1,
                    batch_loss_meter.val,
                    batch_loss_meter.avg,
                    lr_schd.get_last_lr(),
                )
            )
            preds_list_and_edges = [preds] + [labels]  # 7张图片
            height, width = preds_list_and_edges[0].shape[2:]
            interm_image = torch.zeros((len(preds_list_and_edges), 1, height, width))
            for i in range(len(preds_list_and_edges)):
                # 只保存一个批次当中的第一张图片
                interm_image[i, 0, :, :] = preds_list_and_edges[i][0, 0, :, :]
            torchvision.utils.save_image(
                interm_image, os.path.join(save_dir, "batch-{}-1st-image.png".format(batch_index))
            )

    return batch_loss_meter.avg


def train_multicue(
    train_loader, model, opt, lr_schd, print_freq, max_epoch, epoch, save_dir, logger, device, loss, scaler
):
    # 创建目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 设置训练模式
    model.train()

    # 初始化batch_loss记录表
    # batch_loss_meter.sum为一个epoch中的所有batch_loss之和
    # batch_loss_meter.avg为一个epoch的loss的平均值，loss_sum/the number of batch
    batch_loss_meter = AverageMeter()

    progress_bar = tqdm(train_loader, desc="Epoch:{}".format(epoch + 1))
    for batch_index, data in enumerate(progress_bar):
        # 获取图片和标签
        images, labels = data["images"].to(device), data["labels"].to(device)
        # 每一轮迭代先清零梯度
        opt.zero_grad()

        if scaler is not None:
            with autocast():
                preds = model(images)
                batch_loss = loss(preds, labels)

            scaler.scale(batch_loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            preds = model(images)
            batch_loss = loss(preds, labels)
            # 再反向传播更新参数
            batch_loss.backward()
            opt.step()

        # 显示loss
        progress_bar.set_postfix(train_loss_step="{:4f}".format(batch_loss.item()))

        # 记录loss
        batch_loss_meter.update(batch_loss.item())
        if batch_index % print_freq == print_freq - 1:
            logger.info(
                (
                    "Training epoch:{}/{}, batch:{}/{} "
                    + "current batch batch_loss:{}, epoch average batch_loss:{}, learning rate list:{}."
                ).format(
                    epoch + 1,
                    max_epoch,
                    batch_index,
                    len(train_loader),
                    batch_loss_meter.val,
                    batch_loss_meter.avg,
                    lr_schd.get_last_lr(),
                )
            )
            preds_list_and_edges = [preds] + [labels]  # 7张图片
            height, width = preds_list_and_edges[0].shape[2:]
            interm_image = torch.zeros((len(preds_list_and_edges), 1, height, width))
            for i in range(len(preds_list_and_edges)):
                # 只保存一个批次当中的第一张图片
                interm_image[i, 0, :, :] = preds_list_and_edges[i][0, 0, :, :]
            torchvision.utils.save_image(
                interm_image, os.path.join(save_dir, "batch-{}-1st-image.png".format(batch_index))
            )

    return batch_loss_meter.avg
