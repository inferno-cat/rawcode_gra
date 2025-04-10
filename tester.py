import os
import torch
import PIL.Image as Image
from dataset import restore_bsds, crop_bsds, pad_nyud, restore_nyud
import time
from utils import AverageMeter
from tqdm import tqdm
import scipy.io as sio
import torch.nn.functional as F
import numpy as np


def test_bsds(test_loader, model, save_dir, logger, device, multi_scale=True):
    # 创建目录
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # 设置测试模式
    model.eval()

    assert test_loader.batch_size == 1

    """--------------------single_test---------------------"""
    """ single """
    t_time = 0
    t_duration = 0
    length = test_loader.dataset.__len__()

    """single_png"""
    single_png_dir = os.path.join(save_dir, "single_png")
    if not os.path.isdir(single_png_dir):
        os.makedirs(single_png_dir)
    """single_mat"""
    single_mat_dir = os.path.join(save_dir, "single_mat")
    if not os.path.isdir(single_mat_dir):
        os.makedirs(single_mat_dir)

    for batch_index, data in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            images = data["images"].to(device)
            # images = crop_bsds(images)  # BSDS500

            start_time = time.time()
            height, width = images.shape[2:]
            # preds_list = model(images)
            preds_list = slide_sample(model, images)
            preds = preds_list.detach().cpu().numpy().squeeze()  # H*W
            duration = time.time() - start_time
            t_time += duration
            t_duration += 1 / duration
            # preds = restore_bsds(preds, height + 1, width + 1)  # BSDS500
            name = test_loader.dataset.lbl_list[batch_index]
            sio.savemat(os.path.join(single_mat_dir, "{}.mat".format(name)), {"result": preds})
            Image.fromarray((preds * 255).astype(np.uint8)).save(os.path.join(single_png_dir, "{}.png".format(name)))
    logger.info("single test:\t avg_time: {:.3f}, avg_FPS: {:.3f}".format(t_time / length, t_duration / length))

    if multi_scale:
        """-----------------------------multi_test-------------------------------"""
        """multi"""
        t_time = 0
        t_duration = 0

        """multi_png"""
        multi_png_dir = os.path.join(save_dir, "multi_png")
        if not os.path.isdir(multi_png_dir):
            os.makedirs(multi_png_dir)
        """multi_mat"""
        multi_mat_dir = os.path.join(save_dir, "multi_mat")
        if not os.path.isdir(multi_mat_dir):
            os.makedirs(multi_mat_dir)

        for batch_index, data in enumerate(tqdm(test_loader)):
            with torch.no_grad():
                images = data["images"]
                # images = crop_bsds(images)

                height, width = images.shape[2:]
                images_2x = F.interpolate(images, scale_factor=2, mode="bilinear", align_corners=True)
                images_half = F.interpolate(images, scale_factor=0.5, mode="bilinear", align_corners=True)

                start_time = time.time()

                images = images.to(device)
                # preds_list = model(images)
                preds_list = slide_sample(model, images)
                preds = preds_list

                images_2x = images_2x.to(device)
                # preds_2x_list = model(images_2x)
                preds_2x_list = slide_sample(model, images_2x)
                preds_2x_down = F.interpolate(preds_2x_list, size=(height, width), mode="bilinear", align_corners=True)

                images_half = images_half.to(device)
                # preds_half_list = model(images_half)
                preds_half_list = slide_sample(model, images_half)
                preds_half_up = F.interpolate(
                    preds_half_list, size=(height, width), mode="bilinear", align_corners=True
                )

                fuse_final = (preds + preds_2x_down + preds_half_up) / 3
                fuse_final = fuse_final.cpu().detach().numpy().squeeze()
                duration = time.time() - start_time
                t_time += duration
                t_duration += 1 / duration
                # fuse_final = restore_bsds(fuse_final, height + 1, width + 1)
                name = test_loader.dataset.lbl_list[batch_index]
                sio.savemat(os.path.join(multi_mat_dir, "{}.mat".format(name)), {"result": fuse_final})
                Image.fromarray((fuse_final * 255).astype(np.uint8)).save(
                    os.path.join(multi_png_dir, "{}.png".format(name))
                )
        logger.info("multi test:\t avg_time: {:.3f}, avg_FPS: {:.3f}".format(t_time / length, t_duration / length))


def test_nyud(test_loader, model, save_dir, logger, device):
    # 创建目录
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # 设置测试模式
    model.eval()

    assert test_loader.batch_size == 1

    t_time = 0
    t_duration = 0
    length = test_loader.dataset.__len__()

    """png"""
    png_dir = os.path.join(save_dir, "png")
    if not os.path.isdir(png_dir):
        os.makedirs(png_dir)
    """mat"""
    mat_dir = os.path.join(save_dir, "mat")
    if not os.path.isdir(mat_dir):
        os.makedirs(mat_dir)

    for batch_index, data in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            images = data["images"].to(device)
            # images = pad_nyud(images)  # NYUD

            start_time = time.time()
            height, width = images.shape[2:]
            # preds = model(images)
            preds = slide_sample(model, images)
            preds = preds.detach().cpu().numpy().squeeze()  # H*W
            duration = time.time() - start_time
            t_time += duration
            t_duration += 1 / duration
            # preds = restore_nyud(preds, height - 7, width)  # NYUD
            name = test_loader.dataset.lbl_list[batch_index]
            sio.savemat(os.path.join(mat_dir, "{}.mat".format(name)), {"result": preds})
            Image.fromarray((preds * 255).astype(np.uint8)).save(os.path.join(png_dir, "{}.png".format(name)))
    logger.info("single test:\t avg_time: {:.3f}, avg_FPS: {:.3f}".format(t_time / length, t_duration / length))


def test_multicue(test_loader, model, save_dir, logger, device, multi_scale=True, slide_window=True):
    # 创建目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 设置测试模式
    model.eval()

    assert test_loader.batch_size == 1

    t_time = 0
    t_duration = 0
    length = test_loader.dataset.__len__()

    """png"""
    single_png_dir = os.path.join(save_dir, "single_png")
    if not os.path.exists(single_png_dir):
        os.makedirs(single_png_dir)
    """mat"""
    single_mat_dir = os.path.join(save_dir, "single_mat")
    if not os.path.exists(single_mat_dir):
        os.makedirs(single_mat_dir)

    for batch_index, data in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            images = data["images"].to(device)

            start_time = time.time()

            if slide_window:
                preds = slide_sample(model, images)
            else:
                preds = model(images)

            preds = preds.detach().cpu().numpy().squeeze()  # H*W
            duration = time.time() - start_time
            t_time += duration
            t_duration += 1 / duration
            name = test_loader.dataset.lbl_list[batch_index]
            sio.savemat(os.path.join(single_mat_dir, "{}.mat".format(name)), {"result": preds})
            Image.fromarray((preds * 255).astype(np.uint8)).save(os.path.join(single_png_dir, "{}.png".format(name)))
    logger.info("single test:\t avg_time: {:.3f}, avg_FPS: {:.3f}".format(t_time / length, t_duration / length))

    if multi_scale:
        t_time = 0
        t_duration = 0
        """png"""
        multi_png_dir = os.path.join(save_dir, "multi_png")
        if not os.path.exists(multi_png_dir):
            os.makedirs(multi_png_dir)
        """mat"""
        multi_mat_dir = os.path.join(save_dir, "multi_mat")
        if not os.path.exists(multi_mat_dir):
            os.makedirs(multi_mat_dir)

        for batch_index, data in enumerate(tqdm(test_loader)):
            with torch.no_grad():
                images = data["images"]
                height, width = images.shape[2:]
                images_2x = F.interpolate(images, scale_factor=2, mode="bilinear", align_corners=True)
                images_half = F.interpolate(images, scale_factor=0.5, mode="bilinear", align_corners=True)

                start_time = time.time()
                images = images.to(device)
                images_2x = images_2x.to(device)
                images_half = images_half.to(device)

                # inference process
                if slide_window:
                    preds = slide_sample(model, images)
                    preds_2x = slide_sample(model, images_2x)
                    preds_half = slide_sample(model, images_half)
                else:
                    preds = model(images)
                    preds_2x = model(images_2x)
                    preds_half = model(images_half)

                # scale
                preds_2x_down = F.interpolate(preds_2x, size=(height, width), mode="bilinear", align_corners=True)
                preds_half_up = F.interpolate(preds_half, size=(height, width), mode="bilinear", align_corners=True)

                preds_fuse = (preds + preds_2x_down + preds_half_up) / 3
                preds_fuse = preds_fuse.detach().cpu().numpy().squeeze()
                duration = time.time() - start_time
                t_time += duration
                t_duration += 1 / duration
                name = test_loader.dataset.lbl_list[batch_index]
                sio.savemat(os.path.join(multi_mat_dir, "{}.mat".format(name)), {"result": preds_fuse})
                Image.fromarray((preds_fuse * 255).astype(np.uint8)).save(
                    os.path.join(multi_png_dir, "{}.png".format(name))
                )
        logger.info("multi test:\t avg_time: {:.3f}, avg_FPS: {:.3f}".format(t_time / length, t_duration / length))


def slide_sample(model, inputs, crop_size=(320, 320), stride=(240, 240)):
    """
    Perform sliding window inference on the input image.

    Args:
    - inputs (Tensor): Input tensor of shape (batch_size, channels, height, width)
    - crop_size (tuple): Size of the crop window (height, width)
    - stride (tuple): Stride of the sliding window (height_stride, width_stride)

    Returns:
    - seg_logits (Tensor): Segmentation logits for the entire input
    """
    # Unpack stride and crop size
    h_stride, w_stride = stride
    h_crop, w_crop = crop_size

    # Get input dimensions
    batch_size, _, h_img, w_img = inputs.size()
    out_channels = 1  # Assuming single channel output

    # Calculate number of grid cells
    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

    # Initialize prediction and count tensors
    preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
    count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))

    # Iterate over grid cells
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            # Calculate crop coordinates
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)

            # Extract crop from input
            crop_img = inputs[:, :, y1:y2, x1:x2]

            # Perform inference on crop
            crop_seg_logit = model(crop_img)

            # Add crop prediction to overall prediction
            preds += F.pad(crop_seg_logit, (int(x1), int(preds.shape[3] - x2), int(y1), int(preds.shape[2] - y2)))

            # Update count matrix
            count_mat[:, :, y1:y2, x1:x2] += 1

    # Ensure all pixels have been predicted at least once
    assert (count_mat == 0).sum() == 0

    # Average predictions where there are overlaps
    seg_logits = preds / count_mat

    return seg_logits


# def slide_sample(model, inputs, crop_size=(320, 320), stride=(240, 240)):
#     """Inference by sliding-window with overlap.
#
#     If h_crop > h_img or w_crop > w_img, the small patch will be used to
#     decode without padding.
#
#     Args:
#         inputs (tensor): the tensor should have a shape NxCxHxW,
#             which contains all images in the batch.
#         batch_img_metas (List[dict]): List of image metainfo where each may
#             also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
#             'ori_shape', and 'pad_shape'.
#             For details on the values of these keys see
#             `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.
#
#     Returns:
#         Tensor: The segmentation results, seg_logits from model of each
#             input image.
#     """
#
#     h_stride, w_stride = stride
#     h_crop, w_crop = crop_size
#     batch_size, _, h_img, w_img = inputs.size()
#     out_channels = 1
#     h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
#     w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
#     preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
#     aux_out1 = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
#     # aux_out2 = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
#     count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
#     for h_idx in range(h_grids):
#         for w_idx in range(w_grids):
#             y1 = h_idx * h_stride
#             x1 = w_idx * w_stride
#             y2 = min(y1 + h_crop, h_img)
#             x2 = min(x1 + w_crop, w_img)
#             y1 = max(y2 - h_crop, 0)
#             x1 = max(x2 - w_crop, 0)
#             crop_img = inputs[:, :, y1:y2, x1:x2]
#
#             crop_seg_logit = model(crop_img)
#             aux_out = None
#
#             # if isinstance(self.model, nn.parallel.DistributedDataParallel):
#             #     crop_seg_logit = self.model.module.sample(
#             #         batch_size=1, cond=crop_img, mask=mask
#             #     )
#             #     e1 = e2 = None
#             #     aux_out = None
#             # elif isinstance(self.model, nn.Module):
#             #     crop_seg_logit = self.model.sample(
#             #         batch_size=1, cond=crop_img, mask=mask
#             #     )
#             #     e1 = e2 = None
#             #     aux_out = None
#             # else:
#             #     raise NotImplementedError
#             preds += F.pad(crop_seg_logit, (int(x1), int(preds.shape[3] - x2), int(y1), int(preds.shape[2] - y2)))
#             if aux_out is not None:
#                 aux_out1 += F.pad(aux_out, (int(x1), int(aux_out1.shape[3] - x2), int(y1), int(aux_out1.shape[2] - y2)))
#
#             count_mat[:, :, y1:y2, x1:x2] += 1
#     assert (count_mat == 0).sum() == 0
#     # torch.save(count_mat, '/home/yyf/Workspace/edge_detection/codes/Mask-Conditioned-Latent-Space-Diffusion/checkpoints/count.pt')
#     seg_logits = preds / count_mat
#     aux_out1 = aux_out1 / count_mat
#     # aux_out2 = aux_out2 / count_mat
#     if aux_out is not None:
#         return seg_logits, aux_out1
#     return seg_logits


# if __name__ == '__main__':
#     import torch.nn as nn
#
#     def slide_sample(model, inputs, crop_size, stride):
#         """
#         Perform sliding window inference on the input image.
#
#         Args:
#         - inputs (Tensor): Input tensor of shape (batch_size, channels, height, width)
#         - crop_size (tuple): Size of the crop window (height, width)
#         - stride (tuple): Stride of the sliding window (height_stride, width_stride)
#         - mask (Tensor, optional): Mask tensor for conditional sampling
#
#         Returns:
#         - seg_logits (Tensor): Segmentation logits for the entire input
#         """
#         # Unpack stride and crop size
#         h_stride, w_stride = stride
#         h_crop, w_crop = crop_size
#
#         # Get input dimensions
#         batch_size, _, h_img, w_img = inputs.size()
#         out_channels = 1  # Assuming single channel output
#
#         # Calculate number of grid cells
#         h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
#         w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
#
#         # Initialize prediction and count tensors
#         preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
#         count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
#
#         # Iterate over grid cells
#         for h_idx in range(h_grids):
#             for w_idx in range(w_grids):
#                 # Calculate crop coordinates
#                 y1 = h_idx * h_stride
#                 x1 = w_idx * w_stride
#                 y2 = min(y1 + h_crop, h_img)
#                 x2 = min(x1 + w_crop, w_img)
#                 y1 = max(y2 - h_crop, 0)
#                 x1 = max(x2 - w_crop, 0)
#
#                 # Extract crop from input
#                 crop_img = inputs[:, :, y1:y2, x1:x2]
#
#                 # Perform inference on crop
#                 crop_seg_logit = model(crop_img)
#
#                 # Add crop prediction to overall prediction
#                 preds += F.pad(crop_seg_logit, (int(x1), int(preds.shape[3] - x2), int(y1), int(preds.shape[2] - y2)))
#
#                 # Update count matrix
#                 count_mat[:, :, y1:y2, x1:x2] += 1
#
#         # Ensure all pixels have been predicted at least once
#         assert (count_mat == 0).sum() == 0
#
#         # Average predictions where there are overlaps
#         seg_logits = preds / count_mat
#
#         return seg_logits
#
#
# class model(nn.Module):
#     def __init__(self):
#         super(model, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv2d(64, 1, kernel_size=1, stride=1)
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.sigmoid(self.conv3(x))
#
#         return x
#
#
# input = torch.randn(1, 3, 200, 200)
# model = model()
# outputs = slide_sample(model, input, (320, 320), (240, 240))
# print(outputs.shape)