from torch.utils.data import Dataset
import os
import glob
import torch.nn.functional as F
import PIL.Image as Image
import numpy as np
import transforms


def read_from_pair_txt(path, filename):
    with open(os.path.join(path, filename)) as f:
        lines = f.readlines()

    lines = [line.strip() for line in lines]
    pairs = [p.split(" ") for p in lines]
    pairs = [(os.path.join(path, pair[0]), os.path.join(path, pair[1])) for pair in pairs]

    return pairs


def crop_bsds(data):
    crop_result = data[:, :, 1 : data.shape[2], 1 : data.shape[3]]
    return crop_result


def restore_bsds(pred, height, width):
    restore_result = np.zeros((height, width))
    restore_result[1 : restore_result.shape[0], 1 : restore_result.shape[1]] = pred
    return restore_result


def pad_nyud(data):
    pad_result = F.pad(data, pad=(0, 0, 7, 0), mode="reflect")
    return pad_result


def restore_nyud(pred, height, width):
    restore_result = np.zeros((height, width))
    restore_result = pred[7 : pred.shape[0], :]
    return restore_result


class BsdsDataset(Dataset):
    def __init__(self, dataset_path="", flag="train", sub_sample=-1):
        self.dataset_dir = dataset_path
        self.flag = flag

        if self.flag == "train":
            pairs = read_from_pair_txt(self.dataset_dir, "image-train.lst")
            if sub_sample > 0:
                selected_indices = np.random.choice(len(pairs), sub_sample, replace=False)
                self.img_list = [pairs[i][0] for i in selected_indices]
                self.lbl_list = [pairs[i][1] for i in selected_indices]

            else:
                self.img_list = [img_name[0] for img_name in pairs]
                self.lbl_list = [img_name[1] for img_name in pairs]

        elif self.flag == "test":
            assert sub_sample <= 0
            self.img_list = glob.glob(os.path.join(self.dataset_dir, r"images/test/*.jpg"))
            self.lbl_list = [path.split("/")[-1][:-4] for path in self.img_list]

    def __len__(self):
        return len(self.img_list)

    def trans_in_train(self, sample):
        trans = transforms.Compose(
            [
                # transforms.RandomCrop((321, 481)),
                transforms.RandomCrop((320, 320)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        sample = trans(sample)
        return sample

    def trans_in_test(self, sample):
        trans = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )
        sample = trans(sample)
        return sample

    def __getitem__(self, index):
        image = Image.open(self.img_list[index])
        if self.flag == "train":
            label = np.array(Image.open(self.lbl_list[index]).convert("L"))  # HWC
            label = Image.fromarray(label.astype(np.float32) / 255.0)  # 0-255变为0-1二值图像
        elif self.flag == "test":
            label = Image.open(self.img_list[index])

        sample = {"images": image, "labels": label}

        if self.flag == "train":
            sample = self.trans_in_train(sample)
        else:
            sample = self.trans_in_test(sample)

        return sample


class VOCDataset(Dataset):
    def __init__(self, dataset_path="", flag="train", sub_sample=-1):
        self.dataset_dir = dataset_path
        self.flag = flag

        if self.flag == "train":
            pairs = read_from_pair_txt(self.dataset_dir, "image-train.lst")
            if sub_sample > 0:
                selected_indices = np.random.choice(len(pairs), sub_sample, replace=False)
                self.img_list = [pairs[i][0] for i in selected_indices]
                self.lbl_list = [pairs[i][1] for i in selected_indices]
            else:
                self.img_list = [img_name[0] for img_name in pairs]
                self.lbl_list = [img_name[1] for img_name in pairs]

        elif self.flag == "test":
            assert sub_sample <= 0
            self.img_list = glob.glob(os.path.join(self.dataset_dir, r"images/test/*.jpg"))
            self.lbl_list = [path.split("/")[-1][:-4] for path in self.img_list]

    def __len__(self):
        return len(self.img_list)

    # def trans_in_train(self, sample):
    #     trans = transforms.Compose(
    #         [
    #             transforms.RandomCrop((320, 320)),
    #             transforms.ToTensor(),
    #             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    #         ]
    #     )
    #     sample = trans(sample)
    #     return sample

    def trans(self, sample):
        trans = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )
        sample = trans(sample)
        return sample

    def __getitem__(self, index):
        image = Image.open(self.img_list[index])
        if self.flag == "train":
            label = np.array(Image.open(self.lbl_list[index]).convert("L"))  # HWC
            label = Image.fromarray(label.astype(np.float32) / 255.0)  # 0-255变为0-1二值图像
        elif self.flag == "test":
            label = Image.open(self.img_list[index])

        sample = {"images": image, "labels": label}
        sample = self.trans(sample)

        return sample


class NyudDataset(Dataset):
    def __init__(self, dataset_path="", flag="train", rgb=True, sub_sample=-1):
        self.dataset_dir = dataset_path
        self.flag = flag

        if self.flag == "train":
            if rgb:
                pairs = read_from_pair_txt(self.dataset_dir, "image-train.lst")
            else:
                pairs = read_from_pair_txt(self.dataset_dir, "hha-train.lst")

            if sub_sample > 0:
                selected_indices = np.random.choice(len(pairs), sub_sample, replace=False)
                self.img_list = [pairs[i][0] for i in selected_indices]
                self.lbl_list = [pairs[i][1] for i in selected_indices]
            else:
                self.img_list = [img_name[0] for img_name in pairs]
                self.lbl_list = [img_name[1] for img_name in pairs]

        elif self.flag == "test":
            if rgb:
                self.img_list = glob.glob(os.path.join(self.dataset_dir, r"images/test/*.png"))
            else:
                self.img_list = glob.glob(os.path.join(self.dataset_dir, r"hha/test/*.png"))
            self.lbl_list = [path.split("/")[-1][:-4] for path in self.img_list]

    def __len__(self):
        return len(self.img_list)

    def trans_in_train(self, sample):
        trans = transforms.Compose(
            [
                # transforms.RandomCrop((432, 560)),
                transforms.RandomCrop((320, 320)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        sample = trans(sample)
        return sample

    def trans_in_test(self, sample):
        trans = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )
        sample = trans(sample)
        return sample

    def __getitem__(self, index):
        image = Image.open(self.img_list[index])
        if self.flag == "train":
            label = np.array(Image.open(self.lbl_list[index]).convert("L"))  # HWC
            label = Image.fromarray(label.astype(np.float32) / 255.0)  # 0-255变为0-1二值图像
        elif self.flag == "test":
            label = Image.open(self.img_list[index])

        sample = {"images": image, "labels": label}

        if self.flag == "train":
            sample = self.trans_in_train(sample)
        else:
            sample = self.trans_in_test(sample)

        return sample


class MulticueDataset(Dataset):
    def __init__(self, dataset_path="", flag="train", setting=("boundary", "1"), sub_sample=-1):
        assert setting[0] in ("boundary", "edge") and setting[1] in ("1", "2", "3")
        self.dataset_dir = dataset_path
        self.flag = flag

        if self.flag == "train":
            pairs = read_from_pair_txt(self.dataset_dir, "train_pair_{}_set_{}.lst".format(setting[0], setting[1]))
            if sub_sample > 0:
                selected_indices = np.random.choice(len(pairs), sub_sample, replace=False)
                self.img_list = [pairs[i][0] for i in selected_indices]
                self.lbl_list = [pairs[i][1] for i in selected_indices]
            else:
                self.img_list = [img_name[0] for img_name in pairs]
                self.lbl_list = [img_name[1] for img_name in pairs]

        elif self.flag == "test":
            assert sub_sample <= 0
            with open(os.path.join(self.dataset_dir, "test_{}_set_{}.lst".format(setting[0], setting[1])), "r") as f:
                lines = f.readlines()
            self.img_list = [os.path.join(self.dataset_dir, line.strip()) for line in lines]
            self.lbl_list = [path.split("/")[-1][:-4] for path in self.img_list]

    def __len__(self):
        return len(self.img_list)

    def trans_in_train(self, sample):
        trans = transforms.Compose(
            [
                transforms.RandomCrop((320, 320)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        sample = trans(sample)
        return sample

    def trans_in_test(self, sample):
        trans = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )
        sample = trans(sample)
        return sample

    def __getitem__(self, index):
        image = Image.open(self.img_list[index])
        if self.flag == "train":
            label = np.array(Image.open(self.lbl_list[index]).convert("L"))  # HWC
            label = Image.fromarray(label.astype(np.float32) / 255.0)  # 0-255变为0-1二值图像
        elif self.flag == "test":
            label = Image.open(self.img_list[index])

        sample = {"images": image, "labels": label}

        if self.flag == "train":
            sample = self.trans_in_train(sample)
        else:
            sample = self.trans_in_test(sample)

        return sample


class BipedDataset(Dataset):
    def __init__(self, dataset_path="", flag="train", sub_sample=-1):
        self.dataset_dir = dataset_path
        self.flag = flag

        if self.flag == "train":
            pairs = read_from_pair_txt(self.dataset_dir, "image-train.lst")
            if sub_sample > 0:
                selected_indices = np.random.choice(len(pairs), sub_sample, replace=False)
                self.img_list = [pairs[i][0] for i in selected_indices]
                self.lbl_list = [pairs[i][1] for i in selected_indices]
            else:
                self.img_list = [img_name[0] for img_name in pairs]
                self.lbl_list = [img_name[1] for img_name in pairs]

        elif self.flag == "test":
            assert sub_sample <= 0
            self.img_list = glob.glob(os.path.join(self.dataset_dir, r"edges/imgs/test/rgbr/*.jpg"))
            self.lbl_list = [path.split("/")[-1][:-4] for path in self.img_list]

    def __len__(self):
        return len(self.img_list)

    def trans_in_train(self, sample):
        trans = transforms.Compose(
            [
                transforms.RandomCrop((320, 320)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        sample = trans(sample)
        return sample

    def trans_in_test(self, sample):
        trans = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )
        sample = trans(sample)
        return sample

    def __getitem__(self, index):
        image = Image.open(self.img_list[index])
        if self.flag == "train":
            label = np.array(Image.open(self.lbl_list[index]).convert("L"))  # HWC
            label = Image.fromarray(label.astype(np.float32) / 255.0)  # 0-255变为0-1二值图像
        elif self.flag == "test":
            label = Image.open(self.img_list[index])

        sample = {"images": image, "labels": label}

        if self.flag == "train":
            sample = self.trans_in_train(sample)
        else:
            sample = self.trans_in_test(sample)

        return sample
