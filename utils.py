import torch
import logging
import os.path as osp
import numpy as np
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def get_model_parm_nums(model):
    total = sum([param.numel() for param in model.parameters()])
    total = float(total) / 1e6
    return total


def send_email(receiver: str):
    content = "Your Training is Complete!"

    msg = MIMEMultipart()
    msg.attach(MIMEText(content, "plain", "utf-8"))
    msg["Subject"] = "Training Done"
    msg["From"] = "2832941153@qq.com"
    msg["To"] = receiver

    smtp = smtplib.SMTP_SSL("smtp.qq.com", 465)
    smtp.login("2832941153@qq.com", "2003wumen")
    smtp.sendmail("2832941153@qq.com", receiver, msg.as_string())
    smtp.quit()


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


class AverageMeter(object):
    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count


def save_checkpoint(state, path="./checkpoint.pth"):
    """
    Save current state as checkpoint
    :param state: Model state.
    :param path:  Path of checkpoint file.
    """
    torch.save(state, path)


def load_checkpoint(net, opt=None, path="./checkpoint.pth"):
    """
    Load previous pre-trained checkpoint.
    :param net:  Network instance.
    :param opt:  Optimizer instance.
    :param path: Path of checkpoint file.
    :return:     Checkpoint epoch number.
    """
    if osp.isfile(path):
        print("=> Loading checkpoint {}...".format(path))
        checkpoint = torch.load(path)
        net.load_state_dict(checkpoint["model"])
        if opt is not None:
            opt.load_state_dict(checkpoint["opt"])
        return checkpoint["epoch"]
    else:
        raise ValueError("=> No checkpoint found at {}.".format(path))


def convert_model_to_standard_conv(model):
    if isinstance(model, torch.nn.DataParallel):
        model.module.convert_to_standard_conv()
    else:
        model.convert_to_standard_conv()


def save_cpdc(model, path):
    convert_model_to_standard_conv(model)
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)
