import torch
import time
from torchvision import models
from tqdm import tqdm
from pdc_attention_network import PDCNet
import numpy as np


def calculate_fps(model, input_size=(3, 480, 320), num_iterations=200):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # 创建一个随机输入张量
    n = 16
    dummy_input = torch.randn(n, *input_size).to(device)

    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # 计时
    start_time = time.time()
    with torch.no_grad():
        for _ in tqdm(range(num_iterations)):
            _ = model(dummy_input)
    end_time = time.time()

    elapsed_time = end_time - start_time
    fps = num_iterations / elapsed_time

    return n * fps


model = PDCNet(64).to("cuda")
model.load_state_dict(
    torch.load(r'/home/share/liuchangsong/SACANet/Ablation/Convert/PDCNet(64)_Loss=HFL_BS=8/epoch-1-ckpt.pt')
)

# 计算FPS
fps = calculate_fps(model)

print(f"Average FPS: {fps:.2f}")
