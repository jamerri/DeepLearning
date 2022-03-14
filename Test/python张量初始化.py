# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     python张量初始化
   Description :
   Author :       Jamerri
   date：          2022/3/14
-------------------------------------------------
   Change Activity:
                   2022/3/14:
-------------------------------------------------
"""

# 导入 pytorch 库

import torch

# 初始化张量
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device=DEVICE, requires_grad=True)

print(x)
print(x.dtype)
print(x.device)
print(x.requires_grad)

# 其他初始化方法
x = torch.rand((2, 3))  # rand 均匀分布
print(x)

x = torch.randn((2, 3))  # randn 正态分布
print(x)

x = torch.randint(3, 10, (2, 3))
print(x)

input = torch.randn((3, 3))
x = torch.rand_like(input)
print(x)