# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     Pytorch张量索引
   Description :
   Author :       Jamerri
   date：          2022/4/4
-------------------------------------------------
   Change Activity:
                   2022/4/4:
-------------------------------------------------
"""

import torch

x = torch.Tensor([1, 4, 5, 6, 0, 8, 6, 1, 4, 5])
print(x)
print(x[0])  # 第一个元素
print(x[9])  # 第十个元素
print(x[1: 6])  # 输出从第二个元素到第六个元素

x = torch.randn((3, 10))
print(x)
print(x[0])  # size 1*10
print(x[0, :])  # size 1*10
print(x[:, 0])  # size 10*1
print(x[2, 5:9])

# indices = [2, 4, 6]
# print(x[indices])

x = torch.randn((4, 10))
row = [1, 3]
columes = [2, 9]
print(x[row,columes])  # (2,3) (4,10)

x = torch.Tensor([1, 4, 5, 6, 0, 8, 6, 1, 4, 5])
print(x[x > 5])
print(x[(x > 5) & (x <= 6)])
print(x[(x > 5) | (x <= 6)])
print(x[(x > 15)])

x = torch.Tensor([1, 4, 5, 6, 0, 8, 6, 1, 4, 5])
print(torch.where(x > 5, x, x/2))

x = torch.Tensor([1, 4, 5, 6, 0, 8, 6, 1, 4, 5])
print(x.unique())
print(x.numel())
