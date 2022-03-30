# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     Pytorch比较
   Description :
   Author :       Jamerri
   date：          2022/3/31
-------------------------------------------------
   Change Activity:
                   2022/3/31:
-------------------------------------------------
"""
import torch

x = torch.tensor([[-1, 2, 3], [4, 5, 6]], dtype = torch.float32)
y = torch.tensor([[1, 1, 3], [0, 10, 9]], dtype = torch.float32)

out = x < y
print(out)

out = x > y
out = x == y
out = x >= y
out = x <= y

# 其他的常规操作
# x = [-1, 2, 3
#       4, 5, 6]
out = torch.sum(x)
print(out)
out = torch.sum(x, dim=0)
print(out)
out = torch.sum(x, dim=1)
print(out)

# 最大值，最小值，均值
valus, indice = torch.max(x, dim=0)
print(valus)
print(indice)
valus, indice = torch.min(x, dim=0)
print(valus)
print(indice)
out = torch.mean(x, dim=0)
print(out)

# 绝对值
out = torch.abs(x)
print(out)

# 找到最大值，最小值的索引
out = torch.argmax(x, dim=0)
print(out)
out = torch.argmin(x, dim=0)
print(out)

# 排序
valus, indice = torch.sort(x, dim=1, descending=True)
print(valus)
print(indice)
