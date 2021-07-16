#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 16:34:08 2021

@author: tj208
"""

# torch tutorial
# %% 导入pytorch库
import torch
import numpy as np
# %% 创建Tensor数组
# 直接从数组创建
data = [[1,2],[3,4]]
x_data = torch.tensor(data)
# 从numpy数组创建
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
# 从另一个tensor创建
x_ones = torch.ones_like(x_data)
print(f"Ones Tensor: \n{x_ones}\n")
x_rand = torch.rand_like(x_data,dtype=torch.float)
print(f"Random Tensor: \n{x_rand}\n")
# 直接创建torch数组
shape = (2,3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")
# %% 输出tensor的各项属性

tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
# %% 操作移动到GPU上
if torch.cuda.is_available():
    tensor = tensor.to('cuda')
# %% 使用torch的内置方法
tensor = torch.ones(4,4)
tensor[:,1] = 0 # 将tensor的第一个数组变成0
print(tensor)
a = np.array([[1, 1],
              [0, 0]])
# 使用cat命令将几个数组拼接到一起
t1 = torch.cat([tensor,tensor,tensor],dim=1)
print(t1)
# This computes the element-wise product
print(f"tensor.mul(tensor) \n {tensor.mul(tensor)} \n")
# Alternative syntax:
print(f"tensor * tensor \n {tensor * tensor}")
# This computes the matrix multiplication between two tensors
print(f"tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)} \n")
# Alternative syntax:
print(f"tensor @ tensor.T \n {tensor @ tensor.T}")