'''
在该模块下，所有计算的tensor的requires_grad都为False，在反向传播时不会自动求导
（逃避autograd的追踪）
'''
import torch
