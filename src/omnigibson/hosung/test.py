import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json
import cv2
import numpy as np


test = torch.tensor([[1.0,1.2,1.9],
                     [-1.4,-1.5,0.6],
                     [0.7,-0.8,1.9],
                     [0.1,1.1,-3.2]
                     ])

test1 = test/5
print(test1)
test2 = test1 * 1024
print(test2)
test3 = test2 + 511
print(test3)
test4 = test3 // 1
print(test4)
test5 = test4.type(torch.int)
print(test5)

test6 = (test*(1024/5))+511
print(test6)
test7 = test6.type(torch.int)
print(test7)
print(test7[:,2])

test8, indices = torch.sort(test7[:,2], dim = -1)
print(test8)


print(test7[:,0][indices])
print(test7[:,1][indices])
print(test7[:,2][indices])

print('------')
print(test7[indices])

image_width = 1024
image_height = 1024

py = torch.tensor([[x for _ in range(image_width)] for x in range(image_height)])
px = torch.tensor([[y for y in range(image_width)] for _ in range(image_height)])
print(px)