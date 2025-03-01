'''
Hengyuan Xu

generate key matrix

'''


import torch
from torch import nn
from einops import rearrange


def getPi(mask):
    dim=mask.shape[0]
    p=torch.zeros([dim,dim],dtype=torch.float)
    for i in range(dim):
        p[i][mask[i]]=1
    ip = torch.linalg.inv (p)    
    return p,ip

def getPi_Random_Multihead(dim, head):
    p = torch.eye(dim, dtype=torch.float)
    p = rearrange(p, 'r (h l) -> h r l', h=head)
    cols_per_head = dim // head
    for i in range(head):
        p = p[:, :, torch.randperm(cols_per_head)]
        
    p = p[torch.randperm(head), :, :]
    p = rearrange(p, 'h r l -> r (h l)')  
    return p.transpose(0,1), p

# parameters
dim = 512
head = 8

pcs = torch.zeros([10, dim, dim], dtype=torch.float)
ipcs = torch.zeros([10, dim, dim], dtype=torch.float)
for i in range(10):
    maskc = torch.randperm(dim)
    pc,ipc=getPi_Random_Multihead(dim, head)
    pcs[i] = pc
    ipcs[i] = ipc

pcs[0] = torch.eye(dim, dtype=torch.float)
ipcs[0] = torch.eye(dim, dtype=torch.float)
torch.save(pcs,'key.pt')
torch.save(ipcs,'unkey.pt')
key=torch.load('key.pt')
unkey=torch.load('unkey.pt')

print("pcs.shape: ", key.shape)
print('validation:\n',torch.matmul(key[5],unkey[5]))

# import torch
# import time

# def getPi_Random(dim=197):
#     p = torch.eye(dim, dtype=torch.float)
#     generator = torch.Generator()
#     generator.manual_seed(int(time.time_ns()))
#     mask = torch.randperm(dim, generator=generator)
#     p = p[mask]
#     # p.shape = (H, W)
#     ip = torch.transpose(p, 0, 1)
#     return p, ip

# # 设置维度为512
# dim = 768
# # 生成p和ip
# p, ip = getPi_Random(dim)

# # 保存p到key.pt文件
# torch.save(p, 'key.pt')
# # 保存ip到unkey.pt文件
# torch.save(ip, 'unkey.pt')

# print("p已保存到key.pt，ip已保存到unkey.pt")

# # 检查p和ip的形状
# print(f"p的形状: {p.shape}")
# print(f"ip的形状: {ip.shape}")

# # 验证p和ip是否为转置矩阵关系
# is_transpose = torch.allclose(p, ip.T)
# if is_transpose:
#     print("p和ip是转置矩阵关系。")
# else:
#     print("p和ip不是转置矩阵关系。")

# # 检查p和ip相乘是否为单位矩阵
# product = torch.matmul(p, ip)
# identity_matrix = torch.eye(dim, dtype=torch.float)
# is_identity_product = torch.allclose(product, identity_matrix)
# if is_identity_product:
#     print("p和ip相乘的结果是单位矩阵。")
# else:
#     print("p和ip相乘的结果不是单位矩阵。")