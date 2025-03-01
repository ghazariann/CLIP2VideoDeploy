'''
Hengyuan Xu

Get trained ViT-timm model and encrypt it with P_C and P_C^-1

Different models have different names for the parameters, 
so we need to change the names in the code accordingly. 
Print the names and shapes of the parameters, 
find the corresponding names and modify the code accordingly.

Modified by Mingxuan Ma

Modify the codes to fit the Bert model in CN-CLLP.

'''

'''
why does it fail???

the problem is that 
in 11.attention.output.dense.weight with shape torch.Size([768, 768]) is encrypted with P_C W P_C^-1
it prints P_C W P_C^-1
but in fact it is W P_C^-1 ...
'''


import torch

model = torch.load('pretrained_weights/MARVTT/pytorch_model.bin.2')
ori_state_dict = model

# Get the state_dict with "clip.transformer.resblocks." as prefix
new_state_dict = {}
for k, v in ori_state_dict.items():
    if k.startswith('clip.transformer.resblocks.'):
        new_state_dict[k[27:]] = v

# Get the parameters and names in the new state dict 
params = []
names = []
for k, v in new_state_dict.items():
    params.append(v)
    names.append(k)
    # print(k, v.shape)

# Get encryption keys
pc, ipc = torch.load('keys/key.pt')[5], torch.load('keys/unkey.pt')[5]
pc = pc.half().cuda()
ipc = ipc.half().cuda()

# Traverse the names and params, find those with target str in it
for i in range(len(names)):
    if "in_proj_weight" in names[i]:
        para = params[i]
        split_para = list(torch.split(para, 512, dim=0))
        for j in range(len(split_para)):
            # P_C W P_C^-1
            split_para[j] = torch.matmul(split_para[j], ipc)
            split_para[j] = torch.matmul(pc, split_para[j])
        para = torch.cat(split_para, dim=0)
        params[i] = para
        print(names[i], "with shape" , params[i].shape, "is encrypted with P_C W P_C^-1")
    elif "in_proj_bias" in names[i]:
        para = params[i]
        split_para = list(torch.split(para, 512, dim=0))
        for j in range(len(split_para)):
            # B P_C^-1
            split_para[j] = torch.matmul(split_para[j], ipc)
        para = torch.cat(split_para, dim=0)
        params[i] = para
        print(names[i], "with shape", params[i].shape, "is encrypted with B P_C^-1")
    elif "out_proj.weight" in names[i]:
        para = params[i]
        # P_C W P_C^-1
        para = torch.matmul(para, ipc)
        para = torch.matmul(pc, para)
        params[i] = para
        print(names[i], "with shape", params[i].shape, "is encrypted with P_C W P_C^-1")
    elif "out_proj.bias" in names[i]:
        para = params[i]
        # B P_C^-1
        para = torch.matmul(para, ipc)
        params[i] = para
        print(names[i], "with shape", params[i].shape, "is encrypted with B P_C^-1")
    elif "ln_1.weight" in names[i]:
        para = params[i]
        # W P_C^-1
        ipc = ipc.float()
        pc = pc.float()
        para = torch.matmul(para, ipc)
        params[i] = para
        print(names[i], "with shape", params[i].shape, "is encrypted with W P_C^-1")
        ipc = ipc.half()
        pc = pc.half()
    elif "ln_1.bias" in names[i]:
        para = params[i]
        # B P_C^-1
        ipc = ipc.float()
        pc = pc.float()
        para = torch.matmul(para, ipc)
        params[i] = para
        print(names[i], "with shape", params[i].shape, "is encrypted with B P_C^-1")
        ipc = ipc.half()
        pc = pc.half()
    elif "mlp.c_fc.weight" in names[i]:
        para = params[i]
        # W P_C^-1
        para = torch.matmul(para, ipc)
        params[i] = para
        print(names[i], "with shape", params[i].shape, "is encrypted with W P_C^-1")
    elif "mlp.c_fc.bias" in names[i]:
        # skip
        print(names[i], "with shape", params[i].shape, "should not be encrypted")
    elif "mlp.c_proj.weight" in names[i]:
        para = params[i]
        # P_C W
        para = torch.matmul(pc, para)
        params[i] = para
        print(names[i], "with shape", params[i].shape, "is encrypted with P_C W")
    elif "mlp.c_proj.bias" in names[i]:
        para = params[i]
        # B P_C^-1
        para = torch.matmul(para, ipc)
        params[i] = para
        print(names[i], "with shape", params[i].shape, "is encrypted with B P_C^-1")
    elif "ln_2.weight" in names[i]:
        para = params[i]
        # P_C W
        ipc = ipc.float()
        pc = pc.float()
        para = torch.matmul(pc, para)
        params[i] = para
        print(names[i], "with shape", params[i].shape, "is encrypted with P_C W")
        ipc = ipc.half()
        pc = pc.half()
    elif "ln_2.bias" in names[i]:
        para = params[i]
        # B P_C^-1
        ipc = ipc.float()
        pc = pc.float()
        para = torch.matmul(para, ipc)
        params[i] = para
        print(names[i], "with shape", params[i].shape, "is encrypted with B P_C^-1")
        ipc = ipc.half()
        pc = pc.half()
    else:
        raise ValueError("Error: " + names[i] + "should not be here")

# replace the corresponding params in the original state_dict, save it
for i in range(len(names)):
    names[i] = 'clip.transformer.resblocks.' + names[i]
    ori_state_dict[names[i]] = params[i]

# save the new state_dict
torch.save(ori_state_dict, "./pretrained_weights/MARVTT/encrypted_pytorch_model.bin.2")