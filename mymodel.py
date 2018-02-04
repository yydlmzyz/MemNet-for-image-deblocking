import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class BNReLUConv(nn.Sequential):
    def __init__(self, in_channels, channels, kernel_size, inplace=True):
        super(BNReLUConv, self).__init__()
        self.add_module('bn', nn.BatchNorm2d(in_channels, eps=1e-04, momentum=0.9))
        self.add_module('relu', nn.ReLU(inplace=inplace))
        self.add_module('conv', nn.Conv2d(in_channels, channels, kernel_size, 1, (kernel_size-1)/2 ))


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.relu_conv1 = BNReLUConv(channels, channels, 3, True)
        self.relu_conv2 = BNReLUConv(channels, channels, 3, True)  

    def forward(self, x):
        residual = x
        out = self.relu_conv1(x)
        out = self.relu_conv2(out)
        out = out + residual
        return out


class MemoryBlock(nn.Module):
    def __init__(self, channels, num_resblock, num_memblock):
        super(MemoryBlock, self).__init__()
        self.recursive_unit = nn.ModuleList([ResidualBlock(channels) for i in range(num_resblock)])
        self.gate_unit = BNReLUConv((num_resblock+num_memblock) * channels, channels, 1, True)

    def forward(self, x, ys):
        xs = []
        for layer in self.recursive_unit:
            x = layer(x)
            xs.append(x)
        gate_out = self.gate_unit(torch.cat(xs+ys, 1))
        ys.append(gate_out)#Attention
        return gate_out



class MemNet(nn.Module):

    def __init__(self, in_channels, channels, num_memblock, num_resblock):
        super(MemNet, self).__init__()
        self.feature_extractor = BNReLUConv(in_channels, channels, 3, True)
        self.reconstructor = BNReLUConv(channels, in_channels, 3, True)
        self.dense_memory = nn.ModuleList([MemoryBlock(channels, num_resblock, i+1) for i in range(num_memblock)])
        
    def forward(self, x):
        #x= Variable(torch.randn(1,3,12,12), requires_grad=False)
        residual = x
        out = self.feature_extractor(x)
        long_memory = [out]

        for memory_block in self.dense_memory:
            out = memory_block(out, long_memory)
            #long_memory.append(out)

        out = self.reconstructor(out)
        out = out + residual
        
        return out



class MemNet_multi_supervised(nn.Module):

    def __init__(self, in_channels, channels, num_memblock, num_resblock):
        super(MemNet_multi_supervised, self).__init__()
        self.feature_extractor = BNReLUConv(in_channels, channels, 3, True)
        self.reconstructor = BNReLUConv(channels, in_channels, 3, True)
        self.dense_memory = nn.ModuleList([MemoryBlock(channels, num_resblock, i+1) for i in range(num_memblock)])
        self.num_memblock=num_memblock

    def forward(self, x):
        #x= Variable(torch.randn(1,3,12,12), requires_grad=False)
        residual = x
        out = self.feature_extractor(x)
        long_memory = [out]

        #get the out of each block
        for memory_block in self.dense_memory:
            out = memory_block(out, long_memory)
            #long_memory.append(out)

        #delete the first feature extract out
        del(long_memory[0])

        #then get the final output from reconstructor
        outlist=[]
        for memory in long_memory:
            out = self.reconstructor(memory)
            out = out + residual
            outlist.append(out)

        #get the output form multi-supervised
        #weights= Variable(torch.randn(num_memblock), requires_grad=True)
        #nn.init.xavier_normal(weights)
        if torch.cuda.is_available():
            weights= Variable((torch.ones(self.num_memblock)/self.num_memblock).cuda(), requires_grad=True)
        else:
            weights= Variable(torch.ones(self.num_memblock)/self.num_memblock, requires_grad=True)#something wrong
        
        if torch.cuda.is_available():
            final_out= Variable((torch.zeros(outlist[0].size())).cuda(), requires_grad=False)
        else:
            final_out= Variable(torch.zeros(outlist[0].size()), requires_grad=False)

        for i in range(self.num_memblock):
            final_out+=(weights[i]*outlist[i])

        return final_out,outlist,weights#weights is just for test 
