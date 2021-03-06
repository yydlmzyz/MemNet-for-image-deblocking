import argparse
import h5py
import numpy
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable

import mymodel
import myutils

#prepare data
class MyDataset(Dataset):
    def __init__(self,data_file):
        self.file=h5py.File(str(data_file),'r')
        self.inputs=self.file['data'][:].astype(numpy.float32)/255.0#simple normalization in[0,1]
        self.label=self.file['label'][:].astype(numpy.float32)/255.0

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self,idx):
        inputs=self.inputs[idx,:,:,:].transpose(2,0,1)
        label=self.label[idx,:,:,:].transpose(2,0,1)
        inputs=torch.Tensor(inputs)
        label=torch.Tensor(label)
        sample={'inputs':inputs,'label':label}
        return sample
   

def checkpoint(epoch,loss,final_loss,multi_loss,psnr,ssim):
    model.eval()
    if use_gpu:
        model.cpu()#you should save weights on cpu not on gpu

    #save weights
    model_path = str(checkpoint_dir/'{}-{:.4f}-{:.4f}-{:.4f}param.pth'.format(epoch,loss,psnr,ssim))
    torch.save(model.state_dict(),model_path)

    #print and save record
    print('Epoch {} : Avg.loss:{:.4f}'.format(epoch,loss))
    print("Test Avg. PSNR: {:.4f} Avg. SSIM: {:.4f} ".format(psnr,ssim))
    print("Checkpoint saved to {}".format(model_path))

    output = open(str(checkpoint_dir/'train_result.txt'),'a+')
    output.write(('{} {:.4f} {:.4f} {:.4f}'.format(epoch,loss,psnr,ssim))+'\r\n')
    output.write(('{:.4f}'.format(final_loss))+'\t')
    for i in range(len(multi_loss)):
        output.write(('{:.4f}'.format(multi_loss[i]))+'\t')
    output.close()

    if use_gpu:
        model.cuda()#don't forget return to gpu
    model.train()


def wrap_variable(input_batch, label_batch, use_gpu):
        if use_gpu:
            input_batch, label_batch = (Variable(input_batch.cuda()), Variable(label_batch.cuda()))
        else:
            input_batch, label_batch = (Variable(input_batch),Variable(label_batch))
        return input_batch, label_batch



def train_multi_supervised(epoch):
    model.train()
    sum_loss=0.0
    sum_final_loss=0.0
    sum_multi_loss=[0.0,0.0,0.0,0.0,0.0,0.0]

    for iteration, sample in enumerate(dataloader):#difference between (dataloader) &(dataloader,1)
        inputs,label=sample['inputs'],sample['label']

        #Wrap with torch Variable
        inputs,label=wrap_variable(inputs, label, use_gpu)

        #clear the optimizer
        optimizer.zero_grad()

        # forward propagation
        final_out,outlist,weights= model(inputs)

        #calculate the loss
        final_loss=criterion(final_out, label)
         
        #calculate the loss of multi-supervised
        loss=final_loss
        multi_loss=[]

        for i in range(len(outlist)):
            temp=criterion(outlist[i], label)
            loss +=temp#default loss-weights are 1/7
            multi_loss.append(temp)
        loss=loss/(len(outlist)+1)

        #backward propagation and optimize
        loss.backward()
        optimizer.step()

        #monitor
        if iteration%100==0:
            print("===> Epoch[{}]({}/{}):loss: {:.4f} final_loss".format(epoch, iteration, len(dataloader), loss.data[0],final_loss.data[0]))   
            print('weights of each output:', weights.data) 
            for i in range(len(multi_loss)):
                print(('{:.4f}'.format(multi_loss[i].data[0]))+'\t')  

        #caculate the average loss
        sum_loss += loss.data[0]
        sum_final_loss += final_loss.data[0]
        sum_multi_loss=[sum_multi_loss[i]+multi_loss[i].data[0] for i in range(len(outlist))]

    return sum_loss/len(dataloader), sum_final_loss/len(dataloader), [sum_multi_loss[i]/len(dataloader) for i in range(len(sum_multi_loss))]



def test():
    model.eval()
    avg_psnr = 0
    avg_ssim = 0
    for iteration, sample in enumerate(test_dataloader):
        inputs,label=sample['inputs'],sample['label']
        #Wrap with torch Variable
        inputs,label=wrap_variable(inputs, label, use_gpu)
        outputs,outlist,weights = model(inputs)

        psnr = myutils.psnr(outputs, label)
        ssim = torch.sum((myutils.ssim(outputs, label, size_average=False)).data)/args.testbatchsize
        avg_ssim += ssim
        avg_psnr += psnr

    return (avg_psnr / len(test_dataloader)),(avg_ssim / len(test_dataloader))



def main():
    #train & test & record
    for epoch in range(args.epochs):
        loss,final_loss,multi_loss=train_multi_supervised(epoch)
        psnr,ssim = test()
        checkpoint(epoch,loss,final_loss,multi_loss,psnr,ssim)



#---------------------------------------------------------------------------------------------------
# Training settings
parser = argparse.ArgumentParser(description='deblocking using MemMet')
parser.add_argument('--batchsize', type=int, default=64, help='training batch size')
parser.add_argument('--testbatchsize', type=int, default=16, help='testing batch size')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.01')
args = parser.parse_args()

print(args)

#----------------------------------------------------------------------------------------------------
#set other parameters
#1.set cuda
use_gpu=torch.cuda.is_available()


#2.set path and file
save_dir = Path('.')
checkpoint_dir = Path('.') / 'Checkpoints_multi_supervised'#save model parameters and train record
if checkpoint_dir.exists():
    print 'folder esxited'
else:
    checkpoint_dir.mkdir()

#model_weights_file=checkpoint_dir/'42-0.00230-26.70-106.0161param.pth'


#3.set dataset and dataloader
dataset=MyDataset(data_file=save_dir/'Data10.h5')
test_dataset=MyDataset(data_file=save_dir/'TestData10.h5')

dataloader=DataLoader(dataset,batch_size=args.batchsize,shuffle=True,num_workers=0)
test_dataloader=DataLoader(test_dataset,batch_size=args.testbatchsize,shuffle=False,num_workers=0)


#4.set model& criterion& optimizer
model=mymodel.MemNet_multi_supervised(3,64,6,6)

criterion = nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(), lr=args.lr)
#initialize parameters
#model.apply(weights_init)#BUG! you can't apply initialization here!
if use_gpu:
    model = model.cuda()
    criterion = criterion.cuda()
'''
#load parameters
if not use_gpu:
    model.load_state_dict(torch.load(str(model_weights_file), map_location=lambda storage, loc: storage))
else:
    model.load_state_dict(torch.load(str(model_weights_file)))  
'''

#show mdoel&parameters&dataset
print('Model Structure:',model)
params = list(model.parameters())
for i in range(len(params)):
    print('layer:',i+1,params[i].size())

print('length of dataset:',len(dataset))


if __name__=='__main__':
    main()
