import torch
import torch.nn as nn
import torch.nn.functional as F



class Sinple_DenseBlock(nn.Module):
    def __init__(self,input_channel=4,nf=64,upscale=2,filter_size=5):
        super(Sinple_DenseBlock, self).__init__()
        self.nf=nf
        self.upscale=upscale
        self.conv_pre = nn.Conv2d(input_channel, self.nf, kernel_size=3, stride=1, padding=1)
        self.outf=self.nf//2
        self.filter_size=filter_size
    def forward(self,input):
        x = self.conv_pre(input)
        n, c, h, w = x.shape
        # assert c==self.nf
        #block1
        t = nn.BatchNorm2d(self.nf)(x)
        t = nn.ReLU()(t)
        t = nn.Conv2d(in_channels=self.nf, out_channels=self.nf, kernel_size=1, stride=1, padding=0)(t)

        t = nn.BatchNorm2d(self.nf)(t)
        t = nn.ReLU()(t)
        t = nn.Conv2d(in_channels=self.nf, out_channels=self.outf, kernel_size=3, stride=1, padding=1)(t)

        # block2
        self.nf+=self.outf
        x=torch.cat([x,t],1)

        t = nn.BatchNorm2d(self.nf)(x)
        t = nn.ReLU()(t)
        t = nn.Conv2d(in_channels=self.nf, out_channels=self.nf, kernel_size=1, stride=1, padding=0)(t)

        t = nn.BatchNorm2d(self.nf)(t)
        t = nn.ReLU()(t)
        t = nn.Conv2d(in_channels=self.nf, out_channels=self.outf, kernel_size=3, stride=1, padding=1)(t)

        self.nf += self.outf
        x = torch.cat([x, t], 1)

        x= nn.BatchNorm2d(self.nf)(x)
        x = nn.ReLU()(x)
        x = nn.Conv2d(in_channels=self.nf, out_channels=self.nf, kernel_size=3, stride=1, padding=1)(x)
        x = nn.ReLU()(x)

        #resiual
        r=nn.Conv2d(in_channels=self.nf, out_channels=self.nf, kernel_size=1, stride=1, padding=0)(x)
        r = nn.ReLU()(r)
        r = nn.Conv2d(in_channels=self.nf, out_channels=self.upscale*self.upscale, kernel_size=1, stride=1, padding=0)(r)

        #filter
        f = nn.Conv2d(in_channels=self.nf, out_channels=self.nf*2, kernel_size=1, stride=1, padding=0)(x)
        f = nn.ReLU()(f)
        f = nn.Conv2d(in_channels=self.nf*2, out_channels=self.filter_size*self.filter_size* self.upscale* self.upscale, kernel_size=1, stride=1, padding=0)(f)

        #norm
        n,c,h,w=f.shape
        # assert self.filter_size*self.filter_size*self.upscale* self.upscale==c
        f=torch.reshape(f,[n,self.filter_size*self.filter_size,self.upscale*self.upscale,h,w])
        f=f.permute([0,3,4,1,2])
        f=nn.Softmax(dim=3)(f)

        return f,r


class DUF_Wguide_resolutionRGB(nn.Module):
    def __init__(self,upscale=2):
        super(DUF_Wguide_resolutionRGB, self).__init__()

        self.nf=64
        self.filter_size = 5
        self.upscale=upscale
        self.conv_guide_B=Sinple_DenseBlock(nf=self.nf,upscale=upscale,filter_size=self.filter_size)
        self.conv_guide_G1=Sinple_DenseBlock(nf=self.nf,upscale=upscale,filter_size=self.filter_size)
        self.conv_guide_G2=Sinple_DenseBlock(nf=self.nf,upscale=upscale,filter_size=self.filter_size)
        self.conv_guide_R=Sinple_DenseBlock(nf=self.nf,upscale=upscale,filter_size=self.filter_size)
    def forward(self,bggr,W):
        W0=W[:,:,0::2,0::2]
        W1=W[:,:,0::2,1::2]
        W2=W[:,:,1::2,0::2]
        W3=W[:,:,1::2,1::2]


        B = bggr[:, 0:1, :, :]
        G1 = bggr[:, 1:2, :, :]
        G2 = bggr[:, 2:3, :, :]
        R = bggr[:, 3:4, :, :]
        BGGR_L=[B,G1,G2,R]

        W_guide=torch.cat([W0,W1,W2,W3],dim=1)

        filter_B,resiual_B = self.conv_guide_B(W_guide)
        filter_G1,resiual_G1 = self.conv_guide_G1(W_guide)
        filter_G2,resiual_G2 = self.conv_guide_G2(W_guide)
        filter_R,resiual_R = self.conv_guide_R(W_guide)

        filters=[filter_B,filter_G1,filter_G2,filter_R]
        resiuals=[resiual_B,resiual_G1,resiual_G2,resiual_R]
        BGGR_H = []
        for i in range(len(BGGR_L)):
            t=self.DynFilter(BGGR_L[i],filters[i],self.filter_size)
            t=nn.PixelShuffle(upscale_factor=self.upscale)(t)
            r=nn.PixelShuffle(upscale_factor=self.upscale)(resiuals[i])
            BGGR_H.append(t+r)
        BGGR_H=torch.cat(BGGR_H,dim=1)
        return BGGR_H

    def DynFilter(self,x,Filter,filter_size):

        dim=filter_size*filter_size

        filter_localexpand_np = torch.reshape(torch.eye(dim), (dim, 1,filter_size, filter_size))
        x_localexpand=F.conv2d(input=x,weight=filter_localexpand_np,stride=1,padding=self.filter_size//2)
        x_localexpand=x_localexpand.permute(0,2,3,1)
        x_localexpand=x_localexpand[:,:,:,None,:]

        x=torch.matmul(x_localexpand,Filter)
        x=torch.squeeze(x,dim=3)
        x=x.permute(0,3,1,2)
        return x



if __name__=='__main__':
    import numpy as np

    a = np.eye(5, 5)
    DUF=DUF_Wguide_resolutionRGB()
    DUF=DUF.eval()
    input_bggr=torch.randn([1,4,128,128])
    input_w=torch.randn([1,1,256,256])
    output=DUF(input_bggr,input_w)
    print(output.shape)


