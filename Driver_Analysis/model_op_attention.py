import torch.nn as nn
import torch

def conv3x3(in_planes, out_planes):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, 3, 1, 1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_planes, out_planes, 3, 1, 1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))


class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x


class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi


class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )



    def forward(self,x):
        x = self.conv(x)
        return x






class Model(nn.Module):
    def __init__(self):
        n, m = 24, 5 #chanel 5
        
        super(Model, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.maxpool = nn.MaxPool2d(2, 2)


        self.convd1 = conv3x3(1*m, 1*n)
        self.convd2 = conv3x3(1*n, 2*n)
        self.convd3 = conv3x3(2*n, 4*n)
        self.convd4 = conv3x3(4*n, 4*n)
 
 
        self.convu3 = conv3x3(8*n, 4*n) #4+4
        self.convu2 = conv3x3(6*n, 2*n) #4+2
        self.convu1 = conv3x3(3*n, 1*n) #2+1

        
        self.att3 = Attention_block(F_g=4*n,F_l=4*n,F_int=2*n)
        self.att2 = Attention_block(F_g=4*n,F_l=2*n,F_int=n)
        
        
        self.convu0 = nn.Conv2d(n, 1, 3, 1, 1)

    def forward(self, x):

        x1 = x
        x1 = self.convd1(x1)          #in m out n
         # print(x1.size())

      

        x2 = self.maxpool(x1)      
        x2 = self.convd2(x2)   #in n out 2n
        #print(x2.shape)
        # print(x2.size())
 
        x3 = self.maxpool(x2)
        x3 = self.convd3(x3)  #in 2n out4n  
        # print(x3.size())

   

        x4 = self.maxpool(x3)
        x4 = self.convd4(x4) #out4n


        # print(x4.size())

         

        
        y3 = self.upsample(x4) #out:4n
        x3 = self.att3(g=y3,x=x3) #4n 4n  out:4n output size = before feature = x
        y3 = torch.cat([x3, y3], 1)
        y3 = self.convu3(y3) #out:4n
        # print(y3.size())
        y2 = self.upsample(y3) #4n
        x2 = self.att2(g=y2,x=x2) #2n #4n  out2n
        #print(x2.shape)
        y2 = torch.cat([x2, y2], 1) #4+2=6
  
        y2 = self.convu2(y2) #out:2n
        # print(y2.size())

        y1 = self.upsample(y2) #2n
        y1 = torch.cat([x1, y1], 1) #2+1=3
        y1 = self.convu1(y1) #3,1
        # print(y1.size())

        y1 = self.convu0(y1)
        y1 = self.sigmoid(y1)
        # print(y1.size())
        # exit(0)
      
        return y1
        

