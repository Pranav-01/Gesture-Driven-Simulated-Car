import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_block(input_size, output_size):
    block = nn.Sequential(
        nn.Conv2d(input_size, output_size, (5, 5)), nn.ReLU(), nn.BatchNorm2d(output_size), nn.MaxPool2d((3, 3),stride=2,padding=1),nn.Dropout2d(0.4)
    )

    return block

def linear_block(input_size,output_size):
    return nn.Linear(input_size,output_size)

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        
        self.conv1 = conv_block(3, 16)
        self.conv2 = conv_block(16, 32)
        self.conv3 = conv_block(32, 64)
        self.conv4 = conv_block(64,64)
        self.max_pool = nn.MaxPool2d((3, 3),stride=2,padding=1)

        self.ln1 = nn.Linear(in_features=58240,out_features=1024)
        self.ln2 = nn.Linear(42, 42)
        self.ln3 = nn.Linear(42, 32)
        self.ln4 = nn.Linear(32,32)
        self.ln5 = nn.Linear(1056,5)
        # self.ln6 = nn.Linear(48,5)

        self.relu = nn.ReLU()
        # self.soft = F.log_softmax(dim=1)
        self.dropout = nn.Dropout2d(0.4)
        self.dropout2=nn.Dropout2d(0.2)
        self.dropout3=nn.Dropout2d(0.3)

    def forward(self,kp,img):

        #forward image through CNN
        img = self.conv1(img)
        # img= self.dropout2(img)
        img = self.conv2(img)
        # img=self.dropout3(img)        
        img = self.conv3(img)
        img = self.max_pool(img)
        # img=self.dropout(img)   
        # print("img: ",img.shape)
        self.c,self.h,self.w = img.shape[1],img.shape[2],img.shape[3]
        img = img.view(-1, self.c*self.h*self.w)
        img = self.ln1(img)
        img =self.relu(img)
        img=self.dropout2(img)
        # print(img.shape)
       
        # img = img.mean(dim=(3,2))
        # img = img.reshape(img.shape[0], -1)
        # x_pooled = self.global_avg_pool(img)
        # batch_size, num_channels, height, width = x_pooled.size()
        # img = x_pooled.view(batch_size, -1)
        # img = img.reshape(img.shape[0], -1)
        # print(img.shape)

        #forward keypoints through FC
        kp=self.ln2(kp)
        kp=self.relu(kp)
        kp=self.dropout2(kp)
        kp=self.ln3(kp)
        kp=self.relu(kp)
        kp=self.dropout3(kp)
        kp=self.ln4(kp)
        kp=self.relu(kp)
        kp=self.dropout2(kp)

        #concatenation
        out= torch.cat([kp, img], dim=1)

        
        # out=self.ln6(out)
        # out=self.relu(out)
        out=F.log_softmax(self.ln5(out),dim=1)
        # out=self.soft(out)
        
                            

        return out
