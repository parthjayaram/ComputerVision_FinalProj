import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class L2Norm(nn.Module):
    def __init__(self,n_channels, scale=1.0):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.scale = scale
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.weight.data *= 0.0
        self.weight.data += self.scale

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        x = x / norm * self.weight.view(1,-1,1,1)
        return x

class s3fd_original(nn.Module):
    def __init__(self):
        super(s3fd_original, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.fc6     = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=3)
        self.fc7     = nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0)

        self.conv6_1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.conv6_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)

        self.conv7_1 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.conv7_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.conv3_3_norm = L2Norm(256,scale=10)
        self.conv4_3_norm = L2Norm(512,scale=8)
        self.conv5_3_norm = L2Norm(512,scale=5)

       
        self.conv3_3_norm_mbox_conf = nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1)
        self.conv3_3_norm_mbox_loc  = nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1)
        ## gender1

        self.conv4_3_norm_mbox_conf = nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1)
        self.conv4_3_norm_mbox_loc  = nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1)
        ##gender2
        self.conv4_3_norm_gender = nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1)

        self.conv5_3_norm_mbox_conf = nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1)
        self.conv5_3_norm_mbox_loc  = nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1)
        ##gender3
        self.conv5_3_norm_gender = nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1)

        self.fc7_mbox_conf     = nn.Conv2d(1024, 2, kernel_size=3, stride=1, padding=1)
        self.fc7_mbox_loc      = nn.Conv2d(1024, 4, kernel_size=3, stride=1, padding=1)
        ##gender4
        self.fc7_gender        = nn.Conv2d(1024, 2, kernel_size=3, stride=1, padding=1)

        self.conv6_2_mbox_conf = nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1)
        self.conv6_2_mbox_loc  = nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1)
        ##gender5
        self.conv6_2_gender    = nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1)

        self.conv7_2_mbox_conf = nn.Conv2d(256, 2, kernel_size=3, stride=1, padding=1)
        self.conv7_2_mbox_loc  = nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1)
        ##gender6

    def forward(self, x, use_cuda):
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h)); f3_3 = h
        h = F.max_pool2d(h, 2, 2)

        
        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h)); f4_3 = h
        

        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h)); f5_3 = h
        h = F.max_pool2d(h, 2, 2)

        
        h = F.relu(self.fc6(h))
        h = F.relu(self.fc7(h));     ffc7 = h

        h = F.relu(self.conv6_1(h))
        h = F.relu(self.conv6_2(h)); f6_2 = h

        h = F.relu(self.conv7_1(h))
        h = F.relu(self.conv7_2(h)); f7_2 = h

        f3_3 = self.conv3_3_norm(f3_3)
        f4_3 = self.conv4_3_norm(f4_3)
        f5_3 = self.conv5_3_norm(f5_3)

        # outputs from detection layers 1
        cls1 = self.conv3_3_norm_mbox_conf(f3_3)
        reg1 = self.conv3_3_norm_mbox_loc(f3_3)

        # outputs from detections layers 2
        cls2 = self.conv4_3_norm_mbox_conf(f4_3)
        reg2 = self.conv4_3_norm_mbox_loc(f4_3)
        gen2 = self.conv4_3_norm_gender(f4_3)


        #output from detection layers 3
        cls3 = self.conv5_3_norm_mbox_conf(f5_3)
        reg3 = self.conv5_3_norm_mbox_loc(f5_3)
        gen3 = self.conv5_3_norm_gender(f5_3)


        #output from detection layers 4
        cls4 = self.fc7_mbox_conf(ffc7)
        reg4 = self.fc7_mbox_loc(ffc7)
        gen4 = self.fc7_gender(ffc7)


        # output from detection layers 5
        cls5 = self.conv6_2_mbox_conf(f6_2)
        reg5 = self.conv6_2_mbox_loc(f6_2)
        gen5 = self.conv6_2_gender(f6_2)

        # output from detection layers 6
        cls6 = self.conv7_2_mbox_conf(f7_2)
        reg6 = self.conv7_2_mbox_loc(f7_2)


        #max-out background label
        chunk = torch.chunk(cls1,4,1)
        bmax  = torch.max(torch.max(chunk[0],chunk[1]),chunk[2])
        cls1  = torch.cat([bmax,chunk[3]],dim=1)

        face_classes = [cls2,cls3,cls4,cls5]
        gender_classes = [gen2, gen3, gen4, gen5]
        face_output = [F.softmax(face_class, dim=1) for face_class in face_classes]
        gender_output = [F.softmax(gender_class, dim=1) for gender_class in gender_classes]

        gender_classifications = []



        for index in range(int(len(face_output))):
            face_class_output = face_output[index].data
            gender_class_output = gender_output[index].data
            FB, FC, FH, FW = face_class_output.size()
            stride = 2**(index+2)
            anchor = stride * 4
            for Findex in range(FH*FW):
                windex, hindex = Findex % FW, Findex // FW
                axc, ayc = stride / 2 + windex * stride, stride / 2 + hindex * stride
                score = face_class_output[0, 1, hindex, windex]
                score_face = face_class_output[0, 0, hindex, windex]
                gender_score_female = gender_class_output[0, 0, hindex, windex]
                gender_score_male = gender_class_output[0, 1, hindex, windex]
                if score<0.05:
                    continue
                gender_classifications.append([gender_score_female, gender_score_male])

        if use_cuda:
            output = torch.tensor(gender_classifications).cuda()
        else:
            output = torch.tensor(gender_classifications)
        output_var = Variable(output, requires_grad = True)
        return output_var



