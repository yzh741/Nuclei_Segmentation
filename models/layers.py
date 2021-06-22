import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class conv_block(nn.Module):
    '''
        aggregation of conv operation
        conv-bn-relu-conv-bn-relu
        Example:
            input:(B,C,H,W)
            conv_block(C,out)
            conv_block(input)
            rerturn (B,out,H,W)
    '''

    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    '''
        Upsample the featur map and make some non-linear operation
        (B,C,H,W) ---> (B,out,2*H,2*W)
    '''

    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(ch_out),
            nn.GroupNorm(num_groups=32, num_channels=ch_out),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Recurrent_block(nn.Module):
    def __init__(self, ch_out, t=2):
        super(Recurrent_block, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.GroupNorm(3,ch_out), GN可以考虑使用
            # nn.BatchNorm2d(ch_out),
            nn.GroupNorm(num_groups=32, num_channels=ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):

            if i == 0:
                x1 = self.conv(x)

            x1 = self.conv(x + x1)
        return x1


class RRCNN_block(nn.Module):
    def __init__(self, ch_in, ch_out, t=2):
        super(RRCNN_block, self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out, t=t),
            Recurrent_block(ch_out, t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1


class AMSAF_block(nn.Module):
    def __init__(self, ch_in, dilation, group, t=2):
        super(AMSAF_block, self).__init__()
        self.t = t
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch_in, ch_in // 8, kernel_size=1, stride=1, padding=0), # 64/4 = 16
            nn.Conv2d(ch_in // 8, ch_in // 8, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(ch_in // 8, ch_in // 4, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            # nn.BatchNorm2d(ch_in//4)
            nn.GroupNorm(num_groups=group, num_channels=ch_in // 4)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch_in, ch_in // 8, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(ch_in // 8, ch_in // 8, kernel_size=(3, 1), stride=1, padding=(dilation, 0),
                      dilation=(dilation, 1)),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(ch_in // 8, ch_in // 4, kernel_size=(1, 3), stride=1, padding=(0, dilation),
                      dilation=(1, dilation)),
            # nn.BatchNorm2d(ch_in//4)
            nn.GroupNorm(num_groups=group, num_channels=ch_in // 4)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(ch_in, ch_in // 8, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(ch_in // 8, ch_in // 8, kernel_size=(3, 1), stride=1, padding=(dilation * 2, 0),
                      dilation=(dilation * 2, 1)),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(ch_in // 8, ch_in // 4, kernel_size=(1, 3), stride=1, padding=(0, dilation * 4),
                      dilation=(1, dilation * 4)),
            # nn.BatchNorm2d(ch_in//4)
            nn.GroupNorm(num_groups=group, num_channels=ch_in // 4)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(ch_in, ch_in // 8, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(ch_in // 8, ch_in // 8, kernel_size=(3, 1), stride=1, padding=(dilation * 4, 0),
                      dilation=(dilation * 4, 1)),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(ch_in // 8, ch_in // 4, kernel_size=(1, 3), stride=1, padding=(0, dilation * 2),
                      dilation=(1, dilation * 2)),
            # nn.BatchNorm2d(ch_in//4)
            nn.GroupNorm(num_groups=group, num_channels=ch_in // 4)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se_block = nn.Sequential(
            nn.Conv2d(ch_in, ch_in // 16, 1, 1, 0),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(ch_in // 16, ch_in, 1, 1, 0)
        )
        # self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        d1 = torch.cat((x1, x2, x3, x4), dim=1)
        d2 = self.avg_pool(d1)
        d2 = F.sigmoid(self.se_block(d2))  # se 输出         out = d2 * d1  # scale
        out = d2 * d1
        return out


class RAMSAF_block(nn.Module):
    def __init__(self, ch_in, dilation, t=2):
        super(RAMSAF_block, self).__init__()
        self.t = t
        self.conv = AMSAF_block(ch_in, dilation, t)

    def forward(self, x):
        # x = self.Conv_1x1(x)  # c’,h ,w -> c,h,w
        for i in range(self.t):
            if i == 0:
                x1 = self.conv(x)
            x1 = self.conv(x + x1)

        return x1


class RRAMSAF_block(nn.Module):
    def __init__(self, ch_in, ch_out, dilation, t=2):
        super(RRAMSAF_block, self).__init__()
        self.RCNN = nn.Sequential(
            RAMSAF_block(ch_out, dilation, t=t),
            RAMSAF_block(ch_out, dilation, t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1


class single_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class multi_dialiation_scale(nn.Module):
    def __init__(self, ):
        super(multi_dialiation_scale, self).__init__()
