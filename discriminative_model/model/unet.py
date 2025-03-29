import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

    def forward(self, x):
        x = self.conv1(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv2(x)
        return x
    

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.0):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, kernel_size, dropout)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.0, resize='padding'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, kernel_size, dropout)
        self.resize = resize

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        
        if x2 is not None:
            if self.resize == 'padding':
                diffY = x2.size(2) - x1.size(2)
                diffX = x2.size(3) - x1.size(3)
                if diffY > 0 or diffX > 0:
                    x1 = F.pad(x1, [(diffX + 1) // 2, diffX // 2, (diffY + 1) // 2, diffY // 2])

            elif self.resize == 'cropping':
                diffY = x2.size(2) - x1.size(2)
                diffX = x2.size(3) - x1.size(3)
                if diffY > 0 or diffX > 0:
                    x2 = x2[:, :, (diffY + 1) // 2 : x2.size(2) - diffY // 2, (diffX + 1) // 2 : x2.size(3) - diffX // 2]

            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        
        return self.conv(x)
    

class AttentionGate(nn.Module):
    def __init__(self, F_l, F_g, F_int):
        super(AttentionGate, self).__init__()
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

    def forward(self, x, g):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        out = x * psi
        return out


class UpAtt(nn.Module):
    def __init__(self, skip_channels, gating_channels, out_channels, kernel_size=3, dropout=0.0, resize='padding'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.att_gate = AttentionGate(F_l=skip_channels, F_g=gating_channels, F_int=out_channels // 2)
        self.conv = DoubleConv(skip_channels + gating_channels, out_channels, kernel_size, dropout)
        self.resize = resize

    def forward(self, x1, x2):
        x1 = self.up(x1)

        if self.resize == 'padding':
            diffY = x2.size(2) - x1.size(2)
            diffX = x2.size(3) - x1.size(3)
            if diffY > 0 or diffX > 0:
                x1 = F.pad(x1, [(diffX + 1) // 2, diffX // 2, (diffY + 1) // 2, diffY // 2])
        
        elif self.resize == 'cropping':
            diffY = x2.size(2) - x1.size(2)
            diffX = x2.size(3) - x1.size(3)
            if diffY > 0 or diffX > 0:
                x2 = x2[:, :, (diffY + 1) // 2 : x2.size(2) - diffY // 2, (diffX + 1) // 2 : x2.size(3) - diffX // 2]

        x2 = self.att_gate(x2, x1)
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    

class FillUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        # Encoder
        self.inc = DoubleConv(in_channels, 64, 3)
        self.down1 = Down(64,   128, kernel_size=3)
        self.down2 = Down(128,  256, kernel_size=3)
        self.down3 = Down(256,  512, kernel_size=3)
        self.down4 = Down(512, 1024, kernel_size=3)

        # Decoder
        self.up1 = Up(1024 + 512, 512, kernel_size=3)
        self.up2 = Up(512  + 256, 256, kernel_size=3)
        self.up3 = Up(256  + 128, 128, kernel_size=3)
        self.up4 = Up(128  +  64,  64, kernel_size=3)
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)       # ~ [B,   64, 230, 230]
        x2 = self.down1(x1)    # ~ [B,  128, 115, 115]
        x3 = self.down2(x2)    # ~ [B,  256,  57,  57]
        x4 = self.down3(x3)    # ~ [B,  512,  28,  28]
        x5 = self.down4(x4)    # ~ [B, 1024,  14,  14]

        x = self.up1(x5, x4)   # ~ [B,  512,  28,  28]
        x = self.up2(x, x3)    # ~ [B,  256,  57,  57]
        x = self.up3(x, x2)    # ~ [B,  128, 115, 115]
        x = self.up4(x, x1)    # ~ [B,   64, 230, 230]

        logits = self.outc(x)  # ~ [B,    1, 230, 230]
        return logits


class FWIUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        # Encoder
        self.inc = DoubleConv(in_channels, 16, 4)
        self.down1 = Down(16,   32, kernel_size=4, dropout=0.1)
        self.down2 = Down(32,   64, kernel_size=4, dropout=0.2)
        self.down3 = Down(64,  128, kernel_size=4, dropout=0.2)
        self.down4 = Down(128, 256, kernel_size=4, dropout=0.3)

        # Decoder
        self.up1 = Up(256 + 128, 128, kernel_size=4, dropout=0.2, resize='cropping')
        self.up2 = Up(128 +  64,  64, kernel_size=4, dropout=0.2, resize='cropping')
        self.up3 = Up(64  +  32,  32, kernel_size=4, dropout=0.1, resize='cropping')
        self.up4 = Up(32  +  16,  16, kernel_size=4, dropout=0.1, resize='cropping')
        self.outc = nn.Conv2d(16, out_channels, kernel_size=1)

        # Init
        self.apply(self._initialize_weights)

    def _initialize_weights(self, m):
        if isinstance(m, nn.Conv2d):
            # Kaiming He Init for Conv2d
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.inc(x)       # ~ [B,  16, 230, 230]
        x2 = self.down1(x1)    # ~ [B,  32, 115, 115]
        x3 = self.down2(x2)    # ~ [B,  64,  57,  57]
        x4 = self.down3(x3)    # ~ [B, 128,  28,  28]
        x5 = self.down4(x4)    # ~ [B, 256,  14,  14]

        x = self.up1(x5, x4)   # ~ [B, 128,  28,  28]
        x = self.up2(x, x3)    # ~ [B,  64,  56,  56]
        x = self.up3(x, x2)    # ~ [B,  32, 112, 112]
        x = self.up4(x, x1)    # ~ [B,  16, 224, 224]

        logits = self.outc(x)  # ~ [B,   1, 224, 224]
        return logits
    

class FWIUNetAtt(FWIUNet):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        
        # Decoder with Attention
        self.up1 = UpAtt(128, 256, 128, kernel_size=4, dropout=0.0, resize='cropping')
        self.up2 = UpAtt( 64, 128,  64, kernel_size=4, dropout=0.0, resize='cropping')
        self.up3 = UpAtt( 32,  64,  32, kernel_size=4, dropout=0.0, resize='cropping')
        self.up4 = Up(32,  16, kernel_size=4, dropout=0.1, resize='cropping')
    
    def forward(self, x):
        x1 = self.inc(x)       # ~ [B,  16, 230, 230]
        x2 = self.down1(x1)    # ~ [B,  32, 115, 115]
        x3 = self.down2(x2)    # ~ [B,  64,  57,  57]
        x4 = self.down3(x3)    # ~ [B, 128,  28,  28]
        x5 = self.down4(x4)    # ~ [B, 256,  14,  14]

        x = self.up1(x5, x4)   # ~ [B, 128,  28,  28]
        x = self.up2(x, x3)    # ~ [B,  64,  56,  56]
        x = self.up3(x, x2)    # ~ [B,  32, 112, 112]
        x = self.up4(x)    # ~ [B,  16, 224, 224]

        logits = self.outc(x)  # ~ [B,   1, 224, 224]
        return logits