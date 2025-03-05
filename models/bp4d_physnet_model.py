import torch
import torch.nn as nn

class BP4D_PhysNet(nn.Module):
    """
    PhysNet model for BP4D dataset with pseudo-labeling.
    Uses 3D convolutions for temporal modeling.
    """
    def __init__(self):
        super(BP4D_PhysNet, self).__init__()
        
        # Initial convolution block
        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(1, 5, 5), padding=(0, 2, 2)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )
        
        # Downsampling blocks
        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        self.ConvBlock5 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        self.ConvBlock6 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        self.ConvBlock7 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        self.ConvBlock8 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        self.ConvBlock9 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        
        # Upsampling blocks using regular convolutions
        self.upsample = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(4, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        
        self.upsample2 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(4, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        
        # Final convolution
        self.ConvBlock10 = nn.Conv3d(64, 1, kernel_size=(1, 1, 1))
        
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, 3, time, height, width)
            
        Returns:
            Predicted PPG signal
        """
        # Downsampling path
        x1 = self.ConvBlock1(x)
        x2 = self.ConvBlock2(x1)
        x3 = self.ConvBlock3(x2)
        x4 = self.ConvBlock4(x3)
        x5 = self.ConvBlock5(x4)
        x6 = self.ConvBlock6(x5)
        x7 = self.ConvBlock7(x6)
        x8 = self.ConvBlock8(x7)
        x9 = self.ConvBlock9(x8)
        
        # Upsampling path using regular convolutions
        x = self.upsample(x9)
        x = self.upsample2(x)
        
        # Final convolution
        x = self.ConvBlock10(x)
        
        # Global average pooling
        x = torch.mean(x, dim=[2, 3, 4])
        
        return x
