import torch
import torch.nn as nn

class TSCAN(nn.Module):
    def __init__(self, frame_depth=10, drop_rate=0.2):
        super(TSCAN, self).__init__()
        
        # Motion branch (2D convolutions)
        self.motion_conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.motion_conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.motion_conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.motion_conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Appearance branch (2D convolutions)
        self.apperance_conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.apperance_conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.apperance_conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.apperance_conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Attention mechanism
        self.apperance_att_conv1 = nn.Conv2d(32, 1, kernel_size=1)
        self.apperance_att_conv2 = nn.Conv2d(64, 1, kernel_size=1)
        
        # Final dense layers
        self.final_dense_1 = nn.Linear(16384, 128)  # Changed back to 16384
        self.final_dense_2 = nn.Linear(128, 1)
        
        self.dropout = nn.Dropout(drop_rate)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: [batch, channels, frames, height, width]
        B, C, T, H, W = x.shape
        
        # Process each frame independently first
        motion_features = []
        appearance_features = []
        
        for t in range(T):
            frame = x[:, :, t, :, :]  # [batch, channels, height, width]
            
            # Motion branch
            m = self.relu(self.motion_conv1(frame))
            m = self.relu(self.motion_conv2(m))
            m = self.relu(self.motion_conv3(m))
            m = self.relu(self.motion_conv4(m))  # [batch, 64, H, W]
            
            # Appearance branch
            a = self.relu(self.apperance_conv1(frame))
            a = self.relu(self.apperance_conv2(a))
            a = self.relu(self.apperance_conv3(a))
            a = self.relu(self.apperance_conv4(a))  # [batch, 64, H, W]
            
            # Attention mechanism
            att = self.sigmoid(self.apperance_att_conv2(a))  # [batch, 1, H, W]
            
            # Apply attention
            a = a * att
            
            motion_features.append(m)
            appearance_features.append(a)
        
        # Stack features across time
        motion_features = torch.stack(motion_features, dim=1)  # [batch, T, 64, H, W]
        appearance_features = torch.stack(appearance_features, dim=1)  # [batch, T, 64, H, W]
        
        # Combine features
        f = motion_features + appearance_features  # [batch, T, 64, H, W]
        
        # Average across time dimension
        f = torch.mean(f, dim=1)  # [batch, 64, H, W]
        
        # Reshape for final dense layers
        f = f.view(B, -1)  # [batch, 64*H*W]
        f = self.relu(self.final_dense_1(f))
        f = self.dropout(f)
        out = self.final_dense_2(f)  # [batch, 1]
        
        return out
