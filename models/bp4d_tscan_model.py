import torch
import torch.nn as nn

class BP4D_TSCAN(nn.Module):
    """
    TSCAN model for BP4D dataset with pseudo-labeling.
    """
    def __init__(self, frame_depth=3, drop_rate=0.2):
        super(BP4D_TSCAN, self).__init__()
        
        # Motion branch (2D convolutions)
        self.motion_conv1 = nn.Conv2d(frame_depth, 32, kernel_size=3, padding=1)
        self.motion_conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.motion_conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.motion_conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Appearance branch (2D convolutions)
        self.apperance_conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.apperance_conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.apperance_conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.apperance_conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Appearance attention
        self.apperance_att_conv1 = nn.Conv2d(32, 1, kernel_size=1)
        self.apperance_att_conv2 = nn.Conv2d(64, 1, kernel_size=1)
        
        # Final dense layers
        self.final_dense_1 = nn.Linear(16384, 128)  # 16384 = 64 * 16 * 16
        self.final_dense_2 = nn.Linear(128, 1)
        
        # Activation and normalization
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(drop_rate)
        
    def forward(self, x):
        # Split input into motion and appearance streams
        motion_input = x[:, :3, :, :]  # First 3 channels for motion
        appearance_input = x[:, :3, :, :]  # First 3 channels for appearance
        
        # Process motion branch
        motion = self.relu(self.motion_conv1(motion_input))
        motion = self.relu(self.motion_conv2(motion))
        motion = self.relu(self.motion_conv3(motion))
        motion = self.relu(self.motion_conv4(motion))
        
        # Process appearance branch
        appearance1 = self.relu(self.apperance_conv1(appearance_input))
        appearance2 = self.relu(self.apperance_conv2(appearance1))
        appearance3 = self.relu(self.apperance_conv3(appearance2))
        appearance4 = self.relu(self.apperance_conv4(appearance3))
        
        # Calculate attention weights
        att_weights1 = self.sigmoid(self.apperance_att_conv1(appearance2))
        att_weights2 = self.sigmoid(self.apperance_att_conv2(appearance4))
        
        # Apply attention
        appearance2 = appearance2 * att_weights1
        appearance4 = appearance4 * att_weights2
        
        # Reshape for concatenation
        motion = motion.view(motion.size(0), -1)
        appearance = appearance4.view(appearance4.size(0), -1)
        
        # Average the features from both branches
        combined = (motion + appearance) / 2
        
        # Final prediction
        x = self.relu(self.final_dense_1(combined))
        x = self.dropout(x)
        output = self.final_dense_2(x)
        
        return output 