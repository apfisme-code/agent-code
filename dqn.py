import torch.nn as nn
import torch
from env_const import MEMORY_LENGTH, VISION_SIZE, ACTION_SPACE

# ------------------- НЕЙРОННАЯ СЕТЬ DQN -------------------
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        # Свёрточная часть для памяти видений (каналы = MEMORY_LENGTH)
        self.conv = nn.Sequential(
            nn.Conv2d(MEMORY_LENGTH, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # Размер после свёртки: 64 * VISION_SIZE * VISION_SIZE
        conv_out_size = 64 * VISION_SIZE * VISION_SIZE
        # Полносвязная часть для шкал
        self.fc_scalar = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU()
        )
        # Объединение и выход на Q-значения
        self.fc_common = nn.Sequential(
            nn.Linear(conv_out_size + 128, 256),
            nn.ReLU(),
            nn.Linear(256, ACTION_SPACE)
        )

    def forward(self, vision_memory, scalars):
        # vision_memory: (batch, MEMORY_LENGTH, VISION_SIZE, VISION_SIZE)
        conv_out = self.conv(vision_memory)
        scalar_out = self.fc_scalar(scalars)
        combined = torch.cat([conv_out, scalar_out], dim=1)
        q_vals = self.fc_common(combined)
        return q_vals