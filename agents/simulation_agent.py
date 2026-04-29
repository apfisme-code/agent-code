import numpy as np
import torch
from .agent_base import AgentBase
from dqn import DQN, device

class SimulationAgent(AgentBase):
    """Агент, использующий предобученную модель для принятия решений (без обучения)."""
    
    def __init__(self, model_path: str):
        """
        Args:
            model_path: путь к файлу .pth с весами модели (state_dict).
        """
        super().__init__()  # инициализирует память
        self.policy_net = DQN().to(device)
        # Загружаем веса
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.policy_net.load_state_dict(checkpoint)
        self.policy_net.eval()  # режим оценки (отключает dropout, если есть)
    
    def act(self, vision: np.ndarray, scalars: np.ndarray) -> int:
        """Принимает действие без исследования (epsilon=0)."""
        self.update_memory(vision)
        frames = self.get_state_stack()
        frames_t = torch.FloatTensor(frames).unsqueeze(0).to(device)   # (1, MEM, H, W)
        scalars_t = torch.FloatTensor(scalars).unsqueeze(0).to(device)
        with torch.no_grad():
            q_vals = self.policy_net(frames_t, scalars_t)
            return q_vals.argmax().item()
        
