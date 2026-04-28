from collections import deque
import numpy as np
import torch
from agent_base import AgentBase
from dqn import DQN
from env_const import MEMORY_LENGTH, VISION_SIZE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimulationAgent(AgentBase):
    """Агент, использующий предобученную модель для принятия решений (без обучения)."""
    
    def __init__(self, model_path: str):
        """
        Args:
            model_path: путь к файлу .pth с весами модели (state_dict).
        """
        self.policy_net = DQN().to(device)
        # Загружаем веса
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.policy_net.load_state_dict(checkpoint)
        self.policy_net.eval()  # режим оценки (отключает dropout, если есть)
        
        self.memory_frames = deque(maxlen=MEMORY_LENGTH)
        self._fill_initial_memory()
    
    def _fill_initial_memory(self):
        """Заполняет память начальными (пустыми) кадрами, до получения первого наблюдения."""
        dummy_frame = np.zeros((VISION_SIZE, VISION_SIZE), dtype=int)
        for _ in range(MEMORY_LENGTH):
            self.memory_frames.append(dummy_frame)
    
    def update_memory(self, vision: np.ndarray):
        """Добавляет новый кадр в память, вытесняя старый."""
        self.memory_frames.append(vision)
    
    def get_state_stack(self) -> np.ndarray:
        """Возвращает текущий стек кадров (MEMORY_LENGTH, H, W)."""
        return np.array(self.memory_frames)
    
    def reset_memory(self):
        """Очищает память и заполняет пустыми кадрами."""
        self.memory_frames.clear()
        self._fill_initial_memory()
    
    def act(self, vision: np.ndarray, scalars: np.ndarray, **kwargs) -> int:
        """Принимает действие без исследования (epsilon=0)."""
        self.update_memory(vision)
        frames = self.get_state_stack()
        frames_t = torch.FloatTensor(frames).unsqueeze(0).to(device)   # (1, MEM, H, W)
        scalars_t = torch.FloatTensor(scalars).unsqueeze(0).to(device)
        with torch.no_grad():
            q_vals = self.policy_net(frames_t, scalars_t)
            action = q_vals.argmax().item()
        return action