from abc import ABC, abstractmethod
from collections import deque
import numpy as np
from env_const import MEMORY_LENGTH, VISION_SIZE

class AgentBase(ABC):
    """Абстрактный класс для всех агентов."""
    

    def __init__(self):
        self.memory_frames = deque(maxlen=MEMORY_LENGTH)
        self._fill_initial_memory()
    
    def _fill_initial_memory(self):
        """Заполняет память нулевыми кадрами до получения первого наблюдения."""
        dummy_frame = np.zeros((VISION_SIZE, VISION_SIZE), dtype=np.int32)
        for _ in range(MEMORY_LENGTH):
            self.memory_frames.append(dummy_frame)
    
    def update_memory(self, vision: np.ndarray):
        """Добавляет новый кадр в память, вытесняя старый."""
        self.memory_frames.append(vision)
    
    def get_state_stack(self) -> np.ndarray:
        """Возвращает текущий стек кадров (MEMORY_LENGTH, H, W)."""
        return np.array(self.memory_frames)
    
    def reset_memory(self):
        """Очищает память и заполняет пустыми кадрами (перед новым эпизодом)."""
        self.memory_frames.clear()
        self._fill_initial_memory()

    @abstractmethod
    def act(self, vision: np.ndarray, scalars: np.ndarray) -> int:
        """
        Принимает решение на основе текущего наблюдения (видение + шкалы).
        
        Args:
            vision: 2D массив (VISION_SIZE, VISION_SIZE) - что видит агент.
            scalars: 1D массив (3,) - нормализованные значения голода, здоровья, бодрости.        
        Returns:
            int: выбранное действие (0..ACTION_SPACE-1).
        """
        pass
   
