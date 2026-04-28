from abc import ABC, abstractmethod
import numpy as np

class AgentBase(ABC):
    """Абстрактный класс для всех агентов."""
    
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
    
    @abstractmethod
    def reset_memory(self):
        """Сбрасывает внутреннюю память агента (кадры) в начале эпизода."""
        pass