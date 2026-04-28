import os
from collections import deque
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from .agent_base import AgentBase
from dqn import DQN, device
from env_const import (
    REPLAY_SIZE, MEMORY_LENGTH, EPS_START, ACTION_SPACE, BATCH_SIZE,
    EPS_END, EPS_DECAY, LR, GAMMA, VISION_SIZE
)

class TrainingAgent(AgentBase):
    """Агент, способный обучаться (DQN с воспроизведением опыта и целевой сетью)."""
    
    def __init__(self, load_path: str = None):
        """
        Args:
            load_path: если указан, загружает полный чекпоинт (модель + оптимизатор + epsilon и др.).
                      Если None, инициализирует всё с нуля.
        """
        self.policy_net = DQN().to(device)
        self.target_net = DQN().to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        
        self.replay_buffer = deque(maxlen=REPLAY_SIZE)
        self.memory_frames = deque(maxlen=MEMORY_LENGTH)
        self._fill_initial_memory()
        
        self.epsilon = EPS_START
        self.steps_done = 0
        
        # Загружаем чекпоинт, если указан
        if load_path and os.path.isfile(load_path):
            self._load_checkpoint(load_path)
        else:
            # Синхронизируем target-сеть с policy-сетью
            self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def _load_checkpoint(self, load_path: str):
        """Загружает полное состояние (веса, оптимизатор, epsilon, steps_done)."""
        checkpoint = torch.load(load_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.policy_net.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', EPS_START)
            self.steps_done = checkpoint.get('steps_done', 0)
        else:
            # Старый формат (только веса)
            self.policy_net.load_state_dict(checkpoint)
            print("Loaded only model weights, optimizer state is fresh.")
        # Синхронизируем target-сеть
        self.target_net.load_state_dict(self.policy_net.state_dict())
        print(f"Model loaded. epsilon={self.epsilon:.4f}, steps_done={self.steps_done}")
    
    def save_checkpoint(self, save_path: str, episode: int = None):
        """Сохраняет полный чекпоинт, включая состояние оптимизатора и параметры обучения."""
        checkpoint = {
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
        }
        if episode is not None:
            checkpoint['episode'] = episode
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")
    
    def update_memory(self, vision: np.ndarray):
        """Добавляет новый кадр в память (используется для стека)."""
        self.memory_frames.append(vision)
    
    def get_state_stack(self) -> np.ndarray:
        """Возвращает текущий стек кадров."""
        return np.array(self.memory_frames)
    
    def _fill_initial_memory(self):
        """Заполняет память начальными (пустыми) кадрами, до получения первого наблюдения."""
        dummy_frame = np.zeros((VISION_SIZE, VISION_SIZE), dtype=int)
        for _ in range(MEMORY_LENGTH):
            self.memory_frames.append(dummy_frame)

    def reset_memory(self):
        """Очищает память и заполняет пустыми кадрами."""
        self.memory_frames.clear()
        self._fill_initial_memory()
    
    def act(self, vision: np.ndarray, scalars: np.ndarray) -> int:
        self.update_memory(vision)
        frames = self.get_state_stack()
        frames_t = torch.FloatTensor(frames).unsqueeze(0).to(device)
        scalars_t = torch.FloatTensor(scalars).unsqueeze(0).to(device)
        
        if random.random() < self.epsilon:
            return random.randrange(ACTION_SPACE)
        
        with torch.no_grad():
            q_vals = self.policy_net(frames_t, scalars_t)
            return q_vals.argmax().item()
    
    def remember(self, scalars, action, reward, next_vision, next_scalars, done):
        """Сохраняет переход в буфер воспроизведения, используя текущий стек кадров."""
        current_stack = self.get_state_stack().copy()
        # Строим следующий стек: копируем текущую память, заменяем самый старый кадр на next_vision
        next_memory = deque(self.memory_frames, maxlen=MEMORY_LENGTH)
        next_memory.append(next_vision)
        while len(next_memory) < MEMORY_LENGTH:
            next_memory.append(next_vision)
        next_stack = np.array(next_memory)
        self.replay_buffer.append((current_stack, scalars, action, reward, next_stack, next_scalars, done))
    
    def learn(self):
        """Обучает policy_net на случайном батче из буфера."""
        if len(self.replay_buffer) < BATCH_SIZE:
            return
        batch = random.sample(self.replay_buffer, BATCH_SIZE)
        frames_stacks, scalars, actions, rewards, next_frames_stacks, next_scalars, dones = zip(*batch)
        
        frames_t = torch.FloatTensor(np.array(frames_stacks)).to(device)
        next_frames_t = torch.FloatTensor(np.array(next_frames_stacks)).to(device)
        scalars_t = torch.FloatTensor(np.array(scalars)).to(device)
        next_scalars_t = torch.FloatTensor(np.array(next_scalars)).to(device)
        actions_t = torch.LongTensor(actions).to(device)
        rewards_t = torch.FloatTensor(rewards).to(device)
        dones_t = torch.BoolTensor(dones).to(device)
        
        q_values = self.policy_net(frames_t, scalars_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_frames_t, next_scalars_t).max(1)[0]
            target_q = rewards_t + GAMMA * next_q * (~dones_t)
        
        loss = nn.MSELoss()(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Обновление epsilon
        self.epsilon = EPS_END + (EPS_START - EPS_END) * np.exp(-self.steps_done / EPS_DECAY)
        self.steps_done += 1
    
    def update_target_network(self):
        """Копирует веса из policy_net в target_net."""
        self.target_net.load_state_dict(self.policy_net.state_dict())