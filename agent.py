from collections import deque
import torch.optim as optim
import torch
import torch.nn as nn 
import numpy as np
import random
from dqn import DQN
from env_const import REPLAY_SIZE, MEMORY_LENGTH, EPS_START, ACTION_SPACE, BATCH_SIZE, EPS_END, EPS_DECAY, LR, GAMMA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------- АГЕНТ С ПАМЯТЬЮ -------------------
class Agent:
    def __init__(self):
        self.policy_net = DQN().to(device)
        self.target_net = DQN().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.replay_buffer = deque(maxlen=REPLAY_SIZE)
        self.memory_frames = deque(maxlen=MEMORY_LENGTH)  # хранит видения
        self.epsilon = EPS_START
        self.steps_done = 0

    def act(self, vision, scalars, eval_mode=False):
        # Сначала обновляем память текущим кадром
        self.update_memory(vision)
        frames = self.get_state_stack()   # (MEMORY_LENGTH, H, W)
        frames_t = torch.FloatTensor(frames).unsqueeze(0).to(device)   # (1, MEM, H, W)
        scalars_t = torch.FloatTensor(scalars).unsqueeze(0).to(device)
        if not eval_mode and random.random() < self.epsilon:
            return random.randrange(ACTION_SPACE)
        with torch.no_grad():
            q_vals = self.policy_net(frames_t, scalars_t)
            return q_vals.argmax().item()
    
    def remember(self, vision, scalars, action, reward, next_vision, next_scalars, done):
        # Для сохранения текущего стека используем текущую память (уже содержит vision)
        current_stack = self.get_state_stack().copy()   # важно скопировать, т.к. память меняется
        # Строим следующий стек: сначала копируем текущую память, убираем самый старый кадр и добавляем next_vision
        next_memory = deque(self.memory_frames, maxlen=MEMORY_LENGTH)  # копия
        next_memory.append(next_vision)
        # Если ещё не заполнена – дублируем
        while len(next_memory) < MEMORY_LENGTH:
            next_memory.append(next_vision)
        next_stack = np.array(next_memory)
        # Сохраняем в буфер
        self.replay_buffer.append((current_stack, scalars, action, reward, next_stack, next_scalars, done))

    def learn(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return
        batch = random.sample(self.replay_buffer, BATCH_SIZE)
        frames_stacks, scalars, actions, rewards, next_frames_stacks, next_scalars, dones = zip(*batch)
        
        frames_t = torch.FloatTensor(np.array(frames_stacks)).to(device)          # (B, MEM, H, W)
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
        
        self.epsilon = EPS_END + (EPS_START - EPS_END) * np.exp(-self.steps_done / EPS_DECAY)
        self.steps_done += 1

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_memory(self, vision):
        """Добавляет новый кадр в память, автоматически вытесняя старые"""
        self.memory_frames.append(vision)
        # Если памяти недостаточно (начало эпизода), дублируем первый кадр
        while len(self.memory_frames) < MEMORY_LENGTH:
            self.memory_frames.append(vision)

    def get_state_stack(self):
        """Возвращает текущий стек кадров (MEMORY_LENGTH, H, W)"""
        return np.array(self.memory_frames)

    def reset_memory(self):
        """Очищает память в начале нового эпизода"""
        self.memory_frames.clear()