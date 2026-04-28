import numpy as np
import random
from env_const import VISION_SIZE, VISION_RANGE

GRID_SIZE = 10
MAX_HUNGER = 100
MAX_HEALTH = 100
MAX_ENERGY = 100
HUNGER_RESTORE = 30
ENERGY_RESTORE = 30
MAX_STEPS = 500

# ------------------- СРЕДА -------------------
class VirtualWorld:
    def __init__(self):
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        # размещаем объекты: 1-еда, 2-безопасное место
        num_food = 3
        num_safe = 3
        self._place_objects(num_food, 1)
        self._place_objects(num_safe, 2)
        self.reset()

    def _place_objects(self, count, obj_type):
        placed = 0
        while placed < count:
            x, y = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
            if self.grid[x, y] == 0:
                self.grid[x, y] = obj_type
                placed += 1

    def reset(self):
        self.agent_pos = [GRID_SIZE//2, GRID_SIZE//2]  # центр
        self.hunger = MAX_HUNGER
        self.health = MAX_HEALTH
        self.energy = MAX_ENERGY
        self.done = False
        self.steps = 0
        return self.get_state()

    def _get_observation(self):
        # видение: квадрат VISION_SIZE x VISION_SIZE вокруг агента
        vision = np.zeros((VISION_SIZE, VISION_SIZE), dtype=int)
        ax, ay = self.agent_pos
        for i in range(VISION_SIZE):
            for j in range(VISION_SIZE):
                wx = ax + (i - VISION_RANGE)
                wy = ay + (j - VISION_RANGE)
                # периодические границы (тор)
                wx = wx % GRID_SIZE
                wy = wy % GRID_SIZE
                if wx == ax and wy == ay:
                    vision[i, j] = 3   # сам агент
                else:
                    vision[i, j] = self.grid[wx, wy]
        return vision

    def get_state(self):
        vision = self._get_observation()
        # нормализованные шкалы
        state_scalars = np.array([self.hunger/MAX_HUNGER,
                                   self.health/MAX_HEALTH,
                                   self.energy/MAX_ENERGY], dtype=np.float32)
        return vision, state_scalars

    def step(self, action):
        reward = 0
        # перемещение
        if action < 4:
            dx, dy = [(0,1),(0,-1),(-1,0),(1,0)][action]   # вверх,вниз,влево,вправо
            new_x = (self.agent_pos[0] + dx) % GRID_SIZE
            new_y = (self.agent_pos[1] + dy) % GRID_SIZE
            self.agent_pos = [new_x, new_y]
        # взаимодействие
        elif action == 4:
            x, y = self.agent_pos
            cell = self.grid[x, y]
            if cell == 1:   # еда
                self.hunger = min(MAX_HUNGER, self.hunger + HUNGER_RESTORE)
                if self.hunger > 0 and self.energy > 0:
                    self.health = min(MAX_HEALTH, self.health + 1)
                reward += 1.0
            elif cell == 2: # безопасное место
                self.energy = min(MAX_ENERGY, self.energy + ENERGY_RESTORE)
                if self.hunger > 0 and self.energy > 0:
                    self.health = min(MAX_HEALTH, self.health + 1)
                reward += 1.0

        # естественное истощение
        self.hunger = max(0, self.hunger - 1)
        self.energy = max(0, self.energy - 1)
        if self.hunger == 0 or self.energy == 0:
            self.health = max(0, self.health - 1)
            reward -= 1.0   # штраф за истощение

        if self.health <= 0:
            self.done = True
            reward -= 10.0
        else:
            reward += 0.1   # поощрение за выживание

        self.steps += 1
        if self.steps >= MAX_STEPS:
            self.done = True

        next_vision, next_scalars = self.get_state()
        return (next_vision, next_scalars), reward, self.done
