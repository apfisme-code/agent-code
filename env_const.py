# ------------------- ПАРАМЕТРЫ -------------------
GRID_SIZE = 10
VISION_RANGE = 2          # видим квадрат (2*2+1)=5x5
VISION_SIZE = VISION_RANGE * 2 + 1
MEMORY_LENGTH = 4         # храним последние 4 видения
MAX_HUNGER = 100
MAX_HEALTH = 100
MAX_ENERGY = 100
HUNGER_RESTORE = 30
ENERGY_RESTORE = 30
ACTION_SPACE = 5          # 0-вверх,1-вниз,2-влево,3-вправо,4-взаимодействие
EPISODES = 500
MAX_STEPS = 500
BATCH_SIZE = 32
GAMMA = 0.99
LR = 0.001
TARGET_UPDATE = 100
REPLAY_SIZE = 10000
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 200