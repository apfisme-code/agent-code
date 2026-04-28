# ------------------- ПАРАМЕТРЫ -------------------
VISION_RANGE = 2          # видим квадрат (2*2+1)=5x5
VISION_SIZE = VISION_RANGE * 2 + 1
MEMORY_LENGTH = 4         # храним последние 4 видения
ACTION_SPACE = 5          # 0-вверх,1-вниз,2-влево,3-вправо,4-взаимодействие
EPISODES = 500
BATCH_SIZE = 32
GAMMA = 0.99
LR = 0.001
REPLAY_SIZE = 10000
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 200