import numpy as np
from collections import deque

class TrainingMonitor:
    def __init__(self, 
                 window_size=50,          # размер скользящего окна
                 min_episodes_before_stop=100,  # минимальное кол-во эпизодов перед проверкой
                 reward_improvement_threshold=0.1,  # порог улучшения средней награды
                 max_plateau_episodes=200,         # сколько эпизодов ждать улучшения
                 min_avg_reward_threshold=-5.0,    # минимальная средняя награда
                 min_avg_length_threshold=100,     # минимальная средняя длина эпизода (шагов)
                 max_loss_threshold=10.0,          # порог потери
                 verbose=True):
        self.window_size = window_size
        self.min_episodes_before_stop = min_episodes_before_stop
        self.reward_improvement_threshold = reward_improvement_threshold
        self.max_plateau_episodes = max_plateau_episodes
        self.min_avg_reward_threshold = min_avg_reward_threshold
        self.min_avg_length_threshold = min_avg_length_threshold
        self.max_loss_threshold = max_loss_threshold
        self.verbose = verbose

        self.rewards = deque(maxlen=window_size)
        self.lengths = deque(maxlen=window_size)
        self.losses = deque(maxlen=window_size)
        self.q_values = deque(maxlen=window_size)

        self.best_avg_reward = -np.inf
        self.episodes_without_improvement = 0
        self.last_episode = 0

    def update(self, episode, reward, length, loss=None, max_q=None):
        """Добавляет данные одного эпизода."""
        self.rewards.append(reward)
        self.lengths.append(length)
        if loss is not None and not np.isnan(loss) and not np.isinf(loss):
            self.losses.append(loss)
        if max_q is not None and not np.isnan(max_q) and not np.isinf(max_q):
            self.q_values.append(max_q)
        self.last_episode = episode

    def check_and_report(self):
        """Анализирует накопленные данные и возвращает (stop_training: bool, message: str)."""
        if len(self.rewards) < self.window_size:
            return False, "Not enough data yet."

        avg_reward = np.mean(self.rewards)
        avg_length = np.mean(self.lengths)
        avg_loss = np.mean(self.losses) if self.losses else None
        avg_q = np.mean(self.q_values) if self.q_values else None

        # Проверка на NaN/Inf в loss
        if avg_loss is not None and (np.isnan(avg_loss) or np.isinf(avg_loss)):
            return True, f"Loss diverged: avg_loss={avg_loss}"

        # Проверка на минимальную среднюю награду и длину (после начальных эпизодов)
        if self.last_episode > self.min_episodes_before_stop:
            if avg_reward < self.min_avg_reward_threshold:
                return True, f"Avg reward too low ({avg_reward:.2f} < {self.min_avg_reward_threshold}) for last {self.window_size} episodes."
            if avg_length < self.min_avg_length_threshold:
                return True, f"Avg episode length too short ({avg_length:.1f} < {self.min_avg_length_threshold}) for last {self.window_size} episodes."

        # Проверка на стагнацию (plateau) – отсутствие улучшения лучшей средней награды
        if avg_reward > self.best_avg_reward + self.reward_improvement_threshold:
            self.best_avg_reward = avg_reward
            self.episodes_without_improvement = 0
            if self.verbose:
                print(f"  [Monitor] New best avg reward: {avg_reward:.2f}")
        else:
            self.episodes_without_improvement += 1

        if self.episodes_without_improvement >= self.max_plateau_episodes and self.last_episode > self.min_episodes_before_stop:
            return True, f"No improvement for {self.max_plateau_episodes} episodes (best avg reward = {self.best_avg_reward:.2f})"

        # Предупреждения (но не остановка)
        if avg_loss is not None and avg_loss > self.max_loss_threshold:
            if self.verbose:
                print(f"  [Warning] High loss: {avg_loss:.2f} > {self.max_loss_threshold}")

        return False, "OK"

    def get_summary(self):
        """Возвращает строку с текущими метриками."""
        if len(self.rewards) == 0:
            return "No data"
        avg_reward = np.mean(self.rewards)
        avg_length = np.mean(self.lengths)
        avg_loss = np.mean(self.losses) if self.losses else 0
        avg_q = np.mean(self.q_values) if self.q_values else 0
        return (f"avg_reward={avg_reward:6.2f}, avg_len={avg_length:5.1f}, "
                f"avg_loss={avg_loss:6.4f}, avg_Q={avg_q:6.2f}")