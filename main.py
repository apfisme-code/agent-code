# Добавьте в начало файла импорты:
import argparse
import os
from virtual_world import VirtualWorld
from agent import Agent, device 
import torch
from env_const import TARGET_UPDATE


def train(episodes=500, save_path="models/agent_model.pth", load_path=None):

    env = VirtualWorld()
    agent = Agent()
    
    start_episode = 0
    # Если передан путь к существующей модели – загружаем
    if load_path is not None and os.path.isfile(load_path):
        print(f"Loading existing model from {load_path}")
        checkpoint = torch.load(load_path, map_location=device)
        # Проверяем, сохранён ли полный чекпоинт (веса + состояние оптимизатора)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            agent.policy_net.load_state_dict(checkpoint['model_state_dict'])
            # если есть состояние оптимизатора – загружаем
            if 'optimizer_state_dict' in checkpoint:
                agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # если сохранены epsilon и steps_done – можно восстановить
            if 'epsilon' in checkpoint:
                agent.epsilon = checkpoint['epsilon']
            if 'steps_done' in checkpoint:
                agent.steps_done = checkpoint['steps_done']
            if 'start_episode' in checkpoint:
                start_episode = checkpoint['start_episode']
        else:
            # старый формат – только веса модели
            agent.policy_net.load_state_dict(checkpoint)
        # Синхронизируем target-сеть с policy-сетью
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
        print(f"Model loaded. Resuming from episode {start_episode}, epsilon={agent.epsilon:.4f}")
    else:
        print("Starting training from scratch")
    
    for episode in range(episodes):
        vision, scalars = env.reset()
        agent.reset_memory()           # очищаем память перед новым эпизодом
        total_reward = 0
        step = 0
        while not env.done:
            action = agent.act(vision, scalars)
            (next_vision, next_scalars), reward, done = env.step(action)
            agent.remember(vision, scalars, action, reward, next_vision, next_scalars, done)
            agent.learn()
            vision, scalars = next_vision, next_scalars
            total_reward += reward
            step += 1
            if step % TARGET_UPDATE == 0:
                agent.update_target()
        print(f"Episode {episode}, total reward: {total_reward:.2f}, steps: {step}, epsilon: {agent.epsilon:.3f}")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'model_state_dict': agent.policy_net.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'epsilon': agent.epsilon,
        'steps_done': agent.steps_done,
        'start_episode': start_episode + episodes
    }, save_path)
    print(f"Model saved to {save_path}")

def test(load_path="models/agent_model.pth", max_steps=1000):
    env = VirtualWorld()
    agent = Agent()
    checkpoint = torch.load(load_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        agent.policy_net.load_state_dict(checkpoint['model_state_dict'])
    else:
        agent.policy_net.load_state_dict(checkpoint)
    agent.epsilon = 0.0
    vision, scalars = env.reset()
    agent.reset_memory()
    total_reward = 0
    step = 0
    while not env.done and step < max_steps:
        action = agent.act(vision, scalars, eval_mode=True)
        (vision, scalars), reward, done = env.step(action)
        total_reward += reward
        step += 1
        print(f"Step {step}: pos={env.agent_pos}, hunger={env.hunger}, health={env.health}, energy={env.energy}, action={action}")
    print(f"Test finished. Total reward: {total_reward:.2f}, steps: {step}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--model", type=str, default="models/agent_model.pth")
    parser.add_argument("--load", type=str, default=None, help="Путь к модели для продолжения обучения")
    args = parser.parse_args()

    if args.mode == "train":
        train(episodes=1, save_path=args.model, load_path=args.model)

    test(load_path=args.model)