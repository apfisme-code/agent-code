import argparse
import os
from virtual_world import VirtualWorld
from agents.simulation_agent import SimulationAgent
from agents.training_agent import TrainingAgent
from dqn import device
import torch

TARGET_UPDATE = 100


def train(episodes=500, save_path="models/agent_model.pth", load_path=None):

    env = VirtualWorld()
    agent = TrainingAgent(load_path=load_path)
    
    start_episode = 0

    # Если передан путь к существующей модели – загружаем
    if load_path and os.path.isfile(load_path):
        # Проверим, сохранён ли номер эпизода в чекпоинте
        checkpoint = torch.load(load_path, map_location=device)
        if isinstance(checkpoint, dict) and 'episode' in checkpoint:
            start_episode = checkpoint['episode']
            print(f"Resuming from episode {start_episode}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for episode in range(start_episode, start_episode + episodes):
        vision, scalars = env.reset()
        agent.reset_memory()           # очищаем память перед новым эпизодом
        total_reward = 0
        step = 0

        while not env.done:
            action = agent.act(vision, scalars)
            (next_vision, next_scalars), reward, done = env.step(action)
            agent.remember(scalars, action, reward, next_vision, next_scalars, done)
            agent.learn()
            vision, scalars = next_vision, next_scalars
            total_reward += reward
            step += 1
            if step % TARGET_UPDATE == 0:
                agent.update_target_network()

        agent.save_checkpoint(save_path, episode)
        print(f"Episode {episode}, total reward: {total_reward:.2f}, steps: {step}, epsilon: {agent.epsilon:.3f}")


def test(load_path="models/agent_model.pth", max_steps=1000):
    """Тестирует предобученного агента."""
    env = VirtualWorld()
    agent = SimulationAgent(model_path=load_path)

    vision, scalars = env.reset()
    agent.reset_memory()
    total_reward = 0
    step = 0
    done = False
    while not done and step < max_steps:
        action = agent.act(vision, scalars)
        (vision, scalars), reward, done = env.step(action)
        total_reward += reward
        step += 1
        print(f"Step {step}: pos={env.agent_pos}, hunger={env.hunger}, health={env.health}, energy={env.energy}, action={action}")
    print(f"Test finished. Total reward: {total_reward:.2f}, steps: {step}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test", "train_and_test"], default="train_and_test")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--model", type=str, default="models/agent_model.pth")
    parser.add_argument("--load", type=str, default=None, help="Путь к модели для продолжения обучения")
    args = parser.parse_args()

    if args.mode == "train":
        train(episodes=1, save_path=args.model, load_path=args.model)
        test(load_path=args.model)
        #train(episodes=args.episodes, save_path=args.model, load_path=args.model)
    elif args.mode == "test":   
        test(load_path=args.model)
    elif args.mode == "train_and_test":
        train(episodes=1, save_path=args.model, load_path=args.model)
        test(load_path=args.model)