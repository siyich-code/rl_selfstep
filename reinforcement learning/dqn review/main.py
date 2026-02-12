import numpy as np
from tqdm import tqdm
import torch
import random
import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import dqn1
import ale_py
# 直接在这里改环境名；设为 None 时才使用命令行 --env-name
ENV_NAME = "ALE/Breakout-v5"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def moving_average(values, window):
    if len(values) == 0:
        return []
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(float(np.mean(values[start:i + 1])))
    return out


def save_metrics_table(metrics, save_dir):
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / "training_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["episode", "reward", "steps", "avg_loss", "epsilon", "buffer_size"],
        )
        writer.writeheader()
        writer.writerows(metrics)
    return csv_path


def plot_metrics(metrics, save_dir, ma_window):
    save_dir.mkdir(parents=True, exist_ok=True)
    episodes = [m["episode"] for m in metrics]
    rewards = [m["reward"] for m in metrics]
    steps = [m["steps"] for m in metrics]
    epsilons = [m["epsilon"] for m in metrics]
    reward_ma = moving_average(rewards, ma_window)
    steps_ma = moving_average(steps, ma_window)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    axes[0].plot(episodes, rewards, label="reward", alpha=0.5)
    axes[0].plot(episodes, reward_ma, label=f"reward_ma({ma_window})", linewidth=2)
    axes[0].set_ylabel("Reward")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(episodes, steps, label="steps", alpha=0.5, color="tab:orange")
    axes[1].plot(episodes, steps_ma, label=f"steps_ma({ma_window})", linewidth=2, color="tab:red")
    axes[1].set_ylabel("Steps")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(episodes, epsilons, label="epsilon", color="tab:green")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Epsilon")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    fig_path = save_dir / "training_curves.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    return fig_path


def train(args):
    set_seed(args.seed)
    device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


    env_agent = dqn1.env_agent(args.env_name)
    replay_buffer = dqn1.replay_buffer(args.buffer_capacity)
    action_dim = env_agent.env.action_space.n

    agent = dqn1.dqn(
        learning_rate=args.lr,
        gamma=args.gamma,
        epsilon=args.epsilon_start,
        target_update=args.target_update,
        device=device,
        specific_actions=action_dim,
    )

    metrics = []

    for episode in tqdm(range(1, args.num_episodes + 1), desc="Training"):
        state = env_agent.reset()
        done = False
        episode_return = 0.0
        episode_losses = []
        episode_steps = 0

        while not done:
            action = agent.take_action(state)
            next_state, reward, done, _ = env_agent.step(action)
            replay_buffer.add(state, action, reward, next_state, done)

            loss = agent.update(replay_buffer, args.batch_size)
            if loss is not None:
                episode_losses.append(loss)

            state = next_state
            episode_return += reward
            episode_steps += 1

        agent.epsilon = max(args.epsilon_min, agent.epsilon * args.epsilon_decay)

        avg_loss = float(np.mean(episode_losses)) if episode_losses else float("nan")
        metrics.append(
            {
                "episode": episode,
                "reward": float(episode_return),
                "steps": episode_steps,
                "avg_loss": avg_loss,
                "epsilon": float(agent.epsilon),
                "buffer_size": replay_buffer.size(),
            }
        )

        if episode % args.log_interval == 0:
            recent = metrics[-args.log_interval:]
            recent_returns = [m["reward"] for m in recent]
            recent_steps = [m["steps"] for m in recent]
            recent_losses = [m["avg_loss"] for m in recent if not np.isnan(m["avg_loss"])]
            mean_return = float(np.mean(recent_returns))
            mean_steps = float(np.mean(recent_steps))
            mean_loss = float(np.mean(recent_losses)) if recent_losses else float("nan")
            print(
                f"Episode {episode:4d} | "
                f"mean_return={mean_return:8.3f} | "
                f"mean_steps={mean_steps:8.1f} | "
                f"mean_loss={mean_loss:10.6f} | "
                f"epsilon={agent.epsilon:.4f} | "
                f"buffer={replay_buffer.size()}"
            )

    env_agent.env.close()
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train DQN with replay buffer (Atari).")
    parser.add_argument("--env-name", type=str, default="ALE/Breakout-v5")
    parser.add_argument("--num-episodes", type=int, default=300)
    parser.add_argument("--buffer-capacity", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--target-update", type=int, default=1000)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.1)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-dir", type=str, default="reinforcement learning/牛魔鬼怪/runs")
    parser.add_argument("--ma-window", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    if ENV_NAME is not None:
        args.env_name = ENV_NAME
    try:
        metrics = train(args)
        save_dir = Path(args.save_dir)
        csv_path = save_metrics_table(metrics, save_dir)
        fig_path = plot_metrics(metrics, save_dir, args.ma_window)
        print(f"metrics_table={csv_path}")
        print(f"training_plot={fig_path}")
    except RuntimeError as e:
        raise SystemExit(str(e))


if __name__ == "__main__":
    main()
