# 训练可视化说明

运行 `main.py` 后会自动输出两类文件，默认在 `reinforcement learning/牛魔鬼怪/runs`：

- `training_metrics.csv`：每个 episode 一行的训练指标表格
- `training_curves.png`：reward、steps、epsilon 三条曲线图

## CSV 列说明

- `episode`：第几回合
- `reward`：该回合累计奖励
- `steps`：该回合与环境交互步数
- `avg_loss`：该回合内所有更新 step 的平均 TD loss
- `epsilon`：该回合结束后的探索率
- `buffer_size`：回放池当前样本数量

## 如何判断训练效果

- `reward` 或 reward 的移动平均线持续上升，一般表示策略在变好
- `steps` 会因任务不同而变化：
  - 生存型任务常见 `steps` 上升
  - 尽快结束型任务常见 `steps` 下降
- `avg_loss` 不需要单调下降，但应避免长期爆炸或全程 NaN
- `epsilon` 逐步下降后，reward 不应明显崩掉

## 常用参数

- `--save-dir`：修改输出目录
- `--ma-window`：曲线移动平均窗口（默认 20）
- `--log-interval`：终端打印统计的间隔回合数

## 示例

```bash
python3 "reinforcement learning/牛魔鬼怪/main.py" \
  --num-episodes 500 \
  --save-dir "reinforcement learning/牛魔鬼怪/runs/breakout_exp1" \
  --ma-window 30
```
