#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2023 The TARTRL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""""""

from tartrl.envs import make
from tartrl.modules.common import PPONet as Net
from tartrl.runners.common import PPOAgent as Agent

# 创建 环境
render_model = "rgb_array"

env = make("CartPole-v0", render_mode=render_model)
# 创建 神经网络
net = Net(env)
# 初始化训练器
agent = Agent(net, use_wandb=False)
# 开始训练
agent.train(total_time_steps=20000)

# 开始测试环境
render_model = "human"
env = make("CartPole-v0", render_mode=render_model)
obs, info = env.reset()
done = False
step = 0
while not done:
    # 智能体根据 observation 预测下一个动作
    action, _ = agent.act(obs, deterministic=True)
    obs, r, done, info = env.step(action)

    step += 1
    print(f"{step}: action:{action}, reward:{r}")

env.close()
