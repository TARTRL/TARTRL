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

"""
一个简单的训练例子，使用PPO算法训练CartPole-v1环境
"""

import gymnasium as gym

from tartrl.modules.common import PPONet as Net
from tartrl.runners.common import PPOAgent as Agent


# 创建 环境
env = gym.make("CartPole-v1")
# 创建 神经网络
net = Net(env)
# 初始化训练器
agent = Agent(net)
# 开始训练
agent.train(total_time_steps=20000)
