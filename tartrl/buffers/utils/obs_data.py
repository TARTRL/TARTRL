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

import numpy as np
from treevalue import TreeValue, reduce_


class ObsData(TreeValue):
    def flatten(self):
        return reduce_(self, lambda **kwargs: np.concatenate(list(kwargs.values())))

    def step_batch(self, step):
        return_dict = {}
        for str_key in self.keys():
            return_dict[str_key] = np.concatenate(self[str_key][step])
        return return_dict

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.step_batch(key)
        else:
            return super().__getitem__(key)

    def step_flatten(self, step):
        reduce_(
            self,
            lambda **kwargs: np.concatenate(
                [value[step] for value in kwargs.values()], -1
            ),
        )
