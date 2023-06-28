#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
r"""
This file hosts task-specific or trainer-specific environments for trainers.
All environments here should be a (direct or indirect ) subclass of Env class
in habitat. Customized environments should be registered using
``@baseline_registry.register_env(name="myEnv")` for reusability
"""

import os
import random
from datetime import datetime
import gym
import numpy as np
import quaternion
from gym.spaces.dict_space import Dict as SpaceDict
from typing import Any, Dict, Iterator, List, Optional, Tuple, Type, Union

import habitat
from habitat import Config, Dataset
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.constants import scenes, master_scene_dir
from habitat.core.simulator import Observations, Simulator, AgentState
from habitat.sims import make_sim
from habitat.core.logging import logger
from habitat.core.registry import registry


# adapted from detectron2
def seed_all_rng(seed=None):
    """
    Set the random seed for the RNG in torch, numpy and python.
    Args:
        seed (int): if None, will use a strong random seed.
    """
    if seed is None:
        seed = (
            os.getpid()
            + int(datetime.now().strftime("%S%f"))
        )
        seed = int(seed % 1e4)
    # np.random.seed(seed)
    # random.seed(seed)
    # os.environ["PYTHONHASHSEED"] = str(seed)
    return seed


def get_env_class(env_name: str) -> Type[habitat.RLEnv]:
    r"""Return environment class based on name.

    Args:
        env_name: name of the environment.

    Returns:
        Type[habitat.RLEnv]: env class.
    """
    return baseline_registry.get_env(env_name)


@baseline_registry.register_env(name="NavRLEnv")
class NavRLEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._rl_config = config.RL
        self._core_env_config = config.TASK_CONFIG
        self._reward_measure_name = self._rl_config.REWARD_MEASURE
        self._success_measure_name = self._rl_config.SUCCESS_MEASURE

        self._previous_measure = None
        self._previous_action = None
        super().__init__(self._core_env_config, dataset)

    def reset(self):
        self._previous_action = None
        observations = super().reset()
        self._previous_measure = self._env.get_metrics()[
            self._reward_measure_name
        ]
        return observations

    def step(self, *args, **kwargs):
        self._previous_action = kwargs["action"]
        return super().step(*args, **kwargs)

    def get_reward_range(self):
        return (
            self._rl_config.SLACK_REWARD - 1.0,
            self._rl_config.SUCCESS_REWARD + 1.0,
        )

    def get_reward(self, observations):
        reward = self._rl_config.SLACK_REWARD

        current_measure = self._env.get_metrics()[self._reward_measure_name]

        reward += self._previous_measure - current_measure
        self._previous_measure = current_measure

        if self._episode_success():
            reward += self._rl_config.SUCCESS_REWARD

        return reward

    def _episode_success(self):
        return self._env.get_metrics()[self._success_measure_name]

    def get_done(self, observations):
        done = False
        if self._env.episode_over or self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()


@baseline_registry.register_env(name="SimpleRLEnv")
class SimpleRLEnv(habitat.RLEnv):

    observation_space: SpaceDict
    action_space: SpaceDict
    _config: Config
    _sim: Simulator
    _agent_state: AgentState
    _max_episode_steps: int
    _elapsed_steps: int

    def __init__(self, config):
        self._config = config
        self._sim = None
        self._agent = None

        # specify episode information
        self._max_episode_steps = self._config.ENVIRONMENT.MAX_EPISODE_STEPS
        # reward range
        self.reward_range = self.get_reward_range()

        observations = self.reset()

        # specify action and observation space
        self.observation_space = self._sim.sensor_suite.observation_spaces # 256 x 256 x 3
        self.action_space = self._sim.action_space # Discrete(3) starting from 1
    

    def step(self, action):
        # self._sim._prev_sim_obs record previous timestep observation
        observations = self._sim.step(action)
        reward = self.get_reward(observations)
        done = self.get_done(observations)
        info = self.get_info(observations)

        # increment single timestep
        self._elapsed_steps += 1

        return observations, reward, done, info

    def reset(self):
        # cannot use parent reset: self._env.reset()
        # return: init_obs

        # change config node
        self._config.defrost()

        if self._config.ENVIRONMENT.SCENE == "none":
            # random select a scene
            scene_str = self.random_scene() # default training scenes
            # print("initializing scene {}".format(scene_str))
            self._config.SIMULATOR.SCENE = master_scene_dir + scene_str + '.glb'
        else:
            self._config.SIMULATOR.SCENE = master_scene_dir + self._config.ENVIORNMENT.SCENE + '.glb'

        self._config.freeze()

        if self._sim is None:
            # construct habitat simulator
            self._sim = make_sim(
                id_sim=self._config.SIMULATOR.TYPE, config=self._config.SIMULATOR
            )
        else:
            self._sim.reconfigure(self._config.SIMULATOR)
        
        # initialize scene simulator
        observations = self._sim.reset()

        # initialize the agent at a random start state
        self._agent_state = self._sim.get_agent_state()
        self._agent_state.position = self._sim.sample_navigable_point() # location has dimension 3
        random_rotation = np.random.rand(4) # rotation has dimension 4
        random_rotation[1] = 0.0
        random_rotation[3] = 0.0
        self._agent_state.rotation = quaternion.as_quat_array(random_rotation)
        self._sim.set_agent_state(self._agent_state.position, self._agent_state.rotation)
        observations = self._sim.get_observations_at(self._agent_state.position, self._agent_state.rotation, True)
        color_density = 1 - np.count_nonzero(observations['rgb']) / np.prod(observations['rgb'].shape)
        valid_start = color_density < 0.05

        while not valid_start:
            # initialize the agent at a random start state
            self._agent_state = self._sim.get_agent_state()
            self._agent_state.position = self._sim.sample_navigable_point() # location has dimension 3
            random_rotation = np.random.rand(4) # rotation has dimension 4
            random_rotation[1] = 0.0
            random_rotation[3] = 0.0
            self._agent_state.rotation = quaternion.as_quat_array(random_rotation)
            self._sim.set_agent_state(self._agent_state.position, self._agent_state.rotation)
            observations = self._sim.get_observations_at(self._agent_state.position, self._agent_state.rotation, True)
            color_density = 1 - np.count_nonzero(observations['rgb']) / np.prod(observations['rgb'].shape)
            valid_start = color_density < 0.05

        # reset count
        self._elapsed_steps = 0

        return observations
    

    def reset_scene(self, scene_str, seed=None):
        # change config node
        self._config.defrost()

        # set specific random seed
        self._config.SIMULATOR.SEED = seed
        random.seed(self._config.SIMULATOR.SEED)

        # set specific scene
        self._config.SIMULATOR.SCENE = master_scene_dir + scene_str + '.glb'

        self._config.freeze()

        if self._sim is None:
            # construct habitat simulator
            self._sim = make_sim(
                id_sim=self._config.SIMULATOR.TYPE, config=self._config.SIMULATOR
            )
        else:
            self._sim.reconfigure(self._config.SIMULATOR)
        
        # initialize scene simulator
        observations = self._sim.reset()

        # initialize the agent at a random start state
        self._agent_state = self._sim.get_agent_state()
        self._agent_state.position = self._sim.sample_navigable_point() # location has dimension 3
        self._sim.set_agent_state(self._agent_state.position, self._agent_state.rotation)

        # reset count
        self._elapsed_steps = 0

        return observations

    # not used actually, since we have intrinsic reward
    def get_reward_range(self):
        return (-1.0, 1.0)
    
    # no task reward here
    def get_reward(self, observations):
        return 0.0
    
    def get_done(self, observations):
        return self._elapsed_steps + 1 >= self._max_episode_steps

    def get_info(self, observations):
        agent_pos = self._sim.get_agent_state().position
        return {"agent_position": agent_pos}
    
    def close(self):
        self._sim.close()
    
    def random_scene(self, mode='train'):
        return random.choice(scenes[mode])
