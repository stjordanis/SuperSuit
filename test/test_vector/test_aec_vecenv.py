from supersuit.vector import VectorAECWrapper,ProcVectorEnv
from pettingzoo.classic import rps_v0
from pettingzoo.classic import mahjong_v0
from pettingzoo.gamma import knights_archers_zombies_v0
from pettingzoo.mpe import simple_world_comm_v0
from pettingzoo.classic import chess_v0
from pettingzoo.sisl import multiwalker_v0
from pettingzoo.atari import warlords_v0
import numpy as np
import pytest
import random
import time


NUM_ENVS = 5
NUM_CPUS = 2
wrappers = [
    (VectorAECWrapper([rps_v0.env]*NUM_ENVS)),
    (VectorAECWrapper([lambda :mahjong_v0.env(seed=10+i) for i in range(NUM_ENVS)])),
    (VectorAECWrapper([multiwalker_v0.env]*NUM_ENVS)),
    (VectorAECWrapper([simple_world_comm_v0.env]*NUM_ENVS))
]
@pytest.mark.parametrize("vec_env", wrappers)
def test_vec_env(vec_env):
    obs,passes = vec_env.reset()
    print(np.asarray(obs).shape)
    assert len(obs) == NUM_ENVS
    act_space = vec_env.action_spaces[vec_env.agent_selection]
    print(act_space)
    obs,passes,envs_done = vec_env.step([act_space.sample() for _ in range(NUM_ENVS)])
    assert len(obs) == NUM_ENVS
    assert len(vec_env.observe(vec_env.agent_selection)) == NUM_ENVS
    obs,passes = vec_env.reset(False)
    assert obs is None
    vec_env.last()

def test_infos():
    vec_env = VectorAECWrapper([mahjong_v0.env]*NUM_ENVS)
    obs,passes = vec_env.reset()
    infos = vec_env.infos[vec_env.agent_selection]
    assert infos[1]['legal_moves']

def test_some_done():
    vec_env = VectorAECWrapper([mahjong_v0.env]*NUM_ENVS)
    obs,passes = vec_env.reset()
    act_space = vec_env.action_spaces[vec_env.agent_selection]
    assert not any(done for dones in vec_env.dones.values() for done in dones)
    obs,passes,envs_done = vec_env.step([act_space.sample() for _ in range(NUM_ENVS)])
    assert any(done for dones in vec_env.dones.values() for done in dones)
    assert any(rew != 0 for rews in vec_env.rewards.values() for rew in rews)

def select_action(vec_env,passes,i):
    my_info = vec_env.infos[vec_env.agent_selection][i]
    if not passes[i] and 'legal_moves' in my_info:
        return random.choice(my_info['legal_moves'])
    else:
        act_space = vec_env.action_spaces[vec_env.agent_selection]
        return act_space.sample()
