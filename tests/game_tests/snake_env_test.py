import pytest
from game.snake_env import VectorizedSnakeEnv

@pytest.fixture
def env():
    return VectorizedSnakeEnv()

def test_env_reset(env):
    state = env.get_state()
    assert state is not None