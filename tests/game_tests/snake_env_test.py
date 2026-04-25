import pytest
from game.snake_env import Snake_Env

@pytest.fixture
def env():
    return Snake_Env()

def test_env_reset(env):
    state = env.get_state()
    assert state is not None