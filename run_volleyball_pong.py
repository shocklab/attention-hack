from pettingzoo.atari import volleyball_pong_v2
import supersuit

# Create 2 player env
env = volleyball_pong_v2.parallel_env(obs_type='grayscale_image', num_players=2)

def preprocess(env):
    # Preprocessing
    # as per openai baseline's MaxAndSKip wrapper, maxes over the last 2 frames
    # to deal with frame flickering
    env = supersuit.max_observation_v0(env, 2)

    # repeat_action_probability is set to 0.25 to introduce non-determinism to the system
    env = supersuit.sticky_actions_v0(env, repeat_action_probability=0.25)

    # skip frames for faster processing and less control
    # to be compatible with gym, use frame_skip(env, (2,5))
    env = supersuit.frame_skip_v0(env, 4)

    # downscale observation for faster processing
    env = supersuit.resize_v0(env, 84, 84)

    # allow agent to see everything on the screen despite Atari's flickering screen problem
    env = supersuit.frame_stack_v1(env, 4)

    return env


def policy(obs, agent):
    return 0

replay_buffer_1 = 

observations = env.reset()

for step in range(100):
    actions = {agent: policy(observations[agent], agent) for agent in env.agents}
    observations, rewards, dones, infos = env.step(actions)

    print(list(observations.values())[0].shape)