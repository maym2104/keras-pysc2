# Adapted from
# https://github.com/simonmeister/pysc2-rl-agents/blob/master/rl/agents/a2c/runner.py

import numpy as np

from pysc2.lib.actions import FUNCTIONS, FunctionCall
from pysc2.lib.actions import TYPES as ACTION_TYPES
from keras.utils import to_categorical
import tensorflow as tf
import matplotlib.pyplot as plt

class A2CRunner:
    def __init__(self,
                 model,
                 envs,
                 agent,
                 summary_writer=None,
                 train=True,
                 n_steps=16,
                 temporal=False,
                 discount=0.99):
        """
    Args:
      agent: A2CAgent instance.
      envs: SubprocVecEnv instance.
      summary_writer: summary writer to log episode scores.
      train: whether to train the agent.
      n_steps: number of agent steps for collecting rollouts.
      discount: future reward discount.
    """
        self.model = model
        self.envs = envs
        self.agent = agent
        self.summary_writer = summary_writer
        self.train = train
        self.n_steps = n_steps
        self.discount = discount
        self.episode_counter = 0
        self.cumulative_score = 0.0
        self.temporal = temporal

    def reset(self):
        obs_raw = self.envs.reset()
        self.last_obs = [ob[0] for ob in obs_raw]
        self.reset_last_action()
        self.state = None

    def reset_last_action(self):
        pi = np.zeros((self.envs.num_envs, 1, len(FUNCTIONS)))

        args = []
        spatial_types = [ACTION_TYPES.minimap, ACTION_TYPES.screen, ACTION_TYPES.screen2]
        size = self.envs.observation_space.feature_screen
        height, width = size[1:3]
        flatten_size = width * height
        for arg_type in ACTION_TYPES:
            num_classes = flatten_size if arg_type in spatial_types else arg_type.sizes[0]
            arg = np.zeros((self.envs.num_envs, 1, num_classes))
            args.append(arg)

        self.last_action = [pi] + args

    def get_mean_score(self):
        return self.cumulative_score / self.episode_counter

    def _summarize_episode(self, scores, step):
        mean_score = np.asscalar(np.mean(scores))

        print("episode %d: score = %f" % (self.episode_counter, mean_score), flush=True)
        score_summary = tf.Summary(value=[tf.Summary.Value(tag="score", simple_value=mean_score), ])
        self.summary_writer.add_summary(score_summary, global_step=step)

        self.episode_counter += len(scores)
        return mean_score

    def run_batch(self, write_summary=False, step=None):
        """Collect trajectories for a single batch and train (if self.train).

        Args:
          train_summary: return a Summary of the training step (losses, etc.).

        Returns:
          result: None (if not self.train) or the return value of agent.train.
        """
        shapes = (self.n_steps, self.envs.num_envs)
        values = np.zeros(shapes, dtype=np.float32)
        rewards = np.zeros(shapes, dtype=np.float32)
        dones = np.zeros(shapes, dtype=np.float32)
        all_scores = []
        all_samples = []

        obs_raw = self.last_obs
        all_actions = last_action = self.last_action
        all_obs = last_obs = self.transform_obs(obs_raw, last_action)  # self.last_obs
        episode_over = False
        last_state = None

        for n in range(self.n_steps):
            # all_scores = all_outcomes = []
            for t in obs_raw:
                if t.last():
                    score = t.observation["score_cumulative"][0]
                    self.cumulative_score += score
                    all_scores.append(score)
                    episode_over = True

            if self.temporal:
                pi_samples, value_estimates, *args = self.model.predict(last_obs)
                last_state = args[13:]
            else:
                pi_samples, value_estimates, *args = self.model.predict([np.squeeze(ob, axis=1) for ob in last_obs])
                last_state = None
            args = args[:13]
            last_action = self.transform_actions([pi_samples] + args)
            all_samples.append([pi_samples] + args)
            all_actions = [np.concatenate([old_act, new_act], axis=1) for old_act, new_act in zip(all_actions, last_action)]

            actions = self.envs.step_agent(obs_raw, [pi_samples] + args)

            values[n, :] = value_estimates[:, 0]

            obs_raw = self.envs.step_or_reset(actions, dones[n-1, :] > 0.)
            obs_raw = [ob[0] for ob in obs_raw]
            last_obs = self.transform_obs(obs_raw, last_action)
            all_obs = [np.concatenate([old_obs, new_obs], axis=1) for old_obs, new_obs in zip(all_obs, last_obs)]

            rewards[n, :] = [t.reward for t in obs_raw]
            dones[n, :] = [t.last() for t in obs_raw]

            if episode_over:
                self._summarize_episode(all_scores, step)
                break

        self.last_obs = obs_raw

        self.last_action = last_action
        if episode_over:
            self.reset_last_action()
            last_state = None

        if self.temporal:
            predictions = self.model.predict(last_obs)
        else:
            predictions = self.model.predict([np.squeeze(ob, axis=1) for ob in last_obs])
        next_values = predictions[1][:, 0]
        rewards = rewards[:, :n+1]
        values = values[:, :n+1]

        returns, advs = compute_returns_advantages(rewards, dones, values, next_values, self.discount)

        actions = [np.stack(act, axis=1) for act in zip(*all_samples)]  # [l_a[:, 1:] for l_a in all_actions]
        obs = [ob[:, :n+1] for ob in all_obs]
        returns = np.expand_dims(np.transpose(returns), axis=-1)[:, :n+1, :]
        advs = np.expand_dims(np.transpose(advs)[:, :n+1], axis=-1)
        rewards = np.transpose(rewards)

        if not self.temporal:
            flatten_batch_shape = (self.envs.num_envs * (n+1), )
            actions = [np.reshape(act, flatten_batch_shape+act.shape[2:]) for act in actions]
            obs = [np.reshape(ob, flatten_batch_shape+ob.shape[2:]) for ob in obs]
            returns = np.reshape(returns, flatten_batch_shape + returns.shape[2:])
            advs = np.reshape(advs, flatten_batch_shape + advs.shape[2:])
            rewards = np.reshape(rewards, flatten_batch_shape + rewards.shape[2:])

        if self.train:
            loss = self.model.train_reinforcement(obs, actions, returns, advs, write_summary, step, rewards, episode_over, self.state)
            self.state = last_state if last_state is not None or episode_over else self.state

            return loss

        return None,

    def transform_obs(self, obs, actions):
        # obs_distances = np.array([[self.distance_computer.get_distances(ob) for ob in obs_batch] for obs_batch in obs])
        # obs_costs = np.array([[timestep.costs for timestep in obs_batch] for obs_batch in obs])
        obs = [timestep.observation for timestep in obs]  # flatten_first_dims(all_obs)
        obs_screen = np.array([ob['feature_screen'] for ob in obs])
        obs_screen = np.transpose(obs_screen, axes=[0, 2, 3, 1])    # NHWC
        obs_screen = np.expand_dims(obs_screen, axis=1)
        #  TODO consider all observations. Write something more general
        obs_minimap = np.array([ob['feature_minimap'] for ob in obs])
        obs_minimap = np.transpose(obs_minimap, axes=[0, 2, 3, 1])  # NHWC
        obs_minimap = np.expand_dims(obs_minimap, axis=1)

        screen_last_pos = actions[1]
        screen_last_pos = screen_last_pos.reshape((obs_screen.shape[0:4] + (1,)))
        minimap_last_pos = actions[2]
        minimap_last_pos = minimap_last_pos.reshape((obs_minimap.shape[0:4] + (1,)))
        screen2_last_pos = actions[3]
        screen2_last_pos = screen2_last_pos.reshape((obs_screen.shape[0:4] + (1,)))

        obs_minimap = np.concatenate((obs_minimap, minimap_last_pos), axis=-1)  # concatenate last_actions minimap args
        obs_screen = np.concatenate((obs_screen, screen_last_pos, screen2_last_pos), axis=-1)

        obs_available_actions = np.zeros((obs_minimap.shape[0], len(FUNCTIONS),))
        for i, ob in enumerate(obs):
            obs_available_actions[i, ob['available_actions']] = 1.
        obs_available_actions = np.expand_dims(obs_available_actions, axis=1)

        non_spatial_last_actions = [actions[0]] + actions[4:]
        non_spatial_last_actions = np.concatenate(non_spatial_last_actions, axis=2)

        obs_nonspatial = np.array([ob["player"] for ob in obs])
        obs_nonspatial = np.expand_dims(obs_nonspatial, axis=1)

        inputs = [obs_screen, obs_minimap, obs_nonspatial, non_spatial_last_actions, obs_available_actions]

        return inputs

    def transform_actions(self, actions):
        pi, *args = actions
        pi_sample = to_categorical(pi, len(FUNCTIONS))
        pi_sample = np.expand_dims(pi_sample, axis=1)

        arg_samples = []
        spatial_types = [ACTION_TYPES.minimap, ACTION_TYPES.screen, ACTION_TYPES.screen2]
        size = self.envs.observation_space.feature_screen
        height, width = size[1:3]
        flatten_size = width * height
        for arg_type in ACTION_TYPES:
            num_classes = flatten_size if arg_type in spatial_types else arg_type.sizes[0]
            arg = args[arg_type.id]
            mask = np.array([arg_type in FUNCTIONS._func_list[int(sampled_action)].args for sampled_action in pi.flatten().tolist()])
            mask = mask.reshape(arg.shape + (1, ))
            mask = mask.repeat(num_classes, axis=-1)
            arg_sample = to_categorical(arg, num_classes) * mask
            arg_sample = np.expand_dims(arg_sample, axis=1)
            arg_samples.append(arg_sample)

        new_actions = [pi_sample] + arg_samples

        return new_actions


def compute_returns_advantages(rewards, dones, values, next_values, discount):
    """Compute returns and advantages from received rewards and value estimates.

  Args:
    rewards: array of shape [n_steps, n_env] containing received rewards.
    dones: array of shape [n_steps, n_env] indicating whether an episode is
      finished after a time step.
    values: array of shape [n_steps, n_env] containing estimated values.
    next_values: array of shape [n_env] containing estimated values after the
      last step for each environment.
    discount: scalar discount for future rewards.

  Returns:
    returns: array of shape [n_steps, n_env]
    advs: array of shape [n_steps, n_env]
  """
    returns = np.zeros([rewards.shape[0] + 1, rewards.shape[1]])

    returns[-1, :] = next_values
    for t in reversed(range(rewards.shape[0])):
        future_rewards = discount * returns[t + 1, :] * (1 - dones[t, :])
        returns[t, :] = rewards[t, :] + future_rewards

    returns = returns[:-1, :]
    advs = returns - values

    return returns, advs


def flatten_first_dims(x):
    new_shape = [x.shape[0] * x.shape[1]] + list(x.shape[2:])
    return x.reshape(*new_shape)


def show(args):
    screen_arg = args[0][0].reshape((32, 32))
    minimap_arg = args[1][0].reshape((32, 32))
    screen2_arg = args[2][0].reshape((32, 32))
    zeros = np.zeros((32, 32))

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(screen_arg)
    plt.title("screen_arg")

    plt.subplot(2, 2, 2)
    plt.imshow(minimap_arg)
    plt.title("minimap_arg")

    plt.subplot(2, 2, 3)
    plt.imshow(screen2_arg)
    plt.title("screen2_arg")

    plt.subplot(2, 2, 4)
    plt.imshow(zeros)
    plt.title('zeros ref')

    plt.show()
