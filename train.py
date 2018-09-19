# Adapted from
# https://github.com/simonmeister/pysc2-rl-agents/blob/master/run.py

import os
import shutil
import sys
import argparse
from functools import partial

from sc2_dl.a2c.a2cRunner import A2CRunner
from sc2_dl.agents.a2c_agent import A2CAgent
from sc2_dl.enviros.sc2_vec_env import SubprocVecEnv
from sc2_dl.models.fully_conv_model import FullyConvModel
from sc2_dl.models.control_agent_model import ControlAgentModel
from pysc2.env.sc2_env import *
from keras.optimizers import RMSprop, Adam, TFOptimizer
import tensorflow as tf
import numpy as np

# Workaround for pysc2 flags
from absl import flags
FLAGS = flags.FLAGS
FLAGS(['train.py'])


parser = argparse.ArgumentParser(description='Starcraft 2 deep RL agents')
parser.add_argument('experiment_id', type=str, help='identifier to store experiment results')
parser.add_argument('--eval', action='store_true', help='if false, episode scores are evaluated')
parser.add_argument('--ow', action='store_true', help='overwrite existing experiments (if --train=True)')
parser.add_argument('--map', type=str, default='MoveToBeacon', help='name of SC2 map')
parser.add_argument('--vis', action='store_true', help='render with pygame')
parser.add_argument('--max_windows', type=int, default=1, help='maximum number of visualization windows to open')
parser.add_argument('--res', type=int, default=32, help='screen and minimap resolution')
parser.add_argument('--envs', type=int, default=16, help='number of environments simulated in parallel')
parser.add_argument('--step_mul', type=int, default=8, help='number of game steps per agent step')
parser.add_argument('--steps_per_batch', type=int, default=80, help='number of agent steps when collecting trajectories for a single batch')
parser.add_argument('--discount', type=float, default=0.99, help='discount for future rewards')
parser.add_argument('--iters', type=int, default=-1, help='number of iterations to run (-1 to run forever)')
parser.add_argument('--start_point', type=int, default=0, help='iteration at which to (re)start and load weights')
parser.add_argument('--seed', type=int, default=123456, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='gpu device id')
parser.add_argument('--nhwc', action='store_true', help='train/test in NHWC mode. Default to NCHW (recommended for GPU training)')
parser.add_argument('--summary_iters', type=int, default=50, help='record training summary after this many iterations')
parser.add_argument('--save_iters', type=int, default=1000, help='store checkpoint after this many iterations')
parser.add_argument('--max_to_keep', type=int, default=5, help='maximum number of checkpoints to keep before discarding older ones')
parser.add_argument('--entropy_weight', type=float, default=1e-1, help='weight of entropy loss')
parser.add_argument('--value_loss_weight', type=float, default=0.1, help='weight of value function loss')
parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
parser.add_argument('--grad_norm', type=float, default=100.0, help='global gradient norm for clipping')
parser.add_argument('--decay', type=float, default=1e-12, help='Linear decay of learning rate (and entropy weight) parameter. Should be between 0 and 1')
parser.add_argument('--save_dir', type=str, default=os.path.join('out', 'models'), help='root directory for checkpoint storage')
parser.add_argument('--summary_dir', type=str, default=os.path.join('out', 'summary'), help='root directory for summary storage')
parser.add_argument('--use_max', action='store_true', help='Always choose action with max probability')
parser.add_argument('--use_lstm', action='store_true', help='Train an LSTM model')
parser.add_argument('--adam', action='store_true', help='Optimize loss with Adam (default: RMSProp)')

args = parser.parse_args()
# TODO write args to config file and store together with summaries (https://pypi.python.org/pypi/ConfigArgParse)
args.train = not args.eval

#Don't do that on compute canada
#os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


ckpt_path = os.path.join(args.save_dir, args.experiment_id)
summary_type = 'train' if args.train else 'eval'
summary_path = os.path.join(args.summary_dir, args.experiment_id, summary_type)

np.random.seed(args.seed)


def _save_if_training(agent, num_iters):
  if args.train:
    agent.save(ckpt_path, num_iters)
    agent.writer.flush()
    sys.stdout.flush()


def make_sc2env(**kwargs):
    env = SC2Env(**kwargs)
    return env


def main():
    if args.train and args.ow:
        shutil.rmtree(ckpt_path, ignore_errors=True)
        shutil.rmtree(summary_path, ignore_errors=True)

    size_px = args.res
    env_args = dict(
        players=[Agent(Race.terran)],
        map_name=args.map,
        step_mul=args.step_mul,
        game_steps_per_episode=0,
        score_index=0,
        agent_interface_format=parse_agent_interface_format(
            feature_screen=size_px,
            feature_minimap=size_px))
    vis_env_args = env_args.copy()
    vis_env_args['visualize'] = args.vis
    num_vis = min(args.envs, args.max_windows)
    env_fns = [partial(make_sc2env, **vis_env_args)] * num_vis
    num_no_vis = args.envs - num_vis
    if num_no_vis > 0:
        env_fns.extend([partial(make_sc2env, **env_args)] * num_no_vis)

    agent = A2CAgent(
        model_file=args.experiment_id
        )
    envs = SubprocVecEnv(env_fns=env_fns, agent=agent)

    summary_writer = tf.summary.FileWriter(summary_path)
    if args.use_lstm:
        model = ControlAgentModel(
            value_loss_coeff=args.value_loss_weight,
            entropy_coeff=args.entropy_weight,
            learning_rate=args.lr,
            batch_size=args.envs,
            summary_writer=summary_writer,
            seq_length=args.steps_per_batch,
            start_point=args.start_point,
            data_format='channels_last' if args.nhwc else 'channels_first',
            decay=args.decay
        )
    else:
        model = FullyConvModel(
            value_loss_coeff=args.value_loss_weight,
            entropy_coeff=args.entropy_weight,
            learning_rate=args.lr,
            batch_size=args.envs,
            summary_writer=summary_writer,
            start_point=args.start_point,
            data_format='channels_last' if args.nhwc else 'channels_first',
            decay=args.decay
        )

    runner = A2CRunner(
        envs=envs,
        model=model,
        agent=agent,
        train=args.train,
        discount=args.discount,
        summary_writer=summary_writer,
        temporal=args.use_lstm,
        n_steps=args.steps_per_batch)

    lr = args.lr
    decay = args.decay
    if args.adam:
        optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8, clipnorm=args.grad_norm, decay=decay)
        #optimizer = {'class_name': 'adam',
        #             'config': {'lr': lr,
        #                        'beta_1': 0.9,
        #                        'beta_2': 0.999,
        #                        'epsilon': 1e-8,
        #                        'decay': decay,
        #                        'clipnorm': args.grad_norm}}
        #optimizer = TFOptimizer(tf.train.AdamOptimizer(learning_rate=lr))
    else:
        #optimizer = TFOptimizer(tf.train.RMSPropOptimizer(learning_rate=lr, decay=0.99, epsilon=1e-5))
        #optimizer = {'class_name': 'rmsprop',
        #             'config': {'lr': lr,
        #                        'rho': 0.99,
        #                        'epsilon': 1e-5,
        #                        'decay': decay,
        #                        'clipnorm': args.grad_norm}}
        optimizer = RMSprop(lr=lr, rho=0.99, epsilon=1e-5, decay=decay, clipnorm=args.grad_norm)
    model.init_model(envs.observation_space, len(envs.action_space.functions), opt=optimizer, graph_path=summary_path)

    # Loads only the weights for now, hence the prior initialization of the model
    if os.path.exists(ckpt_path):
        model.load(ckpt_path, args.start_point)

    runner.reset()

    i = args.start_point
    try:
        while True:
            write_summary = args.train and i % args.summary_iters == 0

            if i > 0 and i % args.save_iters == 0:
                _save_if_training(model, i)

            assert model.model is not None
            loss = runner.run_batch(write_summary, i)
            if write_summary:
                print('iter %d: loss = %f' % (i, loss), flush=True)
                summary_writer.flush()

            i += 1

            if 0 <= args.iters <= i - args.start_point:
                break

    except KeyboardInterrupt:
        pass

    _save_if_training(model, i)

    envs.close()
    summary_writer.close()

    print('mean score: %f' % runner.get_mean_score(), flush=True)


if __name__ == "__main__":
    main()
