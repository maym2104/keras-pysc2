# Adapted from
# https://github.com/pekaalto/sc2aibot/blob/master/common/multienv.py

from multiprocessing import Process, Pipe
from baselines.common.vec_env import VecEnv, CloudpickleWrapper
from pysc2.lib import protocol

# below (worker, SubprocVecEnv) copied and adapted from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
# with some sc2 specific modifications
def worker(remote, parent_remote, env_fn_wrapper, agent):
    parent_remote.close()
    env = env_fn_wrapper.x()

    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            try:
                timesteps = env.step(data)
            except protocol.ConnectionError:
                env = env_fn_wrapper.x()
                timesteps = env.reset()
            remote.send(timesteps)
        elif cmd == 'reset':
            try:
                timesteps = env.reset()
            except protocol.ConnectionError:
                env = env_fn_wrapper.x()
                timesteps = env.reset()
            agent.reset()
            remote.send(timesteps)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'setup':
            action_space = env.action_spec()
            observation_space = env.observation_spec()
            # assume the "agent" is first index
            agent.setup(observation_space[0], action_space[0])
            remote.send((observation_space[0], action_space[0]))
        elif cmd == 'step_agent':
            obs, *policy = data
            action = agent.step(obs, policy)
            remote.send(action)
        else:
            print(cmd)
            raise NotImplementedError


class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns, agent):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn), agent))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        for r in self.remotes:
            r.send(('setup', None))

        results = [r.recv() for r in self.remotes]
        obs_space, action_space = zip(*results)

        super().__init__(len(env_fns), obs_space[0], action_space[0])
        self.num_actions = len(action_space[0][1])

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            #in multiplayer mode, send list of actions taken by all agents at this step
            #assume single player for now
            remote.send(('step', [action]))
        self.waiting = True

    def step_wait(self):
        timesteps = [remote.recv() for remote in self.remotes]
        self.waiting = False
        return timesteps

    def step_or_reset(self, actions, dones):
        for remote, action, done in zip(self.remotes, actions, dones):
            #in multiplayer mode, send list of actions taken by all agents at this step
            #assume 2 players for now
            cmd = "reset" if done else "step"
            remote.send((cmd, (action, )))

        timesteps = [remote.recv() for remote in self.remotes]
        return timesteps

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return [remote.recv() for remote in self.remotes]

    def step_agent(self, observations, policies):
        for remote, obs, *policy in zip(self.remotes, observations, *policies):
            remote.send(('step_agent', [obs] + policy))
        actions = [remote.recv() for remote in self.remotes]

        return list(actions)

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:            
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True
