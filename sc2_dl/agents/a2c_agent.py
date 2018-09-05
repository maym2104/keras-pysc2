from pysc2.agents import base_agent
from pysc2.lib import actions, features

from keras.backend import *

from absl import logging


class A2CAgent(base_agent.BaseAgent):
    """An agent that uses a model to play the game."""
    def __init__(self, score_index=0, model_path="out\models", model_file="lstm"):
        """
        :param policy: The policy with which the model was built.
        :param model_file: The name of the model file to load the model form.
        :param num_procs: The number of processors.
        :param score_index: The score type, must be >= 0 with 0 being curriculum score.
        :param model_path: The path of the folder containing the model file.
        """
        super().__init__()

        self.load_path = os.path.join(model_path, model_file)

        self.scores = []
        self.score_index = score_index

        self.num_actions = len(actions.FUNCTIONS)

    def setup(self, obs_spec, action_spec):
        """
        :param obs_spec: The observation space.
        :param action_spec: The action space.
        """
        super(A2CAgent, self).setup(obs_spec, action_spec)

        self.size = obs_spec['feature_minimap'][1:3]
        self.available_actions = np.zeros((len(action_spec[1]),))

        logging.info("Agent ready")

    def step(self, timestep, samples=None):
        """
        Uses the current observations as input for the model to get the best action to take.
        :param timestep: A TimeStep that contains the current observation.
        :return: The actions (FunctionCall) to take.
        """
        # TODO: use cumulative score

        super(A2CAgent, self).step(timestep)

        if samples is not None:
            fn_sample, *arg_samples = samples
            act = self.actions_to_pysc2(fn_sample, arg_samples)
        else:
            act = actions.FunctionCall(0, [])

        logging.debug('Action %s', act)

        return act


    def actions_to_pysc2(self, a_0, arg_ids):
        """Convert agent action representation to FunctionCall representation."""
        height, width = self.size
        a_l = []
        for arg_type in actions.FUNCTIONS._func_list[a_0].args:
            arg_id = arg_ids[arg_type.id]
            if arg_type in [actions.TYPES.minimap, actions.TYPES.screen, actions.TYPES.screen2]:
                arg = [arg_id % width, arg_id // height]
            else:
                arg = [arg_id]
            a_l.append(arg)
        action = actions.FunctionCall(a_0, a_l)

        return action

