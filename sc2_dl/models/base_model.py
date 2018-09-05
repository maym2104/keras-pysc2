import os
import numpy as np

from collections import namedtuple

from keras.layers import *
from keras.models import Sequential, Model
import keras.backend as k
from keras.losses import categorical_crossentropy, mse
from tensorflow import split, to_int32
import tensorflow as tf

from pysc2.lib import features, actions


FlatFeature = namedtuple('FlatFeatures', ['index', 'type', 'scale', 'name'])
FLAT_FEATURES = [
  FlatFeature(0,features.FeatureType.CATEGORICAL, 17, 'player_id'),
  FlatFeature(1,  features.FeatureType.SCALAR, 10000, 'minerals'),
  FlatFeature(2,  features.FeatureType.SCALAR, 10000, 'vespene'),
  FlatFeature(3,  features.FeatureType.SCALAR, 200,   'food_used'),
  FlatFeature(4,  features.FeatureType.SCALAR, 200,   'food_cap'),
  FlatFeature(5,  features.FeatureType.SCALAR, 200,   'food_army'),
  FlatFeature(6,  features.FeatureType.SCALAR, 200,   'food_workers'),
  FlatFeature(7,  features.FeatureType.SCALAR, 100,   'idle_worker_count'),
  FlatFeature(8,  features.FeatureType.SCALAR, 1000,  'army_count'),
  FlatFeature(9,  features.FeatureType.SCALAR, 1000,  'warp_gate_count'),
  FlatFeature(10, features.FeatureType.SCALAR, 1000,  'larva_count'),
]

FLAT_ACTION_TYPES = [t for t in actions.TYPES if t not in [actions.TYPES.minimap,
                                                           actions.TYPES.screen,
                                                           actions.TYPES.screen2]]


class BaseModel:
    def __init__(self, summary_writer=tf.summary.FileWriter('./train'), start_point=0, seed=123456,
                 value_loss_coeff=0.1, entropy_coeff=1e-1, batch_size=16, learning_rate=1e-4,
                 data_format='channels_first'):
        self.model = None
        self.target_model = None
        self.value_loss_coeff = value_loss_coeff
        self.batch_size = batch_size
        self.entropy_coeff = entropy_coeff
        self.learning_rate = learning_rate
        self.iterations = start_point
        self.seed = seed
        self.data_format = data_format
        self.channel_axis = -1 if self.data_format == 'channels_last' else -3

        # summary
        self.writer = summary_writer

        ###################################
        # TensorFlow wizardry
        config = tf.ConfigProto()

        # Don't pre-allocate memory; allocate as-needed
        config.gpu_options.allow_growth = True

        # Only allow a total of half the GPU memory to be allocated
        # config.gpu_options.per_process_gpu_memory_fraction = 0.5

        # Create a session with the above options specified.
        k.set_session(tf.Session(config=config))
        k.set_image_data_format(self.data_format)
        k.set_epsilon(1e-12)
        ###################################

    def residual_block(self, input_shape, num_layers=2):
        assert num_layers >= 2

        x = Input(batch_shape=(self.batch_size,) + input_shape)
        y = Conv2D(input_shape[self.channel_axis], (3, 3), padding='same', activation='relu',
                   data_format=self.data_format)(x)

        for _ in range(1, num_layers-1):
            y = Conv2D(input_shape[self.channel_axis], (3, 3), padding='same', activation='relu',
                       data_format=self.data_format)(y)

        y = Conv2D(input_shape[self.channel_axis], (3, 3), padding='same', data_format=self.data_format)(y)
        y = Add()([x, y])
        y = Activation('relu')(y)

        model = Model(inputs=[x], outputs=[y])

        return model

    def two_layer_mlp(self, num_units_1, num_units_2, input_shape):
        model = Sequential()
        model.add(Dense(num_units_1, activation='relu', input_shape=input_shape, batch_size=self.batch_size))
        model.add(Dense(num_units_2))

        return model

    # This method transforms the categorical features in one-hot tensors on the channel dim
    # and log the scalar values to avoid too large values
    def preprocess_spatial_obs(self, feature_types, input_shape, embed_size=10):
        x = Input(batch_shape=(self.batch_size, ) + input_shape)
        layers = []

        def cat_case(feat_type):
            model = Sequential(name=s.name + '_preprocess_model')
            model.add(Lambda(lambda x: k.one_hot(k.cast(x[:, :, :, feat_type.index], dtype='int32'), feat_type.scale),
                             name='preprocess_'+feat_type.name+'_to_one_hot', input_shape=input_shape,
                             batch_size=self.batch_size))
            if self.data_format == 'channels_first':
                model.add(Permute((3, 1, 2), name='preprocess_' + feat_type.name + '_to_NCHW'))
            model.add(Conv2D(embed_size, 1, padding='same', activation='relu',
                             name='preprocess_'+feat_type.name+'_conv', data_format=self.data_format))

            return model

        def scalar_case(feat_type):
            model = Sequential(name=feat_type.name + '_preprocess_model')
            model.add(Lambda(lambda x: k.log(x[:, :, :, feat_type.index:feat_type.index+1]+1.0), name='preprocess_'+feat_type.name+'_log',
                             input_shape=input_shape, batch_size=self.batch_size))
            if self.data_format == 'channels_first':
                model.add(Permute((3, 1, 2), name='preprocess_' + feat_type.name + '_to_NCHW'))

            return model

        for s in feature_types:
            if s.type == features.FeatureType.CATEGORICAL:
                layer = cat_case(s)(x)
            elif s.type == features.FeatureType.SCALAR:
                layer = scalar_case(s)(x)
            else:
                raise NotImplementedError
            layers.append(layer)

        last_actions = Lambda(lambda x: x[:, :, :, len(feature_types):])(x)
        if self.data_format == 'channels_first':
            last_actions = Permute((3, 1, 2), name='preprocess_last_acts_to_NCHW')(last_actions)
        layers.append(last_actions)

        y = Concatenate(axis=self.channel_axis, name='preprocess_spatial_concat')(layers)

        return Model(inputs=[x], outputs=[y], name='preprocess_spatial_model')

    # TODO instead of resulting of a NHWC tensor, its shape should be NHW1 where every channel is merged into a single one
    def preprocess_nonspatial_obs(self, feature_types, input_shape, embed_size=10):
        x = Input(batch_shape=(self.batch_size,) + input_shape)
        layers = []

        def cat_case(feat_type):
            model = Sequential(name=feat_type.name + '_preprocess_model')
            model.add(Lambda(lambda x: k.one_hot(k.cast(x[:, feat_type.index], dtype='int32'), feat_type.scale),
                             name='preprocess_'+feat_type.name+'_to_one_hot', input_shape=input_shape,
                             batch_size=self.batch_size))
            model.add(Dense(embed_size, activation='relu', name='preprocess_'+feat_type.name+'_embed'))

            return model

        def actions_model(start, scale):
            model = Sequential()
            j = start
            model.add(Lambda(lambda x: x[:, j:j+scale], input_shape=input_shape, batch_size=self.batch_size))
            model.add(Dense(embed_size, activation='relu'))

            return model

        for s in feature_types:
            if s.type == features.FeatureType.CATEGORICAL:
                layer = cat_case(s)(x)
            elif s.type == features.FeatureType.SCALAR:
                layer = Lambda(lambda x: k.log(x[:, s.index:s.index+1]+1.0), name='preprocess_'+s.name+'_log')(x)
            else:
                raise NotImplementedError
            layers.append(layer)

        i = len(feature_types)
        # available_actions
        avail_acts = actions_model(i, len(actions.FUNCTIONS))(x)
        layers.append(avail_acts)
        i += len(actions.FUNCTIONS)

        # last_action
        last_act = actions_model(i, len(actions.FUNCTIONS))(x)
        layers.append(last_act)
        i += len(actions.FUNCTIONS)

        # last action args
        for arg_type in FLAT_ACTION_TYPES:
            layer = actions_model(i, arg_type.sizes[0])(x)
            layers.append(layer)
            i += arg_type.sizes[0]

        y = Concatenate(axis=-1, name='preprocess_non_spatial_concat')(layers)

        return Model(inputs=[x], outputs=[y], name='preprocess_non_spatial_model')

    def args_output(self, pi_input_shape, spatial_input_shape):
        pi_embed = Input(batch_shape=(self.batch_size,) + pi_input_shape, name='pi_embed')
        spatial_representation = Input(batch_shape=(self.batch_size,) + spatial_input_shape, name='spatial_rep')

        args = []

        def spatial_case(arg_t):
            model = Sequential(name=arg_t.name + '_model')
            model.add(Conv2D(1, (1, 1), name=arg_t.name+'_conv', data_format=self.data_format,
                             input_shape=spatial_input_shape, batch_size=self.batch_size))
            model.add(Flatten(data_format=self.data_format, name=arg_t.name+'_flat'))
            model.add(Activation('softmax', name=arg_t.name+'_softmax'))

            return model

        for arg_type in actions.TYPES:
            if arg_type in [actions.TYPES.minimap, actions.TYPES.screen, actions.TYPES.screen2]:
                arg = spatial_case(arg_type)(spatial_representation)
            else:
                arg = Dense(arg_type.sizes[0], activation='softmax', name=arg_type.name)(pi_embed)

            args.append(arg)

        args_model = Model(inputs=[pi_embed, spatial_representation], outputs=args, name='args_model')

        return args_model

    def first_conv_block(self, prefix, feature_types, input_shape):
        model = Sequential(name=prefix + '_model')
        model.add(self.preprocess_spatial_obs(feature_types, input_shape))
        model.add(ZeroPadding2D(name='1_1_0_padding_'+prefix, data_format=self.data_format))
        model.add(Conv2D(32, (4, 4), strides=2, activation='relu', name='conv1_'+prefix, data_format=self.data_format))
        res_shape = [16, 16, 16]
        res_shape[self.channel_axis] = 32
        model.add(self.residual_block(input_shape=tuple(res_shape)))
        model.add(MaxPool2D(name='max_pool_'+prefix, data_format=self.data_format))

        return model

    def broadcast_along_channels(self, input_shape, size2d, name='non_spatial'):
        model = Sequential(name=name+'_broadcast_model')
        model.add(RepeatVector(np.asscalar(np.prod(size2d)), name=name+'_tile', input_shape=input_shape, batch_size=self.batch_size))
        model.add(Reshape(size2d + input_shape, name=name+'_unflatten'))

        if self.data_format == 'channels_first':
            model.add(Permute((3, 1, 2), name=name + '_to_NCHW'))

        return model

    # train on a single batch
    # param observations: observations as returned by pysc2 (more or less), wrapped in a batch
    # param returns: nstep sum of rewards + value obtained from taking actions according to policy, according to observations
    # param advantages: returns - values
    # param actions: actions sampled from network policy
    def train_reinforcement(self, observations, actions, returns, advantages,
                            write_summary=False, step=None, rewards=None, reset_states=False, states=None):
        sampled_actions = actions[0]

        labels = [sampled_actions, returns] + actions[1:]  # self.args_mask(actions[1:], action_ids)
        batch_size = returns.shape[0]
        loss = self.trainable_model.train_on_batch(observations + labels + [np.array([step] * batch_size)],
                                                   [np.zeros((batch_size,)) for _ in range(3)])  # dummy targets


        # summary
        if write_summary:
            return_summary = tf.Summary(value=[tf.Summary.Value(tag="return",
                                                                simple_value=np.asscalar(np.mean(returns.flatten())))])
            reward_summary = tf.Summary(value=[tf.Summary.Value(tag="reward",
                                                                simple_value=np.asscalar(np.mean(rewards.flatten())))])
            adv_summary = tf.Summary(value=[tf.Summary.Value(tag="advantage",
                                                             simple_value=np.asscalar(np.mean(advantages.flatten())))])
            self.writer.add_summary(return_summary, global_step=step)
            self.writer.add_summary(reward_summary, global_step=step)
            self.writer.add_summary(adv_summary, global_step=step)
            for name, value in zip(self.trainable_model.metrics_names, loss):
                summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=value), ])
                self.writer.add_summary(summary, global_step=step)

        return loss

    def predict(self, observations):
        pred = self.target_model.predict_on_batch(observations)
        return pred

    def compile(self, opt):
        y_pred_list = self.model.output
        y_true_list = []
        y_true_list.append(Input(shape=(len(actions.FUNCTIONS),), name='pi_sampled'))
        y_true_list.append(Input(shape=(1, ), name='return'))

        for arg_type in actions.TYPES:
            if arg_type in [actions.TYPES.minimap, actions.TYPES.screen, actions.TYPES.screen2]:
                sample = Input(shape=(32*32, ), name=arg_type.name+'_true')
            else:
                sample = Input(shape=(arg_type.sizes[0],), name=arg_type.name+'_true')

            y_true_list.append(sample)

        adv = Lambda(lambda args: k.stop_gradient(args[0] - args[1]))([y_true_list[1], y_pred_list[1]])
        it = Input(shape=(1, ), name='iterations')

        policy_and_args = y_pred_list[:1] + y_pred_list[2:15]
        pi_samples = y_true_list[:1] + y_true_list[2:]
        policy_losses = []
        entropies = []
        for y_pred, y_true in zip(policy_and_args, pi_samples):
            policy_losses.append(Lambda(self.policy_gradient_loss)([adv, y_pred, y_true]))
            entropies.append(Lambda(self.entropy)(y_pred))

        policy_loss = Lambda(k.stack, arguments={'axis': -1})(policy_losses)
        policy_loss = Lambda(k.sum, arguments={'axis': -1})(policy_loss)
        policy_loss = Lambda(k.mean, name='policy_loss')(policy_loss)

        value_loss = Lambda(self.value_loss, name='mse')([y_true_list[1], y_pred_list[1]])
        value_loss = Lambda(k.mean, name='value_loss')(value_loss)

        entropy = Lambda(k.stack, arguments={'axis': -1})(entropies)
        entropy = Lambda(k.sum, arguments={'axis': -1})(entropy)
        entropy = Lambda(lambda x: -k.mean(x), name='entropy')(entropy)

        entropy_coeff = self.entropy_coeff
        entropy_coeff *= (1. / (1. + 0.99 * it[0, 0]))

        self.trainable_model = Model(inputs=self.model.input + y_true_list + [it],
                                     outputs=[policy_loss, value_loss, entropy])
        self.trainable_model.compile(optimizer=opt, loss=lambda yt, yp: yp,
                                     loss_weights=[1., self.value_loss_coeff, entropy_coeff])

    # pickle or json?
    def save(self, name, num_iters):
        # TODO : fix save model to json
        #model_json = self.model.to_yaml()
        #name = os.path.join(name, "model")
        #with open("{}.yaml".format(name), "w") as json_file:
        #    json_file.write(model_json)
        os.makedirs(name, exist_ok=True)
        name = os.path.join(name, "weights")
        self.model.save_weights("{}_{}.h5".format(name, num_iters))

    def load(self, name, num_iters=0):
        # TODO : load model from json when save fixed
        # with open("bin/{}.json".format(name), "r") as json_file:
        #    loaded_model_json = json_file.read()
        # self.model = model_from_json(loaded_model_json)
        name = os.path.join(name, "weights")
        if num_iters is not 0:
            self.model.load_weights("{}_{}.h5".format(name, num_iters))
        else:
            self.model.load_weights("{}.h5".format(name))

    def policy_gradient_loss(self, args):
        advantage, y_pred, y_true = args
        adv = k.squeeze(advantage, axis=-1)

        policy_loss = adv * self.compute_neg_log_prob(y_true, y_pred)

        return policy_loss

    def entropy(self, y_pred):
        entropy = categorical_crossentropy(y_pred, y_pred)

        return entropy

    def value_loss(self, args):
        y_true, y_pred = args
        return mse(y_pred, y_true) / 2.

    def compute_neg_log_prob(self, y_true, y_pred):
        return -k.sum(y_true * k.log(k.clip(y_pred, 1e-12, 1.0)), axis=-1)

    def normalization(self, x):
        return x / k.clip(k.sum(x, axis=-1, keepdims=True), 1e-12, 1.0)


def test_difference(old_w, new_w):
    difference = 0
    counter = 0
    for ow, nw in zip(old_w, new_w):
        assert ow.shape == nw.shape
        local_diff = np.sum(np.abs(ow - nw))
        counter += local_diff == 0
        difference += local_diff

    assert difference > 0
