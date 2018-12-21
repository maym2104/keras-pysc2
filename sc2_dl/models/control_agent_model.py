from sc2_dl.models.base_model import *


class ControlAgentModel(BaseModel):
    def __init__(self, seq_length=16, **kwargs):

        super(ControlAgentModel, self).__init__(**kwargs)

        self.seq_length = None

    def init_model(self, input_shape, num_actions, opt, graph_path=None):
        # minimap layers
        minimap_shape = input_shape['feature_minimap']
        minimap_shape = (self.batch_size, self.seq_length,) + minimap_shape[1:] + (minimap_shape[0] + 1,)
        minimap_model = TimeDistributed(self.first_conv_block('minimap', features.MINIMAP_FEATURES, minimap_shape[2:]),
                                        name='minimap_model_td')
        minimap_layers = Input(batch_shape=minimap_shape, name='feature_minimap')
        minimap_layers_encoded = minimap_model(minimap_layers)

        screen_shape = input_shape['feature_screen']
        screen_shape = (self.batch_size, self.seq_length,) + screen_shape[1:] + (screen_shape[0] + 2,)
        screen_model = TimeDistributed(self.first_conv_block('screen', features.SCREEN_FEATURES, screen_shape[2:]),
                                       name='screen_model_td')
        screen_layers = Input(batch_shape=screen_shape, name='feature_screen')
        screen_layers_encoded = screen_model(screen_layers)

        # non-spatial layers
        size2d = k.int_shape(screen_layers_encoded)
        nonspatial_model = Sequential(name='non_spatial_model')
        nonspatial_shape = (self.batch_size, self.seq_length, len(FLAT_FEATURES)) #+ 2*num_actions + sum([t.sizes[0]
                                                                                     #for t in FLAT_ACTION_TYPES]))
        nonspatial_model.add(TimeDistributed(self.preprocess_nonspatial_obs(FLAT_FEATURES, nonspatial_shape[2:]),
                                                    name='preprocess_non_spatial_td', batch_input_shape=nonspatial_shape))
        nonspatial_model.add(TimeDistributed(self.two_layer_mlp(128, 64,
                                                                input_shape=k.int_shape(nonspatial_model.output[1:])),
                                             name='non_spatial_mlp_td'))  # without batch size

        nonspatial_layers = Input(batch_shape=(self.batch_size, self.seq_length, len(FLAT_FEATURES)),
                                  name='non_spatial')
        #non_spatial_last_actions = Input(batch_shape=(self.batch_size, self.seq_length,
        #                                        num_actions + sum([t.sizes[0] for t in FLAT_ACTION_TYPES])),
        #                                 name='non_spatial_last_acts')
        available_actions = Input(batch_shape=(self.batch_size, self.seq_length, num_actions,), name='avail_acts')

        #all_nonspatial_layers = Concatenate(axis=-1, name='non_spatial_concat')([nonspatial_layers,
        #                                                                         available_actions,
        #                                                                         non_spatial_last_actions])
        #nonspatial_layers_encoded = nonspatial_model(all_nonspatial_layers)
        nonspatial_layers_encoded = nonspatial_model(nonspatial_layers)

        # state_representation
        state_representation = Concatenate(axis=self.channel_axis, name='concat',
                                           batch_size=self.batch_size)([screen_layers_encoded,
                                                                        minimap_layers_encoded])

        self.lstm_layer = ConvLSTM2D(96, (3, 3), padding='same', return_sequences=True, stateful=True, name='conv_lstm',
                                     batch_size=self.batch_size, data_format=self.data_format, return_state=True)
        lstm_state_representation, *lstm_state = self.lstm_layer(state_representation)

        spatial_state_representation = TimeDistributed(Conv2D(32, (3, 3), padding='same', data_format=self.data_format,
                                                              activation='relu'))(lstm_state_representation)
        post_lstm_conv_shape = k.int_shape(spatial_state_representation)[2:]
        spatial_state_representation = TimeDistributed(self.residual_block(input_shape=post_lstm_conv_shape,
                                                                           num_layers=2))(spatial_state_representation)
        spatial_state_representation = TimeDistributed(self.residual_block(input_shape=post_lstm_conv_shape,
                                                                           num_layers=3))(spatial_state_representation)
        spatial_state_representation = TimeDistributed(self.residual_block(input_shape=post_lstm_conv_shape,
                                                                           num_layers=3))(spatial_state_representation)
        spatial_state_representation = TimeDistributed(self.residual_block(input_shape=post_lstm_conv_shape,
                                                                           num_layers=3))(spatial_state_representation)

        fc_state_representation = TimeDistributed(Flatten(data_format=self.data_format,
                                                          name='state_representation_flat'),
                                                  name='state_representation_flat_td')(lstm_state_representation)
        fc_state_representation = TimeDistributed(self.two_layer_mlp(512, 512, input_shape=k.int_shape(fc_state_representation)[1:]),
                                                  name='conv_to_fc_td')(fc_state_representation)
        shared_features = Concatenate(name='concat_shared_features')([fc_state_representation,
                                                                      nonspatial_layers_encoded])

        # deconvolution - opposite operations to get back a size of (32, 32)
        # opposite of MaxPool: UpSampling (copy each element)
        # opposite of valid (4,4) conv stride 2 : valid transpose (4,4) conv stride 2
        # opposite of ZeroPadding (1, 1): Cropping (1, 1) (ideally should be part of conv, but no custom padding avail)
        deconv_state_representation = TimeDistributed(Conv2DTranspose(16, (4, 4), strides=2, name='conv_transpose',
                                                                      data_format=self.data_format, activation='relu'),
                                                      name='conv_transpose_td')(spatial_state_representation)
        deconv_state_representation = TimeDistributed(UpSampling2D(name='up_sampling', data_format=self.data_format),
                                                      name='up_sampling_td')(deconv_state_representation)
        deconv_state_representation = TimeDistributed(Conv2D(16, 5, strides=1, padding='valid', activation='relu',
                                                             data_format=self.data_format))(deconv_state_representation)

        # output
        pi_logits = TimeDistributed(self.two_layer_mlp(256, num_actions, input_shape=k.int_shape(shared_features)[1:]),
                                    name='pi_logits_td')(shared_features)
        pi = TimeDistributed(Activation('softmax', name='pi_softmax'), name='pi_softmax_td')(pi_logits)
        pi_masked = Multiply(name='pi_masked')([pi, available_actions])
        pi_normalized = TimeDistributed(Lambda(lambda x: normalization(x), name='pi'),
                                        name='pi_td')(pi_masked)

        v = TimeDistributed(self.two_layer_mlp(256, 1, input_shape=k.int_shape(fc_state_representation)[1:]),
                            name='v')(fc_state_representation)
        # TODO: find a better solution to this
        pi_embedded = TimeDistributed(Dense(16, activation='relu', name='pi_embedded'),
                                      name='pi_embedded_td')(pi_logits)
        pi_sample = TimeDistributed(Lambda(lambda x: k.argmax(k.log(k.random_uniform((self.batch_size, num_actions))) / x),
                                           name='pi_sample'), name='pi_sample_td')(pi_normalized)

        args_non_spatial_input = Concatenate(name='non_spatial_conditionning')([pi_embedded, shared_features])
        broadcast_pi_embed = TimeDistributed(self.broadcast_along_channels(input_shape=k.int_shape(pi_embedded)[2:],
                                                                           size2d=(32, 32)))(pi_embedded)
        args_spatial_input = Concatenate(name='spatial_conditionning',
                                         axis=self.channel_axis)([deconv_state_representation, broadcast_pi_embed])
        args_model = self.args_output(pi_input_shape=k.int_shape(args_non_spatial_input)[2:],
                                      spatial_input_shape=k.int_shape(args_spatial_input)[2:])
        args = args_model([args_non_spatial_input, args_spatial_input])
        samples = [pi_sample]

        for arg_type in actions.TYPES:
            size = k.int_shape(args[arg_type.id])[2:]
            samples.append(
                Lambda(lambda x: k.argmax(k.log(k.random_uniform((self.batch_size, 1,) + size, seed=self.seed)) / x,
                                          axis=-1), name=arg_type.name + '_sample')(args[arg_type.id]))

        inputs = [screen_layers, minimap_layers, nonspatial_layers, available_actions]

        self.model = Model(inputs=inputs, outputs=[v, pi_normalized]+args)
        self.target_model = Model(inputs=inputs, outputs=[v]+samples+lstm_state)

        self.compile(opt)

        # summary
        self.writer.add_graph(k.get_session().graph)
        if graph_path is not None:
            os.makedirs(graph_path, exist_ok=True)
            #plot_model(self.model, to_file=os.path.join(graph_path, 'model.png'), show_shapes=True, show_layer_names=True, rankdir='LR')

    def args_output(self, pi_input_shape, spatial_input_shape):
        pi_embed = Input(batch_shape=(self.batch_size, self.seq_length) + pi_input_shape, name='pi_embed')
        spatial_representation = Input(batch_shape=(self.batch_size, self.seq_length) + spatial_input_shape, name='spatial_rep')

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
                arg = TimeDistributed(spatial_case(arg_type), name=arg_type.name + '_td')(spatial_representation)
            else:
                arg = TimeDistributed(Dense(arg_type.sizes[0], activation='softmax', name=arg_type.name),
                                      name=arg_type.name + '_name')(pi_embed)
            args.append(arg)

        args_model = Model(inputs=[pi_embed, spatial_representation], outputs=args, name='args_model')

        return args_model

    # return last output of sequence
    def predict(self, observations):
        pred = self.target_model.predict_on_batch(observations)

        return [p[:, -1] for p in pred[:15]] + pred[15:]

    # train on a single batch
    # param observations: observations as returned by pysc2 (more or less), wrapped in a batch
    # param returns: nstep sum of rewards + value obtained from taking actions according to policy, according to observations
    # param advantages: returns - values
    # param actions: actions sampled from network policy
    def train_reinforcement(self, observations, acts, returns, advantages, masks,
                            write_summary=False, step=None, rewards=None, reset_states=False, states=None):
        labels = [returns] + acts  # self.args_mask(actions[1:], action_ids)
        self.reset_states(states)  # it's the same states for both models... let's reset them

        batch_size = returns.shape[0]
        loss = self.trainable_model.train_on_batch(observations + labels + masks + [np.array([step] * batch_size)],
                                                   [np.zeros((batch_size,))])  # dummy targets

        # summary
        if write_summary:
            #keep only last step returns (for now)
            return_summary = tf.Summary(value=[tf.Summary.Value(tag="return", simple_value=np.asscalar(np.mean(returns[:, -1, 0])))])
            reward_summary = tf.Summary(value=[tf.Summary.Value(tag="reward", simple_value=np.asscalar(np.mean(rewards.flatten())))])
            adv_summary = tf.Summary(value=[tf.Summary.Value(tag="advantage", simple_value=np.asscalar(np.mean(advantages.flatten())))])
            self.writer.add_summary(return_summary, global_step=step)
            self.writer.add_summary(reward_summary, global_step=step)
            self.writer.add_summary(adv_summary, global_step=step)
            #for name, value in zip(self.trainable_model.metrics_names, loss):
            summary = tf.Summary(value=[tf.Summary.Value(tag='loss', simple_value=loss), ])
            self.writer.add_summary(summary, global_step=step)

        return loss

    def reset_states(self, states=None):
        self.lstm_layer.reset_states(states)

    def get_states(self):
        return self.lstm_layer.states

    def compile(self, opt):
        y_pred_list = self.model.output
        y_true_list = []
        input_masks = []
        #masks = []
        y_true_list.append(Input(batch_shape=(self.batch_size, self.seq_length, 1, ), name='return'))
        y_true_list.append(Input(batch_shape=(self.batch_size, self.seq_length,), name='pi_sampled', dtype='int32'))

        for arg_type in actions.TYPES:
            if arg_type in [actions.TYPES.minimap, actions.TYPES.screen, actions.TYPES.screen2]:
                sample = Input(batch_shape=(self.batch_size, self.seq_length,), name=arg_type.name+'_true', dtype='int32')
            else:
                sample = Input(batch_shape=(self.batch_size, self.seq_length,), name=arg_type.name+'_true', dtype='int32')

            y_true_list.append(sample)
            mask = Input(batch_shape=(self.batch_size, self.seq_length,), name=arg_type.name + '_mask')
            input_masks.append(mask)
            #mask = Lambda(k.squeeze, arguments={'axis': -1})(mask)
            #y_true_list.append(sample)
            #masks.append(mask)

        ret = Lambda(k.squeeze, arguments={'axis': -1})(y_true_list[0])
        val = Lambda(k.squeeze, arguments={'axis': -1})(y_pred_list[0])
        adv = Lambda(lambda args: k.stop_gradient(args[0] - args[1]))([ret, val])
        it = Input(batch_shape=(self.batch_size, 1, ), name='iterations')

        policy_and_args = y_pred_list[1:15]
        pi_samples = y_true_list[1:15]
        policy_losses = []
        entropies = []
        for y_pred, y_true in zip(policy_and_args, pi_samples):
            y_true = Lambda(k.reshape, arguments={'shape': [-1, 1]})(y_true)
            y_pred = Lambda(k.reshape, arguments={'shape': [k.shape(y_true)[0], -1]})(y_pred)
            policy_loss = Lambda(policy_gradient_loss)([y_true, y_pred])
            policy_loss = Lambda(k.reshape, arguments={'shape': [self.batch_size, -1]})(policy_loss)
            policy_losses.append(policy_loss)
            entropies.append(Lambda(entropy)(y_pred))

        policy_losses = [policy_losses[0]] + [Lambda(lambda args: args[0] * args[1])([p_l, mask])
                                              for p_l, mask in zip(policy_losses[1:], input_masks)]
        policy_loss = Lambda(k.stack, arguments={'axis': -1})(policy_losses)
        policy_loss = Lambda(k.sum, arguments={'axis': -1})(policy_loss)
        policy_loss = Lambda(lambda args: k.mean(args[0]*args[1]), name='policy_loss')([adv, policy_loss])

        value_l = Lambda(lambda args: k.square(args[0] - args[1]) / 2., name='mse')([ret, val])
        value_l = Lambda(k.mean, name='value_loss')(value_l)

        entrop = Lambda(k.stack, arguments={'axis': -1})(entropies)
        entrop = Lambda(k.sum, arguments={'axis': -1})(entrop)
        entrop = Lambda(lambda x: k.mean(x), name='entropy')(entrop)

        entropy_coeff = Lambda(lambda _x:
                               self.entropy_coeff * (1. / (1. + self.decay *
                                                           k.stop_gradient(k.squeeze(k.squeeze(k.slice(_x, [0, 0],
                                                                                                       [1, 1]),
                                                                                               axis=-1),
                                                                                     axis=-1)))))(it)

        loss = Lambda(lambda args: args[0] + self.value_loss_coeff * args[1] - args[3] * args[2])([policy_loss,
                                                                                                   value_l,
                                                                                                   entrop,
                                                                                                   entropy_coeff])

        self.trainable_model = Model(inputs=self.model.input + y_true_list + input_masks + [it],
                                     outputs=[loss])
        self.trainable_model.compile(optimizer=opt, loss=lambda yt, yp: yp)  # ,
                                     # loss_weights=[1., self.value_loss_coeff, entropy_coeff])


def policy_gradient_loss(args):
#    adv, y_true, y_pred = args
    #adv = k.squeeze(advantage, axis=-1)
    # y_true = K.cast(y_true, dtype='int32')
    # policy_loss = k.log(k.clip(tf.gather_nd(y_pred, y_true), 1e-12, 1.0))
    policy_loss = sparse_categorical_crossentropy(y_true, y_pred)  # self.compute_log_prob(y_true, y_pred)
#    policy_loss = adv * categorical_crossentropy(y_true, y_pred)

    return policy_loss