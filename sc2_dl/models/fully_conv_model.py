from sc2_dl.models.base_model import *


class FullyConvModel(BaseModel):
    def __init__(self, **kwargs):

        super(FullyConvModel, self).__init__(**kwargs)

    def init_model(self, input_shape, num_actions, opt, graph_path=None):
        # minimap layers
        minimap_shape = input_shape['feature_minimap']
        minimap_shape = minimap_shape[1:] + (minimap_shape[0], )
        minimap_model = Sequential(name='minimap_model')
        minimap_model.add(self.preprocess_spatial_obs(features.MINIMAP_FEATURES, minimap_shape, 'minimap', embed_size=1))
        minimap_model.add(Conv2D(16, (5, 5), padding='same', activation='relu', name='conv1_minimap',
                                 data_format=self.data_format))
        minimap_model.add(Conv2D(32, (3, 3), padding='same', activation='relu', name='conv2_minimap',
                                 data_format=self.data_format))

        minimap_layers = Input(shape=minimap_shape, name='feature_minimap')
        minimap_layers_encoded = minimap_model(minimap_layers)

        # minimap layers
        screen_shape = input_shape['feature_screen']
        screen_shape = screen_shape[1:] + (screen_shape[0], )
        screen_model = Sequential(name='screen_model')
        screen_model.add(self.preprocess_spatial_obs(features.SCREEN_FEATURES, screen_shape, 'screen', embed_size=1))
        screen_model.add(Conv2D(16, (5, 5), padding='same', activation='relu', name='conv1_screen',
                                data_format=self.data_format))
        screen_model.add(Conv2D(32, (3, 3), padding='same', activation='relu', name='conv2_screen',
                                data_format=self.data_format))

        screen_layers = Input(shape=screen_shape, name='feature_screen')
        screen_layers_encoded = screen_model(screen_layers)

        # non-spatial layers
        size2d = minimap_shape[0:2]

        nonspatial_model = Sequential(name='non_spatial_model')
        nonspatial_shape = (len(FLAT_FEATURES),)
        nonspatial_model.add(self.preprocess_nonspatial_obs(FLAT_FEATURES, nonspatial_shape, embed_size=1))
        nonspatial_model.add(self.broadcast_along_channels((11,), size2d))

        nonspatial_layers = Input(shape=(len(FLAT_FEATURES),), name='non_spatial')
        available_actions = Input(shape=(num_actions,), name='avail_acts')
        nonspatial_layers_encoded = nonspatial_model(nonspatial_layers)

        # state_representation
        state_representation = Concatenate(axis=self.channel_axis, name='concat')([screen_layers_encoded,
                                                                                   minimap_layers_encoded,
                                                                                   nonspatial_layers_encoded])

        fc_state_representation = state_representation
        fc_state_representation = Flatten(data_format=self.data_format,
                                          name='state_representation_flat')(fc_state_representation)
        fc_state_representation = Dense(256, activation='relu', name='fc')(fc_state_representation)  # conv_to_fc

        # output
        pi = Dense(num_actions, activation='softmax', name='pi_softmax')(fc_state_representation)
        pi_masked = Multiply(name='pi_masked')([pi, available_actions])
        pi_normalized = Lambda(lambda x: normalization(x), name='pi')(pi_masked)

        pi_sample = Lambda(lambda x: k.argmax(k.log(k.random_uniform((self.batch_size, num_actions))) / x, axis=-1),
                           name='pi_sample')(pi_normalized)

        v = Dense(1, name='v')(fc_state_representation)

        spatial_input_shape = k.int_shape(state_representation)[1:]
        pi_input_shape = k.int_shape(fc_state_representation)[1:]
        args_model = self.args_output(pi_input_shape=pi_input_shape, spatial_input_shape=spatial_input_shape)
        args = args_model([fc_state_representation, state_representation])
        samples = [pi_sample]

        for arg_type in actions.TYPES:
            size = k.int_shape(args[arg_type.id])[1:]
            samples.append(Lambda(lambda x: k.argmax(k.log(k.random_uniform((self.batch_size,) + size)) / x, axis=-1),
                                  name=arg_type.name + '_sample')(args[arg_type.id]))

        inputs = [screen_layers, minimap_layers, nonspatial_layers, available_actions]
        self.model = Model(inputs=inputs, outputs=[v, pi_normalized] + args)
        self.target_model = Model(inputs=inputs, outputs=[v] + samples)

        self.compile(opt=opt)

        # summary
        self.writer.add_graph(k.get_session().graph)
        if graph_path is not None:
            os.makedirs(graph_path, exist_ok=True)
            #plot_model(self.model, to_file=os.path.join(graph_path, 'model.png'), show_shapes=True, show_layer_names=True, rankdir='LR')

