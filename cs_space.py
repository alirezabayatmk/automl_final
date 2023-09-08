import ConfigSpace as CS
import ConfigSpace.api.types.integer as CSInt
import ConfigSpace.api.types.categorical as CSCat
import ConfigSpace.api.types.float as CSFloat
from ConfigSpace.read_and_write import json as cs_json

def joint_configspace():
    # Create a Configuration Space
    cs = CS.ConfigurationSpace()

    # Hyperparameters for Convolutional Layers
    n_conv_layers = CSInt.Integer(name='n_conv_layers', bounds=(1, 4))
    n_channels_conv_0 = CSInt.Integer(name='n_channels_conv_0', bounds=(32, 512))
    n_channels_conv_1 = CSInt.Integer(name='n_channels_conv_1', bounds=(32, 512))
    n_channels_conv_2 = CSInt.Integer(name='n_channels_conv_2', bounds=(32, 512))
    n_channels_conv_3 = CSInt.Integer(name='n_channels_conv_3', bounds=(32, 512))
    kernel_size = CSInt.Integer(name='kernel_size', bounds=(2, 3))
    use_BN = CSCat.Categorical(name='use_BN', items=[True, False])

    # Hyperparameters for Fully Connected Layers
    n_fc_layers = CSInt.Integer(name='n_fc_layers', bounds=(1, 3))
    n_channels_fc_0 = CSInt.Integer(name='n_channels_fc_0', bounds=(64, 512))
    n_channels_fc_1 = CSInt.Integer(name='n_channels_fc_1', bounds=(64, 512))
    n_channels_fc_2 = CSInt.Integer(name='n_channels_fc_2', bounds=(64, 512))
    dropout_rate = CSFloat.Float(name='dropout_rate', bounds=(0.0, 0.4))

    # Hyperparameters for Global Average Pooling
    global_avg_pooling = CSCat.Categorical(name='global_avg_pooling', items=[True, False])

    # Hyperparameters for batch size and initial learning rate
    batch_size = CSInt.Integer(name='batch_size', bounds=(16, 128))
    learning_rate_init = CSFloat.Float(name='learning_rate_init', bounds=(0.001, 0.01))

    # Define conditions
    conv_channel_conditions = []
    conv_channel_conditions.append(CS.GreaterThanCondition(n_channels_conv_1, n_conv_layers, 1))
    conv_channel_conditions.append(CS.GreaterThanCondition(n_channels_conv_2, n_conv_layers, 2))
    conv_channel_conditions.append(CS.GreaterThanCondition(n_channels_conv_3, n_conv_layers, 3))

    fc_channel_conditions = []
    fc_channel_conditions.append(CS.GreaterThanCondition(n_channels_fc_1, n_fc_layers, 1))
    fc_channel_conditions.append(CS.GreaterThanCondition(n_channels_fc_2, n_fc_layers, 2))

    # Add all hyperparameters to the Configuration Space
    cs.add_hyperparameters([
        n_conv_layers, n_channels_conv_0, n_channels_conv_1, n_channels_conv_2, n_channels_conv_3, kernel_size, use_BN,
        n_fc_layers, n_channels_fc_0, n_channels_fc_1, n_channels_fc_2, dropout_rate, global_avg_pooling,
        batch_size, learning_rate_init
    ])

    # Add conditions to the Configuration Space
    cs.add_conditions(conv_channel_conditions)
    cs.add_conditions(fc_channel_conditions)

    with open('joint_configspace.json', 'w') as f:
        f.write(cs_json.write(cs))


    return cs


def hp_only_configspace():
    # Create a Configuration Space for other hyperparameters
    cs_hyperparameters = CS.ConfigurationSpace()

    # Hyperparameters for kernel size
    kernel_size = CSInt.Integer(name='kernel_size', bounds=(3, 5))

    # Hyperparameters for dropout rate
    dropout_rate = CSFloat.Float(name='dropout_rate', bounds=(0.0, 0.4))

    # Hyperparameters for batch size and initial learning rate
    batch_size = CSInt.Integer(name='batch_size', bounds=(16, 128))
    learning_rate_init = CSFloat.Float(name='learning_rate_init', bounds=(0.001, 0.01))

    # Add all hyperparameters to the Configuration Space
    cs_hyperparameters.add_hyperparameters([kernel_size, dropout_rate, batch_size, learning_rate_init])

    with open('hp_configspace.json', 'w') as f:
        f.write(cs_json.write(cs_hyperparameters))

    return cs_hyperparameters


def arch_only_configspace():
    # Create a Configuration Space
    cs_architecture = CS.ConfigurationSpace()

    # Hyperparameters for Convolutional Layers
    n_conv_layers = CSInt.Integer(name='n_conv_layers', bounds=(1, 4))
    n_channels_conv_0 = CSInt.Integer(name='n_channels_conv_0', bounds=(32, 256))
    n_channels_conv_1 = CSInt.Integer(name='n_channels_conv_1', bounds=(32, 256))
    n_channels_conv_2 = CSInt.Integer(name='n_channels_conv_2', bounds=(32, 256))
    n_channels_conv_3 = CSInt.Integer(name='n_channels_conv_3', bounds=(32, 256))
    use_BN = CSCat.Categorical(name='use_BN', items=[True, False])

    # Hyperparameters for Fully Connected Layers
    n_fc_layers = CSInt.Integer(name='n_fc_layers', bounds=(1, 3))
    n_channels_fc_0 = CSInt.Integer(name='n_channels_fc_0', bounds=(64, 256))
    n_channels_fc_1 = CSInt.Integer(name='n_channels_fc_1', bounds=(64, 256))
    n_channels_fc_2 = CSInt.Integer(name='n_channels_fc_2', bounds=(64, 256))

    # Hyperparameters for Global Average Pooling
    global_avg_pooling = CSCat.Categorical(name='global_avg_pooling', items=[True, False])

    # Define conditions
    conv_channel_conditions = []
    conv_channel_conditions.append(CS.GreaterThanCondition(n_channels_conv_1, n_conv_layers, 1))
    conv_channel_conditions.append(CS.GreaterThanCondition(n_channels_conv_2, n_conv_layers, 2))
    conv_channel_conditions.append(CS.GreaterThanCondition(n_channels_conv_3, n_conv_layers, 3))

    fc_channel_conditions = []
    fc_channel_conditions.append(CS.GreaterThanCondition(n_channels_fc_1, n_fc_layers, 1))
    fc_channel_conditions.append(CS.GreaterThanCondition(n_channels_fc_2, n_fc_layers, 2))

    # Add all hyperparameters to the Configuration Space
    cs_architecture.add_hyperparameters([
        n_conv_layers, n_channels_conv_0, n_channels_conv_1, n_channels_conv_2, n_channels_conv_3, use_BN,
        n_fc_layers, n_channels_fc_0, n_channels_fc_1, n_channels_fc_2, global_avg_pooling
    ])

    # Add conditions to the Configuration Space
    cs_architecture.add_conditions(conv_channel_conditions)
    cs_architecture.add_conditions(fc_channel_conditions)

    with open('arch_configspace.json', 'w') as f:
        f.write(cs_json.write(cs_architecture))

    return cs_architecture


if __name__ == '__main__':
    print('joint_configspace: ')
    joint_configspace()
    """radom_joint_configspace = joint_configspace()
    print(radom_joint_configspace)
    print(radom_joint_configspace.sample_configuration())
    print('-'*50)

    print('hp_only_configspace: ')
    radom_hp_only_configspace = hp_only_configspace()
    print(radom_hp_only_configspace)
    print(radom_hp_only_configspace.sample_configuration())
    print('-'*50)

    print('arch_only_configspace: ')
    radom_arch_only_configspace = arch_only_configspace()
    print(radom_arch_only_configspace)
    print(radom_arch_only_configspace.sample_configuration())
    print('-'*50)"""



