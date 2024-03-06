import numpy as np
import hyperparams as params
import ddpg_train
import time
import tensorflow as tf

max_batch_train = 3

for batch_train in range(0, max_batch_train):
    params.model_name = 'model_' + '{:03}'.format(batch_train + 1)

    if batch_train < max_batch_train:
        params.duration = 160
        params.ac_initial_learning_rate = 0.0005
        params.ac_decay_steps= 50000
        params.ac_decay_rate = 0.9
        params.cr_initial_learning_rate = 0.003
        params.cr_decay_steps = 60000
        params.cr_decay_rate = 0.9
        params.ac_layer = (64,64)
        params.cr_obs_layer = (32,32)
        params.cr_act_layer = (16,16)
        params.cr_joint_layer = (64,64)
        params.discount_factor = 0.95
        params.std_episodes = 9000
        params.std_mul = 0.15
        params.target_update_tau = 0.01
        params.target_update_period = 1
        params.replay_buffer_max_length = 1000000
        params.num_parallel_calls = 2
        params.sample_batch_size = 128
        params.num_steps = 2
        params.prefetch = 3
        params.destination_reward = 100
        params.traditional_ep = 2001
        params.total_episodes = 7001
        params.random_seed = np.random.randint(100000)
        params.DDPG_update_time_steps = 10
        params.DDPG_policy_store_frequency = 1000
        params.DDPG_loss_avg_interval = 100
        params.Kd = 2.0
        params.Kp = 4.0
        params.look_ahead = 2.0
        params.design_para = 0.05

    tf.keras.utils.set_random_seed(params.random_seed)
    tf.config.experimental.enable_op_determinism()
    ddpg_train.ddpg_train(params)

print(f'Execution Completed Successfully')


