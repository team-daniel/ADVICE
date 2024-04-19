import safety_gymnasium
import numpy as np
import os
import joblib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

from encoder_shield_adpt_shield import Encoder_DDPG_Adpt_Shield

# initalise environment
env = safety_gymnasium.make('SafetyCarGoal0-v0', render_mode='rgb_array', camera_name="track")

# declare some vars
state_size = env.observation_space.shape[0]
state_size = 32    # for goal env
action_size = env.action_space.shape[0]
action_high = np.asarray([1.0, 1.0])
action_low = np.asarray([-1.0, -1.0])

total_episodes = 2000


# declare the encoder agent with adaptable shield
enc_ddpg = Encoder_DDPG_Adpt_Shield(env, total_episodes, state_size, action_size, action_high, action_low)

# train
enc_avg_reward_list, enc_ep_reward_list, enc_safety_violations_list, enc_goal_reached_list, enc_shield_activations_list, enc_env_interacts_list, enc_neighbours_list = enc_ddpg.train(1000)

np.savetxt('Results/adpt_encoder_avg_reward.csv', np.asarray(enc_avg_reward_list), delimiter=',')
np.savetxt('Results/adpt_encoder_ep_reward.csv', np.asarray(enc_ep_reward_list), delimiter=',')
np.savetxt('Results/adpt_encoder_safety_violations.csv', np.asarray(enc_safety_violations_list), delimiter=',')
np.savetxt('Results/adpt_encoder_goal_reached.csv', np.asarray(enc_goal_reached_list), delimiter=',')
np.savetxt('Results/adpt_encoder_shield_activations.csv', np.asarray(enc_shield_activations_list), delimiter=',')
np.savetxt('Results/adpt_encoder_env_interactions.csv', np.asarray(enc_env_interacts_list), delimiter=',')
np.savetxt('Results/adpt_encoder_neighbours_list.csv', np.asarray(enc_neighbours_list), delimiter=',')

enc_ddpg.actor.save_weights("Models/adpt_encoder_actor.h5")
enc_ddpg.target_actor.save_weights("Models/adpt_encoder_target_actor.h5")
enc_ddpg.critic.save_weights("Models/adpt_encoder_critic.h5")
enc_ddpg.target_critic.save_weights("Models/adpt_encoder_target_critic.h5")
enc_ddpg.base_encoder.save_weights("Models/adpt_encoder_base_encoder.h5")
enc_ddpg.base_decoder.save_weights("Models/adpt_encoder_base_decoder.h5")
joblib.dump(enc_ddpg.neighbours, 'Models/adpt_encoder_neighbours.joblib')
np.save('Models/adpt_y_array.npy', enc_ddpg.y)

