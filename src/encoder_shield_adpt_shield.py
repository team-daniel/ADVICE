import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import plotly.express as px
import random
from sklearn.neighbors import NearestNeighbors
import gc
from sklearn.metrics import confusion_matrix
from statistics import mode
import seaborn as sns
from tqdm import tqdm

from noise import OUActionNoise
from buffer import Buffer

class Encoder_DDPG_Adpt_Shield():
    def __init__(self, env, total_episodes, state_size, action_size, action_high, action_low):
        # total episodes
        self.total_episodes = total_episodes
        
        # environment
        self.env = env
        
        # rl params
        self.gamma = 0.95
        self.tau = 0.005
        
        # environment params
        self.state_size = state_size
        self.action_size = action_size
        self.action_high = action_high
        self.action_low = action_low
        self.action_range = action_high - action_low
        
        # initialise actor and critic
        self.actor_lr = 0.002
        self.actor = self.get_actor()
        self.target_actor = self.get_actor()
        self.target_actor.set_weights(self.actor.get_weights())
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr)
        
        self.critic_lr = 0.001
        self.critic = self.get_critic()
        self.target_critic = self.get_critic()
        self.target_critic.set_weights(self.critic.get_weights())
        self.critic_optimizer = tf.keras.optimizers.Adam(self.critic_lr)
        
        # initialise noise and buffer objects
        self.noise_object = OUActionNoise(mean=np.zeros(1), std_deviation=float(0.2) * np.ones(1))
        
        self.buffer = Buffer(self.state_size, self.action_size, 250_000, 64, self.actor, self.target_actor, 
                             self.critic, self.target_critic, self.actor_optimizer, self.critic_optimizer, self.gamma)
        
        # initialise the autoencoder model
        self.base_encoder = self.get_encoder()
        self.base_decoder = self.get_decoder()
        
        self.contrastive_autoencoder = self.get_contrastive_autoencoder()
        self.autoencoder_optimizer = tf.keras.optimizers.Nadam(0.01)
        self.contrastive_autoencoder.compile(
            loss=['mse', 'mse', self.contrastive_loss],  # mse for the reconstruction loss, contrastive_loss for the embedding
            loss_weights=[1, 1, 1.25], # Adjust these weights as necessary
            optimizer=self.autoencoder_optimizer
        )
        
        # intial value of k
        self.neighbours_count = 3
        
    # create the actor model
    def get_actor(self):
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        inputs = layers.Input(shape=(self.state_size,))
        x = layers.Dense(256, activation="relu")(inputs)
        x = layers.Dense(256, activation="relu")(x)
        outputs = layers.Dense(self.action_size, activation="tanh", kernel_initializer=last_init)(x)

        outputs = outputs * self.action_high
        model = tf.keras.Model(inputs, outputs)
        return model
    
    # create the critic model
    def get_critic(self):
        state_input = layers.Input(shape=(self.state_size))
        state_out = layers.Dense(16, activation="relu")(state_input)
        state_out = layers.Dense(32, activation="relu")(state_out)
        
        # Action as input
        action_input = layers.Input(shape=(self.action_size))
        action_out = layers.Dense(32, activation="relu")(action_input)
        
        # Both are passed through seperate layer before concatenating
        concat = layers.Concatenate()([state_out, action_out])
        
        out = layers.Dense(256, activation="relu")(concat)
        out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(1)(out)
        
        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], outputs) 
        return model

    # serive the policy's action from a given state and add appropriate noise
    def policy(self, state):
        sampled_actions = tf.squeeze(self.actor(state))
        noise = self.noise_object()     
        sampled_actions = sampled_actions.numpy() + noise      
        legal_action = np.clip(sampled_actions, self.action_low, self.action_high)
    
        return np.squeeze(legal_action)
    
    @tf.function
    def update_target(self, target_weights, weights):
        for a, b in zip(target_weights, weights):
            a.assign(b * self.tau + a * (1 - self.tau)) 
            
    # calculate euclian distance between two vectors.
    def euclidean_distance(self, vects):
        x, y = vects
        sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
        return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))

    # calculate the contrastive loss function according to Equation 1.
    def contrastive_loss(self, y_true, y_pred, margin=1.0):
        square_pred = tf.square(y_pred)
        margin_square = tf.square(tf.maximum(margin - y_pred, 0))
        return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)

    # create the encoder model
    def get_encoder(self):
        inputs = layers.Input(shape=(34,))

        x = layers.Dense(512, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1e-3))(inputs)
        x = layers.LeakyReLU()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        latent = layers.Dense(2, activation='linear')(x)
        model = tf.keras.Model(inputs, latent, name='encoder')
        return model
    
    # create the decoder model
    def get_decoder(self):
        inputs = layers.Input(shape=(2))
        x = layers.Dense(512, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1e-3))(inputs)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.3)(x)
        
        latent = layers.Dense(34, activation='sigmoid')(x)
        model = tf.keras.Model(inputs, latent, name='decoder')
        return model
    
    # create the siamese autoencoder model for training
    def get_contrastive_autoencoder(self):
        input_a = layers.Input(shape=(34,))
        input_b = layers.Input(shape=(34,))

        base_encoder_a = self.base_encoder(input_a)
        base_encoder_b = self.base_encoder(input_b)
        decoded_a = self.base_decoder(base_encoder_a)
        decoded_b = self.base_decoder(base_encoder_b)

        merge_layer = layers.Lambda(self.euclidean_distance, name='lambda')([base_encoder_a, base_encoder_b])
        model = tf.keras.Model([input_a, input_b], outputs=[decoded_a, decoded_b, merge_layer], name='autoencoder')
        return model
    
    # generate pairs of data from the data collected in intial training
    def make_pairs(self, x_train, y_train):
        safe_idxs = [i for i, label in enumerate(y_train) if label == 1]
        unsafe_idxs = [i for i, label in enumerate(y_train) if label == 0]
        
        pairs = []
        labels = []
        
        for idx in range(len(x_train)):
            x1 = x_train[idx]
            safe = y_train[idx]
            
            if safe:
                idx2 = random.choice(safe_idxs)
            else:
                idx2 = random.choice(unsafe_idxs)
            
            x2 = x_train[idx2]
            pairs += [[x1, x2]]
            labels += [1]
            
            idx2 = random.choice(unsafe_idxs if safe else safe_idxs)
            x2 = x_train[idx2]
            pairs += [[x1, x2]]
            labels += [0]
        
        return np.array(pairs), np.array(labels).astype("float32")
    
    # train the autoencoder
    def train_autoencoder(self):
        safe_obs = self.safe_observations
        unsafe_obs = self.unsafe_observations
        
        safe_y = np.ones(len(safe_obs))
        unsafe_y = np.zeros(len(unsafe_obs))
        
        # preprocess data
        safe_obs = np.array([np.concatenate([state, action]) for state, action in safe_obs])
        unsafe_obs = np.array([np.concatenate([state, action]) for state, action in unsafe_obs])
        
        # combine data
        x_train = np.vstack([safe_obs, unsafe_obs])
        y_train = np.hstack([safe_y, unsafe_y])
        
        pairs_train, labels_train = self.make_pairs(x_train, y_train)
        
        x_train_1 = pairs_train[:, 0]
        x_train_2 = pairs_train[:, 1]
        
        x_train_1 = tf.convert_to_tensor(x_train_1, dtype=tf.float32)
        x_train_2 = tf.convert_to_tensor(x_train_2, dtype=tf.float32)
        labels_train = tf.convert_to_tensor(labels_train, dtype=tf.float32)
        
        np.savetxt("x_train_1.csv", x_train_1, delimiter=",")
        np.savetxt("x_train_2.csv", x_train_2, delimiter=",")
        np.savetxt("labels_train.csv", labels_train, delimiter=",")

        # train encoder
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6, verbose=0)
        
        history = self.contrastive_autoencoder.fit(
            [x_train_1, x_train_2],  # Inputs for the contrastive part
            [x_train_1, x_train_2, labels_train],  # Targets for reconstruction and contrastive loss
            batch_size=32,
            epochs=500,
            validation_split=0.2,
            callbacks=[early_stop, reduce_lr],
            shuffle=True,
            verbose=2
        )
        
    # get data for fitting the neighbours model
    def get_testable_data(self):
        safe_obs = self.safe_observations
        unsafe_obs = self.unsafe_observations
        
        safe_obs = np.array([np.reshape(x, (2,)) for x in safe_obs])
        unsafe_obs = np.array([np.reshape(x, (2,)) for x in unsafe_obs])
        safe_obs = np.array([np.array(x).flatten() for x in safe_obs])
        unsafe_obs = np.array([np.array(x).flatten() for x in unsafe_obs])
        safe_obs = np.array([np.concatenate([state, action]) for state, action in safe_obs])
        unsafe_obs = np.array([np.concatenate([state, action]) for state, action in unsafe_obs])
        
        y_safe = np.asarray(["safe"] * len(safe_obs))
        y_unsafe = np.asarray(["unsafe"] * len(unsafe_obs))
        
        x = np.vstack([safe_obs, unsafe_obs])
        y = np.hstack([y_safe, y_unsafe])
        
        np.savetxt("x_plot.csv", x, delimiter=",")
        np.savetxt("y_plot.csv", y, delimiter=",", fmt="%s")
        
        return x, y

    # test the autoencoder, this works if you have some test data from a seperate run but doesn't affect the model. (Just a sanity check)
    def test_autoencoder(self, x, y, episode):
        embeddings = self.base_encoder.predict(x)
        
        # plot embeddings
        label_numeric = np.where(np.array(y) == 'safe', 0, 1)
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1], c=label_numeric, cmap='jet', alpha=0.6)
        cbar = plt.colorbar(scatter, ticks=[0, 1])
        cbar.set_label('Labels')
        cbar.set_ticklabels(['safe', 'unsafe'])
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        plt.title("2D Visualization of Train Embeddings")
        plt.show()
        df = pd.DataFrame(embeddings, columns=['X', 'Y'])
        df['label'] = y
        fig = px.scatter(df, x='X', y='Y', color='label',
                         color_discrete_map={'safe': 'blue', 'unsafe': 'red'},
                         labels={'X': 'X Label', 'Y': 'Y Label'},
                         opacity=0.7)
        path_name = "plot_" + str(episode) + ".html"
        fig.write_html(path_name)
        
        # get neighbours
        self.neighbours = NearestNeighbors(n_neighbors=5)
        self.neighbours.fit(embeddings)
        
        # get test dataset
        x_test = np.loadtxt('Data/x_test.csv', delimiter=',')
        y_test = np.loadtxt('Data/y_test.csv', delimiter=',', dtype=str)
        
        test_embeddings = self.base_encoder.predict(x_test)
        
        label_numeric = np.where(np.array(y_test) == 'safe', 0, 1)       
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(test_embeddings[:, 0], test_embeddings[:, 1], c=label_numeric, cmap='jet', alpha=0.6)
        cbar = plt.colorbar(scatter, ticks=[0, 1])
        cbar.set_label('Labels')
        cbar.set_ticklabels(['safe', 'unsafe'])        
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        plt.title("2D Visualization of Test Embeddings")
        plt.show()      
        # 2d plot using Plotly
        df = pd.DataFrame(test_embeddings, columns=['X', 'Y'])
        df['label'] = y_test
        
        fig = px.scatter(df, x='X', y='Y', color='label',
                         color_discrete_map={'safe': 'blue', 'unsafe': 'red'},
                         labels={'X': 'X Label', 'Y': 'Y Label'},
                         opacity=0.7)
        
        fig.write_html("2d_plot.html")
        
        # Function to predict the class of a new data point
        def predict_class(data_point):
            new_point_embedding = self.base_encoder.predict(np.array([data_point]), verbose=0)
            nearest_neighbor_index = self.neighbours.kneighbors(new_point_embedding, return_distance=False)
            predicted_classes = y[nearest_neighbor_index].flatten()
            mode_result = mode(predicted_classes)
            return mode_result
        
        # Predict classes for test data
        predicted_classes = np.array([predict_class(point) for point in tqdm(x_test)])
        
        # Calculate accuracy
        accuracy = np.mean(predicted_classes == y_test)
        print("")
        print("==================================")
        print("Accuracy: {:.2f}%".format(accuracy * 100))
        
        unique_labels = np.unique(np.concatenate([predicted_classes, y_test]))
        
        # Compute the confusion matrix with explicit labels
        cm = confusion_matrix(y_test, predicted_classes, labels=unique_labels)
        
        # Plot the confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig("conf_matrix.png")
        plt.show()
        
    # the adaptive nearest neighbours module
    def update_neighbors(self, recent_episodes=10, current_window=2):
        # Calculate the differences between consecutive entries (the incremental violations per episode)
        recent_differences = [self.metrics['safety_violations'][i] - self.metrics['safety_violations'][i - 1] for i in range(-recent_episodes, 0)]
        moving_average = sum(recent_differences) / recent_episodes
        moving_std_dev = (sum([(x - moving_average) ** 2 for x in recent_differences]) / recent_episodes) ** 0.5
        
        # current rate of safety violations
        current_rate_window = [self.metrics['safety_violations'][i] - self.metrics['safety_violations'][i - 1] for i in range(-current_window, 0)]
        current_rate = sum(current_rate_window) / current_window

        # Adjust neighbours_count accordingly
        if current_rate > moving_average + moving_std_dev and self.neighbours_count < 5:
            self.neighbours_count += 1
        elif current_rate < moving_average and self.neighbours_count > 3:
            self.neighbours_count -= 1
        
    # the ADVICE shield
    def shield(self, current_state, original_action, step_size=1.0):
        # Process a batch of states and actions to generate embeddings more efficiently
        def get_embeddings(states, actions):
            obs = np.hstack((states, actions))  # Stack states and actions horizontally
            return self.base_encoder.predict(obs, verbose=0)
    
        # Process safety check in batches and only keep safe actions
        def is_safe_action_batch(states, actions):
            safe_actions = []
            embeddings = get_embeddings(states, actions)
            nearest_neighbours_indices = self.neighbours.kneighbors(embeddings, return_distance=False)
            predicted_classes = self.y[nearest_neighbours_indices]
            safe_counts = np.sum(predicted_classes == 'safe', axis=1)
            safe_indices = np.where(safe_counts >= self.neighbours_count)[0]  # Indices of safe actions
            safe_actions = actions[safe_indices]
            return safe_actions
    
        # Generate possible actions on the fly, without storing them all in memory
        def generate_possible_actions(step_size):
            for x in np.arange(-1, 1 + step_size, step_size):
                for y in np.arange(-1, 1 + step_size, step_size):
                    yield np.array([x, y])
                    
        # Check if the original action is safe
        original_embedding = get_embeddings(np.array([current_state]), np.array([original_action]))
        nearest_neighbours_indices = self.neighbours.kneighbors(original_embedding, return_distance=False)
        predicted_classes = self.y[nearest_neighbours_indices].flatten()
        safe_count = np.sum(predicted_classes == 'safe')
        if safe_count >= self.neighbours_count:
            return original_action
    
        # Convert generator to array for batch processing
        possible_actions = np.array(list(generate_possible_actions(step_size)))
        
        # Create repeated states array for batch processing
        repeated_states = np.repeat(current_state[np.newaxis, :], len(possible_actions), axis=0)
    
        # Check all actions for safety in a batch
        safe_actions = is_safe_action_batch(repeated_states, possible_actions)
    
        # Clear memory of large objects that are no longer needed
        del possible_actions, repeated_states
        gc.collect()
    
        # Evaluate safe actions using the critic and select the best one
        if len(safe_actions) > 0:
            critic_values = self.critic.predict([np.tile(current_state, (len(safe_actions), 1)), safe_actions], verbose=0)
            max_index = np.argmax(critic_values)
            return safe_actions[max_index]
        else:
            # If no safe actions were found, return the fallback action
            return -original_action
            
    # the RL loop
    def train(self, encoder_train_every):
        print("Beginning Training (Adaptive Encoder shield)...")
        print("===================================================")
    
        # Initialize lists to store training metrics
        self.metrics = {
            "ep_reward": [],
            "avg_reward": [],
            "safety_violations": [],
            "goal_reaches": [],
            "shield_activations": [],
            "env_interacts": [],
            "neighbours_log" : []
        }
        
        tot_safety_violations = 0
        tot_goal_reaches = 0
        tot_env_interacts = 0
        tot_shield_activations = 0
    
        self.unsafe_observations = []
        self.safe_observations = []
        trained = False
    
        for episode in range(self.total_episodes):
            prev_state, _ = self.env.reset()
            prev_state = prev_state[-32:]
            episode_reward, episode_shield_activations = 0, 0
    
            for step in itertools.count():
                action = self.policy(tf.expand_dims(tf.convert_to_tensor(prev_state), 0))
    
                # Apply shielding if trained
                if trained:
                    orig_action = action
                    action = self.shield(prev_state, action)
                    
                    # check if action was consider safe or unsafe
                    if not np.array_equal(orig_action, action):
                        # penalize unsafe action
                        self.buffer.record((prev_state, orig_action, -0.1, prev_state))
                        episode_shield_activations += 1
                        tot_shield_activations += 1
    
                new_state, reward, cost, terminated, truncated, info = self.env.step(action)
                new_state = new_state[-32:]
                reward = reward - cost
                
                # add to buffers
                self.buffer.record((prev_state, action, reward, new_state))
                episode_reward += reward
    
                # Learning step
                self.buffer.learn()
                self.update_target(self.target_actor.variables, self.actor.variables)
                self.update_target(self.target_critic.variables, self.critic.variables)
    
                # Check for goal reach
                if self.env.task.goal_achieved:
                    print("Reached goal...")
                    tot_goal_reaches += 1
                    self.safe_observations.append((prev_state, action))
                    break
                
                # check for crash
                if cost > 0:
                    print("Crashed...")
                    tot_safety_violations += 1
                    self.unsafe_observations.append((prev_state, action))
                    break
    
                # otherwise max steps were reached
                if terminated or truncated:
                    print("Episode too long...")
                    break
                
                # save the init step 
                if (step == 0):
                   self.safe_observations.append((prev_state, action))
    
                prev_state = new_state
                tot_env_interacts += 1
    
            # Store episode metrics
            self.metrics["ep_reward"].append(episode_reward)
            self.metrics["shield_activations"].append(tot_shield_activations)
            self.metrics["avg_reward"].append(np.mean(self.metrics["ep_reward"][-40:]))
            self.metrics["safety_violations"].append(tot_safety_violations)
            self.metrics["env_interacts"].append(tot_env_interacts)
            self.metrics["goal_reaches"].append(tot_goal_reaches)
            self.metrics["neighbours_log"].append(self.neighbours_count)
            
            if trained:
                # update neighbours value dynamically
                self.update_neighbors()
    
            # Output some results
            print(f"Episode: {episode} -> Avg Reward: {self.metrics['avg_reward'][-1]}")
            if trained:
                print(f"Episode shield activations: {episode_shield_activations}, neighbours: {self.neighbours_count}")
            print("===================================================")
    
            # Train encoder
            if (episode % encoder_train_every == 0) and (episode != 0):
                print("Training the encoder...")
                print("===================================================")
                self.train_autoencoder()
                x, self.y = self.get_testable_data()
                self.test_autoencoder(x, self.y, episode)
                self.unsafe_observations.clear()
                self.safe_observations.clear()
                trained = True
                gc.collect()
    
        print(f"Length of unsafe obs list: {len(self.unsafe_observations)}")
        return (self.metrics["avg_reward"], self.metrics["ep_reward"], self.metrics["safety_violations"],
                self.metrics["goal_reaches"], self.metrics["shield_activations"], self.metrics["env_interacts"], self.metrics["neighbours_log"])
    
    
    
    
    
    
    
    
    
    