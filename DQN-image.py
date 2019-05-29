import random
import gym
import numpy as np
from gym.wrappers import Monitor
from collections import deque
from keras import initializers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Lambda, Flatten
from keras.optimizers import Adam, RMSprop
from keras.callbacks import TensorBoard

class DQNAgent:
    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space
        self.memory = deque(maxlen=200000)
        # learning parameters
        self.gamma = 0.99 # discount rate/factor
        self.epsilon = 1.0 # exploration rate. Will be decreased over time
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.00025 # alpha

        self.model = self.build_model()

    def build_model(self):
        # Neural Network for Deep-Q Learning Model
        model = Sequential()
        # input format after preprocess, 4 frames of (X/2,Y/2)
        model.add(Lambda(lambda x: x/ 255.0, input_shape=(self.state_space[0]/2, self.state_space[1]/2, 4))) # keep values between [0,1]
        model.add(Conv2D(16, (8, 8), strides=(4, 4), activation='relu', kernel_initializer='glorot_normal'))
        model.add(Conv2D(32, (4, 4), strides=(2, 2), activation='relu', kernel_initializer='glorot_normal'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu', kernel_initializer='glorot_normal'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu', kernel_initializer='glorot_normal'))
        model.add(Dense(self.action_space))
        model.compile(loss='mse', optimizer=RMSprop(lr=self.learning_rate, epsilon=0.01))
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        action_values = self.model.predict(state[None,:,:,:])
        return np.argmax(action_values[0]) # o [0] é simplismente pelo modelo de dados usado pelo keras

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                # target prediction "should have predicted this"
                target = (reward + self.gamma * np.amax(target_network.predict(next_state[None,:,:,:])))
                
            # target for training, it will give predictions for all actions, but only the action taken will be changed to 'target'
            target_f = target_network.predict(state[None,:,:,:])
            target_f[0][action] = target

            self.model.fit(state[None,:,:,:], target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

#---------preprocessamento da imagem/estado
def grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)

def downsample(img):
    return img[::2,::2]

def preprocess(img):
    return grayscale(downsample(img))
#---------


if __name__ == '__main__':
    env = gym.make("Breakout-v0")
    env = Monitor(env, './videos/')

    # tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

    state_space = env.observation_space.shape
    action_space = env.action_space.n
    agent = DQNAgent(state_space, action_space)
    target_network = agent.build_model()
    agent.load("./save/breakout-dqn.h5")

    done = False
    batch_size = 32

    state = env.reset()
    state = preprocess(state)
    # state = state.reshape(state.shape, 1)
    frame_buffer_state = np.stack((state, state, state, state), axis=2)

    step = 0
    start_life = 0
    for episode in range(500000):

        if start_life == 0:
            start_life = 5
            state = env.reset()
            state = preprocess(state)
            frame_buffer_state = np.stack((state, state, state, state), axis=2)

        acc_reward = 0
        done = False

        while done != True:

            action = agent.act(frame_buffer_state)

            next_state, reward, done, info = env.step(action)
            next_state = preprocess(next_state)
            next_state = next_state.reshape(next_state.shape[0], next_state.shape[1], 1)
            frame_buffer_next_state = np.append(next_state, frame_buffer_state[:, :, :3], axis=2)

            if start_life > info['ale.lives']:
                done = True
                start_life = info['ale.lives']

            agent.remember(frame_buffer_state, action, reward, frame_buffer_next_state, done)
            acc_reward += reward
            step += 1
            frame_buffer_state = frame_buffer_next_state

            if start_life > info['ale.lives']:
                frame_buffer_state = np.stack((next_state, next_state, next_state, next_state), axis=2) 
                #next state wont be influenced by previous ones after losing a life

            if step % 15000 == 0:
                # print(target_network.get_weights())
                # print(agent.model.get_weights())
                target_network.set_weights(agent.model.get_weights())
            if done:
                # env.render()
                print("Episode {} Accumulated_Reward {} Epsilon {:.4} Step {}"
                .format(episode, acc_reward, agent.epsilon, step))

                if episode % 500 == 0:
                    agent.save("./save/breakout-dqn.h5")

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
