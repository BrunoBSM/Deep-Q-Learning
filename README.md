# Deep-Q-Learning
Deep Q-Learning implementation with Keras and Tensorflow on OpenAI-Gym environments

Based on the [DQN paper](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf) 

### Requirements
* Python 3
* [OpenAI-Gym](https://github.com/openai/gym)
* Keras
* TensorFlow

### Usage
This implementation is of a Deep Q-Learning for image input, in order to change the Environment, make sure it is compatible. 
This means it must have an **image** as *state/observation* and a **discrete** *action space* such as all ATARI 2600 environments. 
In order to choose a different Environment, simply change the following `"Breakout-v0"` to the name of the 
environment you'd like to test.
```
env = gym.make("Breakout-v0")
```
To change the learning parameters and the size for the Replay Memory `maxlen`, change these lines: 
```
self.memory = deque(maxlen=200000)
# learning parameters
self.gamma = 0.99 # discount rate/factor
self.epsilon = 1.0 # exploration rate. Will be decreased over time
self.epsilon_min = 0.1
self.epsilon_decay = 0.9995
self.learning_rate = 0.00025 # alpha
```
For learning purposes I encourage you to test different parameters and compare performances.

### Useful Links
Tutorials and explainations for DQN 

https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26

https://towardsdatascience.com/welcome-to-deep-reinforcement-learning-part-1-dqn-c3cab4d41b6b

https://keon.io/deep-q-learning/

[Vlad Mnih's lecture on DQN](https://www.youtube.com/watch?v=fevMOp5TDQs)
