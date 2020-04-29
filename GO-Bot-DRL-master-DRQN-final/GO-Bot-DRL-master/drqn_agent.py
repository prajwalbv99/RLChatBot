from keras.models import Sequential
from keras.layers import Dense,TimeDistributed,LSTM, Flatten
from keras.optimizers import Adam
import random, copy
import numpy as np
from dialogue_config import rule_requests, agent_actions
import re


# Some of the code based off of https://jaromiru.com/2016/09/27/lets-make-a-dqn-theory/
# Note: In original paper's code the epsilon is not annealed and annealing is not implemented in this code either



class DRQNAgent:
    """The DRQN agent that interacts with the user."""

    def __init__(self, state_size, constants):
        """
        The constructor of DRQNAgent.

        The constructor of DRQNAgent which saves constants, sets up neural network graphs, etc.

        Parameters:
            state_size (int): The state representation size or length of numpy array
            constants (dict): Loaded constants in dict

        """
        
        
        
        
        
        self.C = constants['agent']
        self.memory = []
        self.memory_index = 0
        self.max_memory_size = self.C['max_mem_size']
        self.eps = self.C['epsilon_init']
        self.vanilla = self.C['vanilla']
        self.lr = self.C['learning_rate']
        self.gamma = self.C['gamma']
        self.batch_size = self.C['batch_size']
        self.hidden_size = self.C['dqn_hidden_size']
        
        
        self.time_step = self.C['time_step']                     # adding time step for drqn

        self.load_weights_file_path = self.C['load_weights_file_path']
        self.save_weights_file_path = self.C['save_weights_file_path']
        print(self.load_weights_file_path)

        if self.max_memory_size < self.batch_size:
            raise ValueError('Max memory size must be at least as great as batch size!')

        self.state_size = state_size
        self.possible_actions = agent_actions
        self.num_actions = len(self.possible_actions)

        self.rule_request_set = rule_requests
        self.beh_model = self._build_model()
        self.tar_model = self._build_model()
        

        self._load_weights()

        self.reset()

    def _build_model(self):
        """Builds and returns model/graph of neural network."""

        model = Sequential()
        model.add(TimeDistributed(Dense(self.hidden_size, activation='relu'), input_shape=(self.time_step, self.state_size)))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(64, activation='relu', return_sequences=False))
        model.add(Dense(self.num_actions, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.lr))
        return model

    def reset(self):
        """Resets the rule-based variables."""

        self.rule_current_slot_index = 0
        self.rule_phase = 'not done'

    def get_action(self, state, use_rule=False):
        """
        Returns the action of the agent given a state.

        Gets the action of the agent given the current state. Either the rule-based policy or the neural networks are
        used to respond.

        Parameters:
            state (numpy.array): The database with format dict(long: dict)
            use_rule (bool): Indicates whether or not to use the rule-based policy, which depends on if this was called
                             in warmup or training. Default: False

        Returns:
            int: The index of the action in the possible actions
            dict: The action/response itself

        """

        if self.eps > random.random():
            index = random.randint(0, self.num_actions - 1)
            action = self._map_index_to_action(index)
            return index, action
        else:
            if use_rule:
                return self._rule_action()
            else:
                return self._dqn_action(state)

    def _rule_action(self):
        """
        Returns a rule-based policy action.

        Selects the next action of a simple rule-based policy.

        Returns:
            int: The index of the action in the possible actions
            dict: The action/response itself

        """

        if self.rule_current_slot_index < len(self.rule_request_set):
            slot = self.rule_request_set[self.rule_current_slot_index]
            self.rule_current_slot_index += 1
            rule_response = {'intent': 'request', 'inform_slots': {}, 'request_slots': {slot: 'UNK'}}
        elif self.rule_phase == 'not done':
            rule_response = {'intent': 'match_found', 'inform_slots': {}, 'request_slots': {}}
            self.rule_phase = 'done'
        elif self.rule_phase == 'done':
            rule_response = {'intent': 'done', 'inform_slots': {}, 'request_slots': {}}
        else:
            raise Exception('Should not have reached this clause')

        index = self._map_action_to_index(rule_response)
        return index, rule_response

    def _map_action_to_index(self, response):
        """
        Maps an action to an index from possible actions.

        Parameters:
            response (dict)

        Returns:
            int
        """

        for (i, action) in enumerate(self.possible_actions):
            if response == action:
                return i
        raise ValueError('Response: {} not found in possible actions'.format(response))

    def _dqn_action(self, state):
        """
        Returns a behavior model output given a state.

        Parameters:
            state (numpy.array)

        Returns:
            int: The index of the action in the possible actions
            dict: The action/response itself
        """
        
        index = np.argmax(self._dqn_predict_one(state))
        action = self._map_index_to_action(index)
        return index, action

    def _map_index_to_action(self, index):
        """
        Maps an index to an action in possible actions.

        Parameters:
            index (int)

        Returns:
            dict
        """

        for (i, action) in enumerate(self.possible_actions):
            if index == i:
                return copy.deepcopy(action)
        raise ValueError('Index: {} not in range of possible actions'.format(index))

    def _dqn_predict_one(self, state, target=False):
        """
        Returns a model prediction given a state.

        Parameters:
            state (numpy.array)
            target (bool)

        Returns:
            numpy.array
        """
        #print(state.shape)
        return self._dqn_predict(state.reshape(1, self.time_step, self.state_size), target=target).flatten()

    def _dqn_predict(self, states, target=False):
        """
        Returns a model prediction given an array of states.

        Parameters:
            states (numpy.array)
            target (bool)

        Returns:
            numpy.array
        """
        #print(states.shape)
        if target:
            return self.tar_model.predict(states)
        else:
            return self.beh_model.predict(states)

    def add_experience(self, local_memory):
        """
        Adds an experience tuple made of the parameters to the memory.

        Parameters:
            state (numpy.array)
            action (int)
            reward (int)
            next_state (numpy.array)
            done (bool)

        """

        if len(self.memory) < self.max_memory_size:
            self.memory.append(None)
        self.memory[self.memory_index] = local_memory
        self.memory_index = (self.memory_index + 1) % self.max_memory_size

    def empty_memory(self):
        """Empties the memory and resets the memory index."""

        self.memory = []
        self.memory_index = 0

    def is_memory_full(self):
        """Returns true if the memory is full."""

        return len(self.memory) == self.max_memory_size

    def get_batch(self):
        sampled_epsiodes = random.sample(self.memory, self.batch_size)
        batch = []
        for episode in sampled_epsiodes:
            if len(episode) >= self.time_step:
                point = np.random.randint(0, len(episode)+1-self.time_step)
                batch.append(episode[point:point+self.time_step])
        return batch

    def train(self):
        """
        Trains the agent by improving the behavior model given the memory tuples.

        Takes batches of memories from the memory pool and processing them. The processing takes the tuples and stacks
        them in the correct format for the neural network and calculates the Bellman equation for Q-Learning.

        """

        # Calc. num of batches to run
        num_batches = len(self.memory)*10 // self.batch_size
        for i in range(num_batches):
            batch = self.get_batch()
            current_states = []        #tuple takes the format state, agent_action_index, reward, next_state, done
            actions = []
            next_states = []
            rewards = []
            
            for b in batch:
                cs = []
                act = []
                rew = []
                nex = []
                for item in b:
                    #print(np.array(item).shape)
                    cs.append(item[0])
                    act.append(item[1])
                    rew.append(item[2])
                    nex.append(item[3])
                    #print(np.array(item[0]).shape)
                #print(len(cs))
                #print(np.array(cs).shape)
                
                current_states.append(cs)
                actions.append(act)
                next_states.append(nex)
                rewards.append(rew)
            
            current_states = np.array(current_states)
            next_states = np.array(next_states)
            #print("Current State", current_states.shape)
            #states = np.array([sample[0] for sample in batch])
            #next_states = np.array([sample[3] for sample in batch])

            #assert states.shape == (self.batch_size, self.state_size), 'States Shape: {}'.format(states.shape)
            #assert next_states.shape == states.shape

            beh_state_preds = self._dqn_predict(current_states)  # This take inputs current states i.e ex: [[1,2,3],[4,5,6]] and return two actions
            if not self.vanilla:
                beh_next_states_preds = self._dqn_predict(next_states)  # For indexing for DDQN
            tar_next_state_preds = self._dqn_predict(next_states, target=True)  # For target value for DQN (& DDQN)

            inputs = np.zeros((self.batch_size, self.time_step, self.state_size))
            targets = np.zeros((self.batch_size, self.num_actions))

            for i, item in enumerate(batch):
                t = beh_state_preds[i]                  #t will be an array of the number of actions
                (s,a,r,s,d) = item[-1]
                if not self.vanilla:
                    t[a] = r + self.gamma * tar_next_state_preds[i][np.argmax(beh_next_states_preds[i])] * (not d)
                else:
                    t[a] = r + self.gamma * np.amax(tar_next_state_preds[i]) * (not d)
                for j,item_1 in enumerate(item):
                    inputs[i][j] = item_1[0]
                #inputs[i] = s
                targets[i] = t

            self.beh_model.fit(inputs, targets, epochs=1, verbose=0)

    def copy(self):
        """Copies the behavior model's weights into the target model's weights."""

        self.tar_model.set_weights(self.beh_model.get_weights())

    def save_weights(self):
        """Saves the weights of both models in two h5 files."""

        if not self.save_weights_file_path:
            return
        beh_save_file_path = re.sub(r'\.h5', r'_beh.h5', self.save_weights_file_path)
        self.beh_model.save_weights(beh_save_file_path)
        tar_save_file_path = re.sub(r'\.h5', r'_tar.h5', self.save_weights_file_path)
        self.tar_model.save_weights(tar_save_file_path)

    def _load_weights(self):
        """Loads the weights of both models from two h5 files."""

        if not self.load_weights_file_path:
            return
        beh_load_file_path = re.sub(r'\.h5', r'_beh.h5', self.load_weights_file_path)
        self.beh_model.load_weights(beh_load_file_path)
        tar_load_file_path = re.sub(r'\.h5', r'_tar.h5', self.load_weights_file_path)
        self.tar_model.load_weights(tar_load_file_path)
