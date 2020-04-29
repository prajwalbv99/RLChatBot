# Training a goal oriented chatbot with Deep Q Network variants

This project builds upon the code base present in [GO-BOT] (https://github.com/maxbren/GO-Bot-DRL) , which is covered in [Training a Goal-Oriented Chatbot with Deep Reinforcement Learning] (https://towardsdatascience.com/training-a-goal-oriented-chatbot-with-deep-reinforcement-learning-part-i-introduction-and-dce3af21d383) 
The project implements various deep reinforcement learning algorithms for the task of a goal-oriented chatbot using a simple user simulator. The task chosen is that of movie tickets finder.

## Dependencies
- Python >= 3.5
- Keras >= 2.24 (Earlier versions probably work)
- numpy

## Code Structure
- Each algorithms is stored in a separate folder with their own train and test file. The code for the agent is present in the agent files (dqn_agent, drqn_agent, duellingQNetwork.py) etc.
- Training code is present in train.py 
- Testing Code is present in test.py 

## Training steps.
To train the agent from scratch run 'python train.py'.

To save the weights of the model during training, in the constants.json file, mention the path in save_weights_file_path
For example:
"save_weights_file_path": "weights/model.h5"

If you are planning to train from scratch, make sure that the "load_weights_file_path" in constants.json is empty. Otherwise, it loads the 
model with these weights.

The file constants.json contains all the hyper-parameters required in training .

## Testing steps
For testing run: 'python test.py'
While testing, replace "load_weights_file_path" with the path where you've saved the files. For example, in my case
"load_weights_file_path": "weights/model.h5"
Then run python test.py
For each episode in testing, you will see the reward in that particular episode. And after the testing is done for all the episodes, the
average reward and the average success rate will be displayed.

## Testing with actual user
You can test the agent by inputting your own actions as the user (instead of using a user sim) by setting "usersim” under “run” in 
constants.json to false. You input an action and a success indicator every step of an episode in console. The format for the action input
is: intent/inform slots/request slots.

Example action inputs:
-request/moviename: room, date: friday/starttime, city, theater
-inform/moviename: zootopia/
-request//starttime
-done//

In addition the console will ask for an indicator on whether the agent succeeded yet (other than after the initial action input of an episode). Allowed inputs are -1 for loss, 0 for no outcome yet, 1 for success.