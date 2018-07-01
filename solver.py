#This code is inspired from Udemy's Deep Reinforcement Learning Course
#https://www.udemy.com/deep-reinforcement-learning-in-python/


import numpy as np
from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import Adagrad, SGD
import tensorflow as tf
import rubik as env
import matplotlib.pyplot as plt


BUFFER_SIZE = 10000
TOTAL_ITERATIONS = 0
ITERATIONS_COPY = 20
GAMMA = 0.99
BATCH_SIZE = 32
OUTPUTS = 12


class QModel: #Action Value Model
    def __init__(self, inputs, outputs, hidden_layers, op):
        self.td = TDModel(inputs, outputs, hidden_layers)
        inp = Input(shape = (inputs,))
        x = inp
        for layer in hidden_layers:
            x = Dense(layer, activation = 'tanh')(x)
        x = Dense(outputs)(x)
        self.model = Model(inputs = inp, outputs = x)
        
        self.model.compile(loss = self.custom_loss, optimizer = op)
        self.model.summary()
        
    def predict(self, X): #Predict Scores of each action
        return self.model.predict(X)

    def sample(self, X, eps): #Epsilon Greedy
        if np.random.random() < eps:
            return np.random.choice(OUTPUTS)
        X = np.atleast_2d(X)
        Y = self.predict(X)[0]
        return np.argmax(Y)

    def custom_loss(self, y_true, y_pred): #Loss functon
        y_true = tf.transpose(y_true)
        a = tf.cast(y_true[0], tf.int32) #Actions
        target = y_true[1] #Target Value
        a = tf.one_hot(a, OUTPUTS)
        Qs_a = tf.reduce_sum(tf.multiply(y_pred, a), axis = 1) #Q(s,a)
        return tf.reduce_sum(tf.square(target - Qs_a))

    def partial_fit(self, batch):
        s = [item[0] for item in batch] #State
        #precaution against casting in custom_loss
        a = [item[1] + 0.1 for item in batch] #Action done
        r = [item[2] for item in batch] #Reward
        s_ = np.array([item[3] for item in batch]) #Next state
        not_done = [1.0 - item[4] for item in batch] #Solved or not
        Qs_a_ = np.max(self.td.predict(s_), axis = 1) #max_a(Q(s',a))
        target = np.array(r) + GAMMA*np.array(not_done) * Qs_a_ #target
        #we need same dimensions of y_pred and y_true
        y = [[a[i], target[i],0,0,0,0,0,0,0,0,0,0] for i in range(len(batch))]
        x = np.array(s)
        y = np.array(y)
        self.model.fit(x, y, verbose = 0, shuffle = False)

    def copy_model(self):
        self.td.model.set_weights(self.model.get_weights()) #Copying Model

    def save_model(self, name):
        print("Saving Model.....")
        self.model.save_weights(name)


class TDModel:
    def __init__(self, inputs, outputs, hidden_layers):
        inp = Input(shape = (inputs,))
        x = inp
        for layer in hidden_layers:
            x = Dense(layer, activation = 'tanh')(x)
        x = Dense(outputs)(x)
        self.model = Model(inputs = inp, outputs = x)
        self.model.compile(loss = 'mse', optimizer = 'sgd') #This does not matter

    def predict(self, X):
        return self.model.predict(X)


def scramble(level = 1): 
    #Deciding distribution of number of times cube is..
    #..scrambled at different complexities
    if level == 1:
        return np.random.choice(1) + 1
    if level == 2:
        return np.random.choice(2, p = [0.4,0.6]) + 1
    if level == 4:
        return np.random.choice(4, p = [0.1,0.1,0.4,0.4]) + 1
    if level == 6:
        return np.random.choice(6, p = [0.1,0.1,0.2,0.2,0.2,0.2]) + 1 
        

def gen_buffer(rubik, buffer, min_length):
    #Generating Initial Buffer
    rubik.reset()
    complexity = scramble()
    rubik.scramble(complexity)
    state = rubik.get_state()
    while len(buffer) < min_length:
        action = np.random.choice(OUTPUTS) #Random Action Decided
        rubik.move(int(action/2), action%2) #Action performed
        state_ = rubik.get_state() #Next state 
        reward = rubik.reward() #Reward
        done = rubik.is_solved() 
        buffer.append([state, action, reward, state_, done])
        if done == True:
            print("DONE")
            complexity = scramble()
            rubik.scramble(complexity)
            state = rubik.get_state()
        state = state_


def play_one(rubik, qmodel, buffer, level, eps = 0.0, train = True):
    #Playing one episode
    global TOTAL_ITERATIONS
    global ITERATIONS_COPY
    rubik.reset()
    if not train: #In Testing phase all difficulties are equiprobable
        complexity = 1 + np.random.choice(level)
    else:
        complexity = scramble(level)
    rubik.scramble(complexity) #Scrambling Cube
    state = rubik.get_state()
    done = False
    iter = 0
    R = 0
    while not done and iter < 30:
        action = qmodel.sample(state, eps) #Epsilon Greedy
        rubik.move(int(action/2), action%2) #Action performed
        state_ = rubik.get_state() #Next state
        reward = rubik.reward() #Reward
        done = rubik.is_solved() #Is Cube Solved
        if train:
            if TOTAL_ITERATIONS % ITERATIONS_COPY == 0: #Copying Weights
                qmodel.copy_model()
            buffer.append([state, action, reward, state_, done]) #Buffer
            if len(buffer) > BUFFER_SIZE:
                buffer = buffer[1:]
            batch_indices = np.random.choice(len(buffer), BATCH_SIZE) #Random Sample from buffer
            batch = [buffer[i] for i in batch_indices]
            qmodel.partial_fit(batch)
            TOTAL_ITERATIONS += 1
            if done:
                print("DONE")
        state = state_
        iter += 1
        R += reward #Total Rewards
    return iter, R, complexity

        
def main(
        rubik, #Environment
        architecture, #Architecture
        iter, #Total iterations
        level, #complexity 
        buffer, #Replay Buffer
        max_eps, #Iterations upto which epsilon will decay
        path_qmodel = "", #Path to load weights from
        optimizer = Adagrad, #Optimizer
        lr = 0.01 #Learning Rate
        ):
    op = optimizer(lr)
    qmodel = QModel(54*6, 12, architecture, op)
    if path_qmodel != "":
        qmodel.model.load_weights(path_qmodel)
    length = [] #Episode Lengths 
    score = [] #Total reward obtained
    length_100 = [] #Last 100 episodes average length
    score_100 = [] #Last 100 episodes average rewards
    score_complex = np.zeros((level+1, 2)) #Testing Score

    #TRAIN
    print("TRAINING...")
    for i in range(iter):
        eps = max(1.0 - i/max_eps, 0.1)
        episode_length, episode_score, complexity = play_one(rubik, qmodel,
                buffer, eps = eps, level = level,train = True)
        length.append(episode_length)
        score.append(episode_score)
        length_100.append(np.sum(length[max(i-99,0):i+1])/min(100, i+1))
        score_100.append(np.sum(score[max(i-99,0):i+1])/min(100, i+1))
        print("iter: {0:5d}, length: {1:4d}, score: {2:6.2f}, last_100_length:{3:6.2f}, last_100_score: {4:6.2f}, complexity: {5:4d}".format(i,episode_length, episode_score, length_100[i], score_100[i], complexity))

    #TEST
    print("TESTING...")
    for i in range(1000):
        episode_length, episode_score, complexity = play_one(rubik, qmodel,
                buffer, level = level,train = False)
        if episode_score > 0: #Cube is solved
            score_complex[complexity][0] += 1
        score_complex[complexity][1] += 1 #Occurence of particular complexity

    #Saving weights in format rubik.weights.architecture.iterations.level.optimizer.lr.h5
    name = "rubik.weights."
    for layer in architecture:
        name = name + str(layer) + "."
    name = name + str(iter) + "." + str(level) + "." + str(optimizer).split(".")[2][:-2] + str(lr) + ".h5"
    qmodel.save_model(name)


    #TEST Results
    for i in range(level+1):
        if score_complex[i][1] != 0:
            print("Complexity: {}, Solve Percentage; {}".format(i, 100.0*score_complex[i][0]/score_complex[i][1]))
    print("Overall Solve Percentage " + name + ": ",100.0*np.sum(score_complex[:,0])/1000.0)
    
    plt.figure()
    plt.title(name)
    plt.xlabel("Iterations")
    plt.ylabel("Length")
    plt.plot(range(iter-100), length_100[100:], 'r')
    plt.plot(range(iter-100), (np.cumsum(length)[100:])/(100+np.arange(iter-100)), 'b')
    plt.figure()
    plt.xlabel("Iterations")
    plt.ylabel("Score")
    plt.title(name)
    plt.plot(range(iter-100), score_100[100:], 'g')
    plt.plot(range(iter-100), (np.cumsum(score)[100:])/(100+np.arange(iter-100)), 'b')
    
    return name


if __name__ == "__main__":
    rubik = env.Rubik()
    buffer = []
    gen_buffer(rubik, buffer, 100)
    arch = [1024, 256, 128, 32]
    iter = 0
    level = 6
    max_eps = 0
    path = "rubik.final.h5"
    main(rubik, arch, iter, level, buffer, max_eps, path)
    if iter > 100:
        plt.show()


