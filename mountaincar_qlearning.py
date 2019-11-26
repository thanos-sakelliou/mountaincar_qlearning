import gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def discritize(state, discretize_size):

    # making state-space discrete
    disc_state = (state - env.observation_space.low) * np.array([1, 10])*discretize_size
    disc_state = np.round(disc_state, 0).astype(int)

    return disc_state


def q_learning(env, learning_rate, discount, epsilon, min_eps, episodes, discretize_size):

    statespace_size = discritize(env.observation_space.high, discretize_size) + 1
    Q = np.random.uniform(
        low=-1, high=1, size=(statespace_size[0], statespace_size[1], env.action_space.n))

    print(statespace_size)

    # Initialize variable to track iterations
    iterations_list = []

    # Calculate episodic reduction in epsilon
    reduction = (epsilon - min_eps)/episodes

    # Run Q learning algorithm
    for episode in range(episodes):
        # Initialize parameters
        done = False
        tot_reward, reward = 0, 0
        state = env.reset()

        # Discretize state
        state = discritize(state, discretize_size)

        path = [state]

        iters = 0
        while done != True:
            iters += 1
            # Render environment for last five episodes
            if episode >= (episodes - 50):
                env.render()

            # Determine next action - epsilon greedy strategy
            if np.random.random() < 1 - epsilon:
                action = np.argmax(Q[state[0], state[1]])
            else:
                action = np.random.randint(0, env.action_space.n)

            # Get next state and reward
            state2, reward, done, info = env.step(action)

            # Discretize state2
            state2 = discritize(state2, discretize_size)

            #keep the path
            path.append(state2.tolist())

            # Allow for terminal states
            if done and state2[0] >= 0.5:
                Q[state[0], state[1], action] = reward

            # Adjust Q value for current state
            else:
                # print(state,state2)
                delta = learning_rate * \
                    (reward + discount *
                     np.max(Q[state2[0], state2[1]]) - Q[state[0], state[1], action])
                Q[state[0], state[1], action] += delta

            # Update variables
            tot_reward += reward
            state = state2

        # Decay epsilon
        if epsilon > min_eps:
            epsilon -= reduction

        #PRINTS AND PLOTS--------------
        if (episode+1) % 50 == 0:
            iterations_list.append(iters)

            print("episode :", episode, "- iterations to goal: ", iters)

            #making the path of the car
            path_matrix = np.zeros(statespace_size)
            for i, point in enumerate(path):
                # if path_matrix[point[0], point[1]] == 0:
                path_matrix[point[0], point[1]] = i

            #PLOTS
            fig, (ax1, ax2) = plt.subplots(1, 2)

            plt.rcParams["figure.figsize"] = (15, 15)

            plot1 = sns.heatmap(np.amin(Q, axis=2), ax=ax1, cmap="YlGnBu", square=True)
            plot2 = sns.heatmap(path_matrix, ax=ax2, square=True)
          
            ax1.invert_yaxis()
            ax1.set(xlabel='Car Velocity', ylabel='Car Position')
            ax1.set_title('Max Q value of states')
            ax2.set(xlabel='Car Velocity')
            ax2.set_title('Car state path')
            ax2.invert_yaxis()
            
            plot1.figure.savefig("output"+ str(episode) +".png")

            plt.draw()
            plt.pause(0.001)
            plt.clf()



    env.close()

    return iterations_list


##################################################

gym.envs.register(
    id='MountainCarMyEasyVersion-v0',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=100000,      # MountainCar-v0 uses 200
)
env = gym.make('MountainCarMyEasyVersion-v0')


iterations_list = q_learning(env, 0.4, 0.9, 0.8, 0, 4000, 100)


# Plot Rewards
plt.plot(100*(np.arange(len(iterations_list)) + 1), iterations_list)
plt.xlabel('Episodes')
plt.ylabel('Iterations to goal')
plt.title('Iterations to goal vs Episodes')
# plt.savefig('rewards.jpg')
plt.show()
