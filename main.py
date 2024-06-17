"""
    Robotic pouring task involving Machine Learning (Reinforcement Learning) techniques such as Q-tables
    with potential expansion into Deep Learning with Deep Q Learning.
"""

import sys
sys.path.append('MacAPI')
import numpy as np
import sim


# additional imports for pytorch, keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# potential experimentation with Deep Learning model - requires further testing
def create_model():
  model = tf.keras.Sequential([
      tf.keras.layers.Dense(20, input_shape=(2,), activation='relu'),
      tf.keras.layers.Dense(10, activation='relu'),
      tf.keras.layers.Dense(1, activation='linear')
  ])

  model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
  return model


# Max movement along X
low, high = -0.05, 0.05


def setNumberOfBlocks(clientID, blocks, typeOf, mass, blockLength,
                      frictionCube, frictionCup):
    '''
        Function to set the number of blocks in the simulation
        '''
    emptyBuff = bytearray()
    res, retInts, retFloats, retStrings, retBuffer = sim.simxCallScriptFunction(
        clientID, 'Table', sim.sim_scripttype_childscript, 'setNumberOfBlocks',
        [blocks], [mass, blockLength, frictionCube, frictionCup], [typeOf],
        emptyBuff, sim.simx_opmode_blocking)
    if res == sim.simx_return_ok:
        print(
            'Results: ', retStrings
        )  # display the reply from CoppeliaSim (in this case, the handle of the created dummy)
    else:
        print('Remote function call failed')


def triggerSim(clientID):
    e = sim.simxSynchronousTrigger(clientID)
    step_status = 'successful' if e == 0 else 'error'
    # print(f'Finished Step {step_status}')


def rotation_velocity(rng):
    ''' Set rotation velocity randomly, rotation velocity is a composition of two sinusoidal velocities '''
    #Sinusoidal velocity
    forward = [-0.3, -0.35, -0.4, -0.45, -0.50, -0.55, -0.60, -0.65]
    backward = [0.75, 0.8, 0.85, 0.90]
    freq = 60
    ts = np.linspace(0, 1000 / freq, 1000)
    velFor = rng.choice(forward) * np.sin(2 * np.pi * 1 / 20 * ts)
    velBack = rng.choice(backward) * np.sin(2 * np.pi * 1 / 10 * ts)
    velSin = velFor
    idxFor = np.argmax(velFor > 0)
    velSin[idxFor:] = velBack[idxFor:]
    velReal = velSin
    return velReal


def start_simulation():
    ''' Function to communicate with Coppelia Remote API and start the simulation '''
    sim.simxFinish(-1)  # just in case, close all opened connections
    clientID = sim.simxStart('127.0.0.1', 19000, True, True, 5000,
                             5)  # Connect to CoppeliaSim
    if clientID != -1:
        print('Connected to remote API server')
    else:
        print("fail")
        sys.exit()

    returnCode = sim.simxSynchronous(clientID, True)
    returnCode = sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)

    if returnCode != 0 and returnCode != 1:
        print("something is wrong")
        print(returnCode)
        exit(0)

    triggerSim(clientID)

    # get the handle for the source container
    res, pour = sim.simxGetObjectHandle(clientID, 'joint',
                                        sim.simx_opmode_blocking)
    res, receive = sim.simxGetObjectHandle(clientID, 'receive',
                                           sim.simx_opmode_blocking)
    # start streaming the data
    returnCode, original_position = sim.simxGetObjectPosition(
        clientID, pour, -1, sim.simx_opmode_streaming)
    returnCode, original_position = sim.simxGetObjectPosition(
        clientID, receive, -1, sim.simx_opmode_streaming)
    returnCode, original_position = sim.simxGetJointPosition(
        clientID, pour, sim.simx_opmode_streaming)

    return clientID, pour, receive


def stop_simulation(clientID):
    ''' Function to stop the episode '''
    sim.simxStopSimulation(clientID, sim.simx_opmode_blocking)
    sim.simxFinish(clientID)


def get_object_handles(clientID, pour):
    # Drop blocks in source container
    triggerSim(clientID)
    number_of_blocks = 2
    print('Initial number of blocks=', number_of_blocks)
    setNumberOfBlocks(clientID,
                      blocks=number_of_blocks,
                      typeOf='cube',
                      mass=0.002,
                      blockLength=0.025,
                      frictionCube=0.06,
                      frictionCup=0.8)
    triggerSim(clientID)

    # Get handles of cubes created
    object_shapes_handles = []
    obj_type = "Cuboid"
    for obj_idx in range(number_of_blocks):
        res, obj_handle = sim.simxGetObjectHandle(clientID,
                                                  f'{obj_type}{obj_idx}',
                                                  sim.simx_opmode_blocking)
        object_shapes_handles.append(obj_handle)

    triggerSim(clientID)

    for obj_handle in object_shapes_handles:
        # get the starting position of source
        returnCode, obj_position = sim.simxGetObjectPosition(
            clientID, obj_handle, -1, sim.simx_opmode_streaming)

    returnCode, position = sim.simxGetJointPosition(clientID, pour,
                                                    sim.simx_opmode_buffer)
    returnCode, obj_position = sim.simxGetObjectPosition(
        clientID, pour, -1, sim.simx_opmode_buffer)
    print(f'Pouring Cup Initial Position:{obj_position}')
    # Give time for the cubes to finish falling
    _wait(clientID)
    return object_shapes_handles, obj_position


def set_cup_initial_position(clientID, pour, receive, cup_position, rng):

    # Move cup along x axis
    global low, high
    move_x = low + (high - low) * rng.random()
    cup_position[0] = cup_position[0] + move_x

    returnCode = sim.simxSetObjectPosition(clientID, pour, -1, cup_position,
                                           sim.simx_opmode_blocking)
    triggerSim(clientID)
    returnCode, pour_position = sim.simxGetObjectPosition(
        clientID, pour, -1, sim.simx_opmode_buffer)
    print(f'Pouring Cup Moved Position:{cup_position}')
    returnCode, receive_position = sim.simxGetObjectPosition(
        clientID, receive, -1, sim.simx_opmode_buffer)
    print(f'Receiving Cup Position:{receive_position}')
    return pour_position, receive_position


def get_state(object_shapes_handles, clientID, pour, j):
    ''' Function to get the cubes and pouring cup position '''

    # Get position of the objects
    obj_pos = []
    for obj_handle in object_shapes_handles:
        # get the starting position of source
        returnCode, obj_position = sim.simxGetObjectPosition(
            clientID, obj_handle, -1, sim.simx_opmode_buffer)
        obj_pos.append(obj_position)

    returnCode, cup_position = sim.simxGetObjectPosition(
        clientID, pour, -1, sim.simx_opmode_buffer)

    return obj_pos, cup_position


def move_cup(clientID, pour, action, cup_position, center_position):
    ''' Function to move the pouring cup laterally during the rotation '''

    global low, high
    resolution = 0.001
    move_x = resolution * action
    movement = cup_position[0] + move_x
    if center_position + low < movement < center_position + high:
        cup_position[0] = movement
        returnCode = sim.simxSetObjectPosition(clientID, pour, -1,
                                               cup_position,
                                               sim.simx_opmode_blocking)


def rotate_cup(clientID, speed, pour):
    ''' Function to rotate cup '''
    errorCode = sim.simxSetJointTargetVelocity(clientID, pour, speed,
                                               sim.simx_opmode_oneshot)
    returnCode, position = sim.simxGetJointPosition(clientID, pour,
                                                    sim.simx_opmode_buffer)
    return position


def _wait(clientID):
    for _ in range(60):
        triggerSim(clientID)


def trainingCup():

    rng = np.random.default_rng()

    # Set rotation velocity randomly
    velReal = rotation_velocity(rng)

    # initialize starting Q randomly (replicates weights in nn)
    tableS = [[np.random.uniform(-1,1) for b in range(5)] for c in range(5)]

    # initialize target Q randomly (replicates weights in nn)
    tableT = [[np.random.uniform(-1,1) for b in range(5)] for c in range(5)]

    # initialize replay memory D
    replay = []

    # update targetQ with startingQ every this number of steps - adjustable
    timeUpdate = 50

    # min number inside replay buffer to start sampling - adjustable
    replaybuffminsize = 100

    # sample count from replay buffer - adjustable
    sample_size = 10
    
    # between 0 and 1 - adjustable
    learning_rate = 1

    # between 0 and 1 - adjustable
    gamma = 0

    # between 0 and 1 - epsilon greedy (percentage change of random) - adjustable
    epsilon_greedy = 0.1

    # counter - adjustable
    total_train = 20
    count_train = total_train

    # starting episodes
    for trail in range(total_train):

        print(f'Starting episode {trail + 1}')
        total_reward = 0
        with open('log.txt','a+')  as other_file:
            other_file.write('Episode ' + str(trail + 1) + '\n--- TD Error ---\n')

        # Start simulation
        clientID, pour, receive = start_simulation()
        object_shapes_handles, cup_position = get_object_handles(clientID, pour)

        original_cup_position = cup_position[0]

        # Get initial position of the cups
        cup_position, receive_position = set_cup_initial_position(
            clientID, pour, receive, cup_position, rng)
        _wait(clientID)
        center_position = cup_position[0]

        # get receive position x value
        x_pos_receive = receive_position[0]

        for j in range(velReal.shape[0]):
            # 60HZ
            triggerSim(clientID)
            # Make sure simulation step finishes
            returnCode, pingTime = sim.simxGetPingTime(clientID)

            # Get current state
            cubes_position, cup_position = get_state(object_shapes_handles,
                                                    clientID, pour, j)

            # Rotate cup
            speed = velReal[j]

            # call rotate_cup function and assign the return value to position variable
            position = rotate_cup(clientID, speed, pour)

            # mapping cup position to interval
            index_cup = -10000
            x_pos_pour = cup_position[0]
            if x_pos_pour >= -0.11 + x_pos_receive and x_pos_pour < -0.06 + x_pos_receive:
                index_cup = 0
            elif x_pos_pour >= -0.06 + x_pos_receive and x_pos_pour < -0.01 + x_pos_receive:
                index_cup = 1
            elif x_pos_pour >= -0.01 + x_pos_receive and x_pos_pour < 0.01 + x_pos_receive:
                index_cup = 2
            elif x_pos_pour >= 0.01 + x_pos_receive and x_pos_pour < 0.06 + x_pos_receive:
                index_cup = 3
            elif x_pos_pour >= 0.06 + x_pos_receive and x_pos_pour <= 0.11 + x_pos_receive:
                index_cup = 4

            # Move cup laterally
            actions = [-2, -1, 0, 1, 2]

            # choose max Q
            max_index = -10000
            max_num = -10000
            for indexer in range(5):
                if tableS[index_cup][indexer] > max_num:
                    max_num = tableS[index_cup][indexer]
                    max_index = indexer

            # Call move_cup function based on epsilon greedy
            number = np.random.uniform(0,1)
            if number <= epsilon_greedy:
                action_val = np.random.choice(actions)
            else:
                action_val = actions[max_index]

            move_cup(clientID, pour, action_val, cup_position, center_position)

            # observe reward
            x_pos_pour_after = cup_position[0]
            reward = x_pos_pour_after - x_pos_receive
            if x_pos_pour_after >= -0.01 + x_pos_receive and x_pos_pour_after < 0.01 + x_pos_receive:
                reward = 0
            if reward > 0:
                reward = -1*reward
            total_reward += reward

            # new state index
            index_cup_after = -10000
            if x_pos_pour_after >= -0.11 + x_pos_receive and x_pos_pour_after < -0.06 + x_pos_receive:
                index_cup_after = 0
            elif x_pos_pour_after >= -0.06 + x_pos_receive and x_pos_pour_after < -0.01 + x_pos_receive:
                index_cup_after = 1
            elif x_pos_pour_after >= -0.01 + x_pos_receive and x_pos_pour_after < 0.01 + x_pos_receive:
                index_cup_after = 2
            elif x_pos_pour_after >= 0.01 + x_pos_receive and x_pos_pour_after < 0.06 + x_pos_receive:
                index_cup_after = 3
            elif x_pos_pour_after >= 0.06 + x_pos_receive and x_pos_pour_after <= 0.11 + x_pos_receive:
                index_cup_after = 4

            replay.append({'State': index_cup, 'Action': max_index, 'Reward': reward, 'NState': index_cup_after})

            if len(replay) >= replaybuffminsize:
                random_indices = np.random.choice(len(replay), sample_size, replace=False)
                minibatch = [replay[i] for i in random_indices]
                # sum_variable = 0
                for transition in minibatch:
                    if transition['NState'] == -1:
                        TDtarget = transition['Reward']
                    else:
                        new_state_index = transition['NState']
                        max_index_after = -10000
                        max_num_after = -10000
                        for indexer_after in range(5):
                            if tableT[new_state_index][indexer_after] > max_num_after:
                                max_num_after = tableT[new_state_index][indexer_after]
                                max_index_after = indexer_after
                        TDtarget = transition['Reward'] + gamma*tableT[new_state_index][max_index_after]

                    # update sum_variable
                    TDerror = TDtarget - tableS[transition['State']][transition['Action']]

                    # SGD
                    # note that there are various implementations
                    # some utilize current - target for TDerror and thus have update as val = val - learning * deriv
                    deriv = TDerror
                    tableS[transition['State']][transition['Action']] = tableS[transition['State']][transition['Action']] + (learning_rate * deriv)

                # update targetQ with startingQ
                if j % timeUpdate == 0:
                    tableT = tableS
                    with open('log.txt','a+')  as other_file:
                        other_file.write('Last seen TD Error when updating target Q: ' + str(TDerror) + '\n')

                # calculate loss
                # loss = sum_variable / 10.0

            # Break if cup goes back to vertical position
            if j > 10 and position > 0:
                break

            # Break if either cube falls to ground
            if cubes_position[0][2] < 0.3 or cubes_position[1][2] < 0.3:
                count_train -= 1
                break

        # Stop simulation
        print('Last reward for this episode:', reward)
        print('Accumulated reward for this episode (totaled rewards):', total_reward)
        with open('log.txt','a+')  as other_file:
            other_file.write('--- Reward --- \nLast reward for this episode: ' + str(reward) + '\nAccumulated reward for this episode (totaled rewards): ' + str(total_reward) + '\n\n')
        stop_simulation(clientID)

    print('(Training - Cup) Times cubes went in cup:', count_train)
    print('(Training - Cup) Total runs:', total_train)
    return tableT

def testingCup(table_final):
    answer = input('Start testing?: ')
    rng = np.random.default_rng()

    # Set rotation velocity randomly
    velReal = rotation_velocity(rng)

    # counter - adjustable
    total_test = 100
    count_test = total_test

    # starting episodes
    for trail in range(total_test):

        print(f'Starting test {trail + 1}')

        # Start simulation
        clientID, pour, receive = start_simulation()
        object_shapes_handles, cup_position = get_object_handles(clientID, pour)

        original_cup_position = cup_position[0]

        # Get initial position of the cups
        cup_position, receive_position = set_cup_initial_position(
            clientID, pour, receive, cup_position, rng)
        _wait(clientID)
        center_position = cup_position[0]

        # get receive position x value
        x_pos_receive = receive_position[0]

        for j in range(velReal.shape[0]):
            # 60HZ
            triggerSim(clientID)
            # Make sure simulation step finishes
            returnCode, pingTime = sim.simxGetPingTime(clientID)

            # Get current state
            cubes_position, cup_position = get_state(object_shapes_handles,
                                                    clientID, pour, j)

            # Rotate cup
            speed = velReal[j]

            # call rotate_cup function and assign the return value to position variable
            position = rotate_cup(clientID, speed, pour)

            # mapping cup position to interval
            index_cup = -10000
            x_pos_pour = cup_position[0]
            if x_pos_pour >= -0.11 + x_pos_receive and x_pos_pour < -0.06 + x_pos_receive:
                index_cup = 0
            elif x_pos_pour >= -0.06 + x_pos_receive and x_pos_pour < -0.01 + x_pos_receive:
                index_cup = 1
            elif x_pos_pour >= -0.01 + x_pos_receive and x_pos_pour < 0.01 + x_pos_receive:
                index_cup = 2
            elif x_pos_pour >= 0.01 + x_pos_receive and x_pos_pour < 0.06 + x_pos_receive:
                index_cup = 3
            elif x_pos_pour >= 0.06 + x_pos_receive and x_pos_pour <= 0.11 + x_pos_receive:
                index_cup = 4

            # Move cup laterally
            actions = [-2, -1, 0, 1, 2]

            # choose max Q
            max_index = -10000
            max_num = -10000
            for indexer in range(5):
                if table_final[index_cup][indexer] > max_num:
                    max_num = table_final[index_cup][indexer]
                    max_index = indexer

            # Call move_cup function
            action_val = actions[max_index]
            move_cup(clientID, pour, action_val, cup_position, center_position)

             # information
            print(f'Step: {j}, Cube positions: [{cubes_position[0]}, {cubes_position[1]}], Cup position: {cup_position}, Action taken: {action_val}')

            # Break if cup goes back to vertical position
            if j > 10 and position > 0:
                break

            # Break if either cube falls to ground
            if cubes_position[0][2] < 0.3 or cubes_position[1][2] < 0.3:
                count_test -= 1
                break

        # Stop simulation
        stop_simulation(clientID)

    print('(Testing - Cup) Times cubes went in cup:', count_test)
    print('(Testing - Cup) Total runs:', total_test)
    with open('outcomes.txt','a+')  as other_file:
        other_file.write('(Testing - Cup) Times cubes went in receiving cup out of ' + str(total_test) + ': ' + str(count_test) + '\n')


def trainingCupCube():

    rng = np.random.default_rng()

    # Set rotation velocity randomly
    velReal = rotation_velocity(rng)

    # initialize starting Q
    tableS = [[0 for b in range(5)] for c in range(5)]

    # initialize target Q
    tableT = [[0 for b in range(5)] for c in range(5)]

    # initialize replay memory D
    replay = []

    # update targetQ with startingQ every this number of steps - adjustable
    timeUpdate = 50

    # min number inside replay buffer to start sampling - adjustable
    replaybuffminsize = 50

    # sample count from replay buffer - adjustable
    sample_size = 10
    
    # between 0 and 1 - adjustable
    learning_rate = 1

    # between 0 and 1 - adjustable
    gamma = 0.1

    # between 0 and 1 - epsilon greedy (percentage change of random) - adjustable
    epsilon_greedy = 0.1

    # counter - adjustable
    total_train = 5
    count_train = total_train

    # starting episodes
    for trail in range(total_train):

        print(f'Starting episode {trail + 1}')
        total_reward = 0

        # Start simulation
        clientID, pour, receive = start_simulation()
        object_shapes_handles, cup_position = get_object_handles(clientID, pour)

        original_cup_position = cup_position[0]

        # Get initial position of the cups
        cup_position, receive_position = set_cup_initial_position(
            clientID, pour, receive, cup_position, rng)
        _wait(clientID)
        center_position = cup_position[0]

        # get receive position x value
        x_pos_receive = receive_position[0]

        for j in range(velReal.shape[0]):
            # 60HZ
            triggerSim(clientID)
            # Make sure simulation step finishes
            returnCode, pingTime = sim.simxGetPingTime(clientID)

            # Get current state
            cubes_position, cup_position = get_state(object_shapes_handles,
                                                    clientID, pour, j)

            # Rotate cup
            speed = velReal[j]

            # call rotate_cup function and assign the return value to position variable
            position = rotate_cup(clientID, speed, pour)

            # mapping cup position to interval
            index_cup = -10000
            x_pos_pour = cup_position[0]
            if x_pos_pour >= -0.11 + x_pos_receive and x_pos_pour < -0.06 + x_pos_receive:
                index_cup = 0
            elif x_pos_pour >= -0.06 + x_pos_receive and x_pos_pour < -0.01 + x_pos_receive:
                index_cup = 1
            elif x_pos_pour >= -0.01 + x_pos_receive and x_pos_pour < 0.01 + x_pos_receive:
                index_cup = 2
            elif x_pos_pour >= 0.01 + x_pos_receive and x_pos_pour < 0.06 + x_pos_receive:
                index_cup = 3
            elif x_pos_pour >= 0.06 + x_pos_receive and x_pos_pour <= 0.11 + x_pos_receive:
                index_cup = 4

            # Move cup laterally
            actions = [-2, -1, 0, 1, 2]

            # choose max Q
            max_index = -10000
            max_num = -10000
            for indexer in range(5):
                if tableS[index_cup][indexer] > max_num:
                    max_num = tableS[index_cup][indexer]
                    max_index = indexer

            # Call move_cup function based on epsilon greedy
            number = np.random.uniform(0,1)
            if number <= epsilon_greedy:
                action_val = np.random.choice(actions)
            else:
                action_val = actions[max_index]

            move_cup(clientID, pour, action_val, cup_position, center_position)

            # observe reward
            x_pos_pour_after = cup_position[0]
            reward = x_pos_pour_after - x_pos_receive
            if x_pos_pour_after >= -0.01 + x_pos_receive and x_pos_pour_after < 0.01 + x_pos_receive:
                reward = 0
            if reward > 0:
                reward = -1*reward
            total_reward += reward

            # new state index
            index_cup_after = -10000
            if x_pos_pour_after >= -0.11 + x_pos_receive and x_pos_pour_after < -0.06 + x_pos_receive:
                index_cup_after = 0
            elif x_pos_pour_after >= -0.06 + x_pos_receive and x_pos_pour_after < -0.01 + x_pos_receive:
                index_cup_after = 1
            elif x_pos_pour_after >= -0.01 + x_pos_receive and x_pos_pour_after < 0.01 + x_pos_receive:
                index_cup_after = 2
            elif x_pos_pour_after >= 0.01 + x_pos_receive and x_pos_pour_after < 0.06 + x_pos_receive:
                index_cup_after = 3
            elif x_pos_pour_after >= 0.06 + x_pos_receive and x_pos_pour_after <= 0.11 + x_pos_receive:
                index_cup_after = 4

            replay.append({'State': index_cup, 'Action': max_index, 'Reward': reward, 'NState': index_cup_after})

            if len(replay) >= replaybuffminsize:
                random_indices = np.random.choice(len(replay), sample_size, replace=False)
                minibatch = [replay[i] for i in random_indices]
                # sum_variable = 0
                for transition in minibatch:
                    if transition['NState'] == -1:
                        TDtarget = transition['Reward']
                    else:
                        new_state_index = transition['NState']
                        max_index_after = -10000
                        max_num_after = -10000
                        for indexer_after in range(5):
                            if tableT[new_state_index][indexer_after] > max_num_after:
                                max_num_after = tableT[new_state_index][indexer_after]
                                max_index_after = indexer_after
                        TDtarget = transition['Reward'] + gamma*tableT[new_state_index][max_index_after]

                    # update sum_variable
                    TDerror = TDtarget - tableS[transition['State']][transition['Action']]

                    # SGD
                    deriv = TDerror
                    tableS[transition['State']][transition['Action']] = learning_rate * deriv + tableS[transition['State']][transition['Action']]

                # update targetQ with startingQ
                if j % timeUpdate == 0:
                    tableT = tableS
                    print('(TableT Update) Last TDerror before update', TDerror)

                # calculate loss
                # loss = sum_variable / 10.0

            # Break if cup goes back to vertical position
            if j > 10 and position > 0:
                break

            # Break if either cube falls to ground
            if cubes_position[0][2] < 0.3 or cubes_position[1][2] < 0.3:
                count_train -= 1
                break

        # Stop simulation
        print('Last reward for this episode:', reward)
        print('Accumulated reward for this episode (totaled rewards):', total_reward)
        stop_simulation(clientID)

    print('(Training - Cup/Cube) Times cubes went in cup:', count_train)
    print('(Training - Cup/Cube) Total runs:', total_train)
    return tableT


def testingCupCube(table_final):

    rng = np.random.default_rng()

    # Set rotation velocity randomly
    velReal = rotation_velocity(rng)

    # counter - adjustable
    total_test = 5
    count_test = total_test

    # starting episodes
    for trail in range(total_test):

        print(f'Starting test {trail + 1}')

        # Start simulation
        clientID, pour, receive = start_simulation()
        object_shapes_handles, cup_position = get_object_handles(clientID, pour)

        original_cup_position = cup_position[0]

        # Get initial position of the cups
        cup_position, receive_position = set_cup_initial_position(
            clientID, pour, receive, cup_position, rng)
        _wait(clientID)
        center_position = cup_position[0]

        # get receive position x value
        x_pos_receive = receive_position[0]

        for j in range(velReal.shape[0]):
            # 60HZ
            triggerSim(clientID)
            # Make sure simulation step finishes
            returnCode, pingTime = sim.simxGetPingTime(clientID)

            # Get current state
            cubes_position, cup_position = get_state(object_shapes_handles,
                                                    clientID, pour, j)

            # Rotate cup
            speed = velReal[j]

            # call rotate_cup function and assign the return value to position variable
            position = rotate_cup(clientID, speed, pour)

            # mapping cup position to interval
            index_cup = -10000
            x_pos_pour = cup_position[0]
            if x_pos_pour >= -0.11 + x_pos_receive and x_pos_pour < -0.06 + x_pos_receive:
                index_cup = 0
            elif x_pos_pour >= -0.06 + x_pos_receive and x_pos_pour < -0.01 + x_pos_receive:
                index_cup = 1
            elif x_pos_pour >= -0.01 + x_pos_receive and x_pos_pour < 0.01 + x_pos_receive:
                index_cup = 2
            elif x_pos_pour >= 0.01 + x_pos_receive and x_pos_pour < 0.06 + x_pos_receive:
                index_cup = 3
            elif x_pos_pour >= 0.06 + x_pos_receive and x_pos_pour <= 0.11 + x_pos_receive:
                index_cup = 4

            # Move cup laterally
            actions = [-2, -1, 0, 1, 2]

            # choose max Q
            max_index = -10000
            max_num = -10000
            for indexer in range(5):
                if table_final[index_cup][indexer] > max_num:
                    max_num = table_final[index_cup][indexer]
                    max_index = indexer

            # Call move_cup function
            action_val = actions[max_index]
            move_cup(clientID, pour, action_val, cup_position, center_position)

            # Break if cup goes back to vertical position
            if j > 10 and position > 0:
                break

            # Break if either cube falls to ground
            if cubes_position[0][2] < 0.3 or cubes_position[1][2] < 0.3:
                count_test -= 1
                break

        # Stop simulation
        stop_simulation(clientID)

    print('(Testing - Cup/Cube) Times cubes went in cup:', count_test)
    print('(Testing - Cup/Cube) Total runs:', total_test)
    with open('outcomes.txt','a+')  as other_file:
        other_file.write('(Testing - Cup + Cube) Times cubes went in receiving cup out of ' + str(total_test) + ': ' + str(count_test) + '\n')

if __name__ == '__main__':
    
    # cup position
    output_table = trainingCup()
    testingCup(output_table)

    # cup position and cube position
    output_table = trainingCupCube()
    testingCupCube(output_table)
