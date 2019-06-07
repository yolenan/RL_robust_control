from keras.models import Sequential
from keras.layers import Dense, Activation


def create_SL_model(state_shape, num_actions, mode):
    model = Sequential()
    model.add(Dense(32, input_shape=(state_shape,), activation='relu'))
    if mode == 'veh':
        model.add(Dense(num_actions))
        model.add(Activation('sigmoid'))
    elif mode == 'att':
        model.add(Dense(num_actions))
        model.add(Activation('tanh'))
    model.compile('Adam', 'mse')
    return model