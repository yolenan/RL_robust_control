from keras.models import Sequential
from keras.layers import Dense


def create_SL_model(state_shape, num_actions):
    model = Sequential()
    model.add(Dense(32, input_shape=(state_shape,), activation='relu'))
    model.add(Dense(num_actions))
    model.compile('Adam', 'mse')
    return model