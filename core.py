"""Core classes."""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
#1

class Sample:
    """Represents a reinforcement learning sample.

    Used to store observed experience from an MDP. Represents a
    standard `(s, a, r, s', terminal)` tuple.

    Note: This is not the most efficient way to store things in the
    replay memory, but it is a convenient class to work with when
    sampling batches, or saving and loading samples while debugging.

    Parameters
    ----------
    state: array-like
      Represents the state of the MDP before taking an action. In most
      cases this will be a numpy array.
    action: int, float, tuple
      For discrete action domains this will be an integer. For
      continuous action domains this will be a floating point
      number. For a parameterized action MDP this will be a tuple
      containing the action and its associated parameters.
    reward: float
      The reward received for executing the given action in the given
      state and transitioning to the resulting state.
    next_state: array-like
      This is the state the agent transitions to after executing the
      `action` in `state`. Expected to be the same type/dimensions as
      the state.
    is_terminal: boolean
      True if this action finished the episode. False otherwise.
    """
    def __init__(self, s, a, r, s_hat, is_terminal):
        self.state = s
        self.action = a
        self.reward = r
        self.next_state = s_hat
        self.done = is_terminal


class Preprocessor:
    """Preprocessor base class.

    This is a suggested interface for the preprocessing steps. You may
    implement any of these functions. Feel free to add or change the
    interface to suit your needs.

    Preprocessor can be used to perform some fixed operations on the
    raw state from an environment. For example, in ConvNet based
    networks which use image as the raw state, it is often useful to
    convert the image to greyscale or downsample the image.

    Preprocessors are implemented as class so that they can have
    internal state. This can be useful for things like the
    AtariPreproccessor which maxes over k frames.

    If you're using internal states, such as for keeping a sequence of
    inputs like in Atari, you should probably call reset when a new
    episode begins so that state doesn't leak in from episode to
    episode.
    """

    def process_state_for_network(self, state):
        """Preprocess the given state before giving it to the network.

        Should be called just before the action is selected.

        This is a different method from the process_state_for_memory
        because the replay memory may require a different storage
        format to reduce memory usage. For example, storing images as
        uint8 in memory is a lot more efficient thant float32, but the
        networks work better with floating point images.

        Parameters
        ----------
        state: np.ndarray
          Generally a numpy array. A single state from an environment.

        Returns
        -------
        processed_state: np.ndarray
          Generally a numpy array. The state after processing. Can be
          modified in anyway.
        1. 降维: 210*160 -> 110*84, RGB -> Gray
        2. 切片: 截取中心84*84的活动区域 (GPU卷积计算需要方形输入)
        3. 每次截取4帧作为一个输入
        """
        # state是一个210*160*3的数组
        state_gray = tf.image.rgb_to_grayscale(state)
        # state_gray = state
        state_resize = tf.image.resize_images(state_gray, [110, 84])
        state_square = state_resize[26:110, 0:84]  # 截取下面部分, 除去计分板
        with tf.Session() as sess:
            processed_state = state_square.eval()
        # print(img_.shape)
        # plt.imshow(np.squeeze(img_), cmap='gray')
        # plt.show()
        return processed_state.astype('float16') # 84*84

    def process_state_for_memory(self, state):
        """Preprocess the given state before giving it to the replay memory.

        Should be called just before appending this to the replay memory.

        This is a different method from the process_state_for_network
        because the replay memory may require a different storage
        format to reduce memory usage. For example, storing images as
        uint8 in memory and the network expecting images in floating
        point.

        Parameters
        ----------
        state: np.ndarray
          A single state from an environmnet. Generally a numpy array.

        Returns
        -------
        processed_state: np.ndarray
          Generally a numpy array. The state after processing. Can be
          modified in any manner.

        """
        # state_gray = tf.image.rgb_to_grayscale(state)
        # state_resize = tf.image.resize_images(state_gray, [110, 84])
        # state_square = state_resize[26:110, 0:84]  # 截取下面部分, 除去计分板
        # with tf.Session() as sess:
        #     processed_state = state_square.eval()

        # return processed_state.astype('uint8')  # 84*84
        return state

    def process_batch(self, samples):
        """Process batch of samples.

        If your replay memory storage format is different than your
        network input, you may want to apply this function to your
        sampled batch before running it through your update function.

        Parameters
        ----------
        samples: list(tensorflow_rl.core.Sample)
          List of samples to process

        Returns
        -------
        processed_samples: list(tensorflow_rl.core.Sample)
          Samples after processing. Can be modified in anyways, but
          the list length will generally stay the same.
        """
        return samples

    def process_reward(self, reward):
        """Process the reward.

        Useful for things like reward clipping. The Atari environments
        from DQN paper do this. Instead of taking real score, they
        take the sign of the delta of the score.

        Parameters
        ----------
        reward: float
          Reward to process

        Returns
        -------
        processed_reward: float
          The processed reward
        """
        reward = 1 if reward > 0 else 0
        return reward

    def reset(self):
        """Reset any internal state.

        Will be called at the start of every new episode. Makes it
        possible to do history snapshots.
        """
        pass


class ReplayMemory:
    """Interface for replay memories.

    We have found this to be a useful interface for the replay
    memory. Feel free to add, modify or delete methods/attributes to
    this class.

    It is expected that the replay memory has implemented the
    __iter__, __getitem__, and __len__ methods.

    If you are storing raw Sample objects in your memory, then you may
    not need the end_episode method, and you may want to tweak the
    append method. This will make the sample method easy to implement
    (just randomly draw samples saved in your memory).

    However, the above approach will waste a lot of memory (as states
    will be stored multiple times in s as next state and then s' as
    state, etc.). Depending on your machine resources you may want to
    implement a version that stores samples in a more memory efficient
    manner.

    Methods
    -------
    append(state, action, reward, debug_info=None)
      Add a sample to the replay memory. The sample can be any python
      object, but it is suggested that tensorflow_rl.core.Sample be
      used.
    end_episode(final_state, is_terminal, debug_info=None)
      Set the final state of an episode and mark whether it was a true
      terminal state (i.e. the env returned is_terminal=True), of it
      is is an artificial terminal state (i.e. agent quit the episode
      early, but agent could have kept running episode).
    sample(batch_size, indexes=None)
      Return list of samples from the memory. Each class will
      implement a different method of choosing the
      samples. Optionally, specify the sample indexes manually.
    clear()
      Reset the memory. Deletes all references to the samples.
    """
    def __init__(self, max_size = 100, window_length = 3):
        """Setup memory.

        You should specify the maximum size o the memory. Once the
        memory fills up oldest values should be removed. You can try
        the collections.deque class as the underlying storage, but
        your sample method will be very slow.

        We recommend using a list as a ring buffer. Just track the
        index where the next sample should be inserted in the list.
        """
        self.max_size = max_size
        self.window_length = window_length
        self.replay_buffer = []
        self.indicator = 0
        self.length = 0

    def append(self, state, action, reward=None, next_state=None, is_done=None):
        new_sample = Sample(state, action, reward, next_state, is_done)
        loc = self.indicator % self.max_size
        if len(self.replay_buffer) < self.max_size:
            self.replay_buffer.append(new_sample)
        else:
            self.replay_buffer[loc] = new_sample
        self.indicator += 1
        self.length += 1

    def end_episode(self):
        sample = self.replay_buffer[self.indicator%self.max_size-1]
        if sample.is_terminal:
            return True

    def sample(self, batch_size, indexes=None):
        batch_size_ = min(batch_size, len(self.replay_buffer))
        import random
        if not indexes:
            sample = random.sample(self.replay_buffer, batch_size_)
        else:
            sample = []
            for i in indexes:
                sample.append(self.replay_buffer[i])
        # while batch_size_ < batch_size:
            # new_sample = Sample()
        return sample

    def clear(self):
        self.replay_buffer = []

    def __len__(self):
        return self.length
