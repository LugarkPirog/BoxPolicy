import tensorflow as tf
from BoxModel.main import Env

class PolicyAgent:
    def __init__(self, state_dim, action_dim, name='Policy'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.name = name
        self.sess = tf.Session()
        self.state_input,\
            self.action_output,\
            self.actions,\
            self.net = self.create_network(layer_sizes=(256, 256))

        self.q_val_target,\
            self.action_input,\
            self.loss,\
            self.tr_step = self.create_updater()

        self.sess.run(tf.global_variables_initializer())

    def create_network(self, layer_sizes):
        with tf.variable_scope(self.name):
            state_in = tf.placeholder(tf.float32, (None, self.state_dim), name='state_in')
            with tf.variable_scope('layer1'):
                w1 = tf.get_variable('w', (self.state_dim, layer_sizes[0]))
                b1 = tf.Variable(tf.zeros((layer_sizes[0],)),name='b')
                l1 = tf.nn.relu(tf.matmul(state_in, w1) + b1)
            with tf.variable_scope('layer2'):
                w2 = tf.get_variable('w', (layer_sizes[0], layer_sizes[1]))
                b2 = tf.Variable(tf.zeros((layer_sizes[1],)), name='b')
                l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)
            with tf.variable_scope('layer3'):
                w3 = tf.get_variable('w', (layer_sizes[1], self.action_dim))
                b3 = tf.Variable(tf.zeros((self.action_dim,)), name='b')
                out = tf.nn.sigmoid(tf.matmul(l2, w3) + b3)
            picked = tf.argmax(out, axis=-1)
            vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)
        return state_in, out, picked, vars

    def create_updater(self):
        q_target = tf.placeholder(tf.float32, (None, 1), name='q_target')
        act_placeholder = tf.placeholder(tf.int32, (None, 1), name='action_plh')
        resp_inds = tf.range(0, tf.shape(self.action_output)[0])*self.action_dim + act_placeholder
        resp_outs = tf.gather(tf.reshape(self.action_output, [-1]), resp_inds)

        loss = -tf.log(resp_outs)*q_target
        up = tf.train.AdamOptimizer(1e-3)
        tr_step = up.minimize(loss)
        return q_target, act_placeholder, loss, tr_step

    def get_actions(self, states):
        return self.sess.run(self.actions, feed_dict={self.state_input:states})

    def get_action_distr(self, states):
        return self.sess.run(self.action_output, feed_dict={self.state_input:states})

    def save(self):
        pass

    def load(self):
        pass

    def update(self, states, actions, rewards):
        self.sess.run(self.tr_step, feed_dict={self.state_input:states,
                                               self.action_input:actions,
                                               self.q_val_target:rewards})

def get_discounted_reward(arr, gamma=.98):
    ans = np.zeros_like(arr, dtype=np.float32)
    moving_rew = 0.
    for i in reversed(range(0, ans.size)):
        moving_rew = moving_rew * gamma + arr[i]
        ans[i] = moving_rew
    return ans

def process_distances(arr):
    last = 200
    ans = []
    for i in range(1, len(arr)):
        ans.append(int(abs(arr[i]) < last))
        last = abs(arr[i])
    return ans + ans[-1:]


if __name__ == '__main__':
    import numpy as np
    env = Env()
    agent = PolicyAgent(2, 3)

    rounds = 5000
    verbose = 100
    eps = 0.2
    eps_decay=.99
    total_rewards = []
    for r in range(rounds):
        state = env.get_state()
        target = env.get_target()
        if abs(state - target) < 1:
            env.reset()
            continue
        buffer = []
        while True:
            act = agent.get_actions(np.reshape([state, target], (-1, 2)))
            if np.random.rand() < eps:
                act = np.random.randint(0, 3)
            new_state, distance, done = env.step(act)
            buffer.append([[state, target], act, distance])
            state = new_state
            if done:
                buffer = np.array(buffer)
                states = np.vstack(buffer[:,0])
                actions = np.asarray(buffer[:,1]).reshape((-1,1))
                pd = process_distances(buffer[:, 2])
                rewards = np.asarray(get_discounted_reward(pd)).reshape((-1,1))

                agent.update(states, actions, rewards)
                env.reset()
                break
        total_rewards.append(np.mean(pd))
        eps*=eps_decay
        if r % verbose == verbose - 1:
            print(np.mean(total_rewards[-verbose:]))
