import tensorflow as tf
from BoxModel.main import Env


class PolicyAgent:
    def __init__(self, state_dim, action_dim, name='Policy', savedir='c:/users/sabak/desktop/Policy/model'):
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
            self.grads,\
            self.grad_placeholders,\
            self.tr_step = self.create_updater()

        self.saver = tf.train.Saver(self.net)
        self.savedir = savedir
        self.sess.run(tf.global_variables_initializer())

    def create_network(self, layer_sizes):
        with tf.variable_scope(self.name):
            state_in = tf.placeholder(tf.float32, (None, self.state_dim), name='state_in')
            with tf.variable_scope('layer1'):
                w1 = tf.get_variable('w', (self.state_dim, layer_sizes[0]))
                b1 = tf.Variable(tf.zeros((layer_sizes[0],)), name='b')
                l1 = tf.nn.relu(tf.matmul(state_in / 80., w1) + b1)
            with tf.variable_scope('layer2'):
                w3 = tf.get_variable('w', (layer_sizes[1], self.action_dim))
                b3 = tf.Variable(tf.zeros((self.action_dim,)), name='b')
                out = tf.nn.softmax(tf.matmul(l1, w3) + b3)
            picked = tf.argmax(out, axis=-1)
            vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)
        return state_in, out, picked, vars

    def create_updater(self):
        q_target = tf.placeholder(tf.float32, (None), name='q_target')
        act_placeholder = tf.placeholder(tf.int32, (None), name='action_plh')
        resp_inds = tf.range(0, tf.shape(self.action_output)[0])*self.action_dim + act_placeholder
        resp_outs = tf.gather(tf.reshape(self.action_output, [-1]), resp_inds)

        loss = -tf.reduce_mean(tf.log(resp_outs)*q_target) + tf.log(self.action_output)*self.action_output
        grads = tf.gradients(loss, self.net)
        grad_plh = []
        for var in self.net:
            grad_plh.append(tf.placeholder(tf.float32, name=var.name[:-2]+'_holder'))
        up = tf.train.AdamOptimizer(1e-4)
        tr_step = up.apply_gradients(zip(grad_plh, self.net))
        return q_target, act_placeholder, loss, grads, grad_plh, tr_step

    def get_actions(self, states):
        return self.sess.run(self.actions, feed_dict={self.state_input:states})

    def get_action_distr(self, states):
        return self.sess.run(self.action_output, feed_dict={self.state_input:states})

    def save(self):
        self.saver.save(self.sess, self.savedir)

    def load(self):
        self.saver.restore(self.sess, self.savedir)

    def get_grads(self, states, actions, rewards):
        return self.sess.run(self.grads, feed_dict={self.state_input:states,
                                               self.action_input:actions,
                                               self.q_val_target:rewards})

    def update(self, grads):
        self.sess.run(self.tr_step, feed_dict=dict(zip(self.grad_placeholders,grads)))


def get_discounted_reward(arr, gamma=.99):
    ans = np.zeros_like(arr, dtype=np.float32)
    moving_rew = 0.
    for i in reversed(range(0, ans.size)):
        moving_rew = moving_rew * gamma + arr[i]
        ans[i] = moving_rew
    return ans


def process_distances(arr):
    ans = [-1]*(len(arr))
    return ans #+ [-1] if abs(arr[-1]) > 1.6 else ans + [70]


if __name__ == '__main__1':
    import numpy as np
    env = Env(1.7, 0.05, 0.05, 0.05, max_steps=250, range_=(40., 120.))
    iters = []
    for _ in range(1000):
        env.set_manual_game(120., 40.)
        done = False
        i = 0
        while not done:
            i += 1
            st, dist, done = env.step(1)
        iters.append(i)
    print(np.mean(iters))

if __name__ == '__main__':
    import pickle
    import numpy as np
    env = Env(max_steps=100, range_=(40., 120.))
    agent = PolicyAgent(3, 3)

    rounds = 10000
    update_every = 5
    verbose = 101
    gs = 0
    total_rewards = []
    total_len = []
    #env.set_manual_game(140.,100.)
    env.reset()
    all_buffer = []
    grad_buffer = np.array([tf.zeros_like(k).eval(session=agent.sess) for k in agent.net])
    data = []
    # todo: from 1 state to another with increasing noise and random swap
    try:
        for r in range(rounds):
            gs += 1
            state = env.get_state()
            target = env.get_target()
            if gs % 2 == 0 and state < target:
                env.set_manual_game(target, state)
                state, target = target, state
            elif gs % 2 == 1 and state > target:
                env.set_manual_game(target, state)
                state, target = target, state
            if abs(state - target) < 1:
                #env.set_manual_game(140., 100.)
                env.reset()
                continue
            buffer = []
            state_from, state_to = state, target
            step = 0
            while True:
                step += 1
                act = agent.get_actions(np.reshape([state, target, 80.*np.sign(state-target)], (-1, 3)))[0]
                #act = np.random.choice([0, 1, 2], p=act[0])
                new_state, distance, done = env.step(act)
                buffer.append([[state, target, np.sign(state-target)*80.], act, distance, done])
                state = new_state
                if done:
                    buffer = np.array(buffer)
                    states = np.vstack(buffer[:, 0])
                    actions = np.asarray(buffer[:, 1])
                    #pd = process_distances(buffer[:, 2])
                    #rewards = np.asarray(get_discounted_reward(pd, .99))
                    rewards = -np.abs(np.asarray(get_discounted_reward(buffer[:, 2], .99)))
                    grads = agent.get_grads(states, actions, rewards)
                    for ix, gr in enumerate(grads):
                        grad_buffer[ix] += gr
                    #env.set_manual_game(140., 100.)
                    env.reset()
                    #print(rewards[-1])
                    try:
                        total_rewards.append(rewards[-1])
                        total_len.append(step)
                    except IndexError:
                        print(r, 'fok')
                    break
            data.append((state_from, state_to, step))
            all_buffer.append(buffer)
            if gs % update_every == update_every - 1:
                agent.update(grad_buffer)
            if r % verbose == verbose - 1:
                print(r+1, np.mean(total_rewards[-verbose:]), np.mean(total_len[-verbose:]))
                print(state_from, state_to, state_from - state_to)
                print (actions)
    #    raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass
    finally:
        a = input("save? y/n:")
        if a == 'n':
            exit(0)
        agent.save()
        import pandas as pd
        with open('c:/users/sabak/desktop/buffer1.pkl', 'wb') as f:
            pickle.dump(all_buffer, f)
            f.close()
        pd.DataFrame(data).to_csv('c:/users/sabak/desktop/policy_results1.csv')
