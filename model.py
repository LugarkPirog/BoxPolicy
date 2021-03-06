import tensorflow as tf
import numpy as np
from .replay import ReplayBuffer
from scipy.signal import lfilter

def get_discounted_reward(arr, gamma=.99):
    return lfilter([1], [1, -gamma], np.array(arr)[::-1], 0)[::-1]

def process_distances(arr):
    ans = [-1] * (len(arr))
    return ans

# TODO: Add train interface to policy grad models
class BaseQLearningAgent:

    def __init__(self, max_buffer_len, state_dim, action_dim, name, delta_huber=2., act_replace_dct=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.name = name
        self.delta_huber = delta_huber
        self.sess = tf.Session()
        self.buffer = ReplayBuffer(maxlen=max_buffer_len)
        self.losses = []
        self.test_dists = []
        self.test_lens = []
        self.act_replace_dct = act_replace_dct or {a: a for a in range(self.action_dim)}

        self.state_input, \
        self.q_val, \
        self.actions, \
        self.net = self.create_network(layer_sizes=(256,512,64))

        self.rew_input, \
        self.action_input, \
        self.loss, \
        self.tr_step, \
        self.lr= self.create_updater()

        self.saver = tf.train.Saver(self.net)
        self.sess.run(tf.global_variables_initializer())

    def create_network(self, layer_sizes):
        with tf.variable_scope(self.name):
            act = tf.nn.relu
            sl = tf.contrib.slim
            state_in = tf.placeholder(tf.float32, (None, self.state_dim), name='state_in')
            x = state_in
            for i in range(len(layer_sizes)):
                x = sl.layers.fully_connected(x, layer_sizes[i], act)
            out = sl.layers.fully_connected(x, self.action_dim, tf.nn.softmax)
            picked = tf.argmax(out, axis=-1)
            vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)
        return state_in, out, picked, vars

    def create_updater(self):
        q_target = tf.placeholder(tf.float32, (None), name='q_target')
        act_placeholder = tf.placeholder(tf.int32, (None), name='action_plh')
        lr = tf.placeholder(tf.float32, name='lr')
        resp_inds = tf.range(0, tf.shape(self.q_val)[0]) * self.action_dim + act_placeholder
        resp_outs = tf.gather(tf.reshape(self.q_val, [-1]), resp_inds)

        loss = tf.reduce_mean(tf.abs(resp_outs - q_target))
        # Squared loss
        # loss = tf.reduce_mean(tf.square(resp_outs - q_target))
        # Huber loss
        # loss = tf.cond(err < self.delta_huber, lambda: tf.square(err) / 2, lambda: self.delta_huber * (err - self.delta_huber / 2))
        up = tf.train.AdamOptimizer(lr)
        tr_step = up.minimize(loss)

        return q_target, act_placeholder, loss, tr_step, lr

    def get_actions(self, states):
        return self.sess.run(self.actions, feed_dict={self.state_input: states})

    def get_q_vals(self, states):
        return self.sess.run(self.q_val, feed_dict={self.state_input: states})

    def save(self, savedir='c:/users/sabak/desktop/Policy/model'):
        self.saver.save(self.sess, savedir)

    def load(self, savedir='c:/users/sabak/desktop/Policy/model'):
        self.saver.restore(self.sess, savedir)

    def update_batch(self, states, actions, rewards, lr):
        _, ls = self.sess.run([self.tr_step, self.loss], feed_dict={self.rew_input: rewards,
                                                                    self.action_input: actions,
                                                                    self.state_input: states,
                                                                    self.lr: lr})
        self.losses.append(ls)

    def train(self, steps=30000, batch_size=256, lr=1e-3, disc=0.9, verbose=1000):

        for i in range(steps):
            batch = np.array(self.buffer.sample(batch_size))
            states = np.array([x for x in batch[:, 0]]).reshape(-1, self.state_dim)
            actions = batch[:, 1]
            rewards = batch[:, 2]
            nextstates = np.array([x for x in batch[:, 3]]).reshape(-1, self.state_dim)
            dones = np.array(batch[:, 4])
            x = self.get_q_vals(nextstates)
            nextq = np.max(x, axis=-1)
            target_q = rewards + disc * (nextq) * (1 - dones)
            self.update_batch(states, actions, target_q, lr)
            if i % verbose == verbose - 1:
                print(f'Step {i+1}, loss: {np.mean(self.losses[-verbose:])}')
        print('Trained!')

    def add_observations(self, env, num, greedy_eps=.25):
        states = []
        acts = []
        rews = []
        st1s = []
        dones = []
        i = 0
        while i < num:

            state = env.reset()
            if state is None:
                state = env.get_state()

            while True:
                i += 1
                try:
                    q = self.get_q_vals(state)[0]
                except ValueError:
                    q = self.get_q_vals(np.array(state).reshape([1,-1]))
                a = np.argmax(q)
                if np.random.rand() < greedy_eps:
                    a = np.random.randint(self.action_dim)
                try:
                    st1, rew, done, _ = env.step(self.act_replace_dct[a])
                except ValueError:
                    st1, rew, done = env.step(self.act_replace_dct[a])
                states.append(state[0])
                acts.append(a)
                rews.append(rew)
                st1s.append(st1[0])
                dones.append(int(done))
                state = st1

                if done:
                    rews = (np.array(rews) - np.mean(rews)) / (np.std(rews) + 1e-9)
                    greedy_eps *= .999
                    self.buffer.add(states, acts, get_discounted_reward(rews), st1s, dones)
                    states = []
                    acts = []
                    rews = []
                    st1s = []
                    dones = []
                    break

        print('Buffer ready!')

    def test(self, env, episodes=1000):
        dists = []
        longs = []
        state = env.reset()
        if state is None:
            state = env.get_state()
        for i in range(episodes):
            j = 0
            done = False
            target = env.get_target()
            while not done:
                j += 1
                state, r, done = env.step(self.act_replace_dct[np.argmax(self.get_q_vals(state)[0])])
            dists.append(state[0][2])
            longs.append(j)
            state = env.reset()
            if state is None:
                state = env.get_state()
        print('Mean dist:', np.mean(np.abs(dists)),'\nMean lens:', np.mean(longs))
        self.test_dists.extend(dists)
        self.test_lens.extend(longs)

class BasePolicyAgent:

    def __init__(self, state_dim, action_dim, name='Policy'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.name = name
        self.sess = tf.Session()
        self.state_input, \
        self.action_output, \
        self.actions, \
        self.net = self.create_network(layer_sizes=(256, 256, 128, 128))

        self.q_val_target, \
        self.action_input, \
        self.loss, \
        self.grads, \
        self.grad_placeholders, \
        self.tr_step,\
        self.lr = self.create_updater()

        self.saver = tf.train.Saver(self.net)
        self.sess.run(tf.global_variables_initializer())

    def create_network(self, layer_sizes):
        with tf.variable_scope(self.name):
            act = tf.nn.relu
            sl = tf.contrib.slim
            state_in = tf.placeholder(tf.float32, (None, self.state_dim), name='state_in')
            x = state_in
            for i in range(len(layer_sizes)):
                x = sl.layers.fully_connected(x, layer_sizes[i], act)
            out = sl.layers.fully_connected(x, self.action_dim, tf.nn.softmax)
            picked = tf.argmax(out, axis=-1)
            vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)
        return state_in, out, picked, vars

    def create_updater(self):
        q_target = tf.placeholder(tf.float32, (None), name='q_target')
        lr = tf.placeholder(tf.float32, name='lr')
        act_placeholder = tf.placeholder(tf.int32, (None), name='action_plh')
        resp_inds = tf.range(0, tf.shape(self.action_output)[0]) * self.action_dim + act_placeholder
        resp_outs = tf.gather(tf.reshape(self.action_output, [-1]), resp_inds)

        loss_a = - tf.reduce_mean((tf.log(tf.clip_by_value(resp_outs, 1e-7, 1.-1e-7)) * q_target))  # clipping to prevent log(0) and log(1) as its 0
        loss_b = tf.reduce_mean(tf.reduce_sum(tf.log(tf.clip_by_value(self.action_output, 1e-7, 1.-1e-7)) * self.action_output, axis=-1))  # same
        loss = loss_a + loss_b/50

        grads = tf.gradients(loss, self.net)
        grad_plh = []
        for var in self.net:
            grad_plh.append(tf.placeholder(tf.float32, name=var.name[:-2] + '_holder'))
        up = tf.train.AdamOptimizer(lr)
        tr_step = up.apply_gradients(zip(grad_plh, self.net))
        return q_target, act_placeholder, loss, grads, grad_plh, tr_step, lr

    def get_actions(self, states):
        return self.sess.run(self.actions, feed_dict={self.state_input: states})

    def eval_loss(self, states, actions, rewards):
        return self.sess.run(self.loss, feed_dict={self.state_input: states,
                                                    self.action_input: actions,
                                                    self.q_val_target: rewards})

    def get_action_distr(self, states):
        return self.sess.run(self.action_output, feed_dict={self.state_input: states})

    def save(self, savedir='c:/users/sabak/desktop/Policy/model'):
        self.saver.save(self.sess, savedir)

    def load(self, savedir='c:/users/sabak/desktop/Policy/model'):
        self.saver.restore(self.sess, savedir)

    def get_grads(self, states, actions, rewards):
        return self.sess.run(self.grads, feed_dict={self.state_input: states,
                                                    self.action_input: actions,
                                                    self.q_val_target: rewards})

    def update(self, grads, lr):
        d = dict(zip(self.grad_placeholders, grads))
        d[self.lr] = lr
        self.sess.run(self.tr_step, feed_dict=d)


class PolicyAgent(BasePolicyAgent):
    """Basic RL policy gradient agent"""
    pass


class PPO(BasePolicyAgent):
    def __init__(self, *args, clip_epsilon=.2, **kwargs):
        self.clip_eps = clip_epsilon
        super().__init__(*args, **kwargs)

    def create_updater(self):

        pass


class RecurrentPolicyAgent(BasePolicyAgent):

    def __init__(self, *args, max_actions=40, batch_size=1, **kwargs):
        self.hidden_state = None
        self.bs = batch_size
        self.max_actions = max_actions
        super().__init__(*args, **kwargs)

    def create_network(self, layer_sizes):

        with tf.variable_scope(self.name):
            state_in = tf.placeholder(tf.float32, (None, self.state_dim), name='state_in')
            with tf.variable_scope('layer1'):
                w1 = tf.get_variable('w', (self.state_dim, layer_sizes[0]))
                b1 = tf.Variable(tf.zeros((layer_sizes[0],)), name='b')
                l1 = tf.nn.relu(tf.matmul(state_in, w1) + b1)
            with tf.variable_scope('recurrent'):

                cell = tf.contrib.rnn.LSTMCell(layer_sizes[0], name='lstm')
                lstm_outs = []
                self.hidden_state = cell.zero_state(self.bs, tf.float32)
                self._zero_state = self.hidden_state
                c_in = l1
                for i in range(self.max_actions):
                    # print(c_in)
                    if i > 0:
                        c_in *= 0
                    c_in, self.hidden_state = cell(c_in, self.hidden_state)
                    lstm_outs.append(c_in)
            with tf.variable_scope('layer2'):
                w2 = tf.get_variable('w', (layer_sizes[0], self.action_dim))
                b2 = tf.Variable(tf.zeros((self.action_dim,)), name='b')
                outs = []
                for v in lstm_outs:
                    outs.append(tf.nn.softmax(tf.matmul(v, w2) + b2))
                out = tf.stack(outs, 1)

            vars_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)
            picked = tf.argmax(out, -1)
        return state_in, out, picked, vars_

    def reset_state(self):
        self.hidden_state = self._zero_state

    def get_action_distr(self, state):
        acts = self.sess.run(self.action_output, feed_dict={self.state_input: np.array([state]).reshape((1, -1))})
        self.reset_state()
        return acts

    def get_grads(self, state, actions, rewards):
        return self.sess.run(self.grads, feed_dict={self.state_input: np.array([state]).reshape((1, -1)),
                                                    self.action_input: actions,
                                                    self.q_val_target: rewards})


## Q-learning agent
if __name__ == '__main__':
    import gym
    import numpy as np

    env = gym.make('CartPole-v2')
    agent = BaseQLearningAgent(10000, env.observation_space.shape[0], env.action_space.n, 'QL')

    for ep in range(5):
        print('Ep', ep)
        agent.add_observations(env, num=10000)
        agent.train(steps=10000)
        agent.test(env, episodes=10)

## Recurrent Policy
if __name__ == '__main__1':
    import gym
    import numpy as np

    env = gym.make('CartPole-v2')
    # env = gym.make('Acrobot-v1')
    step = 0
    total_episodes = 10000
    verbose = 50
    parallel_envs = 1  # todo: vectorEnvs - to run many at once
    update_every = 5
    test = 100
    max_ep_steps = 200
    GAMMA_GRAD = .95
    DISC = .99
    agent = RecurrentPolicyAgent(env.observation_space.shape[0], env.action_space.n,
                                 max_actions=max_ep_steps, batch_size=parallel_envs)
    eplen = []
    rr = []
    grad_buffer = np.array([tf.zeros_like(k).eval(session=agent.sess) for k in agent.net])
    for i in range(total_episodes):
        ep_buffer = []
        step += 1
        state = env.reset()
        acts = agent.get_action_distr(state)[0]
        actions = np.array([np.random.choice(range(env.action_space.n), p=probs) for probs in acts])
        rews = np.zeros(len(actions))
        for j, act in enumerate(actions):
            s1, r, done, _ = env.step(act)
            rews[j] = r
            if done:
                rews[j:] = 0.
                break
        rews = get_discounted_reward(rews, DISC)
        grads = agent.get_grads(state, actions, rews)
        for ix, gr in enumerate(grads):
            grad_buffer[ix] += gr
            # grad_buffer[ix] = grad_buffer[ix] * GAMMA_GRAD + (1 - GAMMA_GRAD) * gr
        eplen.append(j)
        rr.append(np.sum(rews))

        if step % update_every == update_every - 1:
            agent.update(grad_buffer)
            grad_buffer = np.zeros_like(grad_buffer)

        if step % verbose == verbose - 1:
            print(f'Step {step+1} '
                  f'mean rew {np.mean(rr[-verbose:])} '
                  f'mean len {np.mean(eplen[-verbose:])} ')

        if step % test == test - 1:
            state = env.reset()
            # rendering
            action_distr = agent.get_action_distr(state)[0]
            actions = np.array([np.random.choice(range(env.action_space.n), p=probs) for probs in acts])
            for j in range(max_ep_steps):
                env.render()
                try:
                    state, reward, done, info = env.step(actions[j])
                except IndexError:
                    print(j)
                    break
                if done:
                    break

## Policy Agent
if __name__ == '__main__1':
    import gym
    import numpy as np

    env = gym.make('CartPole-v2')
    # env = gym.make('Acrobot-v1')
    agent = PolicyAgent(env.observation_space.shape[0], env.action_space.n)
    step = 0
    total_episodes = 10000
    verbose = 50
    update_every = 5
    test = 100
    max_ep_steps = 100
    GAMMA_GRAD = .95
    eplen = []
    rr = []
    grad_buffer = np.array([tf.zeros_like(k).eval(session=agent.sess) for k in agent.net])
    for i in range(total_episodes):
        step += 1
        ep_buffer = []
        state = env.reset()
        for st in range(max_ep_steps):
            action_distr = agent.get_action_distr(np.array([state]).reshape((1, -1)))[0]
            # print(action_distr)
            1
            act = np.random.choice(range(env.action_space.n), p=action_distr)
            state_new, reward, done, info = env.step(act)
            ep_buffer.append([state, act, state_new, reward])

            if done or st == max_ep_steps - 1:
                ep_buffer = np.array(ep_buffer)
                rewards = get_discounted_reward(ep_buffer[:, 3])
                states = np.vstack(ep_buffer[:, 0])
                actions = ep_buffer[:, 1]
                grads = agent.get_grads(states, actions, rewards)
                for ix, gr in enumerate(grads):
                    grad_buffer[ix] += gr
                    # grad_buffer[ix] = grad_buffer[ix] * GAMMA_GRAD + (1 - GAMMA_GRAD) * gr
                eplen.append(st)
                rr.append(np.sum(rewards))
                break

            state = state_new
        if step % update_every == update_every - 1:
            agent.update(grad_buffer)
            grad_buffer = np.zeros_like(grad_buffer)

        if step % verbose == verbose - 1:
            print(f'Step {step+1} '
                  f'mean rew {np.mean(rr[-verbose:])} '
                  f'mean len {np.mean(eplen[-verbose:])} ')

        if step % test == test - 1:
            state = env.reset()
            for j in range(max_ep_steps):
                # rendering
                env.render()
                action_distr = agent.get_action_distr(np.array([state]).reshape((1, -1)))[0]
                act = np.random.choice(range(env.action_space.n), p=action_distr)
                state, reward, done, info = env.step(act)
                if done:
                    break

if __name__ == '__main__1':
    from BoxModel.main import Env
    import pickle
    import numpy as np

    env = Env(max_steps=100, range_=(40., 120.), random=True)  # , rand_probs=(.1, .1, .8))
    agent = PolicyAgent(3, 3)

    rounds = 100000
    update_every = 15
    verbose = 101
    GAMMA_GRAD = .99
    gs = 0
    total_rewards = []
    total_len = []
    # env.set_manual_game(140.,100.)
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
                # env.set_manual_game(140., 100.)
                env.reset()
                continue
            buffer = []
            state_from, state_to = state, target
            step = 0
            while True:
                step += 1
                act = agent.get_actions(np.reshape([state, target, 80. * np.sign(state - target)], (-1, 3)))[0]
                # act = np.random.choice([0, 1, 2], p=act[0])
                new_state, distance, done = env.step(act)
                buffer.append([[state, target, np.sign(state - target) * 80.], act, distance, done])
                state = new_state
                if done:
                    buffer = np.array(buffer)
                    states = np.vstack(buffer[:, 0])
                    actions = np.asarray(buffer[:, 1])
                    # pd = process_distances(buffer[:, 2])
                    # rewards = np.asarray(get_discounted_reward(pd, .99))
                    rewards = -np.abs(np.asarray(get_discounted_reward(buffer[:, 2], .99)))
                    grads = agent.get_grads(states, actions, rewards)
                    for ix, gr in enumerate(grads):
                        grad_buffer[ix] = grad_buffer[ix] * GAMMA_GRAD + (1 - GAMMA_GRAD) * gr
                    # env.set_manual_game(140., 100.)
                    env.reset()
                    # print(rewards[-1])
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
                # grad_buffer = np.array([tf.zeros_like(k).eval(session=agent.sess) for k in agent.net])
            if r % verbose == verbose - 1:
                print(r + 1, np.mean(total_rewards[-verbose:]), np.mean(total_len[-verbose:]))
                print(state_from, state_to, state_from - state_to)
                print(actions)

            if np.mean(total_len[-500:]) < 18:
                print('Converged!')
                break
    #    raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(e)
    finally:
        a = input("save? y/n:")
        if a == 'y':
            agent.save()
            import pandas as pd

            with open('c:/users/sabak/desktop/buffer1.pkl', 'wb') as f:
                pickle.dump(all_buffer, f)
                f.close()
            pd.DataFrame(data).to_csv('c:/users/sabak/desktop/policy_results1.csv')

        preds = []
        xs = np.arange(1, 149, .5)
        ys = np.arange(1, 149, .5)
        import matplotlib.pyplot as plt

        for i in range(len(xs)):
            for j in range(len(ys)):
                pr = agent.get_action_distr(np.array([xs[i], ys[j], 80. * np.sign(xs[i] - ys[j])]).reshape((1, -1)))
                preds.append(pr)
        print('Done!')

        ss = np.array(preds).reshape((148 * 2, 148 * 2, -1))
        plt.imshow(ss)
        plt.show()
        input('Press enter to exit...')