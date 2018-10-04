import tensorflow as tf


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


class BasePolicyAgent:

    def __init__(self, state_dim, action_dim, name='Policy'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.name = name
        self.sess = tf.Session()
        self.state_input,\
            self.action_output,\
            self.actions,\
            self.net = self.create_network(layer_sizes=(256,))

        self.q_val_target,\
            self.action_input,\
            self.loss,\
            self.grads,\
            self.grad_placeholders,\
            self.tr_step = self.create_updater(lr=3e-3)

        self.saver = tf.train.Saver(self.net)
        self.sess.run(tf.global_variables_initializer())

    def create_network(self, layer_sizes):
        with tf.variable_scope(self.name):
            state_in = tf.placeholder(tf.float32, (None, self.state_dim), name='state_in')
            with tf.variable_scope('layer1'):
                w1 = tf.get_variable('w', (self.state_dim, layer_sizes[0]))
                b1 = tf.Variable(tf.zeros((layer_sizes[0],)), name='b')
                l1 = tf.nn.relu(tf.matmul(state_in, w1) + b1)
            with tf.variable_scope('layer2'):
                w3 = tf.get_variable('w', (layer_sizes[0], self.action_dim))
                b3 = tf.Variable(tf.zeros((self.action_dim,)), name='b')
                out = tf.nn.softmax(tf.matmul(l1, w3) + b3)
            picked = tf.argmax(out, axis=-1)
            vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)
        return state_in, out, picked, vars

    def create_updater(self, lr):
        q_target = tf.placeholder(tf.float32, (None), name='q_target')
        act_placeholder = tf.placeholder(tf.int32, (None), name='action_plh')
        resp_inds = tf.range(0, tf.shape(self.action_output)[0])*self.action_dim + act_placeholder
        resp_outs = tf.gather(tf.reshape(self.action_output, [-1]), resp_inds)

        loss = - tf.reduce_mean(tf.log(resp_outs)*q_target)# + 0.1*tf.reduce_mean(tf.log(self.action_output)*self.action_output)
        grads = tf.gradients(-tf.abs(loss), self.net)
        grad_plh = []
        for var in self.net:
            grad_plh.append(tf.placeholder(tf.float32, name=var.name[:-2]+'_holder'))
        up = tf.train.AdamOptimizer(lr)
        tr_step = up.apply_gradients(zip(grad_plh, self.net))
        return q_target, act_placeholder, loss, grads, grad_plh, tr_step

    def get_actions(self, states):
        return self.sess.run(self.actions, feed_dict={self.state_input:states})

    def get_action_distr(self, states):
        return self.sess.run(self.action_output, feed_dict={self.state_input:states})

    def save(self, savedir='c:/users/sabak/desktop/Policy/model'):
        self.saver.save(self.sess, savedir)

    def load(self, savedir='c:/users/sabak/desktop/Policy/model'):
        self.saver.restore(self.sess, savedir)

    def get_grads(self, states, actions, rewards):
        return self.sess.run(self.grads, feed_dict={self.state_input:states,
                                               self.action_input:actions,
                                               self.q_val_target:rewards})

    def update(self, grads):
        self.sess.run(self.tr_step, feed_dict=dict(zip(self.grad_placeholders,grads)))


class PolicyAgent(BasePolicyAgent):
    """Basic RL policy gradient agent"""
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
        acts = self.sess.run(self.action_output, feed_dict={self.state_input:np.array([state]).reshape((1, -1))})
        self.reset_state()
        return acts

    def get_grads(self, state, actions, rewards):
        return self.sess.run(self.grads, feed_dict={self.state_input:np.array([state]).reshape((1, -1)),
                                               self.action_input:actions,
                                               self.q_val_target:rewards})

## Recurrent Policy
if __name__ == '__main__':
    import gym
    import numpy as np
    env = gym.make('LunarLander-v2')
    #env = gym.make('Acrobot-v1')
    step = 0
    total_episodes = 10000
    verbose = 50
    parallel_envs = 1 # todo: vectorEnvs - to run many at once
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
    env = gym.make('LunarLander-v2')
    #env = gym.make('Acrobot-v1')
    agent = PolicyAgent(env.observation_space.shape[0] , env.action_space.n)
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
            #print(action_distr)
            1
            act = np.random.choice(range(env.action_space.n), p=action_distr)
            state_new, reward, done, info = env.step(act)
            ep_buffer.append([state, act, state_new, reward])

            if done or st == max_ep_steps - 1:
                ep_buffer = np.array(ep_buffer)
                rewards = get_discounted_reward(ep_buffer[:,3])
                states = np.vstack(ep_buffer[:, 0])
                actions = ep_buffer[:, 1]
                grads = agent.get_grads(states, actions, rewards)
                for ix, gr in enumerate(grads):
                    grad_buffer[ix] += gr
                    #grad_buffer[ix] = grad_buffer[ix] * GAMMA_GRAD + (1 - GAMMA_GRAD) * gr
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

    env = Env(max_steps=100, range_=(40., 120.), random=True)#, rand_probs=(.1, .1, .8))
    agent = PolicyAgent(3, 3, savedir='home/user/Desktop/py/Policy/model/model.ckpt')

    rounds = 100000
    update_every = 15
    verbose = 101
    GAMMA_GRAD = .99
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
                        grad_buffer[ix] = grad_buffer[ix]*GAMMA_GRAD + (1-GAMMA_GRAD)*gr
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
                #grad_buffer = np.array([tf.zeros_like(k).eval(session=agent.sess) for k in agent.net])
            if r % verbose == verbose - 1:
                print(r+1, np.mean(total_rewards[-verbose:]), np.mean(total_len[-verbose:]))
                print(state_from, state_to, state_from - state_to)
                print (actions)

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