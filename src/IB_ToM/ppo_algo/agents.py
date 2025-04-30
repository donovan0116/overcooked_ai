class RandomAgent:
    def __init__(self, env):
        self.env = env

    def select_action(self, state):
        temp = 0
        return self.env.action_space.sample(), temp