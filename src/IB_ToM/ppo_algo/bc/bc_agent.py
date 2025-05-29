from collections import deque

import numpy as np
import torch
from sklearn.model_selection import KFold

from IB_ToM.ppo_algo.networks import BCMLPActor, BCLSTMActor
from IB_ToM.utils.utils import ParameterManager, bc_process_dataset
from human_aware_rl.human.process_dataframes import get_trajs_from_data

class BCMLPAgent:
    def __init__(self, state_dim, action_dim, hidden_dim, config):
        self.config = config
        self.param = ParameterManager(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = BCMLPActor(state_dim, hidden_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.param.get("lr_actor"))
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)

        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def select_action(self, state):
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            probs, _ = self.actor(state_tensor)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action.item(), log_prob.item()

    def tom_select_action(self, state, tom_latent):
        with torch.no_grad():
            return self.select_action(state)

    def update(self):
        addr_train = self.param.get("bc_data_addr_train")
        assert addr_train is not None
        # addr_test = self.param.get("bc_data_addr_test")
        # assert addr_test is not None
        dataloader = bc_process_dataset(addr_train, self.param)
        epoch = self.param.get("bc_epoch")
        for i in range(epoch):
            for states, actions in dataloader:
                _, logits = self.actor(states)
                loss = self.cross_entropy(logits, actions.squeeze(-1))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()

            print("[BC] epoch:{}, loss: {}".format(i, loss.item()))

    def evaluation(self):
        self.actor.eval()
        addr_test = self.param.get("bc_data_addr_test")
        assert addr_test is not None
        dataloader = bc_process_dataset(addr_test, self.param, train_mode=False)

        total = 0
        correct = 0
        loss_accum = 0

        with torch.no_grad():
            for states, actions in dataloader:
                probs, logits = self.actor(states)
                # dist = torch.distributions.Categorical(probs)
                # preds = dist.sample()
                preds = torch.argmax(probs, dim=-1)
                correct += (preds == actions.squeeze(-1)).sum().item()
                total += actions.size(0)
                loss = self.cross_entropy(logits, actions.squeeze(-1))
                loss_accum += loss.item() * actions.size(0)

        acc = correct / total
        avg_loss = loss_accum / total
        print(f"[Eval] Accuracy: {acc * 100:.2f}%, Cross-Entropy Loss: {avg_loss:.4f}")
        return acc, avg_loss

    def save(self, ckpt_path):
        torch.save(self.actor.state_dict(), ckpt_path)

    def load(self, ckpt_path):
        self.actor.load_state_dict(torch.load(ckpt_path))

class BCLSTMAgent:
    def __init__(self, state_dim, action_dim, hidden_dim, config):
        self.config = config
        self.param = ParameterManager(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = BCLSTMActor(state_dim, hidden_dim, action_dim).to(self.device)

        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.param.get("lr_actor"))
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)
        self.cross_entropy = torch.nn.CrossEntropyLoss()

        self.state_deque = deque(maxlen=self.param.get("bc_seq_len"))

    def select_action(self, state, hidden=None):
        with torch.no_grad():
            self.state_deque.append(state)
            state_tensor = torch.tensor(np.stack(self.state_deque), dtype=torch.float32).unsqueeze(0).to(self.device)
            probs, _ = self.actor(state_tensor, hidden)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action.item(), log_prob.item()

    def tom_select_action(self, state, tom_latent):
        with torch.no_grad():
            return self.select_action(state)

    def update(self):
        addr_train = self.param.get("bc_data_addr_train")
        assert addr_train is not None
        dataloader = bc_process_dataset(addr_train, self.param, bc_use_lstm=True)
        epoch = self.param.get("bc_epoch")
        for i in range(epoch):
            for states, actions in dataloader:
                _, logits = self.actor(states)
                loss = self.cross_entropy(logits, actions.squeeze(-1))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()

            print("[BC] epoch:{}, loss: {}".format(i, loss.item()))

    def evaluation(self):
        self.actor.eval()
        addr_test = self.param.get("bc_data_addr_test")
        assert addr_test is not None
        dataloader = bc_process_dataset(addr_test, self.param, bc_use_lstm=True, train_mode=False)

        total = 0
        correct = 0
        loss_accum = 0

        with torch.no_grad():
            for states, actions in dataloader:
                probs, logits = self.actor(states)
                # dist = torch.distributions.Categorical(probs)
                # preds = dist.sample()
                preds = torch.argmax(probs, dim=-1)
                correct += (preds == actions.squeeze(-1)).sum().item()
                total += actions.size(0)
                loss = self.cross_entropy(logits, actions.squeeze(-1))
                loss_accum += loss.item() * actions.size(0)

        acc = correct / total
        avg_loss = loss_accum / total
        print(f"[Eval] Accuracy: {acc * 100:.2f}%, Cross-Entropy Loss: {avg_loss:.4f}")
        return acc, avg_loss

    def save(self, ckpt_path):
        torch.save(self.actor.state_dict(), ckpt_path)

    def load(self, ckpt_path):
        self.actor.load_state_dict(torch.load(ckpt_path))
        print(f"Loaded checkpoint from {ckpt_path}")


