import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import copy
import os

from habitat_baselines.rl.exploration.utils import weight_init

class InverseDynamics(nn.Module):
    def __init__(
        self,
        encoder, # visual encoder from resnet policy
        action_dim,
        proj_dim,
        hidden_dim,
        num_steps,
        batch_size,
        idm_lr,
        num_update,
        device
    ):
        super(InverseDynamics, self).__init__()

        self.encoder = encoder # assume already on cuda
        self.repr_dim = encoder.output_size # should be 2048 from resnet50
        self.action_dim = action_dim
        self.proj_dim = proj_dim
        self.hidden_dim = hidden_dim
        assert self.proj_dim <= self.repr_dim, "can only project to lower dimension space"

        self.num_steps = num_steps
        self.batch_size = batch_size
        self.device = device

        # projection head: repr_dim -> proj_dim
        if self.proj_dim < self.repr_dim:
            self.proj_head = nn.Sequential(
                nn.Linear(self.repr_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.proj_dim)
            ).to(self.device)
            self.proj_head.apply(weight_init)

        self.act_pred = nn.Sequential(
            nn.Linear(self.proj_dim * (self.num_steps + 1), self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim * self.num_steps)
        ).to(self.device)
        self.act_pred.apply(weight_init)

        self.ce_loss = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.num_update = num_update

        if self.proj_dim < self.repr_dim:
            self.invdyn_opt = torch.optim.Adam(list(self.encoder.parameters()) + list(self.proj_head.parameters()) + list(self.act_pred.parameters()), lr=idm_lr, eps=1e-5)
        else:
            self.invdyn_opt = torch.optim.Adam(list(self.encoder.parameters()) + list(self.act_pred.parameters()), lr=idm_lr, eps=1e-5)

        self.train()
    
    def save(self, folder_path, step):
        if self.proj_dim < self.repr_dim:
            checkpoint = {
                "encoder": self.encoder.state_dict(),
                "proj_head": self.proj_head.state_dict(),
                "act_pred": self.act_pred.state_dict(),
            }
        else:
            checkpoint = {
                "encoder": self.encoder.state_dict(),
                "act_pred": self.act_pred.state_dict(),
            }
        torch.save(
            checkpoint, os.path.join(folder_path, "idm_param_{}.pth".format(int(step)))
        )
    
    def forward(self, img, action_target):
        repr_feat = self.encoder(img)
        repr_feat = self.proj_head(repr_feat)
        repr_feat = repr_feat.view(self.batch_size, -1, repr_feat.shape[1]) # env_steps x num_envs x proj_dim
        # env_steps x num_envs x proj_dim x t ->  env_steps x num_envs x t x proj_dim
        repr_cat = repr_feat.unfold(dimension=0, size=(self.num_steps + 1), step=2).permute(0, 1, 3, 2).contiguous()
        repr_cat = repr_cat.view(repr_cat.shape[0], repr_cat.shape[1], -1) # concatenate (num_steps x proj_sim)
        assert repr_cat.shape[0] == action_target.shape[0] and repr_cat.shape[1] == action_target.shape[2], "shape must match between input and target!"

        # prediction logits
        pred_logits = self.act_pred(repr_cat)
        pred_logits = pred_logits.view(pred_logits.shape[0], pred_logits.shape[1], self.num_steps, self.action_dim).permute(0, 3, 1, 2)
        pred_logits = F.normalize(pred_logits, p=2, dim=1)
        x_loss = self.ce_loss(pred_logits, action_target.argmax(dim=1)) # env_steps x action_dim x num_envs x t

        # prediction accuracy
        with torch.no_grad():
            pred_probs = self.softmax(pred_logits)
            pred_actions = torch.argmax(pred_probs, dim=1) # 0-1 probabilities
            true_actions = torch.argmax(action_target, dim=1) # 0,1 entries
            assert pred_actions.shape == true_actions.shape, "prediction and true label shape must match!"
            pred_acc = (pred_actions == true_actions).sum() / true_actions.numel()
            # print("Acc: {}".format(pred_acc))

        return x_loss, pred_acc
    
    def update_invdyn(self, obs_batch, action_batch):
        r"""
        Function to update inverse dynamics prediction model.

        Input:
        prev_obs_batch: batch observation of previous time step
        next_obs_batch: batch observation of next time step
        action_batch: batch of action taken - (s, a, s')

        Output:
        cross_entropy_loss
        prediction_accuracy
        """
        # env_steps x num_envs x action_dim -> env_steps x num_envs x action_dim x t -> env_steps x action_dim x num_envs x t
        action_mini_batch = action_batch.unfold(dimension=0, size=self.num_steps, step=2).permute(0, 2, 1, 3)
        # env_steps x num_envs x img_size
        obs_mini_batch = obs_batch.view(-1, obs_batch.shape[2], obs_batch.shape[3], obs_batch.shape[4])
        
        for _ in range(self.num_update):
            x_loss, pred_acc = self.forward(obs_mini_batch, action_mini_batch)

            self.invdyn_opt.zero_grad()
            x_loss.backward()
            self.invdyn_opt.step()

        return {"inv_dyn_loss": x_loss, "pred_acc": pred_acc}