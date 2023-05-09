# adapted from https://github.com/WeiChengTseng/Pytorch-PCGrad

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
import numpy as np
import copy
import random


class PCGrad():
    def __init__(self, optimizer):
        self._optim = optimizer
        return

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        '''
        clear the gradient of the parameters
        '''
        return self._optim.zero_grad(set_to_none=True)

    def step(self):
        '''
        update the parameters with the gradient
        '''
        return self._optim.step()

    def pc_backward(self, losses):
        '''
        calculate the gradient of the parameters
        input:
        - objectives: a list of objectives
        '''
        grads, shapes = self._pack_grad(losses)
        pc_grad = self._project_conflicting(grads)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)
        return

    def _project_conflicting(self, grads):
        pc_grad = copy.deepcopy(grads)
        for g_i in pc_grad:
            random.shuffle(grads)
            for g_j in grads:
                g_i_g_j = torch.dot(g_i, g_j)
                if g_i_g_j < 0:
                    g_i -= (g_i_g_j) * g_j / (g_j.norm()**2)
        pc_grad = torch.stack(pc_grad).sum(dim=0)
        return pc_grad

    def _set_grad(self, grads):
        '''
        set the modified gradients to the network
        '''
        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                p.grad = grads[idx]
                idx += 1
        return

    def _pack_grad(self, losses):
        '''
        pack the gradient of the parameters of the network for each objective
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grads, shapes = [], []
        for loss in losses:
            self._optim.zero_grad(set_to_none=True)
            loss.backward(retain_graph=True)
            grad, shape = self._retrieve_grad()
            grads.append(self._flatten_grad(grad, shape))
            shapes.append(shape)
        return grads, shapes

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self):
        '''
        get the gradient of the parameters of the network with specific 
        objective
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''
        grad, shape = [], []
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: tackle the multi-head scenario
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                else:
                    shape.append(p.grad.shape)
                    grad.append(p.grad.clone())
        return grad, shape
