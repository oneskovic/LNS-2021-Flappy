import torch
import numpy as np
from torch import nn
class Organism:
    def __init__(self, hparams):
        input_size = hparams['nn_architecture'][0]
        output_size = hparams['nn_architecture'][-1]
        self.score = 0          # For GA
        self.hparams = hparams
        self.model = nn.Sequential(
            nn.Linear(input_size, 5),
            nn.ReLU(),
            # nn.Linear(13, 8),
            # nn.ReLU(),
            # nn.Linear(8, 13),
            # nn.ReLU(),
            nn.Linear(5, output_size),
        )
        self.weights = []
        self.biases = []
        for i in range(0,3,2):
            # self.weights.append(torch.rand_like(self.model[i].weight.data)*2-1)
            # self.biases.append(torch.rand_like(self.model[i].bias.data)*2-1)
            self.weights.append(self.model[i].weight.data)
            self.biases.append(self.model[i].bias.data)
        # for i in range(4):
        #     self.weights.append(self.model[i].weight.data)
        #     self.biases.append(self.model[i].bias.data)


        self.update_model_params()

    def update_model_params(self):
        with torch.no_grad():
            for i in range(0,3,2):
                self.model[i].weight = nn.Parameter(self.weights[i//2])
                self.model[i].bias = nn.Parameter(self.biases[i//2])
            # for i in range(4):
            #     self.model[i].weight = nn.Parameter(self.weights[i])
            #     self.model[i].bias = nn.Parameter(self.biases[i])

    def get_output(self, input_data: torch.Tensor):
        with torch.no_grad():
            torch_output = self.model(input_data)
            return self.__get_action(torch_output)

    def mutate(self):
        mutation_factor = self.hparams['mutation_factor']
        for i in range(len(self.weights)):
            bin = torch.rand_like(self.weights[i]) <= mutation_factor
            rand_mat = torch.rand_like(self.weights[i])*2 - 1
            self.weights[i] = self.weights[i] * (~bin) + rand_mat * bin

            bin = torch.rand_like(self.biases[i]) <= mutation_factor
            rand_mat = torch.rand_like(self.biases[i])*2 - 1
            self.biases[i] = self.biases[i] * (~bin) + rand_mat * bin
        self.update_model_params()

    def __get_action(self, output_layer: torch.Tensor):
        return torch.argmax(output_layer)
