import time

import gym
import numpy as np
from organism import Organism
from ga import GeneticAlgorithm
from evaluator import Evaluator
import flappy_bird_gym


def preview_env(env):
    state = env.reset()
    done = False
    while not done:
        env.render()
        time.sleep(0.05)
        action = np.array([1])
        state,reward,done,info = env.step(action)
        print(state)
    env.close()

preview_env(flappy_bird_gym.make("FlappyBird-v0"))

env = flappy_bird_gym.make("FlappyBird-v0")
in_dimen = env.observation_space.shape[0]
out_dimen = env.action_space.n
hparams = dict({
    'nn_architecture': [in_dimen, out_dimen],
    'population_size': 20,
    'mutation_factor': 0.1,
    'tournament_size': 2,
    'elite_fraction': 0.2,
    'remove_fraction': 0.2,
    'environment': 'FlappyBird-v0'
})

evaluator = Evaluator(env)
alg = GeneticAlgorithm(hparams, evaluator)
for i in range(1,200):
    if i % 10 == 0:
        evaluator.evaluate(alg.population[0], True, 3)
    alg.step()
    scores = np.array([org.score for org in alg.population])
    #print(scores)
    print(f'Max result: {scores[0]}, mean result: {scores.mean()}')
