import random
import sys

import numpy as np
import torch
import json
import random
import os
from copy import deepcopy

import numpy as np
import math
import random
import torch
import torchvision
import csv
import hashlib
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from medmnist import INFO
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchsummary import summary
from numpy import savetxt
from datetime import datetime
import pandas as pd
import random
import pickle
import matplotlib.pyplot as plt

from evaluate import Evaluate
from mealpy import Tuner
from mealpy.evolutionary_based import DE
from mealpy.evolutionary_based.DE import L_SHADE, BaseDE
from mealpy.evolutionary_based.ES import CMA_ES
from mealpy.evolutionary_based.GA import BaseGA
from mealpy.swarm_based import PSO
from mealpy.swarm_based.ACOR import OriginalACOR
from mealpy.utils import io
from model import NetworkCIFAR
import operations_mapping
from utils import decode_cell, decode_operations
from optimizer import Optimizer

from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.operators.sampling.rnd import IntegerRandomSampling,FloatRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.algorithms.moo.nsga2 import NSGA2


def evaluate_arch(self, ind, dataset, measure):

    return random.randint(10,10)


import numpy as np


def objective_function(individual):
    # Replace this with your actual objective function
    return sum(individual)


def generate_neighbor(current_individual, perturbation_factor=0.1):
    # Generate a random perturbation for each element in the individual
    perturbation = np.random.uniform(-perturbation_factor, perturbation_factor, size=len(current_individual))

    # Create a neighbor by perturbing the current individual
    neighbor = current_individual + perturbation

    return neighbor


def hill_climbing(initial_individual, max_iterations=1000, perturbation_factor=0.1):
    current_individual = initial_individual
    current_score = objective_function(current_individual)

    for iteration in range(max_iterations):
        # Generate a neighbor
        neighbor = generate_neighbor(current_individual, perturbation_factor)

        # Evaluate the neighbor's score
        neighbor_score = objective_function(neighbor)

        # If the neighbor is better, update the current individual and score
        if neighbor_score > current_score:
            current_individual = neighbor
            current_score = neighbor_score

    return current_individual, current_score


# Your given individual vector
initial_individual = np.array([0.21530566, 0.85533325, 0.28166929, 0.08983863, 0.56053246,
                               0.83216755, 0.28182249, 0.16628747, 0.30039807, 0.0,
                               0.30716533, 0.48759968, 0.45569492, 0.0, 0.88215768,
                               0.31858798, 0.0, 0.44383456, 0.50577073, 0.48988354,
                               0.44491873, 0.60778265, 0.00113434, 0.58305548, 0.89792402,
                               0.09271671, 0.47320444, 0.54148738, 0.49932572, 0.3472307,
                               0.43000027, 0.39943375, 0.21688349, 0.58182204, 0.02435816,
                               0.32246534, 0.07770371, 0.52272287, 0.57207401, 0.53663519,
                               0.59136137, 0.81186171, 0.81015222, 0.84866093, 0.34997684,
                               0.99, 0.63237716, 0.85977464])

# Perform hill climbing
best_individual, best_score = hill_climbing(initial_individual)

print("Best Individual:", best_individual)
print("Best Score:", best_score)



class SOGA(Optimizer):
    def __init__(self, population_size, number_of_generations, crossover_prob, mutation_prob, blocks_size, num_classes,
                 in_channels, epochs, batch_size, layers, n_channels, dropout_rate, retrain, resume_train, cutout,
                 multigpu_num,medmnist_dataset,is_medmnist,check_power_consumption=False,evaluation_type='zero_cost'):
        super().__init__(population_size, number_of_generations, crossover_prob, mutation_prob, blocks_size,
                         num_classes, in_channels, epochs, batch_size, layers, n_channels, dropout_rate, retrain,
                         resume_train, cutout, multigpu_num,medmnist_dataset,is_medmnist,check_power_consumption,evaluation_type)

    def objective_function(self,solution):
        info = INFO[self.medmnist_dataset]
        task = info['task']
        n_channels = 3
        n_classes = len(info['label'])

        individual = []
        for i in range(32):
            if i % 2 == 0:
                individual.append(solution[i])
            else:
                individual.append(int(random.choice(self.pop.params_choices[str(i)])))
                # individual.append(random.choice(self.attentions))
        individual.append(int(math.floor(2 + ((self.layers - 2) * solution[-1]))))
        print(individual[-1])
        individuals = np.asarray(individual)

        is_final = False
        # print(decode_operations(individual, self.pop.indexes))
        decoded_individual = NetworkCIFAR(self.n_channels, n_classes, individual[-1], True,
                                          decode_cell(decode_operations(individual[:-1], self.pop.indexes)),
                                          self.is_medmnist, is_final,
                                          self.dropout_rate, 'FP32', False)
        # decoded_individual = NetworkCIFAR(self.n_channels, n_classes, self.layers, True,
        #                                     decode_cell(decode_operations(individual, self.pop.indexes)),self.is_medmnist,
        #                                     self.dropout_rate, 'FP32', False)

        loss = self.evaluator.evaluate_zero_cost(decoded_individual, self.epochs, n_classes)
        return loss['snip']

    def pair_swap(self,arr, index1, index2):
        arr[index1], arr[index2] = arr[index2], arr[index1]

    def inversion(self,arr, index1, index2):
        arr[index1:index2 + 1] = arr[index1:index2 + 1][::-1]

    def insertion(self,arr, index_from, index_to):
        element = arr.pop(index_from)
        arr.insert(index_to, element)

    def displacement(self,arr, index_from, index_to):
        element = arr.pop(index_from)
        arr.insert(index_to, element)

    def generate_neighbor(self,current_individual):
        current_individual
        # Create a neighbor by perturbing the current individual
        neighbor = current_individual

        # Apply operations
        self.pair_swap(current_individual, 1, 3)
        self.inversion(current_individual, 5, 12)
        self.insertion(current_individual, 8, 16)
        self.displacement(current_individual, 20, 4)

        return np.asarray(neighbor)

    def local_search_algo(self,initial_individual, max_iterations=10):
        print("hill_climbing started")
        current_individual = initial_individual
        current_score = objective_function(current_individual)

        for iteration in range(max_iterations):
            # Generate a neighbor
            neighbor = self.generate_neighbor(current_individual.tolist())

            # Evaluate the neighbor's score
            neighbor_score = self.objective_function(neighbor)

            # If the neighbor is better, update the current individual and score
            if neighbor_score > current_score:
                print("score_higherz")
                current_individual = neighbor
                current_score = neighbor_score

        return current_individual, current_score

    def evaluate_fitness_single_mealpy(self, solution):
        info = INFO[self.medmnist_dataset]
        task = info['task']
        n_channels = 3
        n_classes = len(info['label'])
        #zrefined_solution,fitness = self.local_search_algo(solution)
        gpu_ids = '0'
        n_classes = len(info['label'])
        batch_size = 1000
        download = True
        run = 'model1'
        individual = []
        for i in range(32):
            if i % 2 == 0:
                individual.append(solution[i])
            else:
                individual.append(int(random.choice(self.pop.params_choices[str(i)])))
                # individual.append(random.choice(self.attentions))
        individual.append(int(math.floor(2+((self.layers-2)*solution[-1]))))
        #print(individual[-1])
        individuals = np.asarray(individual)

        is_final = False
        #print(decode_operations(individual, self.pop.indexes))
        #print(individuals)
        decoded_individual = NetworkCIFAR(3, n_classes, individual[-1], True,
                                            decode_cell(decode_operations(individual[:-1], self.pop.indexes)))
        # decoded_individual = NetworkCIFAR(self.n_channels, n_classes, self.layers, True,
        #                                     decode_cell(decode_operations(individual, self.pop.indexes)),self.is_medmnist,
        #                                     self.dropout_rate, 'FP32', False)
        if self.evaluation_type == 'zero-cost':
            loss = self.evaluator.evaluate_zero_cost(decoded_individual, self.epochs,n_classes)
            print(loss['snip'])
            return  loss['snip']
            is_final = False
            try:
                decoded_individual = NetworkCIFAR(3, n_classes, individual[-1], True,
                                                  decode_cell(decode_operations(individual[:-1], self.pop.indexes)))
                # First Search for augmentation policy
                # best_combination = self.evaluator.auto_search_daapolicy(decoded_individual, 100, hash_indv=None, grad_clip=5, evaluation='valid', data_flag=data_flag, output_root=output_root,
                #                             num_epochs=num_epochs, gpu_ids=gpu_ids, batch_size=batch_size,is_final=False, download=download, run=run)
                # # evaluation = 'test'
                best_combination = None
                output_root = '\out'
                loss = self.evaluator.train(best_combination, decoded_individual, 100, hash_indv=None, grad_clip=5,
                                            evaluation='val', data_flag=self.medmnist_dataset, output_root=output_root,
                                            num_epochs=25, gpu_ids=gpu_ids, batch_size=self.batch_size, is_final=False,
                                            download=download, run=run)
                return loss
            except Exception as e:
                # Print or log the error message
                print(f"Error: {e}")
                return  0
    def evaluate_ensemble_predictions(self,ensemble,medmnist_dataset):

        return None
    def train_final_individual(self,solution,medmnist_dataset,data_augmentation):
        data_flag = self.medmnist_dataset
        output_root = './output'
        info = INFO[self.medmnist_dataset]
        num_epochs = 300
        gpu_ids = '0'
        n_classes = len(info['label'])
        batch_size = 32
        download = True
        run = 'model1'
        individual = []
        individual = []
        for i in range(32):
            if i % 2 == 0:
                individual.append(solution[i])
            else:
                individual.append(int(random.choice(self.pop.params_choices[str(i)])))
                # individual.append(random.choice(self.attentions))
        individual.append(int(math.floor(2 + ((self.layers - 2) * solution[-1]))))
        print(individual[-1])
        individuals = np.asarray(individual)
        if data_augmentation == False:
            is_final = False
            decoded_individual = NetworkCIFAR(3, n_classes, individual[-1], True,
                                              decode_cell(decode_operations(individual[:-1], self.pop.indexes)))#self.is_medmnist
            #First Search for augmentation policy
            # best_combination = self.evaluator.auto_search_daapolicy(decoded_individual, 100, hash_indv=None, grad_clip=5, evaluation='valid', data_flag=data_flag, output_root=output_root,
            #                             num_epochs=num_epochs, gpu_ids=gpu_ids, batch_size=batch_size,is_final=False, download=download, run=run)
            # # evaluation = 'test'
            best_combination = None
            loss = self.evaluator.train(best_combination,decoded_individual, 100, hash_indv=None, grad_clip=5, evaluation='test', data_flag=data_flag, output_root=output_root,
                                        num_epochs=num_epochs, gpu_ids=gpu_ids, batch_size=batch_size,is_final=False, download=download, run=run)
        else:
            is_final = True
            decoded_individual = NetworkCIFAR(3, n_classes, individual[-1], True,
                                              decode_cell(decode_operations(individual[:-1], self.pop.indexes)))  # self.is_medmnist
            # First Search for augmentation policy
            best_combination = self.evaluator.auto_search_daapolicy(decoded_individual, 25, hash_indv=None, grad_clip=5, evaluation='valid', data_flag=data_flag, output_root=output_root,
                                         num_epochs=num_epochs, gpu_ids=gpu_ids, batch_size=batch_size,is_final=False, download=download, run=run)
            evaluation = 'test'
            #best_combination = None
            loss = self.evaluator.train(best_combination, decoded_individual, 100, hash_indv=None, grad_clip=5,
                                        evaluation='test', data_flag=data_flag, output_root=output_root,
                                        num_epochs=num_epochs, gpu_ids=gpu_ids, batch_size=batch_size, is_final=False,
                                        download=download, run=run)

        print("loss", loss)




        print("Final loss is ",loss)
    def mealypy_evolve(self, algorithm, pop_size=15, epoch=20,strategy = None,medmnist_dataset=None,data_augmentation = False):
        print(" Running Algorithm ## ",algorithm,"# ")
        print(" Running strategy ## ",strategy,"# ")
        ## Design a problem dictionary for multiple objective functions above
        problem_multi = {
            "fit_func": self.evaluate_fitness_single_mealpy,
            "lb": [0 for i in range(32)],
            "ub": [0.99 for i in range(32)],
            "minmax": "max",
            "obj_weights": [1],  # Define it or default value will be [1, 1, 1]
            "save_population": True,
            "log_to": "file",
            "log_file": "result.log",  # Default value = "mealpy.log"
        }
        paras_de = {
            "epoch": [100],
            "pop_size": [100],
            "wf": [0.1, 0.2, 0.4, 0.6, 0.8, 0.9],
            "cr": [0.1, 0.2, 0.4, 0.6, 0.8, 0.9],
            "strategy": [1]
        }
        # term = {
        #     "max_epoch": 2
        # }

        max_time = 60
        term_dict = {
            "max_time": max_time  # 60 seconds to run this algorithm only
        }
        surrogate_params = {'bootstrap': False, 'max_depth': 90, 'max_features': 'sqrt',
                            'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 1800}

        if algorithm == 'pso':
            model = PSO.OriginalPSO(epoch=epoch, pop_size=pop_size)
            best_position, best_fitness = model.solve(problem=problem_multi)
            print(f"Solution: {best_position}, Fitness: {best_fitness}")
            self.train_final_individual(best_position, medmnist_dataset,data_augmentation)
        elif algorithm == 'sade':
            model = DE.SADE(epoch=epoch, pop_size=pop_size)
            best_position, best_fitness = model.solve(problem=problem_multi)
            print(f"Solution: {best_position}, Fitness: {best_fitness}")
            self.train_final_individual(best_position, medmnist_dataset,data_augmentation)
        elif algorithm == 'sap_de':
            model = DE.SAP_DE(epoch=epoch, pop_size=pop_size, branch="ABS")
            best_position, best_fitness = model.solve(problem=problem_multi)
            print(f"Solution: {best_position}, Fitness: {best_fitness}")
            self.train_final_individual(best_position, medmnist_dataset,data_augmentation)
        elif algorithm == 'jade':
            model = DE.JADE(epoch=epoch,pop_size=pop_size)
            best_position, best_fitness = model.solve(problem=problem_multi)
            print(f"Solution: {best_position}, Fitness: {best_fitness}")
            self.train_final_individual(best_position, medmnist_dataset,data_augmentation)
        elif algorithm == 'de':
            wf = 0.7
            cr = 0.9
            #strategy = 0
            model = BaseDE(epoch, pop_size, wf, cr, strategy=strategy)
            #tuner = Tuner(model, paras_de)
            #tuner.execute(problem=problem_multi, n_trials=5, n_jobs=6, mode="parallel", n_workers=6, verbose=True)
            # best_position, best_fitness = model.solve(problem=problem_multi, surrogate_params=surrogate_params,
            #                                           termination=term_dict)
            best_position, best_fitness = model.solve(problem=problem_multi)
            print(f"Solution: {best_position}, Fitness: {best_fitness}")
            self.train_final_individual(best_position, medmnist_dataset, data_augmentation)
        elif algorithm == 'lshade':
            #Trying to create an ensemble of 5 top find networks to improve the perforamnce
            ensemble_models = []
            for i in range(1):
                miu_f = 0.5
                miu_cr = 0.5
                model = L_SHADE(epoch, pop_size, miu_f, miu_cr)
                best_position, best_fitness = model.solve(problem=problem_multi)
                print(f"Solution: {best_position}, Fitness: {best_fitness}")
                ensemble_models.append(best_position)
                self.train_final_individual(best_position, medmnist_dataset,data_augmentation)
            #self.evaluate_ensemble_predictions(ensemble_models,medmnist_dataset)
            #print(ensemble_models)
            #print("Now ensemble predictions from test test")
        elif algorithm == 'ga':
            pc = 0.9
            pm = 0.05
            model1 = BaseGA(epoch, pop_size, pc, pm)
            best_position, best_fitness = model1.solve(problem=problem_multi)
            print(f"Solution: {best_position}, Fitness: {best_fitness}")
        elif algorithm == 'cmaes':
            model = CMA_ES(epoch, pop_size)
            best_position, best_fitness = model.solve(problem=problem_multi)
            print(f"Solution: {best_position}, Fitness: {best_fitness}")
            self.train_final_individual(best_position, medmnist_dataset, data_augmentation)
        elif algorithm == 'aco':
            sample_count = 25
            intent_factor = 0.5
            zeta = 1.0
            model = OriginalACOR(epoch, pop_size, sample_count, intent_factor, zeta)
            best_position, best_fitness = model.solve(problem=problem_multi)
            print(f"Solution: {best_position}, Fitness: {best_fitness}")
        else:
            print("error")
        ## Define the model and solve the problem

        ## Save model to file
        io.save_model(model, "results/model.pkl")
        ## You can access them all via object "history" like this:
        model.history.save_global_objectives_chart(filename="hello/goc")
        model.history.save_local_objectives_chart(filename="hello/loc")

        model.history.save_global_best_fitness_chart(filename="hello/gbfc")
        model.history.save_local_best_fitness_chart(filename="hello/lbfc")

        model.history.save_runtime_chart(filename="hello/rtc")

        model.history.save_exploration_exploitation_chart(filename="hello/eec")

        model.history.save_diversity_chart(filename="hello/dc")

        model.history.save_trajectory_chart(list_agent_idx=[3, 5], selected_dimensions=[3], filename="hello/tc")
    def evolve(self):
        pop_size = 5
        seed = 50
        n_gens = 5
        objectives_list = ['synflow', 'params']
        xl= [ 0.0 for i in range(48)]
        xu= [ 0.99 for i in range(48)]
        xl = np.asarray(xl)
        xu = np.asarray(xu)
        n_obj = len(objectives_list)
        n_var = 48  # NATS-Bench
        problem = NAS(objectives_list=objectives_list, n_var=n_var,
                      n_obj=n_obj,
                      xl=xl, xu=xu,pop = self.pop,population_size=self.population_size, number_of_generations = self.number_of_generations, crossover_prob = self.crossover_prob, mutation_prob= self.mutation_prob, blocks_size=self.blocks_size,
                         num_classes=self.num_classes, in_channels=self.in_channels, epochs=self.epochs, batch_size=self.batch_size, layers=self.layers, n_channels=self.n_channels, dropout_rate=self.dropout_rate, retrain=self.retrain,
                         resume_train=self.resume_train, cutout=self.cutout, multigpu_num=self.multigpu_num)

        algorithm = NSGA2(pop_size=pop_size,
                          sampling=FloatRandomSampling(),
                          crossover=TwoPointCrossover(prob=0.9),
                          mutation=PolynomialMutation(prob=1.0 / n_var),
                          eliminate_duplicates=True)

        stop_criteria = ('n_gen', n_gens)

        results = minimize(
            problem=problem,
            algorithm=algorithm,
            seed=seed,
            save_history=True,
            termination=stop_criteria
        )
        print(results.F)

        # Assuming 'results' contains your optimization results
        # results.F contains the objective values

        n_evals = []  # corresponding number of function evaluations\
        hist_F = []  # the objective space values in each generation
        hist_cv = []  # constraint violation in each generation
        hist_cv_avg = []  # average constraint violation in the whole population

        for algo in hist:
            # store the number of function evaluations
            n_evals.append(algo.evaluator.n_eval)

            # retrieve the optimum from the algorithm
            opt = algo.opt

            # store the least contraint violation and the average in each population
            hist_cv.append(opt.get("CV").min())
            hist_cv_avg.append(algo.pop.get("CV").mean())

            # filter out only the feasible and append and objective space values
            feas = np.where(opt.get("feasible"))[0]
            hist_F.append(opt.get("F")[feas])
