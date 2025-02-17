import math
import random

import numpy as np
import torch
import json
import random
import os
from copy import deepcopy

import numpy as np
import random
import torch
import torchvision
import csv
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from medmnist import INFO
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
from pymoo.indicators.igd import IGD

def evaluate_arch(self, ind, dataset, measure):

    return random.randint(10,10)


class NAS(Problem):
    def __init__(self, n_var=6, n_obj=5, dataset='cifar10', xl=None, xu=None,pop=None,population_size=None, number_of_generations=None, crossover_prob=None, mutation_prob=None, blocks_size=None,
                         num_classes=None, in_channels=None, epochs=None, batch_size=None, layers=None, n_channels=None, dropout_rate=None, retrain=None,
                         resume_train=None, cutout=None, multigpu_num=None,medmnist_dataset=None,is_medmnist=None,check_power_consumption=False,evaluation_type='training', save_dir=None, seed=0,
                 objectives_list=None, args=None):
        super().__init__(n_var=n_var, n_obj=n_obj)
        self.xl = xl
        self.xu = xu
        self._save_dir = save_dir
        self._n_generation = 0
        self._n_evaluated = 0
        self.archive_obj = []
        self.archive_var = []
        self.obj1 = []
        self.obj2 = []
        self.seed = seed
        self.evaluation_type = evaluation_type
        self.dataset = dataset
        self.pop = pop
        self.population_size = population_size
        self.number_of_generations = number_of_generations
        self.crossover_prob =crossover_prob
        self.mutation_prob =mutation_prob
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.epochs = epochs
        self.batch_size = batch_size
        self.layers = layers
        self.n_channels = n_channels
        self.dropout_rate = dropout_rate
        self.resume_train = resume_train
        self.cutout = cutout
        self.multigpu_num = multigpu_num
        self.archive_obj = []
        self.blocks_size =blocks_size
        self.medmnist_dataset = medmnist_dataset
        self.evaluator = Evaluate(self.batch_size,medmnist_dataset,is_medmnist,check_power_consumption)
        self.retrain = retrain
        #self.attentions = [i for i in range(0, len(operations_mapping.attentions))]

    def _evaluate(self, x, out, *args, **kwargs):
        batch_size = 1024
        info = INFO[self.medmnist_dataset]
        task = info['task']
        n_channels = 3
        n_classes = len(info['label'])
        download = True
        data_flag = self.medmnist_dataset
        output_root = './output'
        info = INFO[self.medmnist_dataset]
        num_epochs = 300
        gpu_ids = '0'
        n_classes = len(info['label'])
        batch_size = 32
        download = True
        evaluation = 'val'
        run = 'model1'
        objs = np.full((x.shape[0], self.n_obj), np.nan)
        individuals = []
        for j in range(x.shape[0]):
            indv = []
            for i in range(32):
                if i % 2 == 0:
                    indv.append(x[j][i])
                else:
                    indv.append(int(random.choice(self.pop.params_choices[str(i)])))
                    #indv.append(math.floor(x[j][i] * len(self.attentions)))
            indv.append(int(math.floor(2 + ((self.layers - 2) * x[j][-1]))))
            individuals.append(indv)
        individuals = np.asarray(individuals)
        decoded_individuals = [NetworkCIFAR(self.n_channels, n_classes, self.layers, True,
                                            decode_cell(decode_operations(individuals[i][:-1], self.pop.indexes))) for i in
                               range(0, individuals.shape[0], 1)]

        for i in range(individuals.shape[0]):
            best_combination = None
            if self.evaluation_type == 'training':
                if self.multigpu_num > 1:
                    decoded_individuals[i] = nn.DataParallel(decoded_individuals[i])
                evaluation = 'val'
                loss = self.evaluator.train(best_combination, decoded_individuals[i], 2, hash_indv=None, grad_clip=5,
                                            evaluation=evaluation, data_flag=data_flag, output_root=output_root,
                                            num_epochs=2, gpu_ids=gpu_ids, batch_size=batch_size, is_final=False,
                                            download=download, run=run)
                objs[i][0] = -loss[0]
                objs[i][1] = loss[1]
            else:
                try:
                    loss = self.evaluator.evaluate_zero_cost(decoded_individuals[i], self.epochs,n_classes)
                    # loss = self.evaluator.evaluate_zero_cost(decoded_individuals[i], self.epochs, n_classes)
                    objs[i][0] = -loss['synflow']
                    objs[i][1] = loss['params']
                    if self.evaluation_type == 'training':
                        self.archive_obj.append([objs[i][1], loss[0]])
                    else:
                        self.archive_obj.append([loss['params'], loss['synflow']])
                except  Exception as e:
                    # loss = self.evaluator.evaluate_zero_cost(decoded_individuals[i], self.epochs, n_classes)
                    objs[i][0] =0
                    objs[i][1] = 0
                    if self.evaluation_type == 'training':
                        self.archive_obj.append([objs[i][1], loss[0]])
                    else:
                        self.archive_obj.append([0,0])
            self.obj1.append(objs[i][0])
            self.obj2.append(objs[i][1])
            self.archive_var.append(individuals[i])

        # logging.info('Generation: {}'.format(self._n_generation))
        #

        #
        # for i in range(x.shape[0]):
        #     for j in range(len(self.objectives_list)):
        #         # all objectives assume to be MINIMIZED !!!!!
        #         obj = evaluate_arch(self, ind=x[i], dataset=self.dataset, measure=self.objectives_list[j])
        #
        #         if 'accuracy' in self.objectives_list[j] or self.objectives_list[j] == 'synflow':
        #             objs[i, j] = -1 * obj
        #             print(obj)
        #             print(objectives_list[j])
        #         else:
        #             objs[i, j] = obj
        #     self.archive_obj, self.archive_var = archive_check(objs[i], self.archive_obj, self.archive_var, x[i])
        #     self._n_evaluated += 1
        #
        #     igd_dict, igd_norm_dict = calc_IGD(self, x=x, objs=objs)
        #
        # igd_norm_list = []
        # for dataset in self.datasets:
        #     igd_temp = list(igd_norm_dict[dataset].values())
        #     igd_temp.insert(0, dataset)
        #     igd_norm_list.append(igd_temp)
        #
        # headers = ['']
        # for j in range(1, len(self.objectives_list)):
        #     headers.append('test accuracy - ' + self.objectives_list[j])
        # logging.info(tabulate(igd_norm_list, headers=headers, tablefmt="grid"))
        #
        # self._n_generation += 1
        out["F"] = objs
    def train_final_individual(self,solution,medmnist_dataset,is_medmnist):
        data_flag = self.medmnist_dataset
        output_root = './output'
        info = INFO[self.medmnist_dataset]
        num_epochs = 300
        data_augmentation = False
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
                                              decode_cell(decode_operations(individual[:-1],
                                                                            self.pop.indexes)))  # self.is_medmnist
            # First Search for augmentation policy
            # best_combination = self.evaluator.auto_search_daapolicy(decoded_individual, 100, hash_indv=None, grad_clip=5, evaluation='valid', data_flag=data_flag, output_root=output_root,
            #                             num_epochs=num_epochs, gpu_ids=gpu_ids, batch_size=batch_size,is_final=False, download=download, run=run)
            # # evaluation = 'test'
            best_combination = None
            loss = self.evaluator.train(best_combination, decoded_individual, 100, hash_indv=None, grad_clip=5,
                                        evaluation='test', data_flag=data_flag, output_root=output_root,
                                        num_epochs=num_epochs, gpu_ids=gpu_ids, batch_size=batch_size, is_final=False,
                                        download=download, run=run)
        else:
            is_final = True
            decoded_individual = NetworkCIFAR(3, n_classes, individual[-1], True,
                                              decode_cell(decode_operations(individual[:-1],
                                                                            self.pop.indexes)))  # self.is_medmnist
            # First Search for augmentation policy
            best_combination = self.evaluator.auto_search_daapolicy(decoded_individual, 25, hash_indv=None, grad_clip=5,
                                                                    evaluation='valid', data_flag=data_flag,
                                                                    output_root=output_root,
                                                                    num_epochs=num_epochs, gpu_ids=gpu_ids,
                                                                    batch_size=batch_size, is_final=False,
                                                                    download=download, run=run)
            evaluation = 'test'
            # best_combination = None
            loss = self.evaluator.train(best_combination, decoded_individual, 100, hash_indv=None, grad_clip=5,
                                        evaluation='test', data_flag=data_flag, output_root=output_root,
                                        num_epochs=num_epochs, gpu_ids=gpu_ids, batch_size=batch_size, is_final=False,
                                        download=download, run=run)

        print("loss", loss)

        # data_flag = self.medmnist_dataset
        # output_root = './output'
        # info = INFO[self.medmnist_dataset]
        # num_epochs = 300
        # gpu_ids = '0'
        # n_classes = len(info['label'])
        # batch_size = 32
        # download = True
        # run = 'model1'
        #
        # indv = []
        # for i in range(32):
        #     if i % 2 == 0:
        #         indv.append(individual[i])
        #     else:
        #         indv.append(int(random.choice(self.pop.params_choices[str(i)])))
        #             #indv.append(math.floor(x[j][i] * len(self.attentions)))
        # indv.append(int(math.floor(2 + ((self.layers - 2) * individual[-1]))))
        # individual = np.asarray(indv)
        #
        # decoded_individual = NetworkCIFAR(3, n_classes, self.layers, True,
        #                                     decode_cell(decode_operations(individual[:-1], self.pop.indexes)))
        #
        # evaluation = 'test'
        # best_combination = None
        # loss = self.evaluator.train(best_combination,decoded_individual, 100, hash_indv=None, grad_clip=5, evaluation=evaluation, data_flag=data_flag, output_root=output_root,
        #                             num_epochs=num_epochs, gpu_ids=gpu_ids, batch_size=batch_size,is_final=False, download=download, run=run)
        # print("loss", loss)




        print("Final loss is ",loss)
# Define a function to check if a solution dominates another
# def dominates(x, y):
#     return all(xi <= yi for xi, yi in zip(x, y)) and any(xi < yi for xi, yi in zip(x, y))
# Define a function to check if a solution dominates another
# Define a function to check if a solution dominates another
# def dominates(x, y, minimize_second_objective=True):
#     if minimize_second_objective:
#         return all(xi <= yi for xi, yi in zip(x, y)) and any(xi < yi for xi, yi in zip(x, y))
#     else:
#         return all(xi <= yi for xi, yi in zip(x, y)) and any(xi > yi for xi, yi in zip(x, y))

def dominates(solution1, solution2):
    # Assuming lower values are better for the first objective (minimize)
    # and higher values are better for the second objective (maximize)
    is_solution1_better = (solution1[0] <= solution2[0] and solution1[1] >= solution2[1])
    return is_solution1_better

# Function to calculate the Euclidean distance between two solutions
def euclidean_distance(solution1, solution2):
    return math.sqrt((solution1[0] - solution2[0]) ** 2 + (solution1[1] - solution2[1]) ** 2)


def weighted_sum(obj1,obj2):
    w=0.2
    return w*obj2-(1-w)*obj1

class MOGA(Optimizer):
    def __init__(self, population_size, number_of_generations, crossover_prob, mutation_prob, blocks_size, num_classes,
                 in_channels, epochs, batch_size, layers, n_channels, dropout_rate, retrain, resume_train, cutout,
                 multigpu_num,medmnist_dataset,is_medmnist,check_power_consumption,evaluation_type):
        super().__init__(population_size, number_of_generations, crossover_prob, mutation_prob, blocks_size,
                         num_classes, in_channels, epochs, batch_size, layers, n_channels, dropout_rate, retrain,
                         resume_train, cutout, multigpu_num,medmnist_dataset,is_medmnist,check_power_consumption,evaluation_type)

    def evolve(self):
        pop_size = 20
        seed = 50
        n_gens = 20
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
                         resume_train=self.resume_train, cutout=self.cutout, multigpu_num=self.multigpu_num,medmnist_dataset = self.medmnist_dataset,is_medmnist = self.is_medmnist,check_power_consumption=self.check_power_consumption,evaluation_type=self.evaluation_type)

        algorithm = NSGA2(pop_size=pop_size,
                          sampling=FloatRandomSampling(),
                          # crossover=TwoPointCrossover(prob=0.9),
                          # mutation=PolynomialMutation(prob=1.0 / n_var),
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
        objectives = []
        #objectives.append(problem.obj1)
        #objectives.append(problem.obj2)
        def find_pareto_front(solutions,individuals):
            pareto_front = []
            other_solutions = []
            pareto_indv = []
            other_indv = []

            for solution1 in solutions:
                is_dominated = False
                for solution2 in solutions:
                    if solution1 != solution2 and dominates(solution2, solution1):
                        is_dominated = True
                        break
                if not is_dominated:
                    pareto_front.append(solution1)
                    pareto_indv.append(individuals[solutions.index(solution1)])
                else:
                    other_solutions.append(solution1)
                    other_indv.append(individuals[solutions.index(solution1)])

            return pareto_front, other_solutions, pareto_indv,other_indv

        pareto_front, other_solutions, pareto_indv,other_indv = find_pareto_front(problem.archive_obj,problem.archive_var)

        # Extract objectives from the Pareto front and other instances
        objective1_pareto = [solution[0] for solution in pareto_front]
        objective2_pareto = [solution[1] for solution in pareto_front]
        objective1_other = [solution[0] for solution in other_solutions]
        objective2_other = [solution[1] for solution in other_solutions]
        #Using the weighted sum approach to select the best individual among pareto front
        # wb = [weighted_sum(objective1_pareto[i],objective2_pareto[i]) for i in range(len(objective1_pareto))]
        # get_index= wb.index(min(wb))
        # get_individ = pareto_indv[get_index]
        #Using the maximum zc score
        get_index = objective2_pareto.index(max(objective2_pareto))
        get_individ = pareto_indv[get_index]
        print("Best Individual Params",objective1_pareto[get_index])
        problem.train_final_individual(get_individ,self.medmnist_dataset,self.is_medmnist)
        # Create a scatter plot of the Pareto front and other instances
        plt.figure(figsize=(8, 6))
        plt.scatter(objective1_pareto, objective2_pareto, c='b', label='Pareto Front', marker='o')
        plt.scatter(objective1_other, objective2_other, c='r', label='Other Instances', marker='x')
        plt.xlabel('Objective 1 (Minimize)')
        plt.ylabel('Objective 2 (Maximize)')
        plt.title('Pareto Front and Other Instances')
        plt.legend()
        plt.savefig('figure_pareto_' + self.medmnist_dataset + '.png')
        plt.savefig('figure_pareto_' + self.medmnist_dataset + '.pdf')
        plt.grid(True)
        plt.show()

        # Compute Nadir point
        nadir_point = (max(objective1_pareto), max(objective2_pareto))

        # Compute hypervolume using Nadir point as reference
        hypervolume = 0.0

        for solution in pareto_front:
            hypervolume += max(0, nadir_point[0] - solution[0]) * max(0, nadir_point[1] - solution[1])

        # Compute diversity
        average_distance = 0.0

        for solution1 in pareto_front:
            min_distance = float('inf')
            for solution2 in pareto_front:
                if solution1 != solution2:
                    distance = euclidean_distance(solution1, solution2)
                    if distance < min_distance:
                        min_distance = distance
            average_distance += min_distance

        average_distance /= len(pareto_front)

        # Compute IGD
        igd = 0.0

        for solution in other_solutions:
            min_distance = float('inf')
            for pareto_solution in pareto_front:
                distance = euclidean_distance(solution, pareto_solution)
                if distance < min_distance:
                    min_distance = distance
            igd += min_distance

        igd /= len(other_solutions)

        # Create a scatter plot of the Pareto front and other instances
        plt.figure(figsize=(8, 6))
        plt.scatter(objective1_pareto, objective2_pareto, c='b', label='Pareto Front', marker='o')
        plt.scatter(objective1_other, objective2_other, c='r', label='Other Instances', marker='x')
        plt.xlabel('Objective 1 (Minimize)')
        plt.ylabel('Objective 2 (Maximize)')
        plt.title('Pareto Front and Other Instances')
        plt.legend()
        plt.grid(True)

        # Print metrics
        print(f'Hypervolume: {hypervolume}')
        print(f'Diversity: {average_distance}')
        print(f'IGD: {igd}')

        # Create a bar plot for the metrics
        metrics = ['Hypervolume', 'Diversity', 'IGD']
        values = [hypervolume, average_distance, igd]

        plt.figure(figsize=(8, 6))
        plt.bar(metrics, values)
        plt.ylabel('Value')
        plt.title('Metrics')
        plt.show()
        # objectives = np.array(problem.archive_obj)
        # #objectives = np.reshape(objectives, (objectives.shape[0], objectives.shape[1]))
        # # Calculate the Pareto front
        # pareto_front = []
        #
        # for solution in objectives:
        #     is_dominated = False
        #     to_remove = []
        #
        #     for idx, p in enumerate(pareto_front):
        #         if dominates(p, solution):
        #             is_dominated = True
        #             break
        #         if dominates(solution, p):
        #             to_remove.append(idx)
        #
        #     if not is_dominated:
        #         pareto_front = [p for idx, p in enumerate(pareto_front) if idx not in to_remove]
        #         pareto_front.append(solution)
        #
        # # Separate the Pareto front into individual objective functions
        # pareto_data = np.array(pareto_front)
        #
        # # Extract objective values for each axis
        # x = pareto_data[:, 0]
        # y = pareto_data[:, 1]
        #
        # # Extract other instances for comparison
        # other_instances = objectives[~np.isin(objectives, pareto_data).all(axis=1)]
        #
        # # Scatter plot for the Pareto front
        # plt.figure(figsize=(10, 6))
        # plt.scatter(x, y, c='b', marker='o', label='Pareto Front')
        #
        # # Scatter plot for other instances
        # plt.scatter(other_instances[:, 0], other_instances[:, 1], c='r', marker='x', label='Other Instances')
        #
        # # Set labels for the axes
        # plt.xlabel('Objective 1 (Minimized)')
        # plt.ylabel('Objective 2 (Maximized)')
        #
        # # Set a title for the plot
        # plt.title('Pareto Front vs. Other Instances')
        #
        # # Calculate Hypervolume
        # def hypervolume(front, reference_point):
        #     volume = 0
        #     for solution in front:
        #         if all(solution[i] <= reference_point[i] for i in range(2)):
        #             volume += np.prod(reference_point - solution)
        #     return volume
        #
        # reference_point = np.array([1e5, 1e5])
        # hv_value = hypervolume(pareto_data, reference_point)
        #
        # # Calculate Inverted Generational Distance (IGD)
        # def igd(front, reference_front):
        #     min_distances = []
        #     for ref_point in reference_front:
        #         min_distance = min(np.linalg.norm(solution - ref_point) for solution in front)
        #         min_distances.append(min_distance)
        #     return np.mean(min_distances)
        #
        # igd_value = igd(other_instances, pareto_data)
        #
        # # Calculate Diversity (IGD Plus)
        # def igd_plus(front, reference_front):
        #     distances = []
        #     for solution in front:
        #         distances.append(min(np.linalg.norm(solution - ref_point) for ref_point in reference_front))
        #     return np.mean(distances)
        #
        # diversity_value = igd_plus(other_instances, pareto_data)
        #
        # # Show the metrics
        # print(f'Hypervolume: {hv_value}')
        # print(f'IGD: {igd_value}')
        # print(f'Diversity (IGD Plus): {diversity_value}')
        #
        # # Show the legend
        # plt.legend()
        #
        # # Show the plot
        # plt.grid(True)
        # plt.show()

        # Access the Pareto front and Pareto set
        pareto_front = results.F
        pareto_set = results.X
        # Assuming 'results' contains your optimization results
        # results.F contains the objective values

        n_evals = []  # corresponding number of function evaluations\
        hist_F = []  # the objective space values in each generation
        hist_cv = []  # constraint violation in each generation
        hist_cv_avg = []  # average constraint violation in the whole population

        for algo in results.history:
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


# metric = IGD(pf, zero_to_one=True)
#
# igd = [metric.do(_F) for _F in hist_F]
#
# plt.plot(n_evals, igd,  color='black', lw=0.7, label="Avg. CV of Pop")
# plt.scatter(n_evals, igd,  facecolor="none", edgecolor='black', marker="p")
# plt.axhline(10**-2, color="red", label="10^-2", linestyle="--")
# plt.title("Convergence")
# plt.xlabel("Function Evaluations")
# plt.ylabel("IGD")
# plt.yscale("log")
# plt.legend()
# plt.show()