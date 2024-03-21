from itertools import compress
import random
import time
import matplotlib.pyplot as plt
from data import *


def initial_population(individual_size, population_size):
    return [[random.choice([True, False]) for _ in range(individual_size)] for _ in range(population_size)]


def fitness(items, knapsack_max_capacity, individual):
    total_weight = sum(compress(items['Weight'], individual))
    if total_weight > knapsack_max_capacity:
        return 0
    return sum(compress(items['Value'], individual))


def population_best(items, knapsack_max_capacity, population):
    best_individual = None
    best_individual_fitness = -1
    for individual in population:
        individual_fitness = fitness(items, knapsack_max_capacity, individual)
        if individual_fitness > best_individual_fitness:
            best_individual = individual
            best_individual_fitness = individual_fitness
    return best_individual, best_individual_fitness


def roulette_wheel_selection(items, knapsack_max_capacity, population, n_selection, n_elite):
    individual_fitness = [(fitness(items, knapsack_max_capacity, population[i]), i) for i in range(len(population))]
    population_sum = sum(individual_fit[0] for individual_fit in individual_fitness)
    selection_prob = [fit[0] / population_sum for fit in individual_fitness]

    parents = []
    for _ in range(n_selection):
        end_point = random.uniform(0, 1)
        temp_sum = 0

        for j in range(len(selection_prob)):
            temp_sum += selection_prob[j]
            if temp_sum > end_point:
                parents.append(population[j])
                break
    temp_individuals = sorted(individual_fitness, key=lambda x: x[0])[
                       len(individual_fitness) - n_elite - 1:len(individual_fitness) - 1]
    elite_individuals = [population[temp_individuals[i][1]] for i in range(len(temp_individuals))]

    return parents, elite_individuals


def crossover(parents, population_size, n_elite):
    new_generation = []
    size = population_size - n_elite
    while len(new_generation) < size:
        rand = random.sample(range(0, len(parents)), 2)
        first_child = parents[rand[0]][:len(parents[rand[0]]) // 2] + parents[rand[1]][len(parents[rand[1]]) // 2:]
        second_child = parents[rand[0]][len(parents[rand[0]]) // 2:] + parents[rand[1]][:len(parents[rand[1]]) // 2]
        new_generation.append(first_child)
        if len(new_generation) < size:
            new_generation.append(second_child)

    return new_generation


def mutation(new_generation, mutation_prob):
    for individual in new_generation:
        for i in range(len(individual)):
            if random.random() < mutation_prob:
                individual[i] = not individual[i]
    return new_generation


items, knapsack_max_capacity = get_big()
print(items)

population_size = 100
generations = 200
n_selection = 20
n_elite = 10
mutation_prob = 0.05

start_time = time.time()
best_solution = None
best_fitness = 0
population_history = []
best_history = []
population = initial_population(len(items), population_size)
for _ in range(generations):
    population_history.append(population)

    # TODO: implement genetic algorithm

    parents, elite_individuals = roulette_wheel_selection(items, knapsack_max_capacity, population, n_selection,
                                                          n_elite)
    population = mutation(crossover(parents, population_size, n_elite), mutation_prob)
    population += elite_individuals

    best_individual, best_individual_fitness = population_best(items, knapsack_max_capacity, population)
    if best_individual_fitness > best_fitness:
        best_solution = best_individual
        best_fitness = best_individual_fitness
    best_history.append(best_fitness)

end_time = time.time()
total_time = end_time - start_time
print('Best solution:', list(compress(items['Name'], best_solution)))
print('Best solution value:', best_fitness)
print('Time: ', total_time)

# plot generations
x = []
y = []
top_best = 10
for i, population in enumerate(population_history):
    plotted_individuals = min(len(population), top_best)
    x.extend([i] * plotted_individuals)
    population_fitnesses = [fitness(items, knapsack_max_capacity, individual) for individual in population]
    population_fitnesses.sort(reverse=True)
    y.extend(population_fitnesses[:plotted_individuals])
plt.figure(figsize=(1920 / 100, 1080 / 100), dpi=100)
plt.scatter(x, y, marker='.')
plt.plot(best_history, 'r')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.show()
