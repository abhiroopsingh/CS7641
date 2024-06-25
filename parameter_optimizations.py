import time

import mlrose_hiive
import matplotlib.pyplot as plt


# Helper function to produce optimized fitness and iterations relationship
def get_optimized_iterations(function_name, function, iteration_value, plot_path):
	# Fitness and Problem Setup
	default_discrete_prob = mlrose_hiive.DiscreteOpt(100, function, True)
	default_random_state = 12345

	# Random Hill Climbing
	rhc_state, rhc_fitness, rhc_curve = mlrose_hiive.random_hill_climb(default_discrete_prob, 100, iteration_value, random_state=default_random_state, curve=True)

	# Simulated Annealing
	sa_state, sa_fitness, sa_curve = mlrose_hiive.simulated_annealing(default_discrete_prob, mlrose_hiive.ExpDecay(), 100, iteration_value, random_state=default_random_state, curve=True)

	# Genetic Algorithm
	ga_state, ga_fitness, ga_curve = mlrose_hiive.genetic_alg(default_discrete_prob, max_attempts=100, max_iters=iteration_value, random_state=default_random_state, curve=True)

	plt.title(function_name + ' - Fitness / Iterations Relationship')
	plt.xlabel("Iterations")
	plt.ylabel("Fitness")
	plt.plot(rhc_curve[:, 0], label='Randomized Hill Climbing')
	plt.plot(sa_curve[:, 0], label='Simulated Annealing')
	plt.plot(ga_curve[:, 0], label='Genetic Algorithm')
	plt.legend()
	plt.savefig(plot_path)
	plt.clf()


# Helper function to produce optimized fitness and problem size plots
def get_optimized_problem_size(function_name, function, problem_size_values, plot_path):
	fitness = dict()
	fitness['rhc'] = list()
	fitness['sa'] = list()
	fitness['ga'] = list()

	for problem_size in problem_size_values:
		# Fitness and Problem Setup
		default_discrete_prob = mlrose_hiive.DiscreteOpt(problem_size, function, True)
		default_random_state = 12345

		# Random Hill Climbing
		rhc_state, rhc_fitness, rhc_curve = mlrose_hiive.random_hill_climb(default_discrete_prob, 100, 1000, random_state=default_random_state, curve=True)

		# Simulated Annealing
		sa_state, sa_fitness, sa_curve = mlrose_hiive.simulated_annealing(default_discrete_prob, mlrose_hiive.ExpDecay(), 100, 1000, random_state=default_random_state, curve=True)

		# Genetic Algorithm
		ga_state, ga_fitness, ga_curve = mlrose_hiive.genetic_alg(default_discrete_prob, max_attempts=100, max_iters=1000, random_state=default_random_state, curve=True)

		fitness['rhc'].append(rhc_fitness)
		fitness['sa'].append(sa_fitness)
		fitness['ga'].append(ga_fitness)

	plt.title(function_name + ' - Fitness / Problem Size Relationship')
	plt.xlabel("Problem Size")
	plt.ylabel("Fitness")
	plt.plot(problem_size_values, fitness['rhc'], label='Randomized Hill Climbing')
	plt.plot(problem_size_values, fitness['sa'], label='Simulated Annealing')
	plt.plot(problem_size_values, fitness['ga'], label='Genetic Algorithm')
	plt.legend()
	plt.savefig(plot_path)
	plt.clf()


# Helper function to produce optimized SA fitness and decay schedule relationship
def get_optimized_sa_decay_schedule(function_name, function, plot_path):
	# Fitness and Problem Setup
	default_discrete_prob = mlrose_hiive.DiscreteOpt(100, function, True)
	default_random_state = 12345

	# Simulated Annealing
	arith_state, arith_fitness, arith_curve = mlrose_hiive.simulated_annealing(default_discrete_prob, mlrose_hiive.ArithDecay(), 100, 1000, random_state=default_random_state, curve=True)
	exp_state, exp_fitness, exp_curve = mlrose_hiive.simulated_annealing(default_discrete_prob, mlrose_hiive.ExpDecay(), 100, 1000, random_state=default_random_state, curve=True)
	geom_state, geom_fitness, geom_curve = mlrose_hiive.simulated_annealing(default_discrete_prob, mlrose_hiive.GeomDecay(), 100, 1000, random_state=default_random_state, curve=True)

	plt.title('Simulated Annealing - ' + function_name + ' - Fitness / Decay Function Relationship')
	plt.xlabel("Iterations")
	plt.ylabel("Fitness")
	plt.plot(arith_curve[:, 0], label='ArithDecay()')
	plt.plot(exp_curve[:, 0], label='ExpDecay()')
	plt.plot(geom_curve[:, 0], label='GeomDecay()')
	plt.legend()
	plt.savefig(plot_path)
	plt.clf()


# Helper function to produce optimized fitness and problem size plots
def get_optimized_wall_clock_time(function_name, function, problem_size_values, plot_path):
	wall_clock = dict()
	wall_clock['rhc'] = list()
	wall_clock['sa'] = list()
	wall_clock['ga'] = list()

	for problem_size in problem_size_values:
		# Fitness and Problem Setup
		default_discrete_prob = mlrose_hiive.DiscreteOpt(problem_size, function, True)
		default_random_state = 12345

		# Random Hill Climbing
		rhc_start_time = time.time()
		rhc_state, rhc_fitness, rhc_curve = mlrose_hiive.random_hill_climb(default_discrete_prob, 100, 1000, random_state=default_random_state, curve=True)
		rhc_end_time = time.time()

		# Simulated Annealing
		sa_start_time = time.time()
		sa_state, sa_fitness, sa_curve = mlrose_hiive.simulated_annealing(default_discrete_prob, mlrose_hiive.ExpDecay(), 100, 1000, random_state=default_random_state, curve=True)
		sa_end_time = time.time()

		# Genetic Algorithm
		ga_start_time = time.time()
		ga_state, ga_fitness, ga_curve = mlrose_hiive.genetic_alg(default_discrete_prob, max_attempts=100, max_iters=1000, random_state=default_random_state, curve=True)
		ga_end_time = time.time()

		wall_clock['rhc'].append(rhc_end_time - rhc_start_time)
		wall_clock['sa'].append(sa_end_time - sa_start_time)
		wall_clock['ga'].append(ga_end_time - ga_start_time)

	plt.title(function_name + ' - Time / Problem Size Relationship')
	plt.xlabel("Problem Size")
	plt.ylabel("Time (seconds)")
	plt.plot(problem_size_values, wall_clock['rhc'], label='Randomized Hill Climbing')
	plt.plot(problem_size_values, wall_clock['sa'], label='Simulated Annealing')
	plt.plot(problem_size_values, wall_clock['ga'], label='Genetic Algorithm')
	plt.legend()
	plt.savefig(plot_path)
	plt.clf()


if __name__ == '__main__':
	# Visualization of Fitness / Iterations Relationship
	get_optimized_iterations('FourPeaks', mlrose_hiive.FourPeaks(), 1000, 'images/fitness_100_iterations_four_peaks.png')
	get_optimized_iterations('FlipFlop', mlrose_hiive.FlipFlop(), 1000, 'images/fitness_100_iterations_flip_flop.png')

	# Visualization of Fitness / Problem Size Relationship
	get_optimized_problem_size('FourPeaks', mlrose_hiive.FourPeaks(), range(100, 500, 100), 'images/fitness_problem_size_four_peaks.png')
	get_optimized_problem_size('FlipFlop', mlrose_hiive.FlipFlop(), range(100, 500, 100), 'images/fitness_problem_size_flip_flop.png')

	# Visualization of Simulated Annealing Fitness / Decay Schedule Relationship
	get_optimized_sa_decay_schedule('FourPeaks', mlrose_hiive.FourPeaks(), 'images/fitness_sa_decay_four_peaks.png')
	get_optimized_sa_decay_schedule('FlipFlop', mlrose_hiive.FlipFlop(), 'images/fitness_sa_decay_flip_flop.png')

	# Visualization of Wall Clock / Problem Size Relationship
	get_optimized_wall_clock_time('FourPeaks', mlrose_hiive.FourPeaks(), range(100, 1000, 100), 'images/fitness_wall_clock_four_peaks.png')
	get_optimized_wall_clock_time('FlipFlop', mlrose_hiive.FlipFlop(), range(100, 1000, 100), 'images/fitness_wall_clock_flip_flop.png')
