import time

import mlrose_hiive
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def get_optimized_nn_iterations(x_train, x_test, y_train, y_test):
	default_random_state = 12345
	accuracy_dict = dict()
	accuracy_dict['rhc_train'] = list()
	accuracy_dict['rhc_test'] = list()
	accuracy_dict['sa_train'] = list()
	accuracy_dict['sa_test'] = list()
	accuracy_dict['ga_train'] = list()
	accuracy_dict['ga_test'] = list()

	for iteration_value in range(100, 1000, 100):
		# Random Hill Climbing
		rhc_nn = mlrose_hiive.NeuralNetwork(hidden_nodes=[2], algorithm='random_hill_climb',
		                                    max_iters=iteration_value, early_stopping=True,
		                                    max_attempts=100, random_state=default_random_state)

		rhc_nn.fit(x_train, y_train)

		rhc_nn_y_train_prediction = rhc_nn.predict(x_train)
		rhc_y_train_accuracy = accuracy_score(y_train, rhc_nn_y_train_prediction)
		accuracy_dict['rhc_train'].append(rhc_y_train_accuracy)

		rhc_nn_y_test_prediction = rhc_nn.predict(x_test)
		rhc_y_test_accuracy = accuracy_score(y_test, rhc_nn_y_test_prediction)
		accuracy_dict['rhc_test'].append(rhc_y_test_accuracy)

		# Simulated Annealing
		sa_nn = mlrose_hiive.NeuralNetwork(hidden_nodes=[2], algorithm='simulated_annealing',
		                                   max_iters=iteration_value, early_stopping=True,
		                                   max_attempts=100, random_state=default_random_state)

		sa_nn.fit(x_train, y_train)

		sa_nn_y_train_prediction = sa_nn.predict(x_train)
		sa_y_train_accuracy = accuracy_score(y_train, sa_nn_y_train_prediction)
		accuracy_dict['sa_train'].append(sa_y_train_accuracy)

		sa_nn_y_test_prediction = sa_nn.predict(x_test)
		sa_y_test_accuracy = accuracy_score(y_test, sa_nn_y_test_prediction)
		accuracy_dict['sa_test'].append(sa_y_test_accuracy)

		# Genetic Algorithm
		ga_nn = mlrose_hiive.NeuralNetwork(hidden_nodes=[2], algorithm='genetic_alg',
		                                   max_iters=iteration_value, early_stopping=True,
		                                   max_attempts=100, random_state=default_random_state)

		ga_nn.fit(x_train, y_train)

		ga_nn_y_train_prediction = ga_nn.predict(x_train)
		ga_y_train_accuracy = accuracy_score(y_train, ga_nn_y_train_prediction)
		accuracy_dict['ga_train'].append(ga_y_train_accuracy)

		ga_nn_y_test_prediction = ga_nn.predict(x_test)
		ga_y_test_accuracy = accuracy_score(y_test, ga_nn_y_test_prediction)
		accuracy_dict['ga_test'].append(ga_y_test_accuracy)

	plt.title('[Training] Accuracy / Iterations Relationship')
	plt.xlabel("Iterations")
	plt.ylabel("Accuracy")
	plt.plot(range(100, 1000, 100), accuracy_dict['rhc_train'], label='Randomized Hill Climbing')
	plt.plot(range(100, 1000, 100), accuracy_dict['sa_train'], label='Simulated Annealing')
	plt.plot(range(100, 1000, 100), accuracy_dict['ga_train'], label='Genetic Algorithm')
	plt.legend()
	plt.savefig('images/nn_training_accuracy.png')
	plt.clf()

	plt.title('[Testing] Accuracy / Iterations Relationship')
	plt.xlabel("Iterations")
	plt.ylabel("Accuracy")
	plt.plot(range(100, 1000, 100), accuracy_dict['rhc_test'], label='Randomized Hill Climbing')
	plt.plot(range(100, 1000, 100), accuracy_dict['sa_test'], label='Simulated Annealing')
	plt.plot(range(100, 1000, 100), accuracy_dict['ga_test'], label='Genetic Algorithm')
	plt.legend()
	plt.savefig('images/nn_testing_accuracy.png')
	plt.clf()


def get_optimized_sa_decay_schedule(x_train, x_test, y_train, y_test):
	default_random_state = 12345

	# Simulated Annealing - ArithDecay()
	arith_nn = mlrose_hiive.NeuralNetwork(hidden_nodes=[2], algorithm='simulated_annealing',
	                                      max_iters=1000, early_stopping=True,
	                                      max_attempts=100, random_state=default_random_state,
	                                      schedule=mlrose_hiive.ArithDecay())

	arith_nn.fit(x_train, y_train)

	arith_nn_y_train_prediction = arith_nn.predict(x_train)
	arith_nn_y_train_accuracy = accuracy_score(y_train, arith_nn_y_train_prediction)

	arith_nn_y_test_prediction = arith_nn.predict(x_test)
	arith_nn_y_test_accuracy = accuracy_score(y_test, arith_nn_y_test_prediction)

	# Simulated Annealing - ExpDecay()
	exp_nn = mlrose_hiive.NeuralNetwork(hidden_nodes=[2], algorithm='simulated_annealing',
	                                    max_iters=1000, early_stopping=True,
	                                    max_attempts=100, random_state=default_random_state,
	                                    schedule=mlrose_hiive.ExpDecay())

	exp_nn.fit(x_train, y_train)

	exp_nn_y_train_prediction = exp_nn.predict(x_train)
	exp_nn_y_train_accuracy = accuracy_score(y_train, exp_nn_y_train_prediction)

	exp_nn_y_test_prediction = exp_nn.predict(x_test)
	exp_nn_y_test_accuracy = accuracy_score(y_test, exp_nn_y_test_prediction)

	# Simulated Annealing - GeomDecay()
	geom_nn = mlrose_hiive.NeuralNetwork(hidden_nodes=[2], algorithm='simulated_annealing',
	                                   max_iters=1000, early_stopping=True,
	                                   max_attempts=100, random_state=default_random_state,
	                                   schedule=mlrose_hiive.GeomDecay())
	geom_nn.fit(x_train, y_train)

	geom_nn_y_train_prediction = geom_nn.predict(x_train)
	geom_nn_y_train_accuracy = accuracy_score(y_train, geom_nn_y_train_prediction)

	geom_nn_y_test_prediction = geom_nn.predict(x_test)
	geom_nn_y_test_accuracy = accuracy_score(y_test, geom_nn_y_test_prediction)

	decay_function_list = ['ArithDecay()','ExpDecay()','GeomDecay()']
	training_accuracy_list = [arith_nn_y_train_accuracy, exp_nn_y_train_accuracy, geom_nn_y_train_accuracy]
	testing_accuracy_list = [arith_nn_y_test_accuracy, exp_nn_y_test_accuracy, geom_nn_y_test_accuracy]

	plt.title('[Training] Accuracy / Decay Function Relationship')
	plt.xlabel("Decay Function")
	plt.ylabel("Accuracy")
	plt.plot(decay_function_list, training_accuracy_list, label='Simulated Annealing')
	plt.legend()
	plt.savefig('images/sa_decay_training_accuracy.png')
	plt.clf()

	plt.title('[Testing] Accuracy / Decay Function Relationship')
	plt.xlabel("Decay Function")
	plt.ylabel("Accuracy")
	plt.plot(decay_function_list, testing_accuracy_list, label='Simulated Annealing')
	plt.legend()
	plt.savefig('images/sa_decay_testing_accuracy.png')
	plt.clf()


def get_optimized_nn_wall_clock_time(x_train, y_train):
	default_random_state = 12345
	wall_clock_dict = dict()
	wall_clock_dict['rhc_train'] = list()
	wall_clock_dict['sa_train'] = list()
	wall_clock_dict['ga_train'] = list()

	for iteration_value in range(100, 1000, 100):
		# Random Hill Climbing
		rhc_nn = mlrose_hiive.NeuralNetwork(hidden_nodes=[2], algorithm='random_hill_climb',
		                                    max_iters=iteration_value, early_stopping=True,
		                                    max_attempts=100, random_state=default_random_state)
		rhc_start_time = time.time()
		rhc_nn.fit(x_train, y_train)
		rhc_end_time = time.time()
		wall_clock_dict['rhc_train'].append(rhc_end_time - rhc_start_time)

		# Simulated Annealing
		sa_nn = mlrose_hiive.NeuralNetwork(hidden_nodes=[2], algorithm='simulated_annealing',
		                                   max_iters=iteration_value, early_stopping=True,
		                                   max_attempts=100, random_state=default_random_state)
		sa_start_time = time.time()
		sa_nn.fit(x_train, y_train)
		sa_end_time = time.time()
		wall_clock_dict['sa_train'].append(sa_end_time - sa_start_time)

		# Genetic Algorithm
		ga_nn = mlrose_hiive.NeuralNetwork(hidden_nodes=[2], algorithm='genetic_alg',
		                                   max_iters=iteration_value, early_stopping=True,
		                                   max_attempts=100, random_state=default_random_state)
		ga_start_time = time.time()
		ga_nn.fit(x_train, y_train)
		ga_end_time = time.time()
		wall_clock_dict['ga_train'].append(ga_end_time - ga_start_time)

	plt.title('Training Time / Iterations Relationship')
	plt.xlabel("Iterations")
	plt.ylabel("Training Time (seconds)")
	plt.plot(range(100, 1000, 100), wall_clock_dict['rhc_train'], label='Randomized Hill Climbing')
	plt.plot(range(100, 1000, 100), wall_clock_dict['sa_train'], label='Simulated Annealing')
	plt.plot(range(100, 1000, 100), wall_clock_dict['ga_train'], label='Genetic Algorithm')
	plt.legend()
	plt.savefig('images/nn_training_wall_clock.png')
	plt.clf()


if __name__ == '__main__':
	# Read Data
	df = pd.read_csv("data/heart_failure.csv")
	features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
	df[features] = df[features].astype(object)

	# Usage of Target Column
	x = df.drop(['target'], axis=1)
	y = df['target']

	x = MinMaxScaler().fit_transform(x)

	# Break the dataset into four parts
	xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=4)

	# Visualization of Neural Network Accuracy / Iterations Relationship
	get_optimized_nn_iterations(xtrain, xtest, ytrain, ytest)

	# Visualization of Neural Network Accuracy / SA Decay Schedule Relationship
	get_optimized_sa_decay_schedule(xtrain, xtest, ytrain, ytest)

	# Visualization of Neural Network Wall Clock / Problem Size Relationship
	get_optimized_nn_wall_clock_time(xtrain, ytrain)
