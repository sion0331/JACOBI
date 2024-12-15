import pickle
import os

from population.initial_generation import beautify_system


def save_history(config, history, time_records):
    results = [{'G': config.G, 'N': config.N, 'M': config.M, 'I': config.I, 'J': config.J,
                'allow_composite': config.allow_composite,
                'f0ps': str(config.f0ps), 'ivp_method': config.ivp_method, 'minimize_method': config.minimize_method,
                'elite_rate': config.elite_rate, 'crossover_rate': config.crossover_rate,
                'mutation_rate': config.mutation_rate, 'new_rate': config.new_rate,
                'time_records': time_records}]

    for i, generation in enumerate(history):
        for j, individual in enumerate(generation):
            result = {'Generation': i, 'Population': j, 'Score': individual[0].fun,
                      'System': beautify_system(individual[1]), 'param': individual[0].x}
            results.append(result)

    filename = (f"./data/results/{str(config.target.__class__.__name__)}_{str(config.f0ps)}"
                f"_G{config.G}_N{config.N}_M{config.M}_I{config.I}_J{config.I}_{config.minimize_method}"
                f"_{int(config.elite_rate * 100)}_{int(config.crossover_rate * 100)}"
                f"_{int(config.mutation_rate * 100)}_{int(config.new_rate * 100)}.pkl")
    with open(filename, "wb") as f:
        pickle.dump(results, f)
    print(f"History saved to {filename}.")


def load_history(target):
    directory = "../data/results/" + target
    histories = []
    for filename in os.listdir(directory):
        if filename.endswith('.pkl'):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'rb') as file:
                    history = pickle.load(file)
                    histories.append((filename, history))
                    print(f"Loaded {filename} successfully.")
            except Exception as e:
                print(f"Failed to load {filename}: {e}")
    return histories
