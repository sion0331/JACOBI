from population.genetic_algorithm import generate_new_population
from evaluators.parameter_estimation import *
from population.systems_generation import generate_population
from utils import models
from utils.functions import get_functions
import matplotlib.pyplot as plt

from utils.load_systems import create_ode_function, load_systems
from utils.plots import plot_2D


def main():
    ### Parameters ###
    target_func = models.lotka()
    num_equations, true_betas = models.lotka_defaults
    # plot_2D(target_func, true_betas)
    N = 3  # Maximum number of population
    M = 2  # Maximum number of equations
    I = 2  # Maximum number of terms per equation
    J = 2  # Maximum number of functions per feature
    allow_composite = False  # Set to False if you don't want composite functions
    f0ps = get_functions("5")  # Linear
    iterations = 2
    system_dir = 'data/lotka_equations.txt'  # 'data/differential_equations.txt'

    ### Generate synthetic data ###
    t = np.linspace(0, 10, 100)
    X0 = np.random.rand(num_equations)
    print(f"True parameters: {true_betas}")
    print(f"Initial Condition: {X0}")
    target_data = odeint(target_func, X0, t, args=tuple(true_betas))
    target_data += np.random.normal(0, 0.01, target_data.shape)
    print(f"Target Data: {target_data.shape}")

    ### Generate population ###
    best_model = None
    population = generate_population(N, M, I, J, f0ps, allow_composite)
    for i in range(iterations):
        if i != 0: generate_new_population(population)

        # Load systems
        best_error = float('inf')
        systems = load_systems(system_dir)
        for j, system in enumerate(systems):
            ode_func = create_ode_function(system)

            # Parameter estimation
            estimated_betas = estimate_parameters(ode_func, X0, t, target_data, true_betas)
            print(f"Generation {i} | System {j} | Estimated parameters: {estimated_betas}")

            # Evaluation
            pred_data = simulate_system(ode_func, X0, t, estimated_betas)
            error = calculate_error(pred_data, target_data)
            print(f"Generation {i} | System {j} | Error: {estimated_betas}")
            print("Mean squared error:", error)

            if error < best_error:
                best_error = error
                best_model = ode_func

    ### Plot best model ###
    estimated_betas = estimate_parameters(best_model, X0, t, target_data, true_betas)
    best_pred = simulate_system(best_model, X0, t, estimated_betas)
    plt.figure()
    plt.plot(target_data[:, 0], target_data[:, 1], "-", label="Target")
    plt.plot(best_pred[:, 0], best_pred[:, 1], "-", label="Prediction")
    plt.xlabel("Preys")
    plt.ylabel("Predators")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
