from population.genetic_algorithm import generate_new_population
from evaluators.parameter_estimation import *
from population.initial_generation import generate_population, manual_lotka_systems, beautify_system
from utils.functions import get_functions, funcs_to_str

from utils.load_systems import create_ode_function, load_systems
from utils.models import lotka
from utils.plots import plot_2d_by_func, plot_2d_by_y, plot_loss_by_iteration, plot_invalid_by_iteration
import matplotlib.pyplot as plt
import warnings


class Config:
    def __init__(self):
        self.target = lotka()

        self.G = 10  # Number of generations
        self.N = 50  # Maximum number of population
        self.M = 2  # Maximum number of equations
        self.I = 2  # Maximum number of terms per equation
        self.J = 2  # Maximum number of functions per feature
        self.allow_composite = False  # Composite Functions
        self.f0ps = get_functions("1,2,4,5,6")
        self.ivp_method = 'Radau'
        self.minimize_method = 'Nelder-Mead'

        self.elite_rate = 0.25
        self.crossover_rate = 0.5
        self.mutation_rate = 0.8
        self.new_rate = 0.2

        self.system_load_dir = 'data/differential_equations.txt'  # 'data/lotka_equations.txt'
        self.system_save_dir = 'data/differential_equations.txt'

        self.DEBUG = False


def main():
    #######################################################################
    #                         PARAMETERS                                  #
    #######################################################################

    config = Config()

    #######################################################################
    #                         TARGET DATA                                 #
    #######################################################################

    t = np.linspace(0, 10, 100)
    X0 = np.random.rand(config.target.N) + 1.0  # 1.0~2.0
    print(f"true_betas: {config.target.betas} | Initial Condition: {X0}")

    y_raw = solve_ivp(config.target.func, (t[0], t[-1]), X0, args=config.target.betas, t_eval=t,
                      method=config.ivp_method).y.T
    y_target = y_raw + np.random.normal(0.0, 0.02, y_raw.shape)

    #######################################################################
    #                         INITIAL POPULATION                          #
    #######################################################################

    population = generate_population(config)  # manual_lotka_systems()
    systems = load_systems(config.system_load_dir)

    #######################################################################
    #                         RUN                                     #
    #######################################################################

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        ### Comparing methods
        # ode_func = create_ode_function(systems[0])
        # initial_guess = np.random.uniform(low=-1.0, high=1.0, size=(config.I * config.M,))
        # initial_guess = np.zeros(config.I * config.M)
        # print(f'Initial Guess: {initial_guess}')
        # for min_method in ['Nelder-Mead','Powell','CG','BFGS','L-BFGS-B','COBYLA','COBYQA','SLSQP']: #COBYLA SLSQP
        #     for ivp_method in ['RK45','RK23','DOP853','Radau', 'BDF']: #'LSODA'
        #         start_time = time.time()
        #         solved = estimate_parameters(ode_func, X0, t, y_target, initial_guess, min_method, ivp_method, config.DEBUG)
        #         print(f'min_method:{min_method}, ivp_method:{ivp_method} | Error: {solved.fun} | ts: {time.time()-start_time}s | Solved Parameters: {solved.x}')

        history = []
        for i in range(config.G):
            print("#### SYSTEM EVALUATION ####")
            history.append([])
            for j, system in enumerate(systems):
                ode_func = create_ode_function(system)
                initial_guess = np.zeros(config.I * config.M)  # TODO - improve
                solved = estimate_parameters(ode_func, X0, t, y_target, initial_guess, config.minimize_method,
                                             config.ivp_method, config.DEBUG)
                history[i].append((solved, population[j], ode_func))
                print(
                    f"### Generation {i} #{j} | Error: {round(solved.fun, 4)} | System: {beautify_system(population[j])} | Solved Parameters: {solved.x}")
            if i < config.G - 1:
                population = generate_new_population(history[i], population, config)
                systems = load_systems(config.system_load_dir)

    #######################################################################
    #                         RESULT                                      #
    #######################################################################

    print("\n#### RESULT ####")
    best = None
    min_loss = []
    avg_loss = []
    invalid = []
    for i, generation in enumerate(history):
        loss = []
        for j, individual in enumerate(generation):
            print(
                f'generation {i} {j} | loss:{round(individual[0].fun, 4)} func: {beautify_system(individual[1])} param:{individual[0].x}')
            loss.append(individual[0].fun)
            if best is None or individual[0].fun < best[0].fun:
                best = individual
        min_loss.append(min(loss))
        avg_loss.append(np.mean([l for l in loss if l < 100]))  # Exclude loss > 100
        invalid.append(sum(l >= 100 for l in loss))

    print(f'\nBest | Loss:{best[0].fun} func: {best[1]} param:{best[0].x}')

    # TODO - test with new target data
    y_best = solve_ivp(best[2], (t[0], t[-1]), X0, args=tuple(best[0].x), t_eval=t, method=config.ivp_method).y.T
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    plot_2d_by_func(axs[0, 0], config.target.func, config.target.betas)
    plot_2d_by_y(axs[0, 1], [y_raw, y_target, y_best], ["TARGET_RAW", "TARGET_NOISED", "BEST"])
    plot_loss_by_iteration(axs[1, 0], min_loss, avg_loss)
    plot_invalid_by_iteration(axs[1, 1], invalid)

    note = f""" Target:{type(config.target).__name__} | G:{config.G} N:{config.N} M:{config.M} I:{config.I} J:{config.J} f0ps:{funcs_to_str(config.f0ps)} Composite:{config.allow_composite} | elite:{config.elite_rate} new:{config.new_rate} cross:{config.crossover_rate} mutate:{config.mutation_rate}| ivp:{config.ivp_method} min:{config.minimize_method}
    Best Function: {beautify_system(best[1])} 
    Best Loss: {best[0]['fun']} Best Parameters: {best[0]['x']}"""
    fig.text(0.03, 0.08, note, va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.show()


if __name__ == "__main__":
    main()
