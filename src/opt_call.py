import time

import tqdm

import pandas as pd

import numpy as np

from heuristics import its, grasp, simple_greedy, simple_random

from operations import aux


def generate_experiments(df_pois, distance_matrix):

    pois = np.sort(df_pois[df_pois["General Category"] != "hotel"]["Name"].unique())

    # dataframe without repetition - sem repetição de locais
    df_wr = df_pois.drop_duplicates(subset=["Name"]).sort_values(by="Name")

    # the final route
    places_parameters = {
        "df": df_pois,
        "hotels": df_wr[df_wr["General Category"] == "hotel"]["Name"].values,
        "price": df_wr["Max Price"].values,
        "rating": df_wr["Mean Sentiment"].values,
        "category": df_wr["General Category"].values,
        "distance_matrix": distance_matrix,
        "places_available": pois,
    }

    # we have to explode the parameters
    users_parameters = aux.read_parameters("user.json")
    algorithm_parameters = aux.read_parameters("algorithm.json")
    experiments_parameters = aux.read_parameters("experiments.json")

    techniques = {
        "ITS": {
            "func": its.tabu_search_procedure,
            "parameters": aux.explode_parameters(algorithm_parameters["ITS"]),
        },
        "GRASP": {
            "func": grasp.grasp_heuristic,
            "parameters": aux.explode_parameters(algorithm_parameters["GRASP"]),
        },
        "Greedy": {
            "func": simple_greedy.construct_greedy_route,
            "parameters": [None],
        },
        "Random": {
            "func": simple_random.construct_random_route,
            "parameters": [None],
        },
    }

    users_parameters = aux.explode_parameters(users_parameters)

    for algorithm in techniques.keys():

        algo_func = techniques[algorithm]["func"]

        for algo_parameters in techniques[algorithm]["parameters"]:

            # list of parameters
            for user_parameter in users_parameters:

                user_parameter["travel_end"] = pd.to_datetime(
                    user_parameter["travel_end"], format="%d/%m/%Y %H:%M"
                )

                for _ in range(0, experiments_parameters["amount_executions"]):

                    remove_prob, tabu_size = None, None

                    if "remove_prob" in algorithm_parameters.keys():

                        remove_prob, tabu_size = (
                            algorithm_parameters["remove_prob"],
                            algorithm_parameters["tabu_size"],
                        )

                    start = time.time()

                    if algorithm in ["ITS", "GRASP"]:

                        solution, f = algo_func(
                            places_parameters.copy(),
                            user_parameter.copy(),
                            algo_parameters,
                        )

                    else:

                        solution, f = algo_func(
                            places_parameters.copy(), user_parameter.copy()
                        )

                    end = time.time()

                    aux.write_results(
                        "Results Analysis/results.txt",
                        user_parameter["budget"],
                        user_parameter["total_distance"],
                        end - start,
                        f,
                        solution,
                        remove_prob,
                        tabu_size,
                        algorithm,
                    )
