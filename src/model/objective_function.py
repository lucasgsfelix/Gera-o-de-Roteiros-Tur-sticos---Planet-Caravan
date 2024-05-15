import numpy as np


def measure_fitness(valid_routes, ratings):

    geral_fitness = np.array(
        [
            np.sum(ratings[np.unique(np.array(route, dtype=int))])
            for route in valid_routes
        ]
    )

    return geral_fitness


def retrieve_best_feasible_route(valid_routes, ratings, df=None):

    geral_fitness = measure_fitness(valid_routes, ratings)

    if len(geral_fitness) > 0:

        best_route_index = np.argmax(geral_fitness)

        return np.array(valid_routes[best_route_index])

    else:

        return np.array([])
