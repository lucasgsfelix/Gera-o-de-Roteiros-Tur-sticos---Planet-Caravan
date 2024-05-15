### GRASP Algorithm

import numpy as np

from heuristics import random_greedy_procedure
from model import objective_function
from operations import constructive_operations as co


def verify_local_search_changes(
    route_parameters,
    users_parameters,
    places_parameters,
    old_place,
    new_place,
    index,
    day,
):
    ## temos que verificar o budget
    ## temos que verificar a distância

    distance_matrix = places_parameters["distance_matrix"]

    budget = (
        route_parameters["budget"]
        - places_parameters["price"][old_place]
        + places_parameters["price"][new_place]
    )

    if budget > users_parameters["budget"]:

        return False, route_parameters

    # quer dizer que é ultimo local a ser visitado naquele dia
    if len(route_parameters[day]["route"]) - 1 == index:

        ## o último local é o hotel
        before, after = (
            route_parameters[day]["route"][index - 1],
            route_parameters[day]["route"][0],
        )
    else:

        before, after = (
            route_parameters[day]["route"][index - 1],
            route_parameters[day]["route"][index + 1],
        )

    # tirando a distância gasta visitando aquele local anterior
    distance = (
        route_parameters[day]["daily_distance"]
        - distance_matrix[before][old_place]
        - distance_matrix[old_place][after]
    )

    distance = (
        route_parameters[day]["daily_distance"]
        + distance_matrix[before][new_place]
        + distance_matrix[new_place][after]
    )

    if distance >= users_parameters["total_distance"]:

        return False, route_parameters

    route_parameters[day]["daily_distance"] = distance

    route_parameters[day]["daily_distance"] = budget

    return True, route_parameters


def local_search(route_parameters, users_parameters, places_parameters):

    distance_matrix = places_parameters["distance_matrix"]

    categories = places_parameters["category"]

    ratings = places_parameters["rating"]

    route = co.generate_route_array(users_parameters, route_parameters)

    for day in range(0, users_parameters["amount_days"]):

        for index, place in enumerate(route_parameters[day]["route"][1:]):

            # outros locais que vão ser visitados outros dias não podem estar na rota
            pois = np.setdiff1d(places_parameters["places_available"], route)

            pois = np.setdiff1d(pois, route_parameters[day]["route"])

            if ratings[place] != 5:

                ## então há espaço de melhoria
                ## quais são os locais mais próximos do local atual
                ## que possuem mesma categoria, e possuem uma boa nota

                places_available = pois[ratings[pois] > ratings[place]]

                places_available = places_available[
                    categories[place] == categories[places_available]
                ]

                visit_gain = (
                    distance_matrix[place][places_available] / ratings[places_available]
                )

                if places_available.size > 0:

                    places = places_available[visit_gain == np.min(visit_gain)]

                    # aqui temos que verificar a viabilidade dessa rota

                    (indices,) = np.where(
                        route == route_parameters[day]["route"][index + 1]
                    )

                    new_place = np.random.choice(places, 1)[0]

                    respect_constraints, route_parameters = verify_local_search_changes(
                        route_parameters,
                        users_parameters,
                        places_parameters,
                        place,
                        new_place,
                        index,
                        day,
                    )

                    ## não respeita as restrições
                    if not respect_constraints:

                        continue

                    route[indices] = new_place

                    # tenho que garantir que nota rota é válida
                    # checagem de restrições
                    route_parameters[day]["route"][index + 1] = new_place

    route = co.generate_route_array(users_parameters, route_parameters)

    return route_parameters, objective_function.measure_fitness(
        [route], places_parameters["rating"]
    )


def inverse_probability(probabilities):

    # Passo 2: Subtrair cada probabilidade da soma total
    inverse_proba = np.array([np.sum(probabilities) - p for p in probabilities])

    # Passo 3: Normalizar as probabilidades
    return np.array([p / np.sum(inverse_proba) for p in inverse_proba])


def select_alpha(fitness, alphas):
    """

    Implementação de acordo com o livro do Talbi

    ## z* == fitness
    ## Ai == np.mean(alphas[alpha]) --> media do alpha atual
    ## qj = z*/Ai
    ## qi/ sum Ai
    pi = qi/sum qj

    """

    alpha_mean = np.array(
        [np.mean(alphas[alpha]["results"]) for alpha in alphas.keys()]
    )

    mean_ai = fitness / alpha_mean

    # novas probabilidades
    probability_alpha = [
        fitness / alpha_mean[alpha_index] for alpha_index in range(0, len(alphas))
    ] / np.sum(mean_ai)

    ## dando maior probabilidade para quem o fitness maior
    ## a função feita pelo livro do talbi tem como objetivo a minimização
    ## a função acima tem por objetivo realizar a probabilidade inversa
    ## ou seja, o alpha que antes tinha a menor probabilidade passa a ser maior
    return np.random.choice(
        list(alphas.keys()), 1, p=inverse_probability(probability_alpha)
    )[0]


def generate_daily_array_route(users_parameters, route_parameters):

    route = np.array([], dtype=int)

    for day in range(0, users_parameters["amount_days"]):

        if day == 0:

            route = np.append(route, route_parameters[day]["route"])

        else:

            route = np.append(route, route_parameters[day]["route"])

        route = np.array(route, route[0])

    return route


def grasp_heuristic(places_parameters, users_parameters, algorithm_parameters):

    iteration, best_fitness, best_solution = 0, 0, None

    alphas = {alpha / 10: {"results": np.array([])} for alpha in range(3, 8)}

    alpha_index, count_alpha = 0, 0

    while iteration < algorithm_parameters["max_iterations"]:

        if count_alpha < len(alphas):

            alpha = list(alphas.keys())[alpha_index]

            alpha_index += 1

            count_alpha += 1

        solution = random_greedy_procedure.random_greedy(
            places_parameters, users_parameters, alpha
        )

        route = co.generate_route_array(users_parameters, solution)

        before_ls = objective_function.measure_fitness(
            [route], places_parameters["rating"]
        )

        solution, fitness = local_search(solution, users_parameters, places_parameters)

        alphas[alpha]["results"] = np.append(alphas[alpha]["results"], fitness[0])

        if count_alpha >= len(alphas):
            ## durante as 5 primeiras rodadas utilizamos um alpha de cada
            ## para poder ter insumo e fazer o calculo
            alpha = select_alpha(fitness[0], alphas)

        # selectiona o alpha com base na probabilidade

        if iteration == 0 or fitness > best_fitness:

            best_fitness = fitness

            best_solution = solution

        iteration += 1

    return generate_daily_array_route(users_parameters, best_solution), best_fitness
