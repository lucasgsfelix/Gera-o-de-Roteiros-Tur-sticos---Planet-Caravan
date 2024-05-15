import datetime
import numpy as np


from model import objective_function
from operations import constructive_operations as co


def generate_random_route_array(route, users_parameters):
    ## adiciona o hotel ao longo da rota

    positions = np.array(
        [
            day * (users_parameters["amount_daily_places"])
            for day in range(0, users_parameters["amount_days"])
        ]
    )

    # adiconando o hotel em todas as posições e no final
    return np.append(np.insert(route[1:], positions, route[0]), route[0])


def verify_category_constraints(places_parameters, users_parameters, complete_route):

    categories = places_parameters["category"][complete_route]

    for daily_route in co.split_array_at_value(categories, "hotel"):

        for category in np.unique(categories):

            if len(daily_route[daily_route == category]) > users_parameters["category"]:

                return False

    return True


def generate_distance_feasible_route(route, distance_matrix, users_parameters):

    complete_route, total_distance = np.array([route[0]]), 0

    ## o hotel é a posição 0
    for source_poi, destiny_poi in zip(route[:-1], route[1:]):

        total_distance += distance_matrix[source_poi][destiny_poi]

        if (
            total_distance + distance_matrix[destiny_poi][route[0]]
            < users_parameters["total_distance"]
        ):

            complete_route = np.append(complete_route, destiny_poi)

        else:

            total_distance = 0

            ## então foi o hotel duas vezes, a rota não é viável
            if complete_route[-1] == route[0]:

                return False, complete_route

            complete_route = np.append(complete_route, route[0])

    # o hotel não foi o último visitado
    if route[0] != complete_route[-1]:

        complete_route = np.append(complete_route, route[0])

    return True, complete_route


def verify_complete_route(route, places_parameters, users_parameters):

    total_cost = np.sum(places_parameters["price"][route[1:]]) + (
        places_parameters["price"][route[0]] * users_parameters["amount_days"]
    )

    if total_cost >= users_parameters["budget"]:

        return False

    distance_feasible, complete_route = generate_distance_feasible_route(
        route, places_parameters["distance_matrix"], users_parameters
    )

    if not distance_feasible:

        return False
    ## não precisamos verificar se os locais são únicos, pois garantimos isso na geração
    ## não precisamos verificar a quantidade de locais pois também garantimos na geração

    ## verificando categorias
    return verify_category_constraints(
        places_parameters, users_parameters, complete_route
    )


def generate_random_route(places_parameters, users_parameters):
    """

    Responsável pela geração de uma rota aleatória

    """

    route = np.append(np.array([]), np.random.choice(places_parameters["hotels"], 1)[0])

    # amount_days = (users_parameters['travel_end'] - users_parameters['datetime_start']).days

    selected_places = np.array([])

    hotel = np.random.choice(places_parameters["hotels"], 1)

    amout_places = (
        users_parameters["amount_days"] * users_parameters["amount_daily_places"]
    )

    places = np.random.choice(
        np.unique(places_parameters["places_available"]),
        np.random.randint(users_parameters["amount_days"], amout_places),
    )

    route = np.append(hotel, places)

    if verify_complete_route(route, places_parameters, users_parameters):

        return (
            objective_function.measure_fitness([route], places_parameters["rating"]),
            route,
            True,
        )

    return None, None, False


def construct_random_route(places_parameters, users_parameters):

    feasible = False

    while not feasible:

        fitness, route, feasible = generate_random_route(
            places_parameters, users_parameters
        )

    _, route = generate_distance_feasible_route(
        route, places_parameters["distance_matrix"], users_parameters
    )

    return route, fitness
