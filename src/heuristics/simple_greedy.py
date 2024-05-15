import numpy as np

from model import objective_function
from model import constraints
from operations import constructive_operations as co


def greedy_hotel_selection(places_parameters, users_parameters):
    ### o melhor hotel que é mais barato que o o budget

    hotels = places_parameters["hotels"]

    # tem que ser hotel, e o valor das diárias não pode ultrapassar o budget
    hotels = hotels[
        (places_parameters["rating"][hotels] == 5)
        & (
            places_parameters["price"][hotels] * users_parameters["amount_days"]
            < users_parameters["budget"]
        )
    ]

    # hotels = hotels[places_parameters['rating'] == 5]

    return (
        hotels[0],
        places_parameters["price"][hotels[0]] * users_parameters["amount_days"],
    )


def generate_daily_array_route(users_parameters, route_parameters):

    route = np.array([], dtype=int)

    for day in range(0, users_parameters["amount_days"]):

        if day == 0:

            route = np.append(route, route_parameters[day]["route"])

        else:

            route = np.append(route, route_parameters[day]["route"])

        route = np.array(route, route[0])

    return route


def construct_greedy_route(places_parameters, users_parameters):
    """

    Assumption:
    Dado que estou em current time X
    quem é o melhor local a visitar que esta mais próximo e respeita as restrições?

    """

    hotel, cost = greedy_hotel_selection(places_parameters, users_parameters)

    route_parameters = {"budget": cost}

    for day in range(0, users_parameters["amount_days"]):

        route_parameters[day] = {
            "daily_distance": 0,
            "restaurant": 0,
            "attraction": 0,
            "amount_places": 0,
            "route": np.array([hotel]),
        }

    sort_index = np.flip(
        np.argsort(places_parameters["rating"][places_parameters["places_available"]])
    )

    places_parameters["places_available"] = places_parameters["places_available"][
        sort_index
    ]

    current_place = hotel

    pois, distance_matrix = (
        places_parameters["places_available"],
        places_parameters["distance_matrix"],
    )

    current_day, route, all_visited, current_place = (
        0,
        np.array([hotel]),
        np.array([hotel]),
        hotel,
    )

    while current_day != users_parameters["amount_days"]:

        # remove os locais que já foram visitados

        for place in pois:

            if place in all_visited:

                continue

            feasible, route_parameters = co.verify_constructive_heuristic_constraints(
                route,
                place,
                places_parameters,
                users_parameters,
                route_parameters,
                current_day,
            )

            if feasible:

                route = np.append(route, place)

                all_visited = np.append(all_visited, place)

            if len(route) - 1 == users_parameters["amount_daily_places"]:

                break

        route_parameters[current_day]["route"] = route

        if current_day + 1 >= users_parameters["amount_days"]:

            break

        current_day += 1

        route = route_parameters[current_day]["route"]

    route = co.generate_route_array(users_parameters, route_parameters)

    return (
        generate_daily_array_route(users_parameters, route_parameters),
        objective_function.measure_fitness([route], places_parameters["rating"])[0],
    )
