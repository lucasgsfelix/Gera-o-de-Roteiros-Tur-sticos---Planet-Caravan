import numpy as np
from operations import constructive_operations as co


def build_rcl(
    route,
    general_route,
    users_parameters,
    places_parameters,
    route_parameters,
    current_day,
    alpha=0.5,
):

    distance_matrix = places_parameters["distance_matrix"]

    # removendo locais que já foram visitados
    pois = np.setdiff1d(places_parameters["places_available"], general_route)

    # removendo com preço alto
    visiting_cost = places_parameters["price"][pois] + route_parameters["budget"]

    pois = pois[visiting_cost < users_parameters["budget"]]

    # distância até o local mais a volta para o hotel
    back_to_hotel = (
        route_parameters[current_day]["daily_distance"]
        + distance_matrix[route[-1]][pois]
        + distance_matrix[route[0]][pois]
    )

    candidates = pois[back_to_hotel <= users_parameters["total_distance"]]

    # neste caso não há nenhum local disponível
    if candidates.size == 0:

        return None

    # distancia do local atual para o proximo, dividido pela nota do local
    visit_gain = (
        distance_matrix[route[-1]][candidates] / places_parameters["rating"][candidates]
    )

    # beta = cmin + alpha * (cmax - cmin)

    # cmin
    min_gain = np.min(visit_gain)

    # cmax
    beta = min_gain + alpha * (np.max(visit_gain) - min_gain)

    # locais que são candidatos
    candidates = candidates[visit_gain <= beta]

    selected_place = np.random.choice(candidates, 1)[0]

    return selected_place


def select_hotel(places_parameters, users_parameters):

    hotels, distance_matrix = (
        places_parameters["hotels"],
        places_parameters["distance_matrix"],
    )

    # tem que ser hotel, e o valor das diárias não pode ultrapassar o budget
    hotels = hotels[
        places_parameters["price"][hotels] * users_parameters["amount_days"]
        < users_parameters["budget"]
    ]

    selected_hotel = np.random.choice(hotels, 1)[0]

    return (
        selected_hotel,
        places_parameters["price"][selected_hotel] * users_parameters["amount_days"],
    )


def random_greedy(places_parameters, users_parameters, alpha=0.5):

    size_places = (
        users_parameters["amount_days"] * users_parameters["amount_daily_places"]
    ) + 1

    ## função escolhe hotel
    hotel, cost = select_hotel(places_parameters, users_parameters)

    route_parameters = {"budget": cost}

    for day in range(0, users_parameters["amount_days"]):

        route_parameters[day] = {
            "daily_distance": 0,
            "restaurant": 0,
            "attraction": 0,
            "amount_places": 0,
            "route": np.array([hotel]),
        }

    route, general_route, route_size, count_infeasible, current_day = (
        np.array([hotel]),
        np.array([hotel]),
        0,
        0,
        0,
    )

    while (
        route_size + 1 != size_places
        or current_day + 1 <= users_parameters["amount_days"]
    ):

        ## atualização automática do alpha
        new_place = build_rcl(
            route,
            general_route,
            users_parameters,
            places_parameters,
            route_parameters,
            current_day,
            alpha,
        )

        if not new_place is None:

            feasible, route_parameters = co.verify_constructive_heuristic_constraints(
                route,
                new_place,
                places_parameters,
                users_parameters,
                route_parameters,
                current_day,
            )

            if feasible:

                count_infeasible = 0

                route = np.append(route, new_place)

                general_route = np.append(general_route, new_place)

            else:

                count_infeasible += 1

        if (
            new_place is None
            or len(route) >= users_parameters["amount_daily_places"]
            or count_infeasible >= 5
        ):

            ## não é possível adicionar mais nenhum local
            ## então adicionamos o hotel
            ## e a quantidade dias aumenta
            # route = np.append(route, route[0])
            count_infeasible = 0

            # removing the hotels
            route_size += len(route) - 1

            route_parameters[current_day]["route"] = route

            if current_day + 1 >= users_parameters["amount_days"]:

                break

            current_day += 1

            route = route_parameters[current_day]["route"]

    return route_parameters
