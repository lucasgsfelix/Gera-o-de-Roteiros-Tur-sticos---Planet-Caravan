import numpy as np
import pandas as pd

from model import checkins


def verify_distance_constraint(route_with_hotel, distance_matrix, users_parameters):

    for source_poi, destiny_poi in zip(route_with_hotel[:-1], route_with_hotel[1:]):

        ## toda vez que o usuário volta ao hotel, quer dizer que um novo dia
        ## esta se iniciando
        if source_poi == route_with_hotel[0]:

            total_distance = 0

        total_distance += distance_matrix[source_poi][destiny_poi]

        if total_distance > users_parameters["total_distance"]:

            return False

    return True


def verify_price_constraint(new_route, prices, places_parameters, users_parameters):

    # calculando as diárias do hotel
    total_price = prices[places_parameters["hotel"]] * (
        users_parameters["amount_days"] - 1
    )

    # calculado o preço dos POIs
    total_price += np.sum(
        prices[new_route[new_route != places_parameters["hotel"]].astype(int)]
    )

    return total_price <= users_parameters["budget"]


def verify_category_constraint(new_route, categories, users_parameters):
    """

    Verifica se há locais de categorias repetidas seguidos

    """

    count = 0

    # é possível fazer uma tabela de até que ponto uma rota foi verificada?

    for source_poi, destiny_poi in zip(new_route[:-1], new_route[1:]):

        if categories[source_poi] != categories[destiny_poi]:

            count = 0

        else:  # categorias iguais, o contador vai aumentar pois são seguidos

            count += 1

    return count <= users_parameters["category"]


def verify_duplicates(new_route, hotel):
    """

    Verifica se há duplicadas de locais visitados

    """

    route_without_hotel = new_route[new_route != hotel]

    return route_without_hotel.size == np.unique(route_without_hotel).size


def verify_amount_hotels(new_route, categories):
    """
    Verifica a quantidade de locais que são hotéis visitados

    Isso faz sentido em las vegas?

    """

    visited_places_categories = categories[np.unique(new_route)]

    return len(visited_places_categories[visited_places_categories == "hotel"]) == 1


def verify_amount_places(new_route, users_parameters, places_parameters):
    """
    Verifica a quantidade máxima de locais a serem visitados em um dia
    """

    # -1 pois não podemos considerar o hotel!

    count = 0

    for place in new_route:

        if place == places_parameters["hotel"]:

            count = 0

        else:

            count += 1

        if count > users_parameters["amount_daily_places"]:

            return False

    return True


def verify_constraints(route, destiny_poi, places_parameters, users_parameters):

    # print(route)

    constraints = {}

    route = np.array(route).astype(int)

    hotel = places_parameters["hotel"]

    if route[-1] == hotel and destiny_poi == hotel:
        ## outro hotel?

        return False

    ## new_route é a rota com o POI que nós queremos adicionar
    new_route = np.append(route, int(destiny_poi))

    ## adicionando a volta para o hotel
    route_with_hotel = np.append(new_route, int(hotel))

    # route_with_hotel = route_with_hotel.astype(int)

    constraints["total_distance"] = verify_distance_constraint(
        route_with_hotel, places_parameters["distance_matrix"], users_parameters
    )

    constraints["price"] = verify_price_constraint(
        new_route, places_parameters["price"], places_parameters, users_parameters
    )

    constraints["amount_hotels"] = verify_amount_hotels(
        new_route, places_parameters["category"]
    )

    constraints["category"] = verify_category_constraint(
        new_route, places_parameters["category"], users_parameters
    )

    constraints["duplicates"] = verify_duplicates(new_route, places_parameters["hotel"])

    constraints["amount_places"] = verify_amount_places(
        new_route, users_parameters, places_parameters
    )

    ## verificar horário das atividades
    ## seleção de transport

    if False in set(constraints.values()):

        ## agora fazemos as atribuições de horário
        return False

    feasible, visits = checkins.generate_visit_checkin(
        places_parameters, users_parameters.copy(), new_route
    )

    ## ai nesse caso não há nada a fazer
    ## podemos definir as operações genéticas de acordo com a restrição desrepeitada
    return feasible
