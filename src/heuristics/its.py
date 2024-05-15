import random

import numpy as np

import tqdm

from heuristics import random_greedy_procedure
from operations import constructive_operations as co


# avalia a solução obtida
def evaluate_solution(s, dataset, distance_matrix, users_parameters):

    total_rating = 0
    # os hoteis tão sendo avaliados mais de uma vez
    for day in range(users_parameters["amount_days"]):

        for i in range(len(s)):

            for j in range(len(s)):

                if s[day, i, j] == 1:

                    total_rating += dataset[i][4]  # Índice 4 corresponde ao "rating"

                    total_distance += distance_matrix[i][j]

    return total_rating, total_distance


def verify_category_constraints(places_parameters, users_parameters, complete_route):

    categories = places_parameters["category"][complete_route]

    for daily_route in co.split_array_at_value(categories, "hotel"):

        for category in np.unique(categories):

            if len(daily_route[daily_route == category]) > users_parameters["category"]:

                return False

    return True


def verify_route_size(places_parameters, users_parameters, complete_route):

    categories = places_parameters["category"][complete_route]

    for daily_route in co.split_array_at_value(categories, "hotel"):

        if len(daily_route) - 1 > users_parameters["amount_daily_places"]:

            return False

    return True


def check_all_constraints(s, place_parameters, users_parameters):
    # Verifica todas as restrições estipuladas
    # Distância
    # Dinheiro
    # Categoria
    # Hotel
    # Repetição de locais

    route_array = generate_route_array(s, users_parameters)

    route_without_hotel = route_array[route_array != route_array[0]]

    complete_distance, total_distance = 0, 0

    fitness = (
        np.sum(place_parameters["rating"][route_without_hotel])
        + place_parameters["rating"][route_array[0]]
    )

    # restrição de distância
    for source_poi, destiny_poi in zip(route_array[:-1], route_array[1:]):

        ## toda vez que o usuário volta ao hotel, quer dizer que um novo dia
        ## esta se iniciando
        if source_poi == route_array[0]:

            complete_distance += total_distance
            total_distance = 0

        total_distance += place_parameters["distance_matrix"][source_poi][destiny_poi]

        if total_distance > users_parameters["total_distance"]:

            return False, fitness, complete_distance

    ### Verifica se todos os locais são únicos
    ### Não vamos verificar o hotel
    if len(route_without_hotel) != len(np.unique(route_without_hotel)):

        return False, fitness, complete_distance

    # restrição de custo
    total_cost = np.sum(place_parameters["price"][route_without_hotel]) + (
        place_parameters["price"][route_array[0]] * users_parameters["amount_days"]
    )

    if total_cost >= users_parameters["budget"]:

        return False, fitness, total_distance

    # restrição de categorias
    if not verify_category_constraints(place_parameters, users_parameters, route_array):

        return False, fitness, complete_distance

    if not verify_route_size(place_parameters, users_parameters, route_array):

        return False, fitness, total_distance

    return True, fitness, complete_distance


def remove_place_from_solution(s, day, hotel, tabu_list):

    places_available = np.setdiff1d(np.where(s[day] == 1)[0], [hotel])

    if len(places_available) == 0:

        return s, None, tabu_list

    place_a = np.random.choice(places_available, 1)[0]

    ## Quem aponta para A
    who_points_to_a = np.where(s[day][:, place_a] == 1)[0][0]

    ## Para quem A aponta
    a_points_to = np.where(s[day][place_a] == 1)[0][0]

    operations = [
        [day, who_points_to_a, a_points_to],
        [day, who_points_to_a, place_a],
        [day, place_a, a_points_to],
    ]

    if not verify_tabu_list(operations, tabu_list):

        return s, None, tabu_list

    tabu_list = add_to_tabu_list(operations, tabu_list)

    s[day][who_points_to_a, a_points_to] = 1
    s[day][who_points_to_a, place_a] = 0
    s[day][place_a, a_points_to] = 0

    return s, place_a, tabu_list


def verify_tabu_list(operations, tabu_list):

    for operation in operations:

        ## a operação já esta na lista tabu
        if (
            len(tabu_list) > 0
            and len(operation) > 0
            and np.any(np.all(operation == tabu_list, axis=1))
        ):

            return False

    return True


def add_to_tabu_list(operations, tabu_list):

    for operation in operations:

        if len(tabu_list) == 0:

            tabu_list = np.array(operation, dtype=object)

        else:

            tabu_list = np.vstack((tabu_list, operation))

    return tabu_list


def route_2opt(s, hotel, day, other_day, tabu_list):

    # day == other_day -> intra
    # day != other_day -> intre

    # seleciona dois locais aleatórios e troca de posição
    # não pode ser o hotel
    # dado que o hotel tem que ser o primeiro e o último da lista

    places_available = np.setdiff1d(np.where(s[day] == 1)[0], [hotel])

    if len(places_available) == 0:

        return s, tabu_list

    place_a = np.random.choice(places_available, 1)[0]

    places_available = np.setdiff1d(np.where(s[other_day] == 1)[0], [hotel, place_a])

    if len(places_available) == 0:

        return s, tabu_list

    place_b = np.random.choice(places_available, 1)[0]

    ## agora eu tenho que identificar quem vai para a, e colocar para ir para b
    ## Quem aponta para A
    who_points_to_a = np.where(s[day][:, place_a] == 1)[0][0]

    ## Para quem A aponta
    a_points_to = np.where(s[day][place_a] == 1)[0][0]

    ## Quem aponta para B
    who_points_to_b = np.where(s[other_day][:, place_b] == 1)[0][0]

    ##  Para quem B aponta
    b_points_to = np.where(s[other_day][place_b] == 1)[0][0]

    ### atualizando a lista tabu

    ### verifica se os dias forem iguais, se os items são adjascentes
    if day == other_day and (a_points_to == place_b or b_points_to == place_a):

        return s, tabu_list

    ### Operações que serão feitas, se alguma delas estiver na lista tabu,
    ### nós retornamo a lista da maneira que esta

    operations = [
        [day, who_points_to_a, place_b],
        [day, who_points_to_a, place_a],
        [day, place_b, a_points_to],
        [day, place_a, a_points_to],
        [other_day, who_points_to_b, place_a],
        [other_day, who_points_to_b, place_b],
        [other_day, place_a, b_points_to],
        [other_day, place_b, b_points_to],
    ]

    if not verify_tabu_list(operations, tabu_list):

        return s, tabu_list

    tabu_list = add_to_tabu_list(operations, tabu_list)

    ## Quem apontava para A agora aponta para B
    s[day][who_points_to_a, place_b] = 1
    s[day][who_points_to_a, place_a] = 0

    ## Para quem A apontava, agora é apontado por B
    s[day][place_b, a_points_to] = 1
    s[day][place_a, a_points_to] = 0

    ## Quem apontava para B, agora aponta para A
    s[other_day][who_points_to_b, place_a] = 1
    s[other_day][who_points_to_b, place_b] = 0

    ## Para quem B apontava, agora é apontado por A
    s[other_day][place_a, b_points_to] = 1
    s[other_day][place_b, b_points_to] = 0

    return s, tabu_list


def neighborhood_evaluation(s, users_parameters, place_parameters, hotel, tabu_list):
    # Gera vizinhança
    # edge-exchange local moves
    # Cada um desses podeser intra o inter período
    # intra e inter
    # 2-opt
    # 1-0 relocate 2
    # 1-1 relcate 3

    selected_op = np.random.choice(
        [
            "intra 2-opt",
            "intra 1-0 relocate",
            "inter 1-0 relocate",
            "intra 1-1 relocate",
            "inter 1-1 relocate",
        ],
        1,
    )[0]

    day, other_day = None, None

    # selected_op = 'intra 1-1 relocate'

    while day == other_day:

        day = np.random.randint(0, users_parameters["amount_days"])

        other_day = np.random.randint(0, users_parameters["amount_days"])

    if selected_op == "intra 2-opt":

        s, tabu_list = route_2opt(s, hotel, day, day, tabu_list)

    elif selected_op == "intra 1-0 relocate":
        ## Remove um local

        s, _, tabu_list = remove_place_from_solution(s, day, hotel, tabu_list)

    elif selected_op == "inter 1-0 relocate":
        # relocation - Muda o item de um período para outro
        # inter-period 1-0 Relocate can involve either the relocation of
        # one point from one period to another
        # or
        # the insertion of a currently
        # non-visited optional point to the tour at a given period
        # Insertion - adiciona local em um mesmo período
        selected_op = np.random.choice(["relocation", "addition"], 1)[0]

        if selected_op == "relocation":
            # relocation - Muda o item de um período para outro

            s, place_a, tabu_list = remove_place_from_solution(s, day, hotel, tabu_list)

        elif selected_op == "addition":

            ## e se há já estiver na rota?
            place_a = np.random.choice(place_parameters["places_available"], 1)[0]

        ## Adicionando um local em um novo período
        ## Essa aqui é a parte de addition

        if place_a is None:

            return s, tabu_list

        places_available = np.setdiff1d(
            np.where(s[other_day] == 1)[0], [hotel, place_a]
        )

        if len(places_available) == 0:

            return s, tabu_list

        place_b = np.random.choice(places_available, 1)[0]

        ##  Para quem B aponta
        b_points_to = np.where(s[other_day][place_b] == 1)[0][0]

        operations = [
            [other_day, place_b, place_a],
            [other_day, place_a, b_points_to],
            [other_day, place_b, b_points_to],
        ]

        if not verify_tabu_list(operations, tabu_list):

            return s, tabu_list

        tabu_list = add_to_tabu_list(operations, tabu_list)

        s[other_day][place_b, place_a] = 1
        s[other_day][place_a, b_points_to] = 1
        s[other_day][place_b, b_points_to] = 0

    elif selected_op == "intra 1-1 relocate":

        places_available = np.setdiff1d(np.where(s[day] == 1)[0], [hotel])

        if len(places_available) == 0:

            return s, tabu_list

        ## 1-1 relocate é sequencial - mudança de dois locais sequenciais
        place_a = np.random.choice(places_available, 1)[0]

        ## Quem aponta para A
        who_points_to_a = np.where(s[day][:, place_a] == 1)[0][0]

        ## Para quem A aponta
        a_points_to = np.where(s[day][place_a] == 1)[0][0]

        # print(np.where(s[day][a_points_to] == 1))
        # neste caso termina a avaliação
        if a_points_to == hotel:

            return s, tabu_list

        ## A agora aponta para quem o local que ele apontava, apontava antes
        a_points_to_x_that_points_to_y = np.where(s[day][a_points_to] == 1)[0][0]

        operations = [
            [day, who_points_to_a, a_points_to],
            [day, who_points_to_a, place_a],
            [day, place_a, a_points_to],
            [day, a_points_to, a_points_to_x_that_points_to_y],
            [day, place_a, a_points_to_x_that_points_to_y],
            [day, a_points_to, place_a],
        ]

        if not verify_tabu_list(operations, tabu_list):

            return s, tabu_list

        tabu_list = add_to_tabu_list(operations, tabu_list)

        # B -> C -> A -> D
        # B ->  A- > C -> D
        s[day][who_points_to_a, a_points_to] = 1
        s[day][who_points_to_a, place_a] = 0
        s[day][place_a, a_points_to] = 0
        s[day][a_points_to, a_points_to_x_that_points_to_y] = 0
        s[day][a_points_to, place_a] = 1
        s[day][place_a, a_points_to_x_that_points_to_y] = 1

    elif selected_op == "inter 1-1 relocate":
        # Swap - Troca dois locais de períodos distintos
        # Insertion - Remove um local de um período e adiciona um novo
        # inter-period 1-1 Exchange can involve either the swap
        # of two points visited in two different periods
        # or
        # the insertion of a currently non-visited point
        # with simultaneous removal of a visited point at a given period

        selected_op = np.random.choice(["swap", "insertion"], 1)[0]

        if selected_op == "swap":
            # Swap - Troca dois locais de períodos distintos
            s, tabu_list = route_2opt(s, hotel, day, other_day, tabu_list)

        elif selected_op == "insertion":
            # Insertion - Remove um local Y de um período X e adiciona um novo Z a um período X
            # Dado o mesmo período eu removo um local e adiciono outro
            s, place_a, tabu_list = remove_place_from_solution(s, day, hotel, tabu_list)

            place_a = np.random.choice(place_parameters["places_available"], 1)[0]

            # me dê um local aleatório que eu vou adicionar o place_a entre esse local aleatório e outro
            places_available = np.setdiff1d(np.where(s[day] == 1)[0], [hotel, place_a])

            if len(places_available) > 0:

                place_b = np.random.choice(places_available, 1)[0]

            else:

                return s, tabu_list

            ##  Para quem B aponta
            b_points_to = np.where(s[day][place_b] == 1)[0][0]

            operations = [
                [day, place_b, place_a],
                [day, place_a, b_points_to],
                [day, place_b, b_points_to],
            ]

            if not verify_tabu_list(operations, tabu_list):

                return s, tabu_list

            tabu_list = add_to_tabu_list(operations, tabu_list)

            s[day][place_b, place_a] = 1
            s[day][place_a, b_points_to] = 1
            s[day][place_b, b_points_to] = 0

    return s, tabu_list


def update_tabu_list(tabu_list, algorithm_parameters):
    # Atualiza a lista tabu

    # Limita o tamanho da lista tabu (remove elementos mais antigos, se necessário)
    while len(tabu_list) > algorithm_parameters["tabu_size"]:

        tabu_list = tabu_list[1:]  # Remove o elemento mais antigo

    return tabu_list


def perturbation(s, users_parameters, algorithm_parameters, hotel, tabu_list):
    # Realiza perturbações na solução

    for day in range(0, users_parameters["amount_days"]):

        if random.random() <= algorithm_parameters["remove_prob"]:

            s, _, _ = remove_place_from_solution(s, day, hotel, tabu_list)

    return s


def make_solution_evaluation(solutions_fitness, solution_a, solution_b):

    # queremos uma solução que tenha o fitness maior
    # ou que tenha o fitness igual e ande menos
    return (
        solutions_fitness[solution_a]["fitness"]
        > solutions_fitness[solution_b]["fitness"]
    ) or (
        (
            solutions_fitness[solution_a]["distance"]
            < solutions_fitness[solution_b]["distance"]
        )
        and (
            solutions_fitness[solution_a]["fitness"]
            >= solutions_fitness[solution_b]["fitness"]
        )
    )


def construct_initial_solution(place_parameters, users_parameters):

    ## create the random solution in a greedy fashion

    amount_locations = len(place_parameters["hotels"]) + len(
        place_parameters["places_available"]
    )

    current_solution = np.zeros(
        (users_parameters["amount_days"], amount_locations, amount_locations), dtype=int
    )

    solution = random_greedy_procedure.random_greedy(
        place_parameters, users_parameters, 0.5
    )

    hotel = solution[0]["route"][0]

    for day in range(0, users_parameters["amount_days"]):

        for place_a, place_b in zip(
            solution[day]["route"][:-1], solution[day]["route"][1:]
        ):

            current_solution[day][place_a][place_b] = 1

        last_place = solution[day]["route"][-1]

        current_solution[day][last_place][hotel] = 1

    return current_solution, hotel


def generate_route_array(solution, users_parameters):

    # hotel
    current_place = np.where(solution[0] == 1)[0][0]

    hotel, next_place = current_place, None

    route = np.array([current_place])

    count = 0

    for day in range(0, users_parameters["amount_days"]):

        next_place = None

        count = 0

        while next_place != hotel or next_place is None:

            next_place = np.where(solution[day][current_place] == 1)[0][0]

            route = np.append(route, next_place)

            current_place = next_place

            count += 1

    return route


def iterated_tabu_search(users_parameters, algorithm_parameters, place_parameters):
    # Três soluções principais por iteração
    # S* que é a melhor solução, só atualizamos ela no final
    # S que é uma solução aleatória criada inicialmente - Ela vai sofrer todas as transformações
    # S' que é um backup da solução aleatória incial
    #

    current_solution, hotel = construct_initial_solution(
        place_parameters, users_parameters
    )

    feasible, fitness, distance = check_all_constraints(
        current_solution, place_parameters, users_parameters
    )

    # Removemos todos os locais disponíveis que estão mais distantes do que uma viagem de ida e volta ao hotel
    places_available = place_parameters["places_available"]
    places_available[
        place_parameters["distance_matrix"][hotel][places_available] * 2
        <= users_parameters["total_distance"]
    ]
    place_parameters["places_available"] = places_available

    best_solution, neighbor_solution, tabu_list = (
        current_solution,
        current_solution,
        np.array([]),
    )

    solutions_fitness = {
        "best": {"fitness": fitness, "distance": distance},  # S*
        "neighbor": {"fitness": fitness, "distance": distance},  # S'
        "current": {"fitness": fitness, "distance": distance},  # S
    }

    old_solution = current_solution

    for _ in range(0, algorithm_parameters["max_iterations"]):

        tabu_tenure_counter = 0

        while tabu_tenure_counter <= algorithm_parameters["max_tabu_tenure"]:

            # Aqui, a atualização da solução é feita com base na tabu_list
            new_solution, tabu_list = neighborhood_evaluation(
                current_solution, users_parameters, place_parameters, hotel, tabu_list
            )

            # Entender e finalizar
            # Atualiza lista tabu e remove elementos antigos
            tabu_list = update_tabu_list(tabu_list, algorithm_parameters)

            # Verifica restrições e critérios tabu
            # The tabu search
            # involves the exploration of the solution space by moving at each
            # iteration from a solution s to the best admissible solution s' of the
            # neighborhood structure N_y(S) according to a tabu list
            feasible, fitness, distance = check_all_constraints(
                new_solution, place_parameters, users_parameters
            )

            if feasible:

                current_solution = new_solution

                solutions_fitness["current"]["fitness"] = fitness
                solutions_fitness["current"]["distance"] = distance

                # queremos uma solução que tenha o fitness maior
                # ou que tenha o fitness igual e ande menos
                if make_solution_evaluation(solutions_fitness, "current", "neighbor"):

                    # atualizo a old_solution

                    tabu_tenure_counter = 1

                    neighbor_solution = current_solution

                    solutions_fitness["neighbor"]["fitness"] = solutions_fitness[
                        "current"
                    ]["fitness"]
                    solutions_fitness["neighbor"]["distance"] = solutions_fitness[
                        "current"
                    ]["distance"]

            tabu_tenure_counter += 1

        # Atualiza a força de perturbação com base no limiar de melhoria
        if make_solution_evaluation(solutions_fitness, "neighbor", "best"):

            algorithm_parameters["perturbation_strength"] = 0

            best_solution = neighbor_solution

            solutions_fitness["best"]["fitness"] = solutions_fitness["neighbor"][
                "fitness"
            ]
            solutions_fitness["best"]["distance"] = solutions_fitness["neighbor"][
                "distance"
            ]

            tabu_list = np.array([])
        else:

            tabu_list = np.array([])

            algorithm_parameters["perturbation_strength"] += 1
            # Aplica perturbação na solução e atualiza a melhor solução encontrada
            # Finalizar
            if algorithm_parameters["perturbation_strength"] >= 10:

                current_solution = perturbation(
                    best_solution,
                    users_parameters,
                    algorithm_parameters,
                    hotel,
                    tabu_list,
                )

    return best_solution, solutions_fitness["best"]


def tabu_search_procedure(place_parameters, users_parameters, algorithm_parameters):

    solutions_fitness, best_solution_parameters = {}, {}

    ## multistart
    for iteration in range(0, algorithm_parameters["multistart_iterations"]):

        solution, solutions_fitness["current"] = iterated_tabu_search(
            users_parameters, algorithm_parameters, place_parameters
        )

        if iteration == 0 or make_solution_evaluation(
            {**solutions_fitness, **best_solution_parameters}, "current", "best"
        ):

            best_solution_parameters["best"] = solutions_fitness["current"]

            best_solution = solution

    return (
        generate_route_array(best_solution, users_parameters),
        best_solution_parameters["best"]["fitness"],
    )
