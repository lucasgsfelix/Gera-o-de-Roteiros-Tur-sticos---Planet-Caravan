import json
import itertools


def write_results(
    output_file,
    budget,
    distance,
    time,
    fitness,
    route,
    remove_prob,
    list_size,
    algorithm,
):

    with open(output_file, "a") as file:

        file.write(
            "\t".join(
                [
                    str(algorithm),
                    str(budget),
                    str(distance),
                    str(time),
                    str(fitness),
                    str(route),
                    str(remove_prob),
                    str(list_size),
                ]
            )
            + "\n"
        )


def read_parameters(file_parameters):

    with open("Parameters/" + file_parameters, "r") as json_file:

        parameters_data = json.load(json_file)

    return parameters_data


def explode_parameters(parameters):

    # make a combination between all parameters
    combinations = list(itertools.product(*list(parameters.values())))

    return [dict(zip(parameters.keys(), combination)) for combination in combinations]
