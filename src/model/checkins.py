
import datetime

import pandas as pd

import numpy as np


def generate_walking_time(places_parameters, route_with_hotel):

	## até aqui tá tudo certo, então vamos associar o tempo!
	## ser humano caminha em média 1 metro por segundo	
	## walking time tá em minutos
	walking_time = [0] + [places_parameters['distance_matrix'][source_poi][destiny_poi]/60
					for source_poi, destiny_poi in zip(route_with_hotel[:-1],
														route_with_hotel[1:])]

	return walking_time


def generate_expend_time(places_parameters, route):


	## tá em minutos
	places_time = [0] + [places_parameters['expend_time'][place]
						 for place in route[1:]] + [0]

	return places_time


def define_hotel_times(users_parameters):

	current_time = users_parameters['datetime_start'] + datetime.timedelta(days=1)

	users_parameters['datetime_start'] = users_parameters['datetime_start'] +  datetime.timedelta(days=1)
	# o dia virou
	users_parameters['datetime_end'] = users_parameters['datetime_end'] +  datetime.timedelta(days=1)

	return current_time, users_parameters


def verify_place_open(place_info, visit):

	start, end = (place_info['Working Start'].values[0],
				  place_info['Working End'].values[0])

	start_time, end_time = (visit['start_time'], visit['end_time'])

	if not ((start_time >= start) and (end_time <= end) and (start_time <= end_time)):

		return False

	return True


def define_time_closed_place(place_info, start_time, current_time, expend, walk):
	"""
		Define quando vai começar uma atividade, quando o local esta fechado
	"""

	### caso o local esteja fechado, a gente espera ele abrir
	start, end = (place_info['Working Start'].values[0],
				  place_info['Working End'].values[0])

	# o horário de abertura daquele local é maior que horário atual!
	if start > start_time:

		start_time = pd.to_datetime(start)

		current_time = start_time + datetime.timedelta(minutes=int(expend))

	return current_time, start_time


def generate_visit_checkin(places_parameters, users_parameters, route, print_v=False):


	if places_parameters['hotel'] != route[-1]:

		route_with_hotel = np.append(route, places_parameters['hotel'])

	else:

		route_with_hotel = route


	walking_time = generate_walking_time(places_parameters, route_with_hotel)

	places_time = generate_expend_time(places_parameters, route)

	df, current_time = places_parameters['df'], users_parameters['datetime_start']

	visits, index = [], 0

	for walk, expend, place in zip(walking_time, places_time, route_with_hotel):

		# A atividade deve acabar pelo menos até o horário
		# em que a localidade esta fechadno
		place_info = df[(df['Name'] == place) &
						(df['Working Start'].dt.date == current_time.date()) &
						(df['Working Day'] == True)]


		if place_info.empty:

			return False, visits

		if index == 0: # o usuário tá no hotel

			# horário que começa e que termina a atividade
			visit = {
						'place': place,
					 	'start_time': current_time,
					 	'end_time': current_time
					}

			index += 1

		else:

			## casos básicos
			# a atividade começa dado o tempo atual + o tempo até chegar no local
			start_time = current_time + datetime.timedelta(minutes=walk)

			# e a atividade termina depois que o usuário ficar lá o tempo gasto
			current_time = start_time + datetime.timedelta(minutes=int(expend))

			## o local é um hotel, então o usuário vai ir pra lá, e ficar até no dia seguinte
			if place == places_parameters['hotel']:

				current_time, users_parameters = define_hotel_times(users_parameters)

			elif not place_info.empty:# aquele local esta aberto

				current_time, start_time = define_time_closed_place(place_info, start_time,
																	current_time,
																	expend, walk)

			visit = {
						'place': place,
						'start_time': start_time,
						'end_time': current_time
					}

			if ((current_time > users_parameters['datetime_end']) or
			    (current_time > users_parameters['travel_end'])):

				return False, visits

		# hotel tá sempre aberto!
		if (place != places_parameters['hotel']):# aquele local esta aberto

			feasible = verify_place_open(place_info, visit)

			if not feasible:

				return False, visits

		visits.append(visit)

	return True, visits