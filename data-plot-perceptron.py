import json
import matplotlib
from perceptron import Perceptron
import matplotlib.pyplot as plt

perceptron = Perceptron(3)
matplotlib.rcParams["backend"] = "TkAgg"
plt.switch_backend("TkAgg")

print('Dataset Info:')
with open('players-data.json') as file:
    input_data = []
    forwards_goals = []
    forwards_matches = []
    defenders_goals = []
    defenders_matches = []
    data = json.load(file)
    for player in data:
        print('Name:', player['player_name'])
        print('Last name:', player['player_type'])
        print('Matches Played:', player['player_match_played'])
        print('Goals:', player['player_goals'])
        print('')

        if player['player_type'] == 'Forwards':
            forwards_goals.append(int(player['player_goals']))
            forwards_matches.append(int(player['player_match_played']))

            input_data.append([int(player['player_match_played']), int(player['player_goals']), 1])
        else:
            defenders_goals.append(int(player['player_goals']))
            defenders_matches.append(int(player['player_match_played']))

            input_data.append([int(player['player_match_played']), int(player['player_goals']), 0])

    for _ in range(100):
        for player in input_data:
            output = player[-1]
            inp = [1] + player[0:-1]
            err = perceptron.train(inp, output)

    matches_played = float(input("Introduce la cantidad de partidos: "))
    goals = float(input("Introduce la cantidad de goles: "))

    if perceptron.predict([1, matches_played, goals]) == 1:
        print("Delantero")
    else:
        print("Defensor")
    plt.xlabel('Goals')
    plt.ylabel('Matchs Played')
    plt.plot(forwards_goals, forwards_matches, 'bs', defenders_goals, defenders_matches, 'g^')
    plt.show()
