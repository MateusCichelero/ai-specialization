from dis import dis
import math
from dists import dists, straight_line_dists_from_bucharest


def valuation_calculation(cost, heuristic):
    """
    Retorna o valor da função de avaliação,
    a soma entre custo e heurística"""
    return cost + heuristic


def is_goal(node, goal):
    """
    Testa se a cidade no nó é o objetivo.
    Se for Bucharest, retorna True;"""
    if node == goal:
        return True
    else:
        return False

def insert_in_border(border,node,total_dist, heuristic, parent_node):
    
    value_valuation = valuation_calculation(total_dist, heuristic)
    border[node] = [value_valuation, total_dist, heuristic, parent_node]
    return border

def remove_from_border(border):
    node = min(border, key=border.get)
    previous_city = border[node][3]
    border.pop(node)
    return node, previous_city, border

def is_border_empty(border):

    return not border

def remove_untravelled_paths(explored, start):
    travelled_cities = ["Bucharest"]
    next = "Bucharest"
    while next is not start:
        next = explored[next]
        travelled_cities.append(next)

    return travelled_cities[::-1]




def a_star(start, goal='Bucharest'):
    """
    Retorna uma lista com o caminho de start até 
    goal (somente Bucharest neste caso) segundo o algoritmo A*
    """

    if start not in dists:
        return "City does not exist, try with a valid one"

    node = start
    explored = {}
    travelled_distance = 0
    heuristic_dict = straight_line_dists_from_bucharest
    border={}
    border = insert_in_border(border,node,travelled_distance, heuristic_dict[node], 'start')
    cities = dists
    

    while True:
        if is_border_empty(border):
            return "Fail"

        node, previous_city, border = remove_from_border(border)

        if is_goal(node, goal):
            explored[node]=previous_city

            return remove_untravelled_paths(explored, start)
            
        explored[node]=previous_city

        for city_son in cities[node]:
            city_name = city_son[0]
            total_dist = travelled_distance + city_son[1]
            heuristic = heuristic_dict[city_name]
            value_valuation = valuation_calculation(total_dist, heuristic)

            if (city_name not in border) and (city_name not in explored):
                border = insert_in_border(border,city_name,total_dist, heuristic, node)
                
            elif ((city_name in border) and (border[city_name][0] > value_valuation)):
                border[city_name] = [value_valuation, total_dist, heuristic, node]



def main():
    print(a_star('Timisoara'))


if __name__ == "__main__":
    main()