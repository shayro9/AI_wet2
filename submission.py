import math

from Agent import Agent, AgentGreedy
# from WarehouseEnv import WarehouseEnv, manhattan_distance
from WarehouseEnv import *
import random

INCENTIVE = 10
BATTERY_2_CREDIT = 2


# TODO: section a : 3
def smart_heuristic(env: WarehouseEnv, robot_id: int):
    my_robot = env.get_robot(robot_id)
    pos = my_robot.position

    chargers_dist = dict()
    for c in env.charge_stations:
        chargers_dist[c] = manhattan_distance(c.position, pos)
    best_charger = min(chargers_dist, key=lambda k: chargers_dist[k])

    package_dist = dict()
    for p in env.packages:
        if p.on_board:
            package_dist[p] = manhattan_distance(p.position, pos)
    charge = my_robot.battery
    can_pick = dict()
    for p, d in package_dist.items():
        if charge > d + manhattan_distance(p.position, p.destination):
            can_pick[p] = manhattan_distance(p.position, p.destination) - d
        else:
            can_pick[p] = 0
    if can_pick:
        best_pick = max(can_pick, key=lambda k: can_pick[k])
    has_package = True if my_robot.package else False

    cost, expected_profit = 0, 0
    if has_package:
        expected_profit = manhattan_distance(my_robot.package.position, my_robot.package.destination)
        cost = manhattan_distance(pos, my_robot.package.destination)
    else:
        if package_dist[best_pick] + manhattan_distance(best_pick.position, best_pick.destination) < charge:
            cost = package_dist[best_pick]
        else:
            cost = chargers_dist[best_charger]

    return expected_profit - cost + 2 * my_robot.credit


class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


def closest_station_dist(location: tuple[int], stations: list[ChargeStation]):
    return min(manhattan_distance(location, stations[0].position), manhattan_distance(location, stations[1].position))


def closest_station(location: tuple[int, int], stations: list[ChargeStation]):
    return stations[0] if manhattan_distance(location, stations[0].position) <= manhattan_distance(location, stations[
        1].position) else stations[1]


def set_focus(env: WarehouseEnv, robot_id: int):
    packages: list[Package] = env.packages
    pos = env.get_robot(robot_id).position
    stations: list[ChargeStation] = env.charge_stations
    battery = env.get_robot(robot_id).battery
    best_packages, second_best = list(), list()
    for p in packages:
        if not p.on_board:
            continue

        dist_to_package = manhattan_distance(pos, p.position)
        dist_to_target = manhattan_distance(p.position, p.destination)
        if dist_to_target == 0:
            continue

        dist_target_station = closest_station_dist(p.destination, stations)

        if battery > dist_to_package + dist_to_target + dist_target_station:
            best_packages.append(p)
        elif battery > dist_to_package + dist_to_target:
            second_best.append(p)

    if len(best_packages) >= 2:
        if tie_break(best_packages[0], best_packages[1], stations, pos):
            return best_packages[0]
        else:
            return best_packages[1]
    elif len(best_packages) == 1:
        return best_packages[0]
    elif len(second_best) >= 2:
        if tie_break(second_best[0], second_best[1], stations, pos):
            return second_best[0]
        else:
            return second_best[1]
    elif len(second_best) == 1:
        return second_best[0]
    elif manhattan_distance(packages[0].position, packages[0].destination) \
            > manhattan_distance(packages[1].position, packages[1].destination):
        return packages[0]
    else:
        return packages[1]


def tie_break(pack1: Package, pack2: Package, stations: list[ChargeStation], player_pos):  # maybe add enemy pos as last
    dist1, dist2 = 0, 0
    for s in stations:
        dist1 += manhattan_distance(pack1.destination, s.position)
        dist2 += manhattan_distance(pack2.destination, s.position)

    if manhattan_distance(pack1.position, pack1.destination) == manhattan_distance(pack2.position, pack2.destination):
        if manhattan_distance(player_pos, pack1.position) == manhattan_distance(player_pos, pack2.position):
            return dist1 > dist2
        else:
            return manhattan_distance(player_pos, pack1.position) < manhattan_distance(player_pos, pack2.position)
    else:
        return manhattan_distance(pack1.position, pack1.destination) > manhattan_distance(pack2.position, pack2.destination)


def potential(env: WarehouseEnv, pos, battery):
    p: list[Package] = [pa for pa in env.packages if pa.on_board]
    res = 0
    for pp in p:
        v = manhattan_distance(pp.position, pp.destination)
        d = manhattan_distance(pos, pp.position)

        if v + d < battery:
            res = max(res, 2 * v)

    return res


def smart_heuristic2(env: WarehouseEnv, robot_id: int):
    my_robot: Robot = env.get_robot(robot_id)
    pos = my_robot.position
    has_package = True if my_robot.package else False
    battery: int = my_robot.battery
    stations: list[ChargeStation] = env.charge_stations
    holding_points = 0

    if has_package:
        target = my_robot.package.destination
        dist = manhattan_distance(my_robot.package.position, target)
        holding_points = 2 * dist
        potential_points = potential(env, target, battery - dist) + holding_points
        if manhattan_distance(pos, target) > battery:
            go_to = closest_station(pos, stations).position
        else:
            go_to = target
        if holding_points == 0:
            return -math.inf
    else:  # no package
        potential_points = potential(env, pos, battery)
        best_package = set_focus(env, robot_id)
        go_to = best_package.position
        if manhattan_distance(go_to, pos) > battery:
            go_to = closest_station(pos, stations).position

    return potential_points + BATTERY_2_CREDIT * my_robot.credit - \
        (closest_station_dist(pos, stations) / (holding_points + battery + 1) +
         manhattan_distance(pos, go_to) / (holding_points + 1))


class SecondGreedy(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic2(env, robot_id)


class ForthGreedy(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic2(env, robot_id) - smart_heuristic2(env, (robot_id + 1) % 2)


def potential2(env: WarehouseEnv, robot: Robot, battery):
    p: list[Package] = [pa for pa in env.packages if pa.on_board]
    res = 0
    if robot.package:
        return 2 * manhattan_distance(robot.package.position, robot.package.destination) - manhattan_distance(
            robot.position, robot.package.destination)
    for pp in p:
        v = manhattan_distance(pp.position, pp.destination)
        d = manhattan_distance(robot.position, pp.position)

        if v + d < battery:
            res = max(res, v - d)

    return res


def smart_heuristic3(env: WarehouseEnv, robot_id: int):
    my_robot: Robot = env.get_robot(robot_id)
    enemy_robot: Robot = env.get_robot((robot_id + 1) % 2)
    stations = env.charge_stations

    my_pos = my_robot.position
    enemy_pos = enemy_robot.position

    point_diff = my_robot.credit - enemy_robot.credit
    battery_diff = my_robot.battery - enemy_robot.battery

    station_diff = closest_station_dist(my_pos, stations) - closest_station_dist(enemy_pos, stations)
    potential_diff = potential2(env, my_robot, my_robot.battery) - potential2(env, enemy_robot, enemy_robot.battery)

    return point_diff + potential_diff


class ThirdGreedy(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic3(env, robot_id)


class AgentMinimax(Agent):
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        x = self.min_max(4, agent_id, env, 1)[1]
        return x

    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)

    def min_max(self, d, robot_id, env, turn):
        if turn % 2 == 1:
            operators = env.get_legal_operators(robot_id)
            children = [env.clone() for _ in operators]
            for child, op in zip(children, operators):
                child.apply_operator(robot_id, op)
            if d == 1 or env.num_steps == 1:
                children_val = [self.heuristic(child, robot_id) for child in children]
            else:
                children_val = [self.min_max(d - 1, robot_id, child, turn + 1)[0] for child in children]

            max_child = max(children_val)
            max_index = children_val.index(max_child)
            return max(children_val), operators[max_index]
        else:
            operators = env.get_legal_operators((robot_id + 1) % 2)
            children = [env.clone() for _ in operators]
            for child, op in zip(children, operators):
                child.apply_operator((robot_id + 1) % 2, op)
            if d == 1 or env.num_steps == 1:
                children_val = [self.heuristic(child, robot_id) for child in children]
            else:
                children_val = [self.min_max(d - 1, robot_id, child, turn + 1)[0] for child in children]
            min_child = min(children_val)
            min_index = children_val.index(min_child)
            return min(children_val), operators[min_index]


class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentExpectimax(Agent):
    # TODO: section d : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


# here you can check specific paths to get to know the environment
class AgentHardCoded(Agent):
    def __init__(self):
        self.step = 0
        # specifiy the path you want to check - if a move is illegal - the agent will choose a random move
        self.trajectory = ["move north", "move east", "move north", "move north", "pick_up", "move east", "move east",
                           "move south", "move south", "move south", "move south", "drop_off"]

    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        if self.step == len(self.trajectory):
            return self.run_random_step(env, robot_id, time_limit)
        else:
            op = self.trajectory[self.step]
            if op not in env.get_legal_operators(robot_id):
                op = self.run_random_step(env, robot_id, time_limit)
            self.step += 1
            return op

    def run_random_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators, _ = self.successors(env, robot_id)

        return random.choice(operators)
