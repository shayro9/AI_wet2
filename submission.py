import math
import time
from Agent import Agent, AgentGreedy
# from WarehouseEnv import WarehouseEnv, manhattan_distance
from WarehouseEnv import *
import random

EPSILON = 0.9
BATTERY_2_CREDIT = 4


def smart_heuristic(env: WarehouseEnv, robot_id: int):
    my_robot = env.get_robot(robot_id)
    pos = my_robot.position
    has_package = 1 if my_robot.package else 0

    robot_charge = my_robot.battery
    chargers_dist = dict()
    for c in env.charge_stations:
        chargers_dist[c] = manhattan_distance(c.position, pos)
    best_charger = min(chargers_dist, key=lambda k: chargers_dist[k])

    charge = robot_charge - 0.5 * min(chargers_dist.values())

    package_dist = dict()
    for p in env.packages:
        if p.on_board:
            package_dist[p] = manhattan_distance(p.position, pos)

    can_pick = dict()
    for p, d in package_dist.items():
        if charge > d:
            can_pick[p] = 2 * manhattan_distance(p.position, p.destination) - d
        else:
            can_pick[p] = 0
    if can_pick:
        best_pick = max(can_pick, key=lambda k: can_pick[k])
    has_package = True if my_robot.package else False

    cost, expected_profit = 0, 0
    if has_package:
        expected_profit = 2 * manhattan_distance(my_robot.package.position, my_robot.package.destination)
        cost = manhattan_distance(pos, my_robot.package.destination)
    else:
        expected_profit = manhattan_distance(best_pick.position, best_pick.destination)
        if package_dist[best_pick]< charge:
            cost = 1 * package_dist[best_pick]
        else:
            cost = chargers_dist[best_charger]

    return expected_profit - cost + 2 * my_robot.credit + 0.5 * charge


class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id) - smart_heuristic(env, (robot_id + 1) % 2)


class AgentMinimax(Agent):
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        finish_time = time.time() + time_limit*EPSILON
        d, action = 1, None
        try:
            while True:
                if time.time() > finish_time:
                    raise Exception("Time")
                action = self.min_max(d, agent_id, env, 1, finish_time)[1]
                d += 1
        except:
            return action

    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id) - smart_heuristic(env, (robot_id+1) % 2)

    def min_max(self, d, robot_id, env, turn, time_limit):
        if time.time() > time_limit:
            raise Exception("Time")
        if turn % 2 == 1:
            operators = env.get_legal_operators(robot_id)
            children = [env.clone() for _ in operators]
            for child, op in zip(children, operators):
                child.apply_operator(robot_id, op)
            if d == 1 or env.num_steps == 1:
                children_val = [self.heuristic(child, robot_id) for child in children]
            else:
                children_val = [self.min_max(d - 1, robot_id, child, turn + 1, time_limit)[0] for child in children]

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
                children_val = [self.min_max(d - 1, robot_id, child, turn + 1, time_limit)[0] for child in children]
            min_child = min(children_val)
            min_index = children_val.index(min_child)
            return min(children_val), operators[min_index]


class AgentAlphaBeta(Agent):
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        finish_time = time.time() + time_limit*EPSILON
        d, action = 1, None
        try:
            while True:
                if time.time() > finish_time:
                    raise Exception("Time")
                action = self.min_max_ab(d, agent_id, env, 1, finish_time, -math.inf, math.inf)[1]
                d += 1
        except:
            return action

    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id) - smart_heuristic(env, (robot_id + 1) % 2)

    def min_max_ab(self, d, robot_id, env, turn, time_limit, alpha, beta):
        if time.time() > time_limit:
            raise Exception("Time")

        if turn % 2 == 1:
            curr_max = -math.inf, None
            operators = env.get_legal_operators(robot_id)
            children = [env.clone() for _ in operators]
            for child, op in zip(children, operators):
                child.apply_operator(robot_id, op)
                if d == 1 or env.num_steps == 1:
                    children_val = self.heuristic(child, robot_id)
                else:
                    children_val = self.min_max_ab(d - 1, robot_id, child, turn + 1, time_limit, alpha, beta)[0]
                curr_max = (children_val, op) if children_val > curr_max[0] else curr_max
                alpha = max(curr_max[0], alpha)
                if curr_max[0] >= beta:
                    return math.inf, op
            return curr_max
        else:
            curr_min = math.inf, None
            operators = env.get_legal_operators((robot_id + 1) % 2)
            children = [env.clone() for _ in operators]
            for child, op in zip(children, operators):
                child.apply_operator((robot_id + 1) % 2, op)
                if d == 1 or env.num_steps == 1:
                    children_val = self.heuristic(child, robot_id)
                else:
                    children_val = self.min_max_ab(d - 1, robot_id, child, turn + 1, time_limit, alpha, beta)[0]
                curr_min = (children_val, op) if children_val < curr_min[0] else curr_min
                beta = min(curr_min[0], beta)
                if curr_min[0] <= alpha:
                    return -math.inf, op

            return curr_min


class AgentExpectimax(Agent):
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        finish_time = time.time() + time_limit*EPSILON
        d, action = 1, None
        try:
            while True:
                if time.time() > finish_time:
                    raise Exception("Time")
                action = self.expectimax(d, agent_id, env, 1, finish_time)[1]
                d += 1
        except:
            return action

    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id) - smart_heuristic(env, (robot_id + 1) % 2)

    def expectimax(self, d, robot_id, env, turn, time_limit):
        if time.time() > time_limit:
            raise Exception("Time")
        if turn % 2 == 1:
            operators = env.get_legal_operators(robot_id)
            children = [env.clone() for _ in operators]
            for child, op in zip(children, operators):
                child.apply_operator(robot_id, op)
            if d == 1 or env.num_steps == 1:
                children_val = [self.heuristic(child, robot_id) for child in children]
            else:
                children_val = [self.expectimax(d - 1, robot_id, child, turn + 1, time_limit)[0] for child in children]

            max_child = max(children_val)
            max_index = children_val.index(max_child)
            return max(children_val), operators[max_index]
        else:
            operators = env.get_legal_operators((robot_id + 1) % 2)
            children = [env.clone() for _ in operators]
            mone_list = []
            for child, op in zip(children, operators):
                child.apply_operator((robot_id + 1) % 2, op)
                mone_list += [2 if op == 'move east' or op == 'pick up' else 1]
            probs_sum = sum(mone_list)
            if d == 1 or env.num_steps == 1:
                children_val = [self.heuristic(child, robot_id) for child in children]
            else:
                children_val = [self.expectimax(d - 1, robot_id, child, turn + 1, time_limit)[0] for child in children]
            for val in children_val:
                index = children_val.index(val)
                children_val[index] = val * mone_list[index]/probs_sum
            return sum(children_val), None


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
