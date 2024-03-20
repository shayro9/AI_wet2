import time
from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random

# TODO: section a : 3


def smart_heuristic(env: WarehouseEnv, robot_id: int):
    # robot
    robot = env.get_robot(robot_id)
    # opponent
    opponent = env.get_robot(not robot_id)

    robot_position = robot.position
    robot_battery = robot.battery
    robot_credit = robot.credit
    robot_holding_package = robot.package

    ######## weights #########
    # when holding package:
    robot2destination_weight = -0.3
    # when not holding package:
    robot2package_weight = -0.4
    robot2package_dist_average_weight = 0.3
    robot_is_holding_package = 10;
    credit_weight = 30
    battery_weight = 0.2
    ##########################

    package = None
    robot2package_main_dist = 0
    heuristic = 0
    robot2p_robot2d_min = 9999999
    # If the robot is not holding a package:
    if not robot_holding_package:
        for package_tmp in [p for p in env.packages if p.on_board and not opponent.package == p]:
            robot2package_tmp = manhattan_distance(robot_position, package_tmp.position)
            package2dest_tmp = manhattan_distance(package_tmp.position, package_tmp.destination)
            if robot2package_tmp+package2dest_tmp < robot2p_robot2d_min:
                package = package_tmp
                robot2package = robot2package_tmp
                package2dest = package2dest_tmp
                robot2p_robot2d_min = robot2package_tmp+package2dest_tmp

    if robot_holding_package:
        robot2destination= manhattan_distance(robot_position, robot.package.destination)
        heuristic += robot_is_holding_package
        heuristic += robot2destination * robot2destination_weight
    if package and not robot_holding_package:
        heuristic += robot2package * robot2package_weight

    heuristic += robot_credit * credit_weight + robot_battery * battery_weight
    return heuristic




class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        time_start = time.time()
        robot_id = agent_id
        FlagTime = 0
        for d in range(1,100):
            v, vo, FlagTime = self.run_step_inner(env, robot_id, agent_id, time_limit, time_start,d)
            if FlagTime == 0:
                Action = vo
            else:
                break
        return Action
    def run_step_inner(self, env: WarehouseEnv, robot_id, agent_id, time_limit, time_start,d):
        DeltaTime =  time.time() - time_start
        robot = env.get_robot(robot_id)
        if not robot.battery > 0  or  DeltaTime >= time_limit*0.9 or d == 0:
            FlagTime =  DeltaTime >= time_limit*0.9
            return smart_heuristic(env,agent_id), 'park', FlagTime
        operators, children = self.successors(env,robot_id)
        OtherRobotID = (robot_id + 1) % 2
        if robot_id == agent_id:
            CurMax = float('-inf')
            for o,child in zip(operators,children):
                v, _, _ = self.run_step_inner(child, OtherRobotID, agent_id, time_limit, time_start, d-1)
                if v > CurMax:
                    CurMax = v
                    CurOperator = o
            return CurMax, CurOperator, 0
        else:
            CurMin = float('inf')
            for o,child in zip(operators,children):
                v, _, _ = self.run_step_inner(child, OtherRobotID, agent_id, time_limit, time_start, d-1)
                if v < CurMin:
                    CurMin = v
                    CurOperator = o
            return CurMin, CurOperator, 0

    def successors(self, env: WarehouseEnv, robot_id: int):
        operators = env.get_legal_operators(robot_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(robot_id, op)
        return operators, children


class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        time_start = time.time()
        robot_id = agent_id
        FlagTime = 0
        alpha = float('-inf')
        beta = float('inf')
        for d in range(1,100):
            v, vo, FlagTime = self.run_step_inner(env, robot_id, agent_id, time_limit, time_start,d,alpha,beta)
            if FlagTime == 0:
                Action = vo
            else:
                break
        return Action
    def run_step_inner(self, env: WarehouseEnv, robot_id, agent_id, time_limit, time_start,d,alpha,beta):
        DeltaTime =  time.time() - time_start
        robot = env.get_robot(robot_id)
        if not robot.battery > 0  or  DeltaTime >= time_limit*0.9 or d == 0:
            FlagTime =  DeltaTime >= time_limit*0.9
            return smart_heuristic(env,agent_id), 'park', FlagTime
        operators, children = self.successors(env,robot_id)
        OtherRobotID = (robot_id + 1) % 2
        if robot_id == agent_id:
            CurMax = float('-inf')
            for o,child in zip(operators,children):
                v, _, _ = self.run_step_inner(child, OtherRobotID, agent_id, time_limit, time_start, d-1,alpha,beta)
                if v > CurMax:
                    CurMax = v
                    CurOperator = o
                if CurMax > alpha:
                    alpha = CurMax
                if CurMax >= beta:
                    return float('inf'), None, 0
            try:
                return CurMax, CurOperator, 0
            except:
                return CurMax, None, 0
        else:
            CurMin = float('inf')
            for o,child in zip(operators,children):
                v, _, _ = self.run_step_inner(child, OtherRobotID, agent_id, time_limit, time_start, d-1,alpha,beta)
                if v < CurMin:
                    CurMin = v
                    CurOperator = o
                if CurMin < beta:
                    beta = CurMin
                if CurMin <= alpha:
                    return float('-inf'), None, 0
            try:
                return CurMin, CurOperator, 0
            except:
                return CurMin, None, 0

    def successors(self, env: WarehouseEnv, robot_id: int):
        operators = env.get_legal_operators(robot_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(robot_id, op)
        return operators, children



class AgentExpectimax(Agent):
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        time_start = time.time()
        robot_id = agent_id
        FlagTime = 0
        for d in range(1, 100):
            v, vo, FlagTime = self.run_step_inner(env, robot_id, agent_id, time_limit, time_start, d)
            if FlagTime == 0:
                Action = vo
            else:
                break
        return Action

    def run_step_inner(self, env: WarehouseEnv, robot_id, agent_id, time_limit, time_start, d):
        DeltaTime = time.time() - time_start
        robot = env.get_robot(robot_id)
        if not robot.battery > 0 or DeltaTime >= time_limit * 0.9 or d == 0:
            FlagTime = DeltaTime >= time_limit * 0.9
            return smart_heuristic(env, agent_id), 'park', FlagTime
        operators, children = self.successors(env, robot_id)
        OtherRobotID = (robot_id + 1) % 2
        if robot_id == agent_id:
            CurMax = float('-inf')
            for o, child in zip(operators, children):
                v, _, _ = self.run_step_inner(child, OtherRobotID, agent_id, time_limit, time_start, d - 1)
                if v > CurMax:
                    CurMax = v
                    CurOperator = o
            return CurMax, CurOperator, 0
        else:
            CurMin = float('inf')
            TotalV = []
            operators_prob = self.CalcProb(env, robot_id)
            i = 0
            for o, child in zip(operators, children):
                v, _, _ = self.run_step_inner(child, OtherRobotID, agent_id, time_limit, time_start, d - 1)
                TotalV.append(v*operators_prob[i])
                i += 1
            meanV = sum(TotalV)/len(TotalV)
            return meanV, None, 0

    def successors(self, env: WarehouseEnv, robot_id: int):
        operators = env.get_legal_operators(robot_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(robot_id, op)
        return operators, children

    def CalcProb(self, env: WarehouseEnv, robot_id: int):
        operators = env.get_legal_operators(robot_id)
        operators_prob = []
        n = len(operators)
        if 'move east' not in operators and "pick up" not in operators:
            for operator in operators:
                operators_prob.append(1/n)
        if 'move east' not in operators and "pick up" in operators:
            for operator in operators:
                if operator == "pick up":
                     operators_prob.append(2/(n+1))
                else:
                    operators_prob.append(1/n)
        if 'move east' in operators and "pick up" not in operators:
            for operator in operators:
                if operator == 'move east':
                     operators_prob.append(2/(n+1))
                else:
                    operators_prob.append(1 / (n + 1))
        if 'move east' in operators and "pick up"  in operators:
            for operator in operators:
                if operator == 'move east' or operator == "pick up":
                     operators_prob.append(2/(n+2))
                else:
                    operators_prob.append(1 / (n + 2))
        return operators_prob

# here you can check specific paths to get to know the environment
class AgentHardCoded(Agent):
    def __init__(self):
        self.step = 0
        # specifiy the path you want to check - if a move is illegal - the agent will choose a random move
        self.trajectory = ["move east", "move east", "move east","move east","move east","move east", "move east",
                           "move east", "move east", "move east", "move east", "move east"]

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


