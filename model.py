from gurobipy import Model, GRB
import numpy as np
import matplotlib.pyplot as plt
import os

run_directory = os.environ['RUN_DIRECTORY']


class AgentModel:
    def __init__(self, num_timestep=24, user=0, SoC_max=20, SoC_diff=False):
        self.model = Model(f'Optimization-User{user}')

        self.T = num_timestep

        # cost/feed-in price
        # c_feedin < c_local < c_grid
        self.c_grid = []
        self.c_feedin = []
        self.c_cyc = []

        # battery efficiency coef
        self.n_char = 0.95
        self.n_disc = 0.95
        if SoC_diff:
            self.SoC_max = SoC_max + 2 * int(user)  # [kWh]
        else:
            self.SoC_max = SoC_max
        self.SoC_min = 0  # [kWh]

        # initialise variables
        # p - the amounts of electricity, SoC - electricity storage in battery, e - energy consumption
        # all in [kW]
        self.SoC = {}
        self.p = {}
        self.p_char = {}
        self.p_disc = {}
        self.e = {}
        self.s = {}
        self.z = {}

        for t in range(self.T):
            self.c_grid.append(20.0)
            self.c_feedin.append(10.0)
            self.c_cyc.append(0.1)

        self.p_char_max = 10.0
        self.p_disc_max = 10.0

    def add_variables(self):
        for t in range(self.T):
            v_p = self.model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f'p_local_time{t}')
            self.p[f't{t}'] = v_p

            soc = self.model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f'SoC_time{t}')
            self.SoC[f't{t}'] = soc

            v_p_char = self.model.addVar(vtype=GRB.CONTINUOUS, name=f'p_char_time{t}')
            self.p_char[f't{t}'] = v_p_char

            v_p_disc = self.model.addVar(vtype=GRB.CONTINUOUS, name=f'p_disc_time{t}')
            self.p_disc[f't{t}'] = v_p_disc

        soc = self.model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f'SoC_time{self.T}')
        self.SoC[f't{self.T}'] = soc

        self.model.update()

        return 0

    def set_energy_consumption_profile(self, hourly_consumption, user_index):
        for t in range(self.T):
            self.e[f't{t}'] = hourly_consumption[t]

        print('Set identical hourly energy consumption profile for every users.')
        return 0

    def set_pv_generation_profile(self, power_output, user_index):
        for t in range(self.T):
            if user_index >= num_user/2:
                self.s[f't{t}'] = power_output[t] * 0
            else:
                self.s[f't{t}'] = power_output[t] * 2

        print('Set identical hourly PV generation profile for every users.')
        return 0

    def add_agent_constraint(self):
        self.model.addConstr(self.SoC[f't{0}'] == 0)

        for t in range(self.T):
            self.model.addConstr(self.SoC[f't{t}'] <= self.SoC_max)
            self.model.addConstr(self.SoC[f't{t}'] >= self.SoC_min)

            self.model.addConstr(self.SoC[f't{t+1}'] - self.SoC[f't{t}'] -
                                 self.n_char*self.p_char[f't{t}'] + (1/self.n_disc)*self.p_disc[f't{t}'] == 0)

            self.model.addConstr(self.p[f't{t}'] + self.e[f't{t}'] +
                                 self.p_char[f't{t}'] - self.p_disc[f't{t}'] -
                                 self.s[f't{t}'] == 0)

            self.model.addConstr(self.p_char[f't{t}'] >= 0)
            self.model.addConstr(self.p_char[f't{t}'] <= self.p_char_max)
            self.model.addConstr(self.p_disc[f't{t}'] >= 0)
            self.model.addConstr(self.p_disc[f't{t}'] <= self.p_disc_max)

        self.model.addConstr(self.SoC[f't{self.T}'] <= self.SoC_max)
        self.model.addConstr(self.SoC[f't{self.T}'] >= self.SoC_min)

        self.model.update()

        return 0

    def add_objectives(self, c_local):
        # expense
        j_expense = 0
        for t in range(self.T):
            j_expense -= c_local[t]*self.p[f't{t}']
        # battery
        j_battery = 0
        for t in range(self.T):
            j_battery += ((self.n_char*self.p_char[f't{t}'] + (1/self.n_disc)*self.p_disc[f't{t}']) * self.c_cyc[t])**2

        self.model.setObjective(j_expense + j_battery, sense=GRB.MINIMIZE)
        self.model.update()

        return 0

    def retrieve_results(self):
        r = 0
        # check the solution status
        if self.model.status == GRB.OPTIMAL:
            print("Optimal solution found.")
            r = 1
        elif self.model.status == GRB.INF_OR_UNBD:
            print("Model is infeasible or unbounded.")
            self.model.computeIIS()
            print('\nThe following constraints and variables are in the IIS:')
            for c in self.model.getConstrs():
                if c.IISConstr:
                    print(f'\t{c.constrname}: {self.model.getRow(c)} {c.Sense} {c.RHS}')

            for v in self.model.getVars():
                if v.IISLB:
                    print(f'\t{v.varname} ≥ {v.LB}')
                if v.IISUB:
                    print(f'\t{v.varname} ≤ {v.UB}')
        elif self.model.status == GRB.INFEASIBLE:
            print("Model is infeasible.")
        elif self.model.status == GRB.UNBOUNDED:
            print("Model is unbounded.")
        else:
            print("Optimization was stopped with status", self.model.status)

        return r

    def print_result(self):
        objective_value = self.model.ObjVal
        print("Objective Value:", objective_value)

        for var in self.model.getVars():
            print(f"{var.VarName}: {var.X}")

