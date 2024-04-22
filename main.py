from gurobipy import Model, GRB
import numpy as np
import matplotlib.pyplot as plt
from helper_functions import (
    generate_pv_production,
    generate_electricity_consumption_profile,
    plot_local_price,
    plot_battery_soc,
    plot_iterations,
)
"""
1. All users have batteries installed, but only part of them have PV panels.
2. For PV generation and electricity consumption, a day start from 6 am and end in 6 am the next day
"""

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

    def plot_user_soc_profile(self):
        # profile = {n: [] for n in range(self.N)}
        profile = []

        for t in range(self.T):
            profile.append(self.SoC[f't{t}'].X)

        hour_labels = [(6 + t) % 24 for t in range(self.T)]
        hour_label_positions = range(self.T)

        plt.figure(figsize=(12, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, self.N))

        plt.plot(profile, marker='o', linestyle='-', label=f'User {n}', color=colors[n])

        plt.title('Hourly Battery Profile Over 24 Hours For Each User')
        plt.xlabel('Hour of Day')
        plt.ylabel('User SoC Profile (kWh)')
        plt.xticks(hour_label_positions, hour_labels)

        plt.grid(True)
        plt.legend()
        plt.show()

        return 0


# def plot_battery_profile(battery_profile):

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # breakpoint()
    num_user = 4
    num_timestep = 24
    pv_profile = generate_pv_production(num_timestep)
    consumption_profile = generate_electricity_consumption_profile(num_timestep)

    c_grid = 20.0
    c_feedin = 10.0
    epsilon = 1e-3
    z = {}
    z['-1'] = [1] * num_timestep
    c_local = [0.5 * (c_feedin + c_grid)] * num_timestep
    delta = [5] * num_timestep   #[(c_grid - c_feedin) * 0.5] * num_timestep
    c_local_list = [c_local.copy()]
    delta_list = [delta.copy()]
    z_list = []

    # Initialize all agent models
    model_list = {f'User{n}': AgentModel(num_timestep=num_timestep, user=n, SoC_max=40) for n in range(num_user)}
    k = 0  # Iteration counter

    # Begin the iterative process
    while True:
        p = {n: [] for n in range(num_user)}
        # Step 1: Solve independent optimization problem for each user
        for i in range(num_user):
            agent_model = model_list[f'User{i}']
            agent_model.set_pv_generation_profile(pv_profile, user_index=i)
            agent_model.set_energy_consumption_profile(consumption_profile, user_index=i)
            if k == 0:
                agent_model.add_variables()
                agent_model.add_agent_constraint()

            agent_model.add_objectives(c_local)
            agent_model.model.update()

            agent_model.model.optimize()

            result = agent_model.retrieve_results()

            # Retrieve the solution for user i
            for t in range(num_timestep):
                p[i].append(agent_model.p[f't{t}'].X)

            agent_model.print_result()

        # Step 2: Update z based on the solutions p from all users
        z[str(k)] = np.zeros(num_timestep)  # Sum over users
        for t in range(num_timestep):
            for i in range(num_user):
                z[str(k)][t] += p[i][t]

        # Step 3: Update the local price for each time period
        for t in range(num_timestep):
            if z[str(k)][t] > 0:
                c_local[t] = max(c_feedin, c_local[t] - delta[t])
            elif z[str(k)][t] < 0:
                c_local[t] = min(c_grid, c_local[t] + delta[t])
            if z[str(k)][t] * z[str(k-1)][t] < 0:
                delta[t] = 0.5 * delta[t]

        delta_list.append(delta.copy())
        c_local_list.append(c_local.copy())
        z_list.append(z[str(k)].copy())

            # If z_new[t] == z[t], c_local remains unchanged
        print(f"For iteration {k}, the local trading price is \n{[(t, c_local[t]) for t in range(len(c_local))]}")
        print(delta)
        print(z[str(k)])

        # Check for convergence
        if np.linalg.norm(np.array(delta), np.inf) < epsilon:
            break
        if k >= 1000:
            # plot_price_iteration(z)
            break

        # Update iteration counter and z
        k += 1


    # Plotting
    plot_local_price(c_local, z[str(k)])
    plot_iterations(delta_list, label="Delta")
    plot_iterations(c_local_list, label="Local Price")
    plot_iterations(z_list, label="Aggregator z")

    battery_profile = {i: [] for i in range(num_user)}
    for i in range(num_user):
        agent_model = model_list[f'User{i}']
        for t in range(num_timestep+1):
            battery_profile[i].append(agent_model.SoC[f't{t}'].X)

    plot_battery_soc(battery_profile, num_user, num_timestep)






"""    if result:
        agent_model.print_result()
        agent_model.plot_user_soc_profile()"""

