from gurobipy import Model, GRB
import numpy as np
import matplotlib.pyplot as plt

"""
All users have batteries installed, but only part of them have PV panels.
"""


def generate_pv_production(timestep=24):
    # Constants
    peak_irradiance = 1000  # W/m² at solar noon
    panel_efficiency = 0.15  # 15%
    system_size = 50  # kW

    # Generate a simulated solar irradiance curve over 24 hours (simplified)
    hours = np.arange(timestep)
    irradiance = peak_irradiance * np.cos((hours - 12) * np.pi / 12) ** 2
    irradiance[irradiance < 0] = 0  # No negative irradiance
    irradiance[:6] = 0  # No generation before 6 am
    irradiance[17:] = 0  # No generation after 8 pm

    # Calculate power output
    power_output = irradiance * panel_efficiency * system_size / 1000  # kW

    # Plot
    plt.plot(hours, power_output, label='Solar Power Output')
    plt.fill_between(hours, 0, power_output, alpha=0.3)
    plt.xlabel('Hour of Day')
    plt.ylabel('Power Output (kW)')
    plt.title('Simulated Solar Panel Power Output Over 24 Hours')
    plt.legend()
    plt.grid(True)
    plt.show()

    print(power_output)
    return power_output


def generate_electricity_consumption_profile(num_timestep:int):
    hourly_consumption = np.array([0.3] * 6 + [1.5] * 2 + [0.7] * 7 + [1.8] * 4 + [0.5] * 2 + [0.3] * 3) * 5

    hours = np.arange(num_timestep)  # 0-23 hours
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(hours, hourly_consumption, marker='o', linestyle='-', color='royalblue')
    plt.fill_between(hours, 0, hourly_consumption, color='lightblue', alpha=0.4)
    plt.title('General Hourly Energy Consumption Profile Over 24 Hours')
    plt.xlabel('Hour of Day')
    plt.ylabel('Energy Consumption (kWh)')
    plt.xticks(hours)
    plt.grid(True)
    plt.show()
    print(hourly_consumption)

    return hourly_consumption


def plot_local_price(c_local):
    hours = np.arange(24)
    plt.figure(figsize=(12, 6))
    plt.plot(hours, c_local, marker='o', linestyle='-', color='royalblue')
    plt.fill_between(hours, 0, c_local, color='lightblue', alpha=0.4)
    plt.title('General Hourly Energy Consumption Profile Over 24 Hours')
    plt.xlabel('Hour of Day')
    plt.ylabel('Energy Consumption (kWh)')
    plt.xticks(hours)
    plt.grid(True)
    plt.show()
    print(c_local)

    return 0


class AgentModel:
    def __init__(self, num_timestep=24, user=0):
        self.model = Model(f'Optimization-User{user}')
        ## set constants
        # time period
        self.T = num_timestep
        # total number of user
        # self.N = num_user

        # cost/feed-in price
        # c_feedin < c_local < c_grid
        self.c_local = []
        self.c_grid = []
        self.c_feedin = []
        self.c_cyc = []

        # battery efficiency coef
        self.n_char = 0.95
        self.n_disc = 0.95
        self.SoC_max = 10  # [kWh]
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
            # if 8 <= t <= 18:
            #     self.c_local.append(0.8)
            # else:
            #     self.c_local.append(1.2)

            self.c_grid.append(2.0)
            self.c_local = self.c_grid.copy()
            self.c_feedin.append(0.5)
            self.c_cyc.append(0.01)

        self.p_char_max = 4.0
        self.p_disc_max = 4.0

    def add_variables(self):
        for t in range(self.T):
            v_p = self.model.addVar(vtype=GRB.CONTINUOUS, name=f'p_local_time{t}')
            self.p[f't{t}'] = v_p

            soc = self.model.addVar(vtype=GRB.CONTINUOUS, name=f'SoC_time{t}')
            self.SoC[f't{t}'] = soc

            v_p_char = self.model.addVar(vtype=GRB.CONTINUOUS, name=f'p_char_time{t}')
            self.p_char[f't{t}'] = v_p_char

            v_p_disc = self.model.addVar(vtype=GRB.CONTINUOUS, name=f'p_disc_time{t}')
            self.p_disc[f't{t}'] = v_p_disc

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
                self.s[f't{t}'] = power_output[t] * 0.5
            else:
                self.s[f't{t}'] = power_output[t]

        print('Set identical hourly PV generation profile for every users.')
        return 0

    def add_constraints(self):
        self.model.addConstr(self.SoC[f't{0}'] == 0)

        for t in range(self.T):
            self.model.addConstr(self.SoC[f't{t}'] <= self.SoC_max)
            self.model.addConstr(self.SoC[f't{t}'] >= self.SoC_min)

            if t != self.T-1:
                self.model.addConstr(self.SoC[f't{t+1}'] - self.SoC[f't{t}'] -
                                     self.n_char*self.p_char[f't{t}'] + (1/self.n_disc)*self.p_disc[f't{t}'] == 0)

            self.model.addConstr(self.p[f't{t}'] + self.e[f't{t}'] +
                                 self.p_char[f't{t}'] - self.p_disc[f't{t}'] -
                                 self.s[f't{t}'] == 0)

            self.model.addConstr(self.p_char[f't{t}'] >= 0)
            self.model.addConstr(self.p_char[f't{t}'] <= self.p_char_max)
            self.model.addConstr(self.p_disc[f't{t}'] >= 0)
            self.model.addConstr(self.p_disc[f't{t}'] <= self.p_disc_max)

        self.model.update()

        return 0

    def add_objectives(self):
        # expense
        j_expense = 0
        for t in range(self.T):
            j_expense -= self.c_local[t]*self.p[f't{t}']
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

        hours = np.arange(self.T)  # 0-23 hours

        plt.figure(figsize=(12, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, self.N))

        plt.plot(hours, profile, marker='o', linestyle='-', label=f'User {n}', color=colors[n])

        plt.title('Hourly Battery Profile Over 24 Hours For Each User')
        plt.xlabel('Hour of Day')
        plt.ylabel('User SoC Profile (kWh)')
        plt.xticks(hours)
        plt.grid(True)
        plt.legend()
        plt.show()

        return 0

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # breakpoint()
    num_user = 4
    num_timestep = 24
    pv_profile = generate_pv_production(num_timestep)
    consumption_profile = generate_electricity_consumption_profile(num_timestep)

    ## initialize all agent model
    for n in range(num_user):
        agent_model = AgentModel(num_timestep=num_timestep)
        agent_model.set_pv_generation_profile(pv_profile, user_index=n)
        agent_model.set_energy_consumption_profile(consumption_profile, user_index=n)
        agent_model.add_variables()
        agent_model.add_constraints()
        agent_model.add_objectives()
        plot_local_price(agent_model.c_local)

    # breakpoint()
    c_grid = 2.0
    c_feedin = 1.0
    epsilon = 1e-4
    z = {}
    z['-1'] = [1] * num_timestep
    c_local_last = [c_grid] * num_timestep
    c_local = [0.5 * (c_feedin + c_grid)] * num_timestep
    p = {n: [] for n in range(num_user)}

    # Initialize the algorithm
    model_list = {f'User{n}': AgentModel(num_timestep=num_timestep, user=n) for n in range(num_user)}
    k = 0  # Iteration counter

    # Begin the iterative process
    while True:
        # Step 1: Solve independent optimization problem for each user
        for i in range(num_user):
            agent_model = model_list[f'User{i}']
            agent_model.set_pv_generation_profile(pv_profile, user_index=i)
            agent_model.set_energy_consumption_profile(consumption_profile, user_index=i)
            agent_model.c_local = c_local.copy()
            agent_model.add_variables()
            agent_model.add_constraints()
            agent_model.add_objectives()
            agent_model.model.update()

            agent_model.model.optimize()
            result = agent_model.retrieve_results()
            
            # Retrieve the solution for user i
            for t in range(num_timestep):
                p[i].append(agent_model.p[f't{t}'].X)

        # Step 2: Update z based on the solutions p from all users
        z[str(k)] = np.zeros(num_timestep)  # Sum over users
        for t in range(num_timestep):
            for i in range(num_user):
                z[str(k)][t] += p[i][t]

        # Step 3: Update the local price for each time period
        for t in range(num_timestep):
            if z[str(k)][t] * z[str(k-1)][t] > 0:
                temp = c_local[t]
                c_local[t] = c_local[t] + 0.5 * (c_local[t] - c_local_last[t])
                c_local_last[t] = temp
            elif z[str(k)][t] * z[str(k-1)][t] < 0:
                temp = c_local[t]
                c_local[t] = c_local[t] - 0.5 * (c_local[t] - c_local_last[t])
                c_local_last[t] = temp
            else:
                c_local_last[t] = c_local[t]
            # If z_new[t] == z[t], c_local remains unchanged
        print(f"For iteration {k}, the local trading price is \n{c_local} ")
        print(c_local_last)

        # Check for convergence
        if np.linalg.norm(np.array(c_local) - np.array(c_local_last), np.inf) < epsilon:
            break

        # Update iteration counter and z
        k += 1
        z[str(k)] = []




"""    if result:
        agent_model.print_result()
        agent_model.plot_user_soc_profile()"""

