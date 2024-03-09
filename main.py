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


def generate_electricity_consumption_profile():
    hourly_consumption = np.array([0.3] * 6 + [1.5] * 2 + [0.7] * 7 + [1.8] * 4 + [0.5] * 2 + [0.3] * 3) * 5

    hours = np.arange(24)  # 0-23 hours

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


class AgentModel:
    def __init__(self, num_user=4, num_timestep=24):
        self.model = Model("model")

        ## set constants
        # time period
        self.T = num_timestep
        # total number of user
        self.N = num_user

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
        self.p_grid = {}
        self.p_local = {}
        self.p_sell = {}
        self.p_char = {}
        self.p_disc = {}
        self.e = {}

        for _ in range(self.T):
            self.c_local.append(1.0)
            self.c_grid.append(2.0)
            self.c_feedin.append(0.5)
            self.c_cyc.append(0.1)

        self.p_char_max = []
        self.p_disc_max = []

        for n in range(self.N):
            self.p_char_max.append(2.0)
            self.p_disc_max.append(2.0)

    def add_variables(self):
        for n in range(self.N):
            for t in range(self.T):
                v_p_grid = self.model.addVar(vtype=GRB.CONTINUOUS, name=f'p_grid_time{t}_user{n}')
                self.p_grid[f'u{n}-t{t}'] = v_p_grid

                soc = self.model.addVar(vtype=GRB.CONTINUOUS, name=f'SoC_time{t}_user{n}')
                self.SoC[f'u{n}-t{t}'] = soc

                v_p_sell = self.model.addVar(vtype=GRB.CONTINUOUS, name=f'p_sell_time{t}_user{n}')
                self.p_sell[f'u{n}-t{t}'] = v_p_sell

                v_p_char = self.model.addVar(vtype=GRB.CONTINUOUS, name=f'p_char_time{t}_user{n}')
                self.p_char[f'u{n}-t{t}'] = v_p_char

                v_p_disc = self.model.addVar(vtype=GRB.CONTINUOUS, name=f'p_disc_time{t}_user{n}')
                self.p_disc[f'u{n}-t{t}'] = v_p_disc

        self.model.update()

        return 0

    def set_energy_consumption_profile(self, hourly_consumption):
        for n in range(self.N):
            for t in range(self.T):
                self.e[f'u{n}-t{t}'] = hourly_consumption[t]

        print('Set identical hourly energy consumption profile for every users.')
        return 0

    def set_pv_generation_profile(self, power_output):
        for n in range(self.N):
            for t in range(self.T):
                self.p_local[f'u{n}-t{t}'] = power_output[t]

        print('Set identical hourly PV generation profile for every users.')
        return 0

    def add_constraints(self):
        for n in range(self.N):
            self.model.addConstr(self.SoC[f'u{n}-t{0}'] == 0)

            for t in range(self.T):
                self.model.addConstr(self.SoC[f'u{n}-t{t}'] <= self.SoC_max)
                self.model.addConstr(self.SoC[f'u{n}-t{t}'] >= self.SoC_min)

                if t != self.T-1:
                    self.model.addConstr(self.SoC[f'u{n}-t{t+1}'] - self.SoC[f'u{n}-t{t}'] +
                                         self.n_char*self.p_char[f'u{n}-t{t}'] - (1/self.n_disc)*self.p_disc[f'u{n}-t{t}'] == 0)

                self.model.addConstr(self.p_grid[f'u{n}-t{t}'] + self.p_local[f'u{n}-t{t}'] -
                                     self.p_sell[f'u{n}-t{t}'] - self.e[f'u{n}-t{t}'] -
                                     self.p_char[f'u{n}-t{t}'] - self.p_disc[f'u{n}-t{t}'] == 0)

                self.model.addConstr(self.p_grid[f'u{n}-t{t}'] >= 0)
                self.model.addConstr(self.p_sell[f'u{n}-t{t}'] >= 0)

                self.model.addConstr(self.p_char[f'u{n}-t{t}'] >= 0)
                self.model.addConstr(self.p_char[f'u{n}-t{t}'] <= self.p_char_max[n])
                self.model.addConstr(self.p_disc[f'u{n}-t{t}'] >= 0)
                self.model.addConstr(self.p_disc[f'u{n}-t{t}'] <= self.p_disc_max[n])

        self.model.update()

        return 0

    def add_objectives(self):
        # expense
        j_expense = 0
        for n in range(self.N):
            for t in range(self.T):
                j_expense += self.c_local[t]*self.p_local[f'u{n}-t{t}'] + self.c_grid[t]*self.p_grid[f'u{n}-t{t}'] - \
                             self.c_feedin[t]*self.p_sell[f'u{n}-t{t}']
        # battery
        j_battery = 0
        for n in range(self.N):
            for t in range(self.T):
                j_battery += ((self.n_char*self.p_char[f'u{n}-t{t}'] + (1/self.n_disc)*self.p_disc[f'u{n}-t{t}']) * self.c_cyc[t])**2

        self.model.setObjective(j_expense + j_battery, sense=GRB.MINIMIZE)
        self.model.update()

        return 0

    def retrieve_results(self):
        # check the solution status
        if self.model.status == GRB.OPTIMAL:
            print("Optimal solution found.")
        elif self.model.status == GRB.INF_OR_UNBD:
            print("Model is infeasible or unbounded.")
        elif self.model.status == GRB.INFEASIBLE:
            print("Model is infeasible.")
        elif self.model.status == GRB.UNBOUNDED:
            print("Model is unbounded.")
        else:
            print("Optimization was stopped with status", self.model.status)

        objective_value = self.model.ObjVal
        print("Objective Value:", objective_value)

        for var in self.model.getVars():
            print(f"{var.VarName}: {var.X}")

        return 0


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # breakpoint()
    num_user = 4
    num_timestep = 24
    pv_profile = generate_pv_production(num_timestep)
    # breakpoint()
    consumption_profile = generate_electricity_consumption_profile()
    agent_model = AgentModel(num_user=num_user, num_timestep=num_timestep)
    agent_model.set_pv_generation_profile(pv_profile)
    agent_model.set_energy_consumption_profile(consumption_profile)
    agent_model.add_variables()
    agent_model.add_constraints()
    agent_model.add_objectives()
    agent_model.model.optimize()
    agent_model.retrieve_results()

