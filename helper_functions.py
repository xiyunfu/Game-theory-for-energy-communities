import numpy as np
import matplotlib.pyplot as plt
import os
import json


def write_text(value, name: str, path, file_name: str, wandb=None):
    params_filename = os.path.join(path, file_name)
    with open(params_filename, 'a') as file:
        file.write(f'{name}: {value}\n')
    if wandb:
        wandb.log({name: value})


def save_dict(dict, path, name: str, wandb=None):
    """
    :param dict:
    :param path:
    :param name:
    :return:
    Save the parameters in a text file within the run directory
    """
    params_filename = os.path.join(path, name)
    with open(params_filename, 'w') as file:
        for key, value in dict.items():
            file.write(f'{key}: {value}\n')
    dict_name = name.split(".")[0]
    if wandb:
        if dict_name:
            wandb.log({dict_name: dict})
        else:
            wandb.log(dict)

def save_json(soc, path):
    params_filename = os.path.join(path, 'soc.json')
    with open(params_filename, 'w') as json_file:
        json.dump(soc, json_file)


class Handler:
    def __init__(self):
        self.run_directory = os.environ['RUN_DIRECTORY']
        self.wandb = None

    def _set_wandb(self, wandb=None):
        if wandb is None:
            raise ValueError("wandb must be provided and cannot be None.")
        self.wandb = wandb

    def _write_text(self, value, name: str, file_name: str):
        params_filename = os.path.join(self.run_directory, file_name)
        with open(params_filename, 'a') as file:
            file.write(f'{name}: {value}\n')
        if self.wandb:
            self.wandb.log({name: value})

    def generate_pv_production(self, timestep=24, size=100):
        peak_irradiance = 1000  # W/m² at solar noon
        panel_efficiency = 0.15  # 15%
        system_size = size  # m²

        hours = np.arange(timestep) + 6
        irradiance = peak_irradiance * np.cos((hours - 12) * np.pi / 12) ** 2  # W/m² for 1h
        irradiance[irradiance < 0] = 0  # No negative irradiance
        # irradiance[:6] = 0  # No generation before 6 am
        irradiance[12:] = 0  # No generation after 6 pm

        power_output = irradiance * panel_efficiency * system_size / 1000  # kWh

        hour_labels = [(6 + t) % 24 for t in range(timestep)]
        hour_label_positions = range(timestep)
        plt.figure(figsize=(12, 6))
        plt.plot(power_output, label='Solar Power Output')
        plt.fill_between(hour_label_positions, 0, power_output, alpha=0.3)
        plt.xlabel('Hour of Day')
        plt.ylabel('Power Output (kW)')
        plt.xticks(hour_label_positions, hour_labels)
        plt.title('Simulated Solar Panel Power Output Over 24 Hours')
        plt.legend()
        plt.grid(True)
        filename = "Simulated_PV_Output.png"
        full_path = os.path.join(self.run_directory, filename)
        plt.savefig(full_path)
        if self.wandb:
            self.wandb.log({"Simulated_PV_Output": self.wandb.Image(full_path)})
        plt.show()
        plt.close()

        return power_output

    def generate_electricity_consumption_profile(self, num_timestep: int):
        hourly_consumption = np.array([1.2] * 2 + [0.7] * 7 + [1.2] * 4 + [0.5] * 2 + [0.3] * 3 + [0.3] * 6) * 5
        hours = np.arange(num_timestep)  # 0-23 hours
        hour_labels = [(6 + t) % 24 for t in range(num_timestep)]
        hour_label_positions = range(num_timestep)
        # Plotting
        plt.figure(figsize=(12, 6))
        plt.plot(hourly_consumption, marker='o', linestyle='-', color='royalblue')
        plt.fill_between(hours, 0, hourly_consumption, color='lightblue', alpha=0.4)
        plt.title('General Hourly Energy Consumption Profile Over 24 Hours')
        plt.xlabel('Hour of Day')
        plt.ylabel('Energy Consumption (kWh)')
        plt.xticks(hour_label_positions, hour_labels)
        plt.grid(True)
        # print(hourly_consumption)

        filename = "Energy_Consumption_Profile.png"
        full_path = os.path.join(self.run_directory, filename)
        plt.savefig(full_path)
        if self.wandb:
            self.wandb.log({"Energy_Consumption_Profile": self.wandb.Image(full_path)})
        plt.show()
        plt.close()

        return hourly_consumption

    def plot_single_local_price(self, c_local):
        """
        Not in used
        :param c_local:
        :return:
        """
        hours = np.arange(24)
        plt.figure(figsize=(12, 6))
        plt.plot(hours, c_local, marker='o', linestyle='-', color='royalblue')
        plt.fill_between(hours, 0, c_local, color='lightblue', alpha=0.4)
        plt.title('Local Price')
        plt.xlabel('Hour of Day')
        plt.ylabel('Energy Consumption (kWh)')
        plt.xticks(hours)
        plt.grid(True)
        print(c_local)

        filename = "Local_Price.png"
        full_path = os.path.join(self.run_directory, filename)
        plt.savefig(full_path)
        if self.wandb:
            self.wandb.log({"Local_Price": self.wandb.Image(full_path)})
        plt.show()
        plt.close()

        return 0

    def plot_user_profile(
            self,
            user_profile,
            type: str = "Battery",
            num_user=4,
            num_timestep=24,
            ):
        # Plotting
        extra_timestep = 0
        if type == "Battery":
            extra_timestep = 1
        hour_labels = [(6 + t) % 24 for t in range(num_timestep + extra_timestep)]
        hour_label_positions = range(num_timestep + extra_timestep)  # Positions corresponding to the time steps
        plt.figure(figsize=(10, 6))
        for i in range(num_user):
            plt.plot(user_profile[i], label=f'User {i+1}')

        plt.title(f'{type} Profile for Each User')
        plt.xlabel('Time Step')
        plt.ylabel('State of Charge (SoC)')
        plt.xticks(hour_label_positions, hour_labels)
        plt.legend()
        plt.grid(True)

        filename = f"{type}_Profile_for_Each_User.png"
        full_path = os.path.join(self.run_directory, filename)
        plt.savefig(full_path)
        # Log the plot to wandb
        if self.wandb:
            self.wandb.log({f"{type} Profile for Each User": self.wandb.Image(full_path)})
        plt.show()
        plt.close()


    def plot_local_price(self, c, z):
        assert len(c) == len(z)  # assert the price vector and the sum of energy trading vector have same timestamps
        hours = np.arange(len(c))
        hour_labels = [(6 + t) % 24 for t in range(len(c))]
        hour_label_positions = range(len(c))
        fig, ax1 = plt.subplots(figsize=(12, 6))

        color = 'royalblue'
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('Electricity price per unit ($/kWh)', color=color)
        ax1.plot(c, marker='o', linestyle='-', color=color)
        ax1.fill_between(hours, 0, c, color='lightblue', alpha=0.4)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True)

        ax2 = ax1.twinx()
        color = 'darkred'
        ax2.set_ylabel('Amount of electricity traded in EC', color=color)
        ax2.plot(z, marker='o', linestyle='-', color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.axhline(y=0, color='darkred', linewidth=1.5, linestyle='--')

        plt.title('Local price and Electricity trading amounts')
        plt.xticks(hour_label_positions, hour_labels)

        filename = "Local_price_and_Electricity_trading_amounts.png"
        full_path = os.path.join(self.run_directory, filename)
        plt.savefig(full_path)
        if self.wandb:
            self.wandb.log({"Local_price_and_Electricity_trading_amounts": self.wandb.Image(full_path)})
        plt.show()
        plt.close()

        return 0

    def plot_iterations(self, iteration_list: list, label: str = "Delta"):
        num_timestep = len(iteration_list[0])
        iter_array = np.array(iteration_list)
        plt.figure(figsize=(12, 6))
        for i in range(num_timestep):
            plt.plot(iter_array[:, i], label=f'Hour {(i + 6) % 24}')

        plt.title(f'{label} Change Over Iterations for Each Timestep')
        plt.xlabel('Iteration Number')
        plt.ylabel(f'{label} Value')
        plt.legend()
        plt.grid(True)
        plt.gca().set_xscale('log')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()

        filename = f"Iteration {label}.png"
        full_path = os.path.join(self.run_directory, filename)
        plt.savefig(full_path)
        if self.wandb:
            self.wandb.log({f"Iteration {label}": self.wandb.Image(full_path)})
        plt.show()
        plt.close()

    def plot_expense_diff(self, expense):
        out_utility = []
        in_utility = []
        users = list(expense.keys())
        for i in users:
            out_utility.append(round(-expense[i][0], 1))
            in_utility.append(round(-expense[i][-1], 1))

        x = np.arange(len(users))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width / 2, out_utility, width, label='Out-community Utility')
        rects2 = ax.bar(x + width / 2, in_utility, width, label='In-community Utility')

        ax.set_xlabel('Users')
        ax.set_ylabel('Payoff')
        ax.set_title('Comparison of In- and Out-community Payoff by User')
        ax.set_xticks(x+1)
        ax.set_xticklabels(users)
        ax.legend()

        total_mean_diff = np.mean(np.array(in_utility) - np.array(out_utility))
        mid_index = int(len(users)/2)
        pv_mean_diff = np.mean(np.array(in_utility[:mid_index]) - np.array(out_utility[:mid_index]))
        nopv_mean_diff = np.mean(np.array(in_utility[mid_index:]) - np.array(out_utility[mid_index:]))
        first_user_diff = np.array(in_utility[0]) - np.array(out_utility[0])
        last_user_diff = np.array(in_utility[-1]) - np.array(out_utility[-1])

        self._write_text(total_mean_diff, name="Average Utility Difference", file_name="configuration.txt")
        self._write_text(pv_mean_diff, name="Average Utility Difference for user with PV", file_name="configuration.txt")
        self._write_text(nopv_mean_diff, name="Average Utility Difference for user without PV", file_name="configuration.txt")
        self._write_text(first_user_diff, name="First User Utility Difference", file_name="configuration.txt")
        self._write_text(last_user_diff, name="Last User Utility Difference", file_name="configuration.txt")

        # Function to attach a text label above each bar in *rects*, displaying its height.
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        autolabel(rects1)
        autolabel(rects2)
        fig.tight_layout()
        filename = "Expense differences for Each User.png"
        full_path = os.path.join(self.run_directory, filename)
        plt.savefig(full_path)
        if self.wandb:
            self.wandb.log({"Expense differences for Each User": self.wandb.Image(full_path)})
        plt.show()
        plt.close()

