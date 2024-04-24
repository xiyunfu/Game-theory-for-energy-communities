import numpy as np
import matplotlib.pyplot as plt
import os

run_directory = os.environ['RUN_DIRECTORY']


def generate_pv_production(timestep=24):
    # Constants
    peak_irradiance = 1000  # W/m² at solar noon
    panel_efficiency = 0.15  # 15%
    system_size = 50  # kW

    # Generate a simulated solar irradiance curve over 24 hours (simplified)
    hours = np.arange(timestep) + 6
    irradiance = peak_irradiance * np.cos((hours - 12) * np.pi / 12) ** 2
    irradiance[irradiance < 0] = 0  # No negative irradiance
    # irradiance[:6] = 0  # No generation before 6 am
    irradiance[12:] = 0  # No generation after 6 pm

    # Calculate power output
    power_output = irradiance * panel_efficiency * system_size / 1000  # kW

    # Plot
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
    plt.show()

    print(power_output)

    filename = "Simulated_PV_Output.png"
    full_path = os.path.join(run_directory, filename)
    plt.savefig(full_path)
    plt.close()

    return power_output


def generate_electricity_consumption_profile(num_timestep:int):
    hourly_consumption = np.array([1.5] * 2 + [0.7] * 7 + [1.2] * 4 + [0.5] * 2 + [0.3] * 3 + [0.3] * 6) * 5

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
    plt.show()
    print(hourly_consumption)

    filename = "Energy_Consumption_Profile.png"
    full_path = os.path.join(run_directory, filename)
    plt.savefig(full_path)
    plt.close()

    return hourly_consumption


def plot_single_local_price(c_local):
    hours = np.arange(24)
    plt.figure(figsize=(12, 6))
    plt.plot(hours, c_local, marker='o', linestyle='-', color='royalblue')
    plt.fill_between(hours, 0, c_local, color='lightblue', alpha=0.4)
    plt.title('Local Price')
    plt.xlabel('Hour of Day')
    plt.ylabel('Energy Consumption (kWh)')
    plt.xticks(hours)
    plt.grid(True)
    plt.show()
    print(c_local)

    filename = "Local_Price.png"
    full_path = os.path.join(run_directory, filename)
    plt.savefig(full_path)
    plt.close()

    return 0


def plot_battery_soc(battery_profile, num_user=4, num_timestep=24):
    # Plotting
    hour_labels = [(6 + t) % 24 for t in range(num_timestep + 1)]
    hour_label_positions = range(num_timestep + 1)  # Positions corresponding to the time steps
    plt.figure(figsize=(10, 6))
    for i in range(num_user):
        plt.plot(battery_profile[i], label=f'User {i}')

    plt.title('Battery Profile for Each User')
    plt.xlabel('Time Step')
    plt.ylabel('State of Charge (SoC)')
    plt.xticks(hour_label_positions, hour_labels)
    plt.legend()
    plt.grid(True)
    plt.show()

    filename = "Battery_Profile_for_Each_User.png"
    full_path = os.path.join(run_directory, filename)
    plt.savefig(full_path)
    plt.close()


def plot_local_price(c, z):
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

    plt.show()

    filename = "Local_price_and_Electricity_trading_amounts.png"
    full_path = os.path.join(run_directory, filename)
    plt.savefig(full_path)
    plt.close()

    return 0


# def plot_delta(delta_list):
#     num_timestep = len(delta_list[0])
#     delta_array = np.array(delta_list)
#     plt.figure(figsize=(12, 6))
#     for i in range(num_timestep):
#         plt.plot(delta_array[:, i], label=f'Hour {(i + 6) % 24}')
#
#     plt.title('Delta Change Over Iterations for Each Timestep')
#     plt.xlabel('Iteration Number')
#     plt.ylabel('Delta Value')
#     plt.legend()
#     plt.grid(True)
#     plt.gca().set_xscale('log')
#     plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
#     plt.tight_layout()
#     plt.show()
#

def plot_iterations(iteration_list: list, label: str = "Delta"):
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
    plt.show()

    filename = f"Iteration {label}.png"
    full_path = os.path.join(run_directory, filename)
    plt.savefig(full_path)
    plt.close()

