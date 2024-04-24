import numpy as np
from helper_functions import (
    generate_pv_production,
    generate_electricity_consumption_profile,
    plot_local_price,
    plot_battery_soc,
    plot_iterations,
)
from model import AgentModel
import os
import datetime

"""
1. All users have batteries installed, but only part of them have PV panels.
2. For PV generation and electricity consumption, a day start from 6 am and end in 6 am the next day
"""

# def plot_battery_profile(battery_profile):

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # create directory
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    project_directory = '/Users/fxy/Game-theory-for-energy-communities'  # Adjust this to your project directory
    run_directory = os.path.join(project_directory, current_time)
    os.makedirs(run_directory, exist_ok=True)

    # Set the environment variable
    os.environ['RUN_DIRECTORY'] = run_directory

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


