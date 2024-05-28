import numpy as np
from gurobipy import GRB
import time
import wandb
from helper_functions import Handler, save_dict, write_text
from model import AgentModel
from config import initialize_run_directory

"""
1. All users have batteries installed, but only part of them have PV panels.
2. For PV generation and electricity consumption, a day start from 6 am and end in 6 am the next day
"""

def main():
    start_time = time.time()
    params = {
        'num_user': 4,
        'num_timestep': 24,
        'c_grid': 30.0,  # 30 Swiss cents (CHF) per kWh
        'c_feedin': 10.0,  # ~9 Swiss cents (CHF) per kWh
        'c_cyc': 0.1,
        'init_delta': 0.001 * 2 ** 10,
        'epsilon': 1e-3,
        'SoC_max': 500,  # kWh
        'SoC_diff': False,
        'consumption_rd_mag': 2,
        'generation_rd_mag': 0,
        'z_threshold': 0.01,
        'min_delta': 0.001,
        'max_iteration': 1e4,
        'pv_size': 220,  # m^2
        'first_user_pv_size_augment': 0,  # kWh
        'first_user_battery_size_augment': 0,  # kWh
        'last_user_battery_size_augment': 0,  # kWh
        'tau': 0.001,
    }

    run_directory = initialize_run_directory(params)
    save_dict(params, run_directory, name='configuration.txt')

    num_user = params['num_user']
    num_timestep = params['num_timestep']
    c_grid = params['c_grid']
    c_feedin = params['c_feedin']
    epsilon = params['epsilon']
    soc_max = params['SoC_max']
    soc_diff = params['SoC_diff']
    tau = params['tau']

    handler = Handler()
    pv_profile = handler.generate_pv_production(num_timestep, size=params['pv_size'])
    consumption_profile = handler.generate_electricity_consumption_profile(num_timestep)

    z = {}
    z['-1'] = [1] * num_timestep
    # c_local = [[0.5 * (c_feedin + c_grid)] * num_timestep] * num_user
    delta = [params['init_delta']] * num_timestep
    c_local_list = []
    delta_list = [delta.copy()]
    z_list = []
    user_consumption_profile = {i: [] for i in range(num_user)}
    user_generation_profile = {i: [] for i in range(num_user)}
    expense = {n: [] for n in range(num_user)}

    # Initialize all agent models
    c_local_old = [[c_grid/num_user for _ in range(num_timestep)] for _ in range(num_user)]
    model_list = {f'User{n}': AgentModel(params, user=n)
                  for n in range(num_user)}
    k = 0
    # Begin the iterative process
    while True:
        p_d = {n: [] for n in range(num_user)}
        soc = {n: [] for n in range(num_user)}

        # Step 1: Solve independent optimization problem for each user
        for i in range(num_user):
            agent_model = model_list[f'User{i}']
            agent_model.set_pv_generation_profile(
                params,
                pv_profile,
                user_index=i,
                num_user=num_user,
                random_mag=params['generation_rd_mag']
            )
            agent_model.set_energy_consumption_profile(
                consumption_profile,
                user_index=i,
                random_mag=params['consumption_rd_mag']
            )
            if k == 0:
                agent_model.add_variables()
                agent_model.add_agent_constraint()
                agent_model.add_objectives([c_grid / num_user] * num_timestep)  # set the uniform price
                for t in range(num_timestep):
                    user_consumption_profile[i].append(agent_model.e[f't{t}'])
                    user_generation_profile[i].append(agent_model.s[f't{t}'])
            else:
                agent_model.add_objectives(c_local[i])
            agent_model.model.update()
            agent_model.model.optimize()
            assert agent_model.model.status == GRB.OPTIMAL
            expense[i].append(agent_model.model.getObjective().getValue())

            # Retrieve the solution for user i
            for t in range(num_timestep):
                p_d[i].append(agent_model.p_d[f't{t}'].X)
                soc[i].append(agent_model.SoC[f't{t}'].X)

        if k == 0:
            handler.plot_user_profile(user_consumption_profile, type="Consumption", num_user=num_user,
                                      num_timestep=num_timestep)
            handler.plot_user_profile(user_generation_profile, type="Generation", num_user=num_user,
                                      num_timestep=num_timestep)

        # Step 2: Update z based on the solutions p from all users
        z[str(k)] = np.zeros(num_timestep)
        for t in range(num_timestep):
            for i in range(num_user):
                z[str(k)][t] += p_d[i][t]

        # Step 3: Update the local price for each time period
        c_local = [[0.0 for _ in range(num_timestep)] for _ in range(num_user)]
        for t in range(num_timestep):
            for i in range(num_user):
                c_local[i][t] = (1 - tau) * c_local_old[i][t] + tau * c_grid * (p_d[i][t] / z[str(k)][t])
            # if z[str(k)][t] > params['z_threshold']:
            #     c_local[t] = max(c_feedin, c_local[t] - delta[t])
            # elif z[str(k)][t] < - params['z_threshold']:
            #     c_local[t] = min(c_grid, c_local[t] + delta[t])
            # if z[str(k)][t] * z[str(k - 1)][t] < 0:
            #     delta[t] = max(0.5 * delta[t], params['min_delta'])

        # delta_list.append(delta.copy())
        c_local_list.append(c_local.copy())
        z_list.append(z[str(k)].copy())
        c_local_old = c_local.copy()

        print(f"For iteration {k}, the local trading price is \n{[(t, c_local[t]) for t in range(len(c_local))]}")
        print(delta)
        print(z[str(k)])

        # Check for convergence
        if k > 1:
            if np.linalg.norm(np.array(c_local_list[-1]) - np.array(c_local_list[-2]), np.inf) < epsilon:
                convergency = True
                break
        if k >= params['max_iteration']:
            convergency = False
            break

        k += 1

    end_time = time.time()
    elapsed_time = end_time - start_time
    write_text(elapsed_time, "Elapsed Time", run_directory, file_name='configuration.txt')
    write_text(k, "Final iterations", run_directory, file_name='configuration.txt')

    wandb.init(
        project="game-theory_sp",
        name=f"SoCMax={params['SoC_max']}-SoCDiff={params['SoC_diff']}-PVSize={params['pv_size']}-zT={params['z_threshold']}",
        config=params,
    )
    wandb.log({"Covengency": convergency})
    handler._set_wandb(wandb)

    total_consumption = 0
    total_generation = 0
    for i in range(num_user):
        agent_model = model_list[f'User{i}']
        for t in range(num_timestep):
            total_consumption += agent_model.e[f't{t}']
            total_generation += agent_model.s[f't{t}']

    gamma = round(total_generation / total_consumption, 2)
    write_text(total_consumption, "Total electricity consumption", run_directory, file_name='configuration.txt',
               wandb=wandb)
    write_text(total_generation, "Total electricity generation", run_directory, file_name='configuration.txt',
               wandb=wandb)
    write_text(gamma, "Electricity consumption/generation ratio", run_directory, file_name='configuration.txt',
               wandb=wandb)

    # Plotting
    # handler.plot_iterations(delta_list, label="Delta")
    c_local_array = np.array(c_local_list)
    for i in range(num_user):
        handler.plot_iterations(c_local_array[:, i, :], label=f"Electricity Price of user {i}")
        # handler.plot_local_price(c_local_array[:, i, :], z[str(k)])
    handler.plot_iterations(z_list, label="Aggregator z of electricity demand")

    battery_profile = {i: [] for i in range(num_user)}
    trading_profile = {i: [] for i in range(num_user)}

    for i in range(num_user):
        agent_model = model_list[f'User{i}']
        for t in range(num_timestep + 1):
            battery_profile[i].append(agent_model.SoC[f't{t}'].X)
            if not t == num_timestep:
                trading_profile[i].append(agent_model.p_d[f't{t}'].X)

    handler.plot_user_profile(battery_profile, type="Battery", num_user=num_user, num_timestep=num_timestep)
    handler.plot_user_profile(trading_profile, type='Demand', num_user=num_user, num_timestep=num_timestep)
    handler.plot_expense_diff(expense)

    save_dict(battery_profile, run_directory, name='final_battery_profile.txt', wandb=wandb)
    save_dict(soc, run_directory, name='last_iteration_SoC.txt', wandb=wandb)
    save_dict(p_d, run_directory, name='last_iteration_p.txt', wandb=wandb)

    wandb.finish()


if __name__ == '__main__':
    main()