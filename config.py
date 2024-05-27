import datetime
import os


def initialize_run_directory(params: dict):
    current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")
    config = f"SoCMax={params['SoC_max']}-SoCDiff={params['SoC_diff']}-PVSize={params['pv_size']}"
    path = current_time + '-' + config
    project_directory = '/Users/fxy/Game-theory-for-energy-communities/runs'  # Adjust this to your project directory
    run_directory = os.path.join(project_directory, path)
    os.makedirs(run_directory, exist_ok=True)

    os.environ['RUN_DIRECTORY'] = run_directory

    return run_directory
