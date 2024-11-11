import json
import numpy as np
from set_training_clients import sample_clients, get_client_ids, parse_arguments, load_config

def main():
    args = parse_arguments()
    folder_path = args.dir
    config = load_config()

    client_ids = get_client_ids(f"{folder_path}/trainpt")
    training_sample = sample_clients(client_ids, 100, 100)

    if isinstance(training_sample, np.ndarray):
        training_sample = training_sample.tolist()
    training_samples = {i: training_sample for i in range(100)}

    output_file = f"{folder_path}/training_samples_ctrl_experiment.json"

    with open(output_file, 'w') as f:
        json.dump(training_samples, f, indent=4)

if __name__ == "__main__":
    main()