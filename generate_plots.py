import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_tb_data(log_dir, tags):
    """Extract scalar data for given tags from all tfevents files recursively in log_dir."""
    event_files = glob.glob(os.path.join(log_dir, "**", "events.out.tfevents.*"), recursive=True)
    
    data = {tag: [] for tag in tags}
    
    for e_file in event_files:
        run_name = os.path.basename(os.path.dirname(e_file))
        # Keep it distinct for different runs
        try:
            ea = EventAccumulator(e_file, size_guidance={'scalars': 0})
            ea.Reload()
            
            for tag in tags:
                if tag in ea.Tags()['scalars']:
                    events = ea.Scalars(tag)
                    df = pd.DataFrame([(e.step, e.value) for e in events], columns=['step', 'value'])
                    df['run'] = run_name
                    data[tag].append(df)
        except Exception as e:
            print(f"Error reading {e_file}: {e}")
            
    # Concat all dataframes per tag
    for tag in tags:
        if data[tag]:
            data[tag] = pd.concat(data[tag], ignore_index=True)
        else:
            data[tag] = pd.DataFrame(columns=['step', 'value', 'run'])
            
    return data

def plot_curves(tag, algorithms_data, title, ylabel, filename):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Training Steps")
    plt.ylabel(ylabel)
    
    has_data = False
    for algo, data_dict in algorithms_data.items():
        if tag in data_dict and not data_dict[tag].empty:
            df = data_dict[tag]
            # Plot individual runs
            for run_name, run_df in df.groupby('run'):
                plt.plot(run_df['step'], run_df['value'], label=f"{algo} ({run_name})", alpha=0.5)
            
            # Alternative: plot mean, but let's stick to individual for hyperparam variation visibility
            has_data = True

    if has_data:
        # Simplify legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        # Only show up to 15 items in legend to avoid cluttering
        if len(by_label) > 15:
            subset = {k: by_label[k] for k in list(by_label.keys())[:15]}
            plt.legend(subset.values(), subset.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')
            
        plt.tight_layout()
        plt.grid(True, alpha=0.3)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
    else:
        print(f"No data found for {tag}, skipping {filename}")
    plt.close()

def main():
    base_log_dir = "logs"
    if not os.path.exists(base_log_dir):
        print("logs directory not found!")
        return
        
    os.makedirs("plots", exist_ok=True)
    
    # We will look for dqn, ppo, reinforce, improved_ppo_v2 etc inside logs/ and logs/improved/
    algorithms = {}
    
    # Scan standard logs
    for d in os.listdir(base_log_dir):
        path = os.path.join(base_log_dir, d)
        if os.path.isdir(path) and d != "improved":
            algorithms[f"standard_{d}"] = path
            
    # Scan improved logs
    imp_dir = os.path.join(base_log_dir, "improved")
    if os.path.exists(imp_dir):
        for d in os.listdir(imp_dir):
            path = os.path.join(imp_dir, d)
            if os.path.isdir(path):
                algorithms[f"improved_{d}"] = path

    print(f"Found log directories: {list(algorithms.keys())}")
    
    tags = ['rollout/ep_rew_mean', 'train/loss', 'train/entropy_loss']
    
    # Extract data for all algorithms
    extracted_data = {}
    for algo, path in algorithms.items():
        print(f"Extracting data for {algo}...")
        extracted_data[algo] = extract_tb_data(path, tags)
        
    # Plot Cumulative Reward Curves
    plot_curves(
        'rollout/ep_rew_mean', 
        extracted_data, 
        "Cumulative Reward Curves for Different Hyperparameter Runs", 
        "Mean Reward", 
        "plots/cumulative_reward_curves.png"
    )
    
    # Plot DQN Objective Loss
    dqn_data = {k: v for k, v in extracted_data.items() if 'dqn' in k.lower()}
    plot_curves(
        'train/loss', 
        dqn_data, 
        "DQN Training Loss (Objective Curves)", 
        "Loss", 
        "plots/dqn_objective_curves.png"
    )
    
    # Plot PG Entropy
    pg_data = {k: v for k, v in extracted_data.items() if 'dqn' not in k.lower()}
    plot_curves(
        'train/entropy_loss', 
        pg_data, 
        "Policy Gradient Entropy Curves", 
        "Entropy Loss", 
        "plots/pg_entropy_curves.png"
    )

    print("Finished generating plots in the 'plots/' directory.")

if __name__ == "__main__":
    main()
