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

def plot_episodes_converge(algorithms_data_paths, filename):
    plt.figure(figsize=(10, 6))
    plt.title("Episodes to Converge (Reward vs Episodes)")
    plt.xlabel("Training Episodes")
    plt.ylabel("Mean Reward")
    
    has_data = False
    for algo, log_dir in algorithms_data_paths.items():
        event_files = glob.glob(os.path.join(log_dir, "**", "events.out.tfevents.*"), recursive=True)
        for e_file in event_files:
            run_name = os.path.basename(os.path.dirname(e_file))
            try:
                ea = EventAccumulator(e_file, size_guidance={'scalars': 0})
                ea.Reload()
                if 'rollout/ep_rew_mean' in ea.Tags()['scalars']:
                    ev_y = ea.Scalars('rollout/ep_rew_mean')
                    df = pd.DataFrame([(e.step, e.value) for e in ev_y], columns=['step', 'reward'])
                    
                    # Try to get episodes directly
                    if 'time/episodes' in ea.Tags()['scalars']:
                        ev_ep = ea.Scalars('time/episodes')
                        df_ep = pd.DataFrame([(e.step, e.value) for e in ev_ep], columns=['step', 'episodes'])
                        df = pd.merge(df, df_ep, on='step', how='outer').sort_values('step').interpolate().dropna()
                    elif 'rollout/ep_len_mean' in ea.Tags()['scalars']:
                        ev_len = ea.Scalars('rollout/ep_len_mean')
                        df_len = pd.DataFrame([(e.step, e.value) for e in ev_len], columns=['step', 'ep_len'])
                        df = pd.merge(df, df_len, on='step', how='inner')
                        # Approximate total episodes = total steps / current mean episode length
                        df['episodes'] = df['step'] / (df['ep_len'] + 1e-8)
                    else:
                        df['episodes'] = df['step'] / 200.0  # fallback constant

                    if not df.empty:
                        plt.plot(df['episodes'], df['reward'], label=f"{algo} ({run_name})", alpha=0.5)
                        has_data = True
            except Exception as e:
                pass

    if has_data:
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
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
        print(f"Could not merge episodes/reward data, skipping {filename}")
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

    # Plot Episodes to Converge
    print("Generating Episodes to Converge plot...")
    plot_episodes_converge(algorithms, "plots/episodes_to_converge.png")

    print("Finished generating plots in the 'plots/' directory.")

if __name__ == "__main__":
    main()
