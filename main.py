# pyrefly: ignore [missing-import]
import simpy
import random
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# --- CONFIGURATION ---
NUM_TASKS_TRAIN = 1000   # Data to train the RF
NUM_TASKS_SIM = 500      # Tasks to run in the simulation
NUM_VMS = 2              # Number of Cloud Servers (k)
ARRIVAL_RATE = 0.5       # Tasks per second (Lambda) - Poisson process
RANDOM_SEED = 42

# --- PHASE 1: DATA GENERATION (Stochastic Modeling) ---
def generate_synthetic_data(num_samples):
    """
    Creates a synthetic dataset representing cloud jobs.
    Features: CPU cores, RAM, UserType
    Target: Actual Execution Time (with noise)
    """
    np.random.seed(RANDOM_SEED)
    data = []
    for i in range(num_samples):
        # Features
        cpu_req = np.random.randint(1, 9)     # 1 to 8 cores
        ram_req = np.random.randint(1, 32)    # 1 to 32 GB
        user_type = np.random.randint(0, 3)   # 0=Student, 1=Dev, 2=Admin
        
        # The "Real" Formula (Unknown to the scheduler)
        # More CPU = Faster? No, usually bigger jobs request more CPU.
        base_time = (cpu_req * 2) + (ram_req * 0.5) + (user_type * 5)
        
        # Add Stochastic Noise (The "Uncertainty" factor)
        noise = np.random.normal(0, 2) 
        actual_time = max(1, base_time + noise) # Time can't be negative
        
        data.append([i, cpu_req, ram_req, user_type, actual_time])
        
    return pd.DataFrame(data, columns=['Job_ID', 'CPU', 'RAM', 'User', 'Actual_Time'])

# Generate Data
print(">>> Generating Synthetic Workload...")
df = generate_synthetic_data(NUM_TASKS_TRAIN + NUM_TASKS_SIM)
train_data = df.iloc[:NUM_TASKS_TRAIN]
sim_data = df.iloc[NUM_TASKS_TRAIN:]

# --- PHASE 2: THE ORACLE (Random Forest) ---
print(">>> Training Random Forest Predictor...")
X = train_data[['CPU', 'RAM', 'User']]
y = train_data['Actual_Time']

rf_model = RandomForestRegressor(n_estimators=50, random_state=RANDOM_SEED)
rf_model.fit(X, y)

# Evaluate accuracy just to be sure
preds = rf_model.predict(X)
print(f"   Training-set MAE: {mean_absolute_error(y, preds):.2f} seconds (not held-out accuracy)")

# --- PHASE 3: THE SIMULATION (SimPy) ---

class CloudDatacenter:
    def __init__(self, env, num_vms, policy):
        self.env = env
        # PriorityResource sorts queue automatically!
        # If policy is SJF, we use predicted time as priority.
        # If policy is FIFO, we use arrival time as priority.
        self.servers = simpy.PriorityResource(env, capacity=num_vms)
        self.policy = policy
        self.wait_times = []

    def process_task(self, task_features, actual_duration, arrival_time):
        
        # 1. PREDICTION STEP
        predicted_duration = rf_model.predict([task_features])[0]
        
        # 2. SCHEDULING POLICY DECISION
        if self.policy == 'RF-SJF':
            # SJF: Priority = Shortest Prediction (Lower number = Higher Priority)
            priority = predicted_duration
        else: # FCFS / FIFO
            # FIFO: Priority = Arrival Time (Earlier time = Higher Priority)
            priority = arrival_time

        # 3. QUEUEING
        request = self.servers.request(priority=priority)
        yield request # Wait here until a VM is free

        # 4. EXECUTION
        wait = self.env.now - arrival_time
        self.wait_times.append(wait)
        
        # Simulate processing time
        yield self.env.timeout(actual_duration)

def task_generator(env, datacenter, task_list):
    """Generates tasks based on a Poisson Process"""
    for index, row in task_list.iterrows():
        # Inter-arrival time (Poisson process)
        inter_arrival = random.expovariate(ARRIVAL_RATE)
        yield env.timeout(inter_arrival)
        
        # Create a process for the task
        features = row[['CPU', 'RAM', 'User']].values
        actual_time = row['Actual_Time']
        env.process(datacenter.process_task(features, actual_time, env.now))

# --- PHASE 4: RUNNING THE EXPERIMENT ---

def run_simulation(policy_name):
    env = simpy.Environment()
    dc = CloudDatacenter(env, NUM_VMS, policy=policy_name)
    
    # Start the generator
    env.process(task_generator(env, dc, sim_data))
    
    # Run simulation
    env.run()
    
    avg_wait = np.mean(dc.wait_times)
    print(f"[{policy_name}] Average Wait Time: {avg_wait:.4f} seconds")
    return avg_wait

print("\n>>> STARTING BENCHMARK <<<")
wait_fcfs = run_simulation("FCFS")
wait_rf = run_simulation("RF-SJF")

# --- RESULTS ---
print("\n>>> Legacy demo values are illustrative only: runs do not share paired arrivals or test data.")

import matplotlib.pyplot as plt

# --- PHASE 5: VISUALIZATION ---

def plot_results(wait_fcfs, wait_rf, rf_model, X, y):
    """
    Generates two charts:
    1. Performance Comparison (The 'Money' Shot)
    2. Prediction Accuracy (The 'Proof' of AI)
    """
    
    # Setup the figure with 2 subplots side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # CHART 1: Bar Graph of Wait Times
    strategies = ['Legacy FCFS', 'Legacy RF-SJF']
    times = [wait_fcfs, wait_rf]
    colors = ['#ff9999', '#66b3ff'] # Red for slow, Blue for fast
    
    ax1.bar(strategies, times, color=colors, edgecolor='black', alpha=0.7)
    ax1.set_ylabel('Average Wait Time (seconds)')
    ax1.set_title('Illustrative Legacy Run Values (Not Paired Evidence)')
    
    # Add text labels on top of bars
    for i, v in enumerate(times):
        ax1.text(i, v + 0.1, f"{v:.2f}s", ha='center', fontweight='bold')

    # CHART 2: Scatter Plot of RF Accuracy
    # We take a sample of 50 points to keep the chart clean
    sample_size = 50
    preds = rf_model.predict(X[:sample_size])
    actuals = y[:sample_size]
    
    ax2.scatter(actuals, preds, color='purple', alpha=0.6, label='Task Predictions')
    
    # Draw the "Perfect Prediction" line (y=x)
    max_val = max(max(actuals), max(preds))
    ax2.plot([0, max_val], [0, max_val], 'k--', lw=2, label='Perfect Accuracy')
    
    ax2.set_xlabel('Actual Execution Time')
    ax2.set_ylabel('Predicted Execution Time')
    ax2.set_title('Random Forest Training-Set Fit (Not Test Accuracy)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

# --- EXECUTE PLOT ---
print(">>> Generating Charts...")
plot_results(wait_fcfs, wait_rf, rf_model, X, y)
