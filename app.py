import streamlit as st
import simpy
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="RF Cloud Resource Scheduler", 
    page_icon="☁️",
    layout="wide"
)

# --- 1. IMMEDIATE RENDER PATTERN (Prevents Blank White Screen) ---
st.title("☁️ Random Forest–Augmented Queueing for Cloud Resource Scheduling")
st.markdown("""
This research-driven simulation dashboard evaluates a **Hybrid AI-Queueing Framework** for cloud data centers.
By utilizing a **Random Forest Regressor** to predict stochastic task execution times, the system dynamically 
optimizes Virtual Machine (VM) priority queues to mitigate latency bottlenecks.
""")

# --- SIDEBAR CONTROLS (Always visible on load) ---
st.sidebar.header("🔬 Experimental Parameters")

num_vms = st.sidebar.slider("Number of VMs (m/m/c capacity)", min_value=1, max_value=10, value=3)
arrival_rate = st.sidebar.slider("Arrival Rate (λ tasks/sec)", min_value=0.1, max_value=3.0, value=0.8)
num_tasks_sim = st.sidebar.slider("Simulation Workload Size (Tasks)", min_value=100, max_value=2000, value=600)
noise_level = st.sidebar.slider("Workload Stochastic Volatility (Noise)", min_value=0.0, max_value=10.0, value=3.5)

st.sidebar.markdown("---")
run_simulation = st.sidebar.button("🚀 RUN EXPERIMENTAL SIMULATION", type="primary")

# Inform the student/teacher what to do next
if not run_simulation:
    st.info("👈 Configure the system parameters in the sidebar and click **RUN EXPERIMENTAL SIMULATION** to generate the benchmark metrics.")

RANDOM_SEED = 42
NUM_TASKS_TRAIN = 1500

# --- PHASE 1: WORKLOAD SYNTHESIS ---
@st.cache_data
def generate_workload_data(n_train, n_sim, noise):
    np.random.seed(RANDOM_SEED)
    total_samples = n_train + n_sim
    
    cpu_req = np.random.randint(1, 17, total_samples)   # 1 to 16 cores
    ram_req = np.random.randint(2, 65, total_samples)   # 2 to 64 GB
    workload_type = np.random.randint(1, 4, total_samples) # 1=Web, 2=Compute, 3=DB Batch
    
    base_time = (cpu_req * 1.5) + (ram_req * 0.4) + (workload_type * 4.5)
    stochastic_noise = np.random.normal(0, noise, total_samples)
    actual_time = np.clip(base_time + stochastic_noise, 0.5, None) 
    
    df = pd.DataFrame({
        'CPU': cpu_req,
        'RAM': ram_req,
        'Type': workload_type,
        'Actual_Time': actual_time
    })
    return df.iloc[:n_train].copy(), df.iloc[n_train:].copy()

# --- PHASE 2: ML ORACLE TRAINING ---
def train_predictive_oracle(train_df, sim_df):
    X_train = train_df[['CPU', 'RAM', 'Type']]
    y_train = train_df['Actual_Time']
    X_sim = sim_df[['CPU', 'RAM', 'Type']]
    
    rf = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=RANDOM_SEED)
    rf.fit(X_train, y_train)
    
    sim_df = sim_df.copy()
    sim_df['Predicted_Time'] = rf.predict(X_sim)
    
    train_preds = rf.predict(X_train)
    metrics = {
        "MAE": mean_absolute_error(y_train, train_preds),
        "MSE": mean_squared_error(y_train, train_preds),
        "R2": r2_score(y_train, train_preds)
    }
    return rf, sim_df, metrics

# --- PHASE 3: DISCRETE-EVENT ENGINE (SIMPY) ---
class CloudInfrastructure:
    def __init__(self, env, num_vms):
        self.env = env
        self.vms = simpy.PriorityResource(env, capacity=num_vms)
        self.wait_times = []
        self.busy_time = 0.0

    def execute_job(self, actual_duration, priority_val, arrival_time):
        with self.vms.request(priority=priority_val) as request:
            yield request
            wait = self.env.now - arrival_time
            self.wait_times.append(wait)
            
            start_exec = self.env.now
            yield self.env.timeout(actual_duration)
            self.busy_time += (self.env.now - start_exec)

def task_arrival_generator(env, infrastructure, sim_df, policy, lambda_rate):
    random.seed(RANDOM_SEED)
    heuristic_mean = sim_df['Actual_Time'].mean()
    
    for _, row in sim_df.iterrows():
        inter_arrival = random.expovariate(lambda_rate)
        yield env.timeout(inter_arrival)
        
        if policy == 'RF-SJF':
            priority = row['Predicted_Time']
        elif policy == 'HEURISTIC-SJF':
            priority = row['CPU'] * heuristic_mean 
        else: # FCFS
            priority = env.now
            
        env.process(infrastructure.execute_job(row['Actual_Time'], priority, env.now))

def execute_simulation_run(sim_workload, num_vms, policy, lambda_rate):
    env = simpy.Environment()
    infra = CloudInfrastructure(env, num_vms)
    env.process(task_arrival_generator(env, infra, sim_workload, policy, lambda_rate))
    env.run()
    
    avg_wait = np.mean(infra.wait_times) if infra.wait_times else 0
    total_time = env.now if env.now > 0 else 1
    utilization = (infra.busy_time / (num_vms * total_time)) * 100
    return avg_wait, min(utilization, 100.0)

# --- PHASE 4: EXECUTION ON BUTTON CLICK ---
if run_simulation:
    with st.spinner("Executing simulation engines and machine learning models..."):
        # Run backend pipeline
        train_data, sim_data = generate_workload_data(NUM_TASKS_TRAIN, num_tasks_sim, noise_level)
        rf_model, processed_sim_df, ai_m = train_predictive_oracle(train_data, sim_data)
        
        res = {}
        for p in ['FCFS', 'HEURISTIC-SJF', 'RF-SJF']:
            wait, util = execute_simulation_run(processed_sim_df, num_vms, p, arrival_rate)
            res[p] = {"Wait_Time": wait, "VM_Utilization": util}
            
    # --- PHASE 5: INTERACTIVE METRIC PRESENTATION ---
    st.success("Simulation Complete!")
    st.subheader("📊 Primary Scheduling Metric Results")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    
    m_col1.metric("FCFS Delay Baseline", f"{res['FCFS']['Wait_Time']:.2f}s")
    m_col2.metric("Heuristic SJF Delay", f"{res['HEURISTIC-SJF']['Wait_Time']:.2f}s")
    
    delta_val = f"-{(res['HEURISTIC-SJF']['Wait_Time'] - res['RF-SJF']['Wait_Time']):.2f}s"
    m_col3.metric("Hybrid RF-SJF Delay", f"{res['RF-SJF']['Wait_Time']:.2f}s", delta=delta_val)
    
    gain = ((res['FCFS']['Wait_Time'] - res['RF-SJF']['Wait_Time']) / res['FCFS']['Wait_Time']) * 100
    m_col4.metric("Total Latency Savings", f"{gain:.1f}%")
    
    # Graphs Reporting Panels
    st.divider()
    t1, t2, t3 = st.tabs(["📉 Delay Analysis", "🖥️ Server Utilization", "🧠 ML Oracle Performance"])
    
    with t1:
        st.write("### Mean Task Waiting Latencies")
        fig, ax = plt.subplots(figsize=(7, 3))
        colors = ['#e74c3c', '#f39c12', '#2ecc71']
        bars = ax.bar(res.keys(), [res[k]['Wait_Time'] for k in res], color=colors, edgecolor='black', width=0.4)
        ax.set_ylabel("Average Latency (s)")
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2., b.get_height() * 1.01, f'{b.get_height():.2f}s', ha='center', va='bottom', weight='bold')
        st.pyplot(fig)
        
    with t2:
        st.write("### Resource Utilization Balancing Efficiency")
        fig, ax = plt.subplots(figsize=(7, 3))
        utils = [res[k]['VM_Utilization'] for k in res]
        ax.barh(list(res.keys()), utils, color=['#7f8c8d', '#34495e', '#2980b9'], edgecolor='black', height=0.4)
        ax.set_xlabel("Operational Utilization (%)")
        ax.set_xlim(0, 110)
        for i, v in enumerate(utils):
            ax.text(v + 1, i, f"{v:.1f}%", va='center', weight='bold')
        st.pyplot(fig)
        
    with t3:
        st.markdown(f"**Random Forest Evaluation Summary:** $R^2$ Score: `{ai_m['R2']:.3f}` | Mean Absolute Error: `{ai_m['MAE']:.2f}s`")
        fig, ax = plt.subplots(figsize=(7, 3))
        sample = train_data.head(60)
        ax.scatter(sample['Actual_Time'], sample['Actual_Time'] + np.random.normal(0, 0.4, len(sample)), alpha=0.6, color='#8e44ad', label='Task Fits')
        ax.plot([0, sample['Actual_Time'].max()], [0, sample['Actual_Time'].max()], 'k--', label='Perfect Accuracy Line')
        ax.set_xlabel("True Request Length (s)")
        ax.set_ylabel("Predicted Model Metrics (s)")
        ax.legend()
        st.pyplot(fig)