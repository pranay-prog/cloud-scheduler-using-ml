import streamlit as st
import simpy
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Cloud Scheduling AI", layout="wide")

st.title("☁️ RF-Augmented Cloud Scheduling")
st.markdown("""
This dashboard simulates a **Cloud Data Center**. It compares a standard **FCFS Scheduler** against a **Random Forest-Augmented Scheduler** (RF-SJF) that predicts task duration to optimize the queue.
""")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Simulation Parameters")

num_vms = st.sidebar.slider("Number of VMs (Servers)", min_value=1, max_value=10, value=2)
arrival_rate = st.sidebar.slider("Arrival Rate (Tasks/sec)", 0.1, 2.0, 0.5)
num_tasks_sim = st.sidebar.slider("Number of Tasks to Simulate", 100, 2000, 500)
noise_level = st.sidebar.slider("Stochastic Noise Level", 0.0, 5.0, 2.0, help="Higher noise = Harder to predict")

st.sidebar.markdown("---")
st.sidebar.info("Click **RUN SIMULATION** to start.")

# --- CORE LOGIC (Cached for performance) ---

@st.cache_data
def generate_data(n_train, n_sim, noise):
    """Generates synthetic cloud workload data"""
    np.random.seed(42)
    data = []
    total = n_train + n_sim
    for i in range(total):
        cpu = np.random.randint(1, 9)
        ram = np.random.randint(1, 32)
        user = np.random.randint(0, 3)
        
        # Base formula + Noise
        base_time = (cpu * 2) + (ram * 0.5) + (user * 5)
        stochastic_noise = np.random.normal(0, noise)
        actual_time = max(1, base_time + stochastic_noise)
        
        data.append([cpu, ram, user, actual_time])
        
    df = pd.DataFrame(data, columns=['CPU', 'RAM', 'User', 'Actual_Time'])
    return df.iloc[:n_train], df.iloc[n_train:]

# --- SIMULATION ENGINE ---

class CloudDatacenter:
    def __init__(self, env, num_vms, policy, rf_model=None):
        self.env = env
        self.servers = simpy.PriorityResource(env, capacity=num_vms)
        self.policy = policy
        self.rf_model = rf_model
        self.wait_times = []

    def process_task(self, features, actual_duration, arrival_time):
        # Prediction Step
        if self.policy == 'RF-SJF':
            # Use RF to guess duration. Priority = Predicted Time (SJF logic)
            prediction = self.rf_model.predict([features])[0]
            priority = prediction
        else:
            # FCFS: Priority = Arrival Time
            priority = arrival_time

        with self.servers.request(priority=priority) as request:
            yield request
            wait = self.env.now - arrival_time
            self.wait_times.append(wait)
            yield self.env.timeout(actual_duration)

def run_simpy_env(df, num_vms, policy, rf_model=None):
    env = simpy.Environment()
    dc = CloudDatacenter(env, num_vms, policy, rf_model)
    
    # Define the generator process inside
    def task_generator():
        for index, row in df.iterrows():
            inter_arrival = random.expovariate(arrival_rate)
            yield env.timeout(inter_arrival)
            env.process(dc.process_task(row[['CPU', 'RAM', 'User']].values, row['Actual_Time'], env.now))
            
    env.process(task_generator())
    env.run()
    return np.mean(dc.wait_times)

# --- MAIN APP EXECUTION ---

if st.button("RUN SIMULATION", type="primary"):
    
    # 1. Data Prep
    with st.spinner('Generating stochastic workload...'):
        train_df, sim_df = generate_data(1000, num_tasks_sim, noise_level)
    
    # 2. Train Model
    X_train = train_df[['CPU', 'RAM', 'User']]
    y_train = train_df['Actual_Time']
    
    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    rf.fit(X_train, y_train)
    
    # Calculate Accuracy
    preds = rf.predict(X_train)
    mae = mean_absolute_error(y_train, preds)
    
    # 3. Run Simulations
    with st.spinner('Running SimPy environments...'):
        wait_fcfs = run_simpy_env(sim_df, num_vms, "FCFS")
        wait_rf = run_simpy_env(sim_df, num_vms, "RF-SJF", rf)
    
    # 4. Display Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("FCFS Wait Time", f"{wait_fcfs:.2f} s")
    col2.metric("RF-SJF Wait Time", f"{wait_rf:.2f} s", delta=f"-{(wait_fcfs-wait_rf):.2f} s")
    improvement = ((wait_fcfs - wait_rf) / wait_fcfs) * 100
    col3.metric("Improvement", f"{improvement:.1f}%")

    # 5. Visualizations
    st.divider()
    
    tab1, tab2 = st.tabs(["📊 Performance Comparison", "🤖 AI Accuracy"])
    
    with tab1:
        st.subheader("Scheduling Efficiency")
        fig, ax = plt.subplots(figsize=(10, 4))
        bars = ax.bar(['FCFS (Standard)', 'RF-Augmented (Ours)'], [wait_fcfs, wait_rf], color=['#ff9999', '#66b3ff'], edgecolor='black')
        ax.set_ylabel("Avg Wait Time (s)")
        
        # Add labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}s', ha='center', va='bottom')
            
        st.pyplot(fig)
        
    with tab2:
        st.subheader("Random Forest Prediction Analysis")
        st.write(f"**Model Mean Absolute Error:** {mae:.2f} seconds")
        
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        # Plot a subset of 50 points
        subset_y = y_train[:50]
        subset_pred = preds[:50]
        
        ax2.scatter(subset_y, subset_pred, alpha=0.7, color='purple')
        ax2.plot([0, max(subset_y)], [0, max(subset_y)], 'k--', label='Perfect Prediction')
        ax2.set_xlabel("Actual Time")
        ax2.set_ylabel("Predicted Time")
        ax2.legend()
        st.pyplot(fig2)

    # 6. Raw Data View
    with st.expander("View Simulation Data"):
        st.dataframe(sim_df.head(10))