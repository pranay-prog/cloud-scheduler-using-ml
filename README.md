https://cloud-scheduler-using-ml-gkjv7qkr4zj9fpvauz6x2p.streamlit.app/   this is the link to access the website
Title: Random Forest–Augmented Queueing for Cloud Resource Scheduling


1. The "Stochastic" Part (generate_synthetic_data): In real life, you don't know exactly how long a task takes. Here, we simulate that reality. We calculate a base_time based on CPU/RAM, but then we add noise. This noise is critical—it's why simple math fails and why you need AI.

2. The "Priority" Hack (simpy.PriorityResource): This is the smartest part of the simulation.

SimPy has a built-in resource manager.

By setting priority=predicted_duration, the simulation automatically shuffles the queue so the shortest jobs go first.

By setting priority=arrival_time, it acts as a standard First-Come-First-Serve queue.

3. The Comparison: When you run this, you will see two numbers.

FCFS: Standard queue. Long jobs will clog the VMs, making small jobs wait.

RF-SJF: Your model. It predicts the small jobs and sneaks them in front, drastically lowering the average wait time.
Efficient resource allocation in cloud data centers is critical for meeting Quality of Service (QoS) requirements under stochastic workload conditions. Traditional scheduling algorithms, such as First-Come-First-Served (FCFS) and Round Robin (RR), often fail to optimize task waiting times due to their inability to account for variability in task execution durations. While Shortest Job First (SJF) theoretically minimizes waiting time, its practical implementation is hindered by the unavailability of a priori knowledge regarding task execution times. To address this limitation, this study proposes a hybrid Random Forest–Augmented Queueing (RF-AQ) framework. We model the cloud data center as a $G/G/k$ queueing system and integrate a Random Forest regressor to predict task execution times based on resource request features (CPU, RAM, and User ID). These predictions drive a dynamic dispatching policy that approximates the optimal SJF schedule. The proposed model is evaluated using a Discrete Event Simulation (DES) built with SimPy and scikit-learn. Experimental results demonstrate that the RF-AQ approach significantly outperforms baseline FCFS strategies, achieving a 73.2% reduction in average waiting time while maintaining high throughput under high-load stochastic scenarios. This work bridges the gap between predictive machine learning and classical operations research, offering a robust solution for real-time cloud resource management.
