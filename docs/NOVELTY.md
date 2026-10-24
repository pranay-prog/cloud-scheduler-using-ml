# Novelty status

This repository implements an experimental combination of established ideas:
Random Forest runtime regression, a global residual-quantile runtime offset,
non-preemptive deadline and shortest-job queue priorities, and a SimPy cloud
queueing simulator. The implementation has not been compared against the full
research literature or reproduced a published evaluation. No novelty claim is
made.

The residual offset is a simple empirical upper estimate, not a formal coverage
guarantee. The scheduler policy should be evaluated against stronger
feature-conditional uncertainty methods and additional published baselines
before making research contribution claims.
