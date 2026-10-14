# The Nature 2022 recipe and what this task keeps from it

Degrave et al., *Magnetic control of tokamak plasmas through deep reinforcement
learning*, Nature 602 (2022); Tracey et al., *Towards practical reinforcement
learning for tokamak magnetic control*, arXiv:2307.11546 (2023). TCV, simulator FGE
(closed source, MEQ suite).

| Ingredient | Nature 2022 / Tracey 2023 | This task |
| --- | --- | --- |
| Plant | FGE: free-boundary GS + circuit equations + vessel, TCV description | FreeGSNKE: same physics class, MAST-U-like description |
| Actions | 19 coil voltage commands, 10 kHz | 12 coil voltages, 2 kHz (0.5 ms solver step) |
| Observations | 34 flux loops, 38 probes, 19 coil currents + target references | 102 flux loops, 398 pickups, 12 currents, Ip + target points/X-points/Ip |
| Algorithm | MPO, asymmetric actor (small MLP, real-time) / critic (recurrent, privileged) | starter: SAC with privileged critic; MPO/recurrent critic are natural follow-ups |
| Reward | weighted SmoothMax of per-objective SoftPlus terms with good/bad thresholds (shape points, X-points, Ip, coil limits) | starter reward: linear penalties on shape RMS, X-point, Ip, coil limits, voltage effort; termination penalty (training signal only) |
| Robustness | randomized plasma resistivity Rp, βp and qA (log-uniform, ranges identified from TCV data) and a per-episode power-supply offset; sensor delay/noise and supply bias/delay are fixed values identified from data | randomized profile parameters, plasma resistivity, passive resistance, initial currents, probe offsets/noise, supply delay, handover pre-roll, with ranges from FAIR-MAST (MAST) data; the sealed plant is wider and adds time-varying plasma, supply non-idealities, sensor faults and modelling differences |
| Training | 5,000 parallel actors (one FGE simulator each), training times of 1-3 days per task | 32-64 FreeGSNKE workers, ~4 min per 700-step episode per worker |
| Speed-ups (2023) | episode chunking with task-specific starts (3x), two-stage reward thresholds (65% shape accuracy), integrator feedback for Ip bias, warm-starting related tasks | not in the starter; all are legal research directions |
| Evaluation | real TCV discharges; shape RMSE and Ip error from LIUQE reconstructions against a "reasonable aim" of 2 cm and 3%; terminations enforce coil-current and q95 limits | sealed perturbed simulator; success fraction under MAST-U's 1 cm steady-state requirement and 3% Ip; coil-current limits terminate; PID/virtual-circuit reference |

What the task deliberately changes: one policy across all cases (Nature trained
one per configuration), sealed perturbation evaluation (Nature's judge was the real
device), open simulator and machine description, and a delivery contract that keeps
the actor deployable (NumPy/ONNX, no torch, no privileged inputs, 0.5 ms median
computation per step). Nature's main ablation is relevant to the starter: "the main
benefit comes from the recurrent design in the critic", and the starter's SAC critic is
feed-forward.
