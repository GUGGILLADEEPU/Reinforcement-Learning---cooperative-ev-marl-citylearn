# Cooperative Multi-Agent Reinforcement Learning for EV Charging Coordination

A reinforcement learning project that investigates cooperative electric vehicle (EV) charging coordination using a shared-parameter Deep Q-Network (DQN) in the CityLearn simulation environment.

The project evaluates how coordinated charging decisions affect electricity grid demand while examining the trade-off between grid performance and EV charging service.

## Project Overview

Uncoordinated EV charging can increase grid electricity import and peak demand. This project implements and evaluates a cooperative EV-level charging strategy that uses local vehicle information and shared energy-system features to select charging and discharging actions.

The framework is evaluated against four alternative strategies to understand its strengths and limitations.

## Objectives

- Develop a shared-parameter DQN framework for EV charging coordination.
- Represent eight EV charging slots using a 22-feature state representation.
- Select among five discrete charging and discharging actions.
- Compare five charging strategies under a common simulation setup.
- Evaluate grid demand alongside EV charging outcomes.
- Analyse the trade-off between reducing grid demand and maintaining charging service.

## System Architecture

The workflow consists of five main stages:

1. **Simulation Environment:** Configure CityLearn and the selected EV-enabled dataset.
2. **State Construction:** Build a 22-feature state for each EV slot.
3. **Shared DQN Learning:** Estimate action values using a common neural network.
4. **Environment Interaction and Training:** Apply actions, collect rewards, store experiences, and update the network.
5. **Evaluation:** Compare charging strategies across five random seeds.

### Workflow

`Observe State → Select Actions → Apply Actions → Simulate Transition → Compute Reward → Store Experience → Update DQN → Evaluate Strategies`

## Environment and Configuration

- **Simulator:** CityLearn 3.0.2
- **Dataset:** `citylearn_challenge_2022_phase_all_plus_evs`
- **Building-level observation/action spaces:** 17
- **EV charging slots:** 8 across 7 buildings
- **State features per EV:** 22
- **State batch shape:** `(8, 22)`
- **DQN architecture:** `22 → 128 → 128 → 64 → 5`
- **Trainable parameters:** 28,037
- **Action space:** Five discrete actions
- **Evaluation seeds:** `0, 42, 123, 777, 2026`
- **Evaluation horizon:** 23 transitions per episode

## DQN Architecture and Action Space

The shared DQN contains three hidden layers with 128, 128, and 64 units. Its output layer estimates the values of five possible actions.

| Action | Meaning |
|---:|---|
| `-1.0` | Strong discharging |
| `-0.5` | Moderate discharging |
| `0` | Idle |
| `+0.5` | Moderate charging |
| `+1.0` | Strong charging |

Negative values represent discharging, positive values represent charging, and zero represents idle. These are discrete action labels, not power values in kilowatts. Disconnected EVs are restricted to the idle action.

The same DQN parameters are shared across the eight EV slots. This provides shared-parameter EV-level coordination; separate station-level and grid-level learning agents are potential future extensions.

## Charging Strategies Compared

1. **Uncontrolled Charging**
2. **Rule-Based Smart Charging**
3. **Independent DQN**
4. **Cooperative MARL-DQN**
5. **Service-Aware Cooperative MARL-DQN**

## Evaluation Metrics

The project evaluates performance using:

- **Grid import (kWh):** Total electricity imported from the grid.
- **Peak load (kW):** Maximum observed grid demand.
- **Peak-to-average ratio (PAR):** Maximum load divided by average load.
- **Charging energy (kWh):** Energy delivered to EVs.
- **Final state of charge (SOC, %):** Final EV battery charge.
- **Charging completion (%):** Proportion of EVs meeting the configured completion criterion.

Results are summarised using the mean and standard deviation across five random seeds.

## Results Summary

The following results are based on the recorded five-seed evaluation.

| Method | Grid Import (kWh) | Peak Load (kW) | Charging Energy (kWh) |
|---|---:|---:|---:|
| Uncontrolled Charging | 836.96 ± 51.53 | 101.44 ± 5.59 | 1347.60 ± 335.07 |
| Rule-Based Smart Charging | 576.01 ± 21.95 | 60.10 ± 5.41 | 326.26 ± 16.92 |
| Independent DQN | 748.19 ± 120.81 | 96.30 ± 4.62 | 986.82 ± 208.23 |
| Cooperative MARL-DQN | **301.65 ± 55.25** | **48.68 ± 12.51** | 228.68 ± 34.80 |
| Service-Aware Cooperative MARL-DQN | 791.66 ± 39.13 | 99.02 ± 2.92 | 926.80 ± 32.92 |

### Key Findings

- Cooperative MARL-DQN achieved the lowest mean grid import and peak load among the evaluated methods.
- Compared with uncontrolled charging, mean grid import decreased by approximately **64%**, while peak load decreased by approximately **52%**.
- The cooperative method also delivered substantially less charging energy, highlighting a trade-off between grid-demand reduction and EV charging service.
- The service-aware variant delivered more charging energy than the standard cooperative method, but its grid import and peak load were higher.
- No single strategy performed best across every metric.

**Conclusion:** Grid-demand reduction should be evaluated alongside final SOC and charging completion to ensure that improved grid performance does not come at the expense of EV charging requirements.

## Technology Stack

- Python
- PyTorch
- CityLearn
- Deep Q-Networks (DQN)
- Experience replay
- Target network learning
- NumPy and data-analysis tools

## Repository Structure

A suggested structure for organising the project:

```text
cooperative-ev-marl-citylearn/
├── README.md
├── requirements.txt
├── 01_environment/
├── 02_agents/
├── 03_training/
├── 04_experiments/
├── 05_results/
├── 06_figures/
├── 07_models/
├── 08_report/
└── 09_presentation/
```

The folders can be adapted to match the final implementation. Include only files that are actually present in the repository.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/cooperative-ev-marl-citylearn.git
cd cooperative-ev-marl-citylearn
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows**
```bash
.venv\Scripts\activate
```

**Linux/macOS**
```bash
source .venv/bin/activate
```

### 3. Install dependencies

If a `requirements.txt` file is provided:

```bash
pip install -r requirements.txt
```

Use the dependency versions tested with your final implementation. The exact installation command may vary depending on the Python and PyTorch versions.

### 4. Run the project

Run the environment setup, training, or evaluation script provided in the repository. Update this section with the exact notebook or script path and command once those files have been added.

## Reproducibility

The evaluation uses five random seeds:

`0, 42, 123, 777, 2026`

Reported metrics are expressed as mean ± standard deviation. Results may vary with dependency versions, simulator configuration, hardware, and randomisation.

## Limitations

- The current implementation focuses on EV-level coordination using a shared DQN.
- Lower grid import does not necessarily indicate better overall performance if EV charging service is reduced.
- The service-aware variant does not outperform the standard cooperative method on every grid metric.
- Separate station-level and grid-level agents have not been implemented as independent learning agents in the current framework.

## Future Work

- Improve charging completion and final SOC while maintaining grid performance.
- Refine reward balancing and charging-service constraints.
- Extend the framework to explicit EV, charging-station, and grid-level agents.
- Evaluate longer episodes and additional energy-system scenarios.
- Investigate performance across a broader range of simulation conditions.

## References

1. Tuchnitz, F., Ebell, N., Schlund, J., & Pruckner, M. (2021). Development and evaluation of a smart charging strategy for an electric vehicle fleet based on reinforcement learning. *Applied Energy, 285*, Article 116382. https://doi.org/10.1016/j.apenergy.2020.116382
2. Zhou, L., Huo, D., Chen, J., Bo, B., & Li, H. (2025). Federated reinforcement learning with constrained Markov decision processes and graph neural networks for fair and grid-constrained coordination of large-scale electric vehicle charging networks. *Scientific Reports, 15*, Article 39593. https://doi.org/10.1038/s41598-025-22482-5
3. Li, Y., Zhao, H., Song, Z., Chong, X., Chen, R., Li, L., & Wang, X. (2026). Coordinated interaction of electric vehicles, charging stations, and the power grid using multi-agent reinforcement learning. *Advances in Applied Energy, 22*, Article 100274. https://doi.org/10.1016/j.adapen.2026.100274
4. CityLearn. Official documentation. https://www.citylearn.net/
5. Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature, 518*, 529–533. https://doi.org/10.1038/nature14236

## Author

**Deepu Guggilla**  
B.Tech — Computer Science and Engineering (Artificial Intelligence and Engineering)

## License

Choose a license before distributing or reusing the project. For example, the MIT License may be suitable if you want others to reuse the code, subject to its terms.
