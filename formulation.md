# Hardware-Aware Adaptive Inference with Percentile-Based Latency Control

## Problem Formulation

The proposed system consists of a set of deployable subnet configurations:

$$\\mathcal{C} = \\{c_i=(w_i,b_i)\\}\_{i=1}^{N}$$

where \\(w_i\\) represents the network width and \\(b_i\\) represents the quantization precision. In this work,

$$w_i \\in \\{0.25,0.5,0.75,1.0\\},\\qquad b_i \\in \\{4,8,16,32\\}.$$

Each configuration provides a different trade-off between inference accuracy and computational cost. The accuracy of configuration \\(c_i\\) is represented by \\(A(c_i)\\).

At each inference step \\(t\\), the controller observes the current hardware state:

$$s_t = \[u_t,m_t,\\theta_t,e_t,v_t,r_t,g_t\]$$

where the state contains CPU utilization, available memory, temperature, device speed, number of CPU cores, total RAM, and GPU availability.

The system also receives a latency budget \\(B_t\\), which specifies the maximum acceptable inference latency.

The controller selects a configuration according to

$$c_t = \\pi(s_t,B_t,c\_{t-1}),$$

where \\(c\_{t-1}\\) is the configuration selected during the previous inference step.

The main objective is to select the **highest-accuracy configuration that satisfies the latency requirement**. Since inference latency can vary during execution, the controller uses latency percentiles rather than only average latency.

The deadline constraint is defined as

$$Q\_{1-\\delta}(L(c_t,s_t)) \\leq B_t,$$

where \\(Q\_{1-\\delta}\\) represents the required latency percentile. For example, P50 corresponds to \\(\\delta=0.50\\), P95 corresponds to \\(\\delta=0.05\\), and P99 corresponds to \\(\\delta=0.01\\).

A safety margin can also be applied:

$$Q\_{1-\\delta}(L(c_t,s_t)) \\leq \\rho B_t,$$

where \\(0 \\lt \\rho \\leq 1\\).

If no configuration satisfies this constraint, the latency budget is considered **infeasible**. This means that the requested deadline is lower than the minimum achievable latency of the available configurations.

## Latency Modeling

Before runtime adaptation, the latency of each subnet configuration is measured under the available hardware conditions. Multiple measurements are collected for each configuration:

$$\\mathcal{D}\_i = \\{\\ell\_{i,1},\\ell\_{i,2},\\ldots,\\ell\_{i,M}\\}.$$

From these measurements, latency percentiles such as P50, P95, and P99 are calculated.

The system considers both normal execution and the additional latency that may occur when switching from one configuration to another.

To estimate latency for the current hardware state, a latency prediction function is used:

$$\\widehat{L}\_{i,t} = f\_{\\phi}(c_i,s_t).$$

The predictor can use either a measured lookup table or a machine-learning model such as Random Forest.

The input features include the configuration properties and hardware characteristics, such as:

- width multiplier,
- quantization precision,
- estimated FLOPs,
- parameter count,
- CPU utilization,
- available memory,
- temperature,
- device speed,
- number of CPU cores, and
- GPU availability.

This allows the latency prediction to adapt to changes in the underlying hardware conditions.

## Online Latency Calibration

The predicted latency may differ from the actual latency because of changing system conditions, background processes, or other runtime effects. Therefore, the controller continuously updates its latency estimates using observed inference latency.

After executing a configuration, the relative prediction error is calculated as

$$e_t = \\frac{\\ell_t-\\widehat{L}\_t}{\\max(\\widehat{L}\_t,\\epsilon)},$$

where \\(\\ell_t\\) is the measured latency and \\(\\epsilon\\) is a small value used to avoid division by zero.

The error is clipped to prevent a single abnormal latency measurement from causing a large change in the prediction:

$$\\bar{e}\_t = \\operatorname{clip}(e_t,-\\eta\_{\\mathrm{low}},\\eta\_{\\mathrm{high}}).$$

A separate calibration factor is maintained for every configuration:

$$k\_{c_t,t+1} = \\operatorname{clip}\\left(k\_{c_t,t}+\\alpha\\bar{e}\_t,\\;k\_{\\min},k\_{\\max}\\right).$$

Only the configuration that was executed is updated. This prevents an unusual latency measurement for one subnet from affecting the predictions of other configurations.

The calibrated latency prediction is therefore

$$\\widetilde{L}\_{i,t} = k\_{i,t}\\widehat{L}\_{i,t}.$$

## Adaptive Configuration Selection

For each configuration, the controller calculates a risk-adjusted latency:

$$R\_{i,t} = k\_{i,t}\\widehat{L}\_{i,t} + C\_{\\mathrm{switch}}(c_i,c\_{t-1}),$$

where \\(C\_{\\mathrm{switch}}\\) represents the additional cost of switching to a different configuration.

A configuration is considered feasible when

$$R\_{i,t} \\leq \\rho B_t.$$

Among all feasible configurations, the controller selects the one with the highest accuracy:

$$c_t = \\arg\\max\_{c_i\\in\\mathcal{C}} A(c_i)\\quad\\text{subject to}\\quad R\_{i,t} \\leq \\rho B_t.$$

If multiple configurations provide the same accuracy, the configuration with the lower predicted latency is selected.

If no configuration satisfies the latency constraint, the controller selects the configuration with the lowest predicted latency:

$$c_t = \\arg\\min\_{c_i\\in\\mathcal{C}} R\_{i,t}.$$

The decision is then recorded as an **infeasible-budget event**.

This distinction is important because a deadline can be missed for two different reasons:

1. The requested deadline was impossible to satisfy with the available configurations.
2. A feasible configuration existed, but the actual latency exceeded the deadline.

## Proposed Adaptive Inference Algorithm

The complete procedure is summarized below:

```
Input:
    Configuration set C
    Accuracy table A(c)
    Latency profiles
    Current hardware state s_t
    Latency budget B_t
    Previous configuration c_(t-1)

1. Measure the current hardware state.
2. For every configuration c_i:
       Predict its latency.
       Apply configuration-specific calibration.
       Add switching overhead if required.
3. Determine whether the latency budget is feasible.
4. Select the highest-accuracy feasible configuration.
5. If no configuration is feasible,
       select the configuration with the lowest predicted latency.
6. Execute inference using the selected configuration.
7. Measure the actual inference latency.
8. Update the calibration factor of the selected configuration.
9. Store the decision and runtime measurements.
```

The controller therefore adapts the computational cost of the model according to both the current hardware state and the required latency budget.

## Evaluation Metrics

The proposed controller is evaluated using accuracy, latency, and adaptation-related metrics.

### Deadline Miss Rate

The overall deadline miss rate is calculated as

$$\\mathrm{MissRate} = \\frac{1}{T}\\sum\_{t=1}^{T}\\mathbf{1}\[\\ell_t>B_t\].$$

### Infeasible Budget Rate

The percentage of requests for which no available configuration can satisfy the latency requirement is measured as

$$\\mathrm{InfeasibleRate} = \\frac{1}{T}\\sum\_{t=1}^{T}\\mathbf{1}\[\\text{budget is infeasible}\].$$

### Feasible-Only Miss Rate

To evaluate the controller separately from impossible deadlines, the miss rate is also calculated only for feasible budgets:

$$\\mathrm{FeasibleMissRate} = \\frac{\\sum_t \\mathbf{1}\[\\ell_t>B_t\]\\,\\mathbf{1}\[\\text{budget is feasible}\]}{\\sum_t \\mathbf{1}\[\\text{budget is feasible}\]}.$$

### Accuracy

Classification accuracy is measured as

$$\\mathrm{Accuracy} = \\frac{\\text{Number of correct predictions}}{\\text{Total number of predictions}}.$$

### Latency Prediction Error

The accuracy of the latency prediction model is evaluated using mean absolute error:

$$\\mathrm{MAE} = \\frac{1}{T}\\sum\_{t=1}^{T}|\\ell_t-\\widehat{L}\_t|.$$

### Violation Magnitude

For missed deadlines, the average amount by which the actual latency exceeds the required budget is measured as

$$\\mathrm{ViolationMagnitude} = \\frac{1}{|\\mathcal{M}|}\\sum\_{t\\in\\mathcal{M}}(\\ell_t-B_t),$$

where \\(\\mathcal{M}\\) is the set of missed deadlines.

### Tail Latency

The overall latency distribution is reported using

$$Q\_{0.50}(\\ell),\\qquad Q\_{0.95}(\\ell),\\qquad Q\_{0.99}(\\ell).$$

These values represent the median, P95, and P99 latency, respectively.

### Switching Rate

The frequency of configuration changes is measured using

$$\\mathrm{SwitchRate} = \\frac{1}{T-1}\\sum\_{t=2}^{T}\\mathbf{1}\[c_t\\neq c\_{t-1}\].$$

A lower switching rate indicates that the controller is making fewer configuration changes during inference.

## Research Hypothesis

The proposed approach is based on the following hypothesis:

> A hardware-aware controller using percentile-based latency prediction and configuration-specific online calibration can reduce deadline violations while maintaining higher inference accuracy compared with static and average-latency configuration selection methods.

The following policies are evaluated:

1. Static fastest configuration
2. Static highest-accuracy configuration
3. Average-latency controller
4. P50 controller
5. P95 controller
6. P99 controller
7. P95 with configuration-specific calibration
8. P99 with configuration-specific calibration

The policies are compared using accuracy, total deadline miss rate, feasible-only miss rate, latency prediction error, tail latency, and configuration switching rate.

The proposed method aims to provide adaptive inference that selects an appropriate subnet according to the current hardware condition and latency requirement while explicitly distinguishing infeasible deadlines from controller-induced deadline violations.