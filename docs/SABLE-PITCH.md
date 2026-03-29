# SABLE - Infrastructure Diagnostics That Think

## The Problem

Your monitoring stack tells you **what** broke. SABLE tells you **why**, **what's next**, and **what to do about it** - before the cascade reaches your customers.

Current tools generate alerts. SABLE generates understanding.

## What It Does

SABLE watches your infrastructure through three lenses simultaneously:

**Structure** - understands how your components depend on each other. When a core switch degrades, SABLE already knows which VMs, services, and applications sit downstream.

**Decisions** - plans diagnostic actions under uncertainty. With partial monitoring coverage and delayed telemetry, SABLE figures out what to check next to find the root cause fastest.

**Time** - predicts where the cascade is heading. Two ticks of data, and SABLE forecasts the next thirty - which nodes will fail, how severe the impact will be, and when things will stabilize.

These three perspectives fuse into a single assessment with honest confidence. When SABLE is uncertain, it says so.

## How It's Different

| | Traditional Monitoring | AIOps Platforms | SABLE |
|---|---|---|---|
| Detection | Threshold alerts | Anomaly detection | Structural + temporal reasoning |
| Root cause | Manual investigation | Statistical correlation | Causal graph inference |
| Prediction | None | Trend extrapolation | Cascade simulation |
| Confidence | Binary (alert/no alert) | Hidden scores | Variance-based honest uncertainty |
| Feedback | None | Black box retraining | Operator corrections improve the model |
| Infrastructure | Flat metric lists | Metric + log correlation | Full dependency graph awareness |

## What an Operator Sees

1. An incident begins. Monitoring shows a core switch at 95% CPU.
2. SABLE immediately identifies 12 downstream nodes at risk, predicts cascade severity at 0.73, and names the root cause with 89% confidence.
3. Recommended action: "INVESTIGATE ROOT CAUSE - Core Switch 1 [CORE_SWITCH]. First failure at tick 0. 5 total failed nodes likely depend on this."
4. The operator clicks a node, sees the probability distribution across states, and can correct the prediction if SABLE is wrong - feeding the correction back into the model.

Time from incident to actionable diagnosis: **under 25ms**.

## Technical Specs

- 5.4M parameters (runs on commodity hardware)
- 5ms per inference, 25ms with Monte Carlo confidence
- Ingests Prometheus, SNMP, Netbox, or any metric source
- Adapts to your topology via YAML config or auto-discovery
- Fine-tunes on your environment's data in under a week
- Honest confidence via MC dropout - measures actual model uncertainty

## The Ask

We need one environment with real monitoring data to prove this on live infrastructure. Read access to a Prometheus instance and a topology map. Two weeks of data. We build the adapter, fine-tune, and show you results on a held-out third week.

The architecture is built. The training pipeline is built. The telemetry adapter is built. What's missing is your data.

## Contact

Keith Burns - b1tr0n1n
