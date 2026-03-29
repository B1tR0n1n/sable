# SABLE — Path to Production

## Where We Are

The model is a 5.4M parameter system that classifies infrastructure node states across 5 classes through three specialized pillars fused with a temporal feedback loop. It runs at 5ms per inference on a single RTX 5090.

**On data from its own pipeline:** 99.9% accuracy. Near-perfect. The confidence is real and earned.

**On unseen topologies using the same feature pipeline:** 86-97% accuracy. The temporal chain helps. Confidence is honest — 0.93-0.99 where it's right, drops to 0.3-0.5 where it's uncertain.

**On adversarial conditions:** 59-84% accuracy depending on scenario. It degrades but doesn't collapse.

**The gap:** The model learned the *feature distributions* of sable_sim, not the underlying infrastructure reasoning. When the distribution shifts — different topology generator, different failure modes, real telemetry instead of simulated — accuracy drops. The model is a very good sable_sim classifier. It's not yet a general infrastructure diagnostics system.

---

## What It Would Take

### 1. Real Telemetry Data

This is the biggest gap. Everything the model has ever seen came from sable_sim — simulated health values, simulated cascades, simulated fog-of-war. Real infrastructure monitoring produces different signals:

- CPU/memory/disk utilization time series from Prometheus or Datadog
- Response time distributions from APM tools
- Network flow data from switches
- Log anomaly signals from ELK/Splunk
- SNMP trap patterns from network devices

To handle real data, you need an **ingestion adapter** that converts real telemetry into the feature space the three pillars understand. The GNN needs real topology data (from a CMDB or auto-discovery). The POMDP needs real observation streams (from monitoring agents). The Mamba needs real time-series state snapshots.

**How to pull it off:** Partner with one enterprise customer. Get read access to their monitoring stack for one environment — even a lab. Build the adapter that maps their Prometheus metrics + ServiceNow CMDB + Cisco topology into the pillar feature format. Fine-tune on 2 weeks of their data. Validate on a third week they hold out.

**Effort:** 2-4 weeks for the adapter + fine-tuning pipeline. The model architecture doesn't change. The feature encoding does.

### 2. Diverse Training Topologies

sable_sim generates random topologies with `build_random_topology`. Real infrastructure has structure — three-tier architectures, hub-and-spoke networks, microservice meshes, hybrid cloud patterns. The model needs to see all of these during training.

**How to pull it off:** We already have Topology Zoo (261 real ISP networks) and the Microservice Dataset (20 real service graphs). Generate 10x more cascade scenarios on those real topologies — different failure points, different cascade depths, different recovery patterns. Train on the full diversity of real graph structures, not just random generation.

**Effort:** 1-2 days. The infrastructure is built. It's a data generation run + retraining pass.

### 3. Longer, More Complex Cascade Sequences

sable_sim cascades resolve in 1-3 ticks. Real infrastructure failures play out over hours. The temporal chain was trained on 3-tick sequences. It needs 20-50 tick sequences with multiple phases — initial degradation, cascade, partial recovery, secondary failure, stabilization.

**How to pull it off:** Modify sable_sim's PropagationEngine to support slower cascade dynamics — delayed propagation, gradual health decay, intermittent recovery. Generate sequences where the cascade takes 15-20 ticks to fully resolve and another 10-15 to partially recover. The temporal chain will learn longer-horizon patterns.

**Effort:** 1 week. The engine changes are moderate. The retraining is a day.

### 4. Online Learning / Operator Feedback Loop

The most honest path to out-of-distribution accuracy: let the system learn from corrections. When an operator sees the system classify a node as "degraded" and knows it's actually "oscillating," that correction becomes a training signal. Over time, the model adapts to the specific infrastructure it's monitoring.

**How to pull it off:** Add a feedback endpoint to the API — `/api/feedback` that takes `{node_id, correct_state}`. Accumulate corrections. When you have 100+, run a fine-tuning pass on the corrections with frozen backbone, only updating the fusion heads and temporal chain. This is standard active learning.

**Effort:** The API endpoint is a day. The fine-tuning loop is a day. The hard part is the operational workflow — getting operators to actually provide feedback consistently. That's a UX/process problem, not a technical one.

### 5. Confidence That Means Something on New Data

The confidence problem isn't calibration — it's that the model correctly identifies uncertainty on data it hasn't seen. The fix:

- **Ensemble disagreement:** Run 3-5 models with different random seeds. When they agree, confidence is high. When they disagree, confidence is low. This is computationally expensive (5x inference cost) but gives honest uncertainty estimates on any data.

- **Monte Carlo dropout:** Run inference with dropout enabled multiple times. Variance in predictions = uncertainty. Same idea as ensembles but cheaper. We already have dropout in the fusion layers.

- **Conformal prediction:** Given a calibration set, compute prediction sets that are guaranteed to contain the true label at a specified rate (e.g., 90%). Instead of a point confidence, the system says "I'm 90% sure the answer is one of {degraded, failed}." Mathematically guaranteed coverage.

**How to pull it off:** Monte Carlo dropout is the cheapest — run 5 forward passes with dropout at inference time, take the mean prediction and the variance as confidence. ~25ms instead of 5ms per inference. Still real-time. Implement as a flag in `SableEngine.infer(mc_samples=5)`.

**Effort:** Half a day for implementation. The variance-based confidence is honest by construction — it measures the model's actual uncertainty, not a learned scalar.

---

## The Real Answer

The model will be accurate on truly unseen data when it's been trained on data that represents the deployment distribution. There's no shortcut to that. You can pretrain on simulated data (which we did), but the last mile is always real data from real infrastructure.

The fastest path to a product that works honestly:

1. Ship the current demo as a proof of architecture — "this is what the system does on simulated infrastructure"
2. Land one design partner with real monitoring data
3. Build the telemetry adapter (2-4 weeks)
4. Fine-tune on their data (1 week)
5. Deploy with Monte Carlo dropout for honest confidence (1 day)
6. Add operator feedback loop for continuous improvement (1 week)

Total: 5-7 weeks from design partner agreement to production pilot. The architecture is built. The training pipeline is built. What's missing is real data and the adapter to ingest it.
