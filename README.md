# Analog (Dose-Response) Backdoor against WiFi-based Human Pose Estimation

Implementation of a data-poisoning backdoor on WiFi-CSI human pose estimation whose
**payload magnitude is a continuous function of the trigger's dose** (impossible in a
classification backdoor). The kinematics-derived micro-Doppler trigger's intensity
sets, via forward-kinematics, the displacement of a chosen limb in the predicted
skeleton. See `wifi_hpe_backdoor_design.md` for the full design rationale.

## Layout

```
attack/
  trigger.py     dose-parameterized micro-Doppler antenna-differential trigger
  payload.py     FK joint-localized dose->rotation payload (bone-length preserving)
  poison.py      poisoning pipeline + dataset wrapper + eval modes
models/
  hpeli.py       HPE-Li victim (offline)         sk_network.py
  metafi.py      MetaFi++ victim (needs torchvision; pretrained weights need network)
  channel_trans.py  factory.py
data_utils/
  feeder.py      real Person-in-WiFi-3D feeder (load_raw / normalize split)
  synth_dataset.py  synthetic real-format data + synthetic action (for smoke test)
eval/
  metrics.py     MPJPE/PA-MPJPE/PCK + dose-response (Spearman, step-contrast) + ASR
train_backdoor.py  main train+eval driver
verify_joints.py   prints bone tree / candidate sub-chains (run FIRST)
smoke_test.py      end-to-end pipeline test on synthetic data
configs/attack.yaml
```

## Quick start (smoke test, no real data)

```bash
pip install torch torchvision pyyaml scipy numpy   # torchvision only for MetaFi
python smoke_test.py
```
Validates: FK bone-length preservation, antenna-differential trigger, dose=0 identity,
monotone payload, both victims build/forward, and a full poison→train→eval run. It does
**NOT** test attack efficacy (synthetic CSI, 2 epochs) — the dose-response curve will be
flat; that is expected.

## Running on the real dataset

1. Request Person-in-WiFi-3D (aiotgroup) and run its `process.py`. The feeder reads
   **`csi_ap/<name>.npy`** of shape `(3,180,20)` = `concat(amplitude[3,90,20], phase[3,90,20])`.
   The dataset's `process.py` writes `csi_amplitude` and `csi_phase` separately — concat
   them into `csi_ap` (its commented `csi_ap` block does exactly this).
2. `python verify_joints.py` — confirm `pivot` selects the intended limb (default 7 →
   rotates {11,13}). **Do not trust the repo's PCK joint-name comments.**
3. Set `dataset_root` in `configs/attack.yaml` to your `data/person_in_wifi_3d`.
4. `python train_backdoor.py --config configs/attack.yaml`

## Experiment #1 (the headline necessity test)

The analog claim must be shown **graded and monotone**, not a thresholded on/off. The
evaluator reports, per dose grid: target-limb `displacement`, `nontarget_mpjpe`
(localization), `plausibility`, and a `dose_response` block with `spearman` (monotonicity)
and `ramp_minus_step` (R² of a monotone fit minus the best step fit; **>0 means graded,
not binary**). Sweep `rho`, `eps`, `theta_max_deg`, `dose_mode`, and `pivot` (sub-chain
size ablation). Run both `model: hpeli` and `model: metafi` for the cross-architecture claim.

## Honest scope / caveats

- Digital attack: the trigger is injected into stored (sanitized) CSI before
  normalization; sanitization does not re-run, so survival is automatic. The
  antenna-differential micro-Doppler structure is kept for physical plausibility and to
  stay distinct from an additive (BadNet) trigger (pilot: common-mode survival 0.008 vs
  0.66 antenna-differential — relevant if a physical variant is attempted later).
- ASR thresholds (`tau_*` in `eval/metrics.py`) are dataset-unit dependent; calibrate
  them once on the real data scale.
- MetaFi++ needs ImageNet-pretrained ResNet weights (`pretrained: true`) for intended
  performance — that download requires network access.
- The smoke test's synthetic CSI validates wiring only; all attack-success numbers are
  meaningful only after a real-data training run.
```
```
