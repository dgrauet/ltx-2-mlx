# Known Issues

## IC-LoRA

- IC-LoRA pipeline works with LoRA fusion but has only been tested at low resolution (192x128). Higher resolution E2E testing pending.
- Only tested with `LTX-2.3-22b-IC-LoRA-Union-Control`. Motion-Track-Control not tested.

## Tiled VAE

- Tiled decode works but the first temporal chunk can be empty due to causal overlap (guard added). Full pixel-by-pixel equivalence with standard decode not yet verified.

## Memory

- On 32GB Macs, maximum resolution depends on frame count. Approximate limits with q4:
  - 384x256 @ 97 frames
  - 512x320 @ 49 frames
  - Higher resolutions require reducing frame count or using q4 with fewer steps.

## Trainer

- Training loop tested with DummyDataset only. Full LoRA fine-tuning on real data not yet validated end-to-end.
