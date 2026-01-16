---
name: Bug Fix - Feature Dimension Mismatch
about: Dynamic projection layers created on every forward pass causing memory leak
title: "[BUG] Fix #6: Cache dynamic projections in OmniFusionModel"
labels: bug, high-priority, fusion-model, memory-leak
assignees: ''
---

## Description

When feature dimensions don't match expected values in OmniFusionModel, a new `nn.Linear` projection layer is created on every forward pass. These layers are not tracked in `model.parameters()`.

## Evidence

```python
# fusion_network.py:355-357 (BEFORE fix)
elif features.dim() == 2:
    proj = nn.Linear(features.shape[1], self.hidden_dim).to(features.device)
    encoded_features[name] = proj(features)  # New layer EVERY call!
```

## Impact

- **Memory Leak**: Projection layers accumulate in memory
- **No Gradient Flow**: Layers not in `model.parameters()`, not updated during training
- **Inconsistent Projections**: Different random weights between batches
- **Potential Runtime Errors**: Shape mismatches when expected != actual dimensions

## Proposed Fix

1. Add `_dynamic_projections: nn.ModuleDict` to track cached layers
2. Create `_get_or_create_projection()` method for lazy initialization
3. Cache projections by `{name}_{input_dim}` key
4. Register in ModuleDict so they're in `model.parameters()`

## Code Change

```python
def _get_or_create_projection(self, name: str, input_dim: int, device) -> nn.Module:
    key = f"{name}_{input_dim}"
    if key not in self._dynamic_projections:
        proj = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim),
        ).to(device)
        self._dynamic_projections[key] = proj
    return self._dynamic_projections[key]
```

## Acceptance Criteria

- [ ] Dimension mismatches don't crash the model
- [ ] Number of parameters stays constant after multiple forwards
- [ ] Gradient flows through dynamic projections
- [ ] Unit tests verify caching behavior

## Files to Modify

- `src/omni_mercury_engine/ml/fusion_network.py`
- `tests/test_fusion_training.py`

## Branch

`fix/issue-6-dynamic-projections`

## References

- [Rectification Plan](../docs/RECTIFICATION_PLAN.md)
