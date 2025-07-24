# Agent DevLog

## 2025-07-26

### Today's Progress

- **Pivoted Gradient Capture to `torch.autograd.grad`**: Based on the previous day's findings, completely removed the `backward_hook` based `GradientInterceptor`.
- **Refactored `TrainerEngine`**: The core training loop in `_hyper_step` now uses `torch.autograd.grad` to explicitly calculate per-expert gradients for the `surprise` metric. This is a non-invasive method that avoids the complexities and incompatibilities encountered with hooks.
- **Simplified Model Outputs**: Modified `tiny_onn/modular.py` to have the MoE layers directly return the token-to-expert assignments, removing the need for complex hook-based state management. Custom dataclasses for model outputs were created to support this.
- **Removed `training/hooks.py`**: The entire module is now obsolete and has been deleted.
- **Fixed Configuration Loading**: Corrected the `model_path` in `configs/meta_train_v1.yaml` to point to the local weights directory, resolving the `RepositoryNotFoundError`.
- **Added `pi_alpha` and `pi_gamma` to Config**: Added the necessary hyperparameters for the PI score calculation to `training/config.py` and `configs/meta_train_v1.yaml`.
- **Corrected Optimizer and Scheduler Steps**: Fixed a recurring `UserWarning` by ensuring `lr_scheduler.step()` is called after `grad_scaler.step(optimizer)`.
- **Developed New Tests**:
    - Created `tests/test_autograd_surprise.py` to provide a basic unit test for the new `autograd.grad` implementation.
    - Created `tests/test_e2e_meta_learning.py` for a full end-to-end validation of the meta-learning pipeline, including distillation loss, surprise calculation, and router loss.
- **Validation**: All new and existing tests are now passing, confirming the correctness of the refactored training loop.
- **Created `exp/autograd_poc.py`**: A proof-of-concept script was created and run to visually confirm that valid gradients are being captured for expert parameters.

### Next Steps

- Continue with the training and analysis of the model, now that the core training mechanism is stable and validated.
