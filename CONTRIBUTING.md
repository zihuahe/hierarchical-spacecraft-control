# Contributing

Start with [current status](docs/STATUS.md). Code in an open PR is proposed work; it is not automatically available on `main`.

## Useful first contributions

1. Record the exact Linux launch command, available scenario, and one real state/action trace.
2. Connect local Kev through the chosen controller boundary.
3. Save a complete episode and a readable trajectory or replay.
4. Compare controllers using the [experiment protocol](docs/EXPERIMENTS.md).

Agree ownership before editing overlapping controller files. Keep a documentation change separate from changes to physics, controller behavior, or experiment settings when practical.

## For coding agents

Read `README.md`, `docs/STATUS.md`, `docs/SETUP.md`, and `docs/CONTROL_TASK.md` before changing code. Inspect the actual checkout and any existing repository instructions.

The smallest milestone is a real observation, a local Kev decision, a valid executed command, and the next observation. Reuse the working environment. Preserve the programmed controller as a baseline. Do not replace the simulator to avoid an integration problem.

Choose one execution harness: the existing `KSPDGBaseAgent`/runner path, or the proposed direct `env.step()` bridge. Their timing differs. Confirm the chosen path's stop behavior and log it. Model or simulator failure must produce a diagnostic and supported runner/environment shutdown; returning `None` in the standard runner preserves the previous action.

## Before opening a PR

- Explain the concrete behavior changed and how it was checked.
- Label real KSPDG, stand-in simulation, real model calls, and mocked responses separately.
- Check all seven action mappings, frames, units, and configured limits when changing control code.
- Include a trace and reproducible command for an execution claim.
- Keep credentials, personal host paths, private recordings, and the licensed game archive out of the PR.
- Update the relevant guide and [status](docs/STATUS.md). Include failures and unfinished work.

Use a conventional commit title. Documentation-only example: `docs(readme): explain spacecraft control setup [skip ci]`.

## Keeping these docs useful

Keep the public overview in `README.md`, commands in `docs/SETUP.md`, the interface in `docs/CONTROL_TASK.md`, and evidence in `docs/STATUS.md`. Link between them instead of copying long sections. Update the status snapshot after code merges or new execution evidence.
