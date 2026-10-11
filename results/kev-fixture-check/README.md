# Real Kev on the lbg1_lite TEST FIXTURE: not Kerbal Space Program

These traces come from `python -m hsc.kspdg_loop --env fixture --episode`. The local Kev-0.8B server
(`jaredpalmer/kev-0.8b@v1.0`, CPU, float32) made every decision. The environment is the pure-Python
`lbg1_lite` stand-in, so they are **not** a KSPDG run and say nothing about flight in the game. The
fixture's Guard is a proxy that captures at 5 m. The real LG7 env intercepts at 10 m. A fixture step
also takes no wall time, while KSPDG burns for `burn_vec[3]` wall-clock seconds.

They show the loop working end to end: warm-up, the observation record, the Kev request, validation,
the command, the next observation, history, the terminal or error reason and cleanup, all with real
model answers.
