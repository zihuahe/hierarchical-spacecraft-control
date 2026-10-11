# Hierarchical Spacecraft Control

Can a fast AI controller react quickly while slower models keep it focused on the mission?

Built at [Sundai Hack 144: Autonomous Spacecraft](https://www.sundai.club/events/boston/hack-144-autonomous-spacecraft-hack), MIT, October 11, 2026.

## Team

**Heiko, Robert, Enrique, and Zi Hua.** Thanks to the **Sundai team** for bringing us all together and making this experiment possible.

## The mission

We control the **Bandit**, a simulated spacecraft trying to reach the **Lady** while avoiding interception by the **Guard**. Capture means crossing a proximity threshold, not docking.

The environment is **Kerbal Space Program Differential Games (KSPDG)** inside Kerbal Space Program. The intended task lasts up to four minutes. Our original target is `LBG1_LG7_I2_V1`; the effective scenario and rules must be confirmed on the running host. See [current status](docs/STATUS.md).

## The idea

Heiko's original pitch: a global model sets mission objectives and constraints, a mid-level model reviews recent states and updates short-term objectives, and a fast model chooses immediate actions.

| Layer | Responsibility | Build plan |
| --- | --- | --- |
| Mission strategy | Set objectives and constraints | Fixed guidance first; Opus later |
| Tactical planning | Update short-term goals | Haiku later |
| Immediate control | Select the next action | Local Kev-0.8B |

The pitch originally used Jev. We chose open-source Kev-0.8B with pretrained weights for the first build. No model training is planned. Hosted planners would still require a connection.

**Observe → choose → validate → thrust → observe again.** Python prepares numerical state and maps Kev's choice to a simulator command. The simulator handles physics.

## Where we are

The team reports a Linux simulation running with a programmed controller. Stored artifacts in an open PR record Kev responding to synthetic and stand-in simulation states. In the latest synthetic check, all 20 timed responses chose `COAST`; this establishes API integration, not useful navigation. A real KSPDG flight controlled by Kev and a benefit from hierarchical planning remain unverified.

Our next milestone is one real observation → Kev decision → executed command → next observation, with a saved trace. Then complete an episode and compare programmed control, Kev alone, and hierarchical guidance under matched conditions.

## Start here

| Document | Use it to |
| --- | --- |
| [Setup](docs/SETUP.md) | Select a runtime path and reproduce a connection test |
| [Control task](docs/CONTROL_TASK.md) | Understand observations, actions, timing, and the agent boundary |
| [Experiments](docs/EXPERIMENTS.md) | Define a fair comparison and record evidence |
| [Status](docs/STATUS.md) | Check integration progress and open tasks |
| [Contributing](CONTRIBUTING.md) | Make a small, reviewable contribution |

## Thanks

Thanks to Alejandro Carrasco Aragón, MIT AeroAstro PhD student, for the talk and simulator materials, and to the KSPDG and Kev maintainers for their tools.

- [KSPDG scenario guide](https://github.com/mit-ll/spacegym-kspdg/blob/main/src/kspdg/lbg1/README.md)
- [Speaker's Docker setup](https://github.com/ARCLab-MIT/kspdg-2026/tree/main/docker)
- [Kev source](https://github.com/jaredpalmer/kev) and [model card](https://huggingface.co/jaredpalmer/kev-0.8b)
