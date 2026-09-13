# tsch-scheduling

Two Q-learning scheduling functions for TSCH, built on a fork of the
[6TiSCH Simulator](https://github.com/openwsn-berkeley/6tisch-simulator):
DynQ (dynamic discretisation) and Q-static (fixed thresholds). They are
compared against MSF (RFC 9033), EMSF and RL-SF.

## Setup

You need Docker. You do not need Python on your machine.

```bash
docker compose build
```

Every command below runs from the repository root. Simulations write to
`simulator/bin/simData/`, which git ignores.

## Parameters

The results use these settings, all versioned in `simulator/bin/`:

| File | What it holds |
| --- | --- |
| `config.json` | The scenario: 101-slot slotframes, 10 ms slots, 16 channels, queue of 5 packets, RPL with OF0, reward weights 0.8, 0.2, 0.8, 0.01. |
| `anchor_escolhido.json` | The core learning parameters chosen by the sensitivity sweep. DynQ: α = 0.1, γ = 0.95, ε_min = 0.15. Q-static: γ = 0.7. |
| `traffic_queue_charge_parameters.json` | DynQ's remaining parameters, such as the ε decay rate. |
| `qlearningSBRC24_parameters.json` | Q-static's remaining parameters, including the ε threshold of 0.3. |

The comparison reads `config.json`, applies the scheduler's parameters file,
then applies the anchor on top. The factorial applies only the anchor, so its
cells take every other setting from `config.json`. Each run writes the config
it used to `simulator/bin/config_<name>.json`.

## Reproducing the results

All experiments use 10 runs of 15000 slotframes each. Runs are seeded by run
index, so run 3 of every scheduler sees the same topology. That is what makes
the comparisons paired.

### 1. Scheduler comparison

```bash
docker compose run --rm sim python runComparison.py \
  --motes 50 \
  --arms dynq,qstatic,rlsf,msf,emsf \
  --runs 10 \
  --cpus 10 \
  --slotframes 15000 \
  --anchor anchor_escolhido.json
```

Each scheduler lands in `simData/<arm>_n<motes>/` and its KPIs are computed at
the end. On 10 cores, one scheduler at 50 motes takes 15 to 35 minutes.

The other scenarios are the same command with one change:

| Scenario | Change |
| --- | --- |
| 100 motes | `--motes 100` |
| Linear topology | `--conn-class Linear` |
| Periodic traffic | `--app AppPeriodic` |

The default traffic is `AppRandom`: every 60 s, each mote sends a burst of 1 to
10 packets. The default topology is `Random`, with link quality from the
Pister-Hack model. Output folders have the same names in every scenario, so run
each one in a fresh `simData/` or move the previous one away first.

`runComparison.py` flags:

| Flag | Meaning |
| --- | --- |
| `--motes` | Network sizes, as a list. The area grows with the count, so density stays the same. |
| `--arms` | Comma-separated schedulers: `dynq`, `qstatic`, `rlsf`, `msf`, `emsf`. |
| `--runs` | Runs per scheduler. |
| `--cpus` | Cores. Runs are spread across them. |
| `--slotframes` | Length of each run. |
| `--anchor` | JSON with the core learning parameters. |
| `--app` | Traffic class, overriding `config.json`. |
| `--conn-class` | Topology class, overriding `config.json`. |

## Scheduling functions

| Arm | `sf_class` | Scheduler |
| --- | --- | --- |
| `dynq` | `Qlearning` | DynQ, dynamic Q-learning |
| `qstatic` | `QlearningSBRC24` | Q-static, threshold-based Q-learning |
| `rlsf` | `RLSF` | RL-SF (Pratama and Chung, 2022) |
| `msf` | `MSF` | Minimal Scheduling Function, RFC 9033 |
| `emsf` | `EMSF` | Enhanced Minimal Scheduling Function |

## Working inside the container

```bash
docker compose run --rm sim bash
```

You land in `/sim/bin` with the repository mounted, so edits on your machine
apply immediately.

## Running without Docker

Needs Python 2.7 on linux/x86_64.

```bash
python2 -m virtualenv venv
source venv/bin/activate
pip install -r simulator/requirements.txt
cd simulator/bin
```

Then drop the `docker compose run --rm sim` prefix from the commands above.
