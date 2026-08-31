# tsch-scheduling

Two Q-learning scheduling functions for TSCH, built on a fork of the
[6TiSCH Simulator](https://github.com/openwsn-berkeley/6tisch-simulator).

## Setup

You need Docker. You do not need Python on your machine.

```bash
docker compose build
```

## Running an experiment

```bash
docker compose run --rm sim python runExperiments.py \
  -cb 50 \
  -nr 10 \
  -nc 8 \
  -sf Qlearning \
  -app AppBurst \
  -cc Random \
  -nslots 3750 \
  -of results_dynq \
  -is_min minimization
```

Results are written to the output folder inside your working copy.

Use `-is_min 2k` for the 2³ factorial experiment. Everything else stays the same.

### Required arguments

| Flag | Meaning |
| --- | --- |
| `-cb` | Network sizes to simulate. Takes a list: `-cb 20 50 100` runs all three. |
| `-nr` | Runs per network size. Results are averaged over these. |
| `-nc` | CPU cores to use. |
| `-sf` | Scheduling function. See the table below. |
| `-app` | Traffic pattern. `AppBurst`, `AppPeriodic`, `AppRandom`. |
| `-cc` | Topology. `Random`, `Linear`, `FullyMeshed`, `K7`. |
| `-nslots` | Simulation length in slotframes. A slotframe is 10 ms. |
| `-of` | Output folder name. |
| `-is_min` | `minimization` or `2k`. |

### Optional arguments

| Flag | Meaning |
| --- | --- |
| `-fc` | Which state factors to use, comma separated: `traffic,queue,charge`. Defaults to all three. |
| `-ne` | Number of evaluations in the Bayesian optimization. |
| `-nrs` | Number of random starts before the optimizer takes over. |
| `-af` | Acquisition function for `gp_minimize`. |
| `-sr` | Collect synchronization info during the run. |

### Scheduling functions

| `-sf` value | Scheduler |
| --- | --- |
| `Qlearning` | DynQ, dynamic Q-learning |
| `QlearningSBRC24` | Q-static, threshold based Q-learning |
| `MSF` | Minimal Scheduling Function, RFC 9033 |
| `EMSF` | Enhanced Minimal Scheduling Function |
| `SFNone` | No scheduling function |

## Working inside the container

```bash
docker compose run --rm sim bash
```

You land in `/sim/bin` with the repository mounted, so edits on your machine
apply immediately.

## Running without Docker

Needs Python 2.7 on linux/x86_64. The pinned dependencies have no wheels for
Apple Silicon and will not build there.

```bash
python2 -m virtualenv venv
source venv/bin/activate
pip install -r simulator/requirements.txt
cd simulator/bin
```

Then drop the `docker compose run --rm sim` prefix from the commands above.
