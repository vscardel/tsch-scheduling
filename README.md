# tsch-scheduling

Repository containing two new scheduling functions for the TSCH protocol. Those
functions are described in detail in the articles referenced in this project
README.

The simulator is a fork of the [6TiSCH Simulator](https://github.com/openwsn-berkeley/6tisch-simulator)
and runs on **Python 2.7**.

---

## Setup

The simulator depends on a pinned scientific stack (`numpy 1.16.6`,
`scipy 1.2.3`, `matplotlib 2.2.5`, ...) that only publishes Python 2 wheels for
`linux/x86_64`. Rather than ask you to find a working Python 2 toolchain, the
repository ships a container that reproduces the exact environment used to
produce the published results.

**Requirements:** Docker (with Compose v2). Nothing else — no local Python 2.

Build the image once:

```bash
docker compose build
```

On Apple Silicon this runs under emulation, which is slower but numerically
identical. The build takes a couple of minutes; after that it is cached.

### Working inside the container

```bash
docker compose run --rm sim bash
```

This drops you in `/sim/bin` with the repository bind-mounted, so edits you make
on the host apply immediately and any results the simulator writes appear in your
working copy. To run a single command without an interactive shell, put it after
`sim`:

```bash
docker compose run --rm sim python runSim.py
```

### Tests

```bash
docker compose run --rm -w /sim sim python -m pytest tests/
```

---

## Running experiments

Both commands below run from `/sim/bin`, which is the container's default
working directory.

### Minimization experiment

```bash
docker compose run --rm sim python runExperiments.py \
  -nc <combinations> \
  -nr <num_runs> \
  -sf <scheduling_function> \
  -app <app> \
  -of <output_folder_name> \
  -cc <conn_class> \
  --num_slots <num_slots> \
  --experiment_type minimization
```

### 2³ factorial experiment

Same invocation with `--experiment_type 2k`:

```bash
docker compose run --rm sim python runExperiments.py \
  -nc <combinations> \
  -nr <num_runs> \
  -sf <scheduling_function> \
  -app <app> \
  -of <output_folder_name> \
  -cc <conn_class> \
  --num_slots <num_slots> \
  --experiment_type 2k
```

`-sf` accepts the scheduling function class name, which must match a module in
`simulator/SimEngine/Mote/scheduling_functions/`:

| Value             | Scheduling function                       |
| ----------------- | ----------------------------------------- |
| `Qlearning`       | DynQ — dynamic Q-learning SF              |
| `QlearningSBRC24` | Q-static — threshold-based Q-learning SF  |
| `MSF`             | Minimal Scheduling Function (RFC 9033)    |
| `EMSF`            | Enhanced Minimal Scheduling Function      |
| `SFNone`          | No scheduling function                    |

Note that `bin/config.json` holds defaults that `runExperiments.py` overrides at
run time. The values that matter for a given experiment are the command-line
arguments, plus the per-run config each experiment writes out.

---

## Running without Docker

Not recommended, and unsupported on Apple Silicon, where the pinned dependencies
have no wheels and will not build. If you have a working Python 2.7 on
`linux/x86_64`:

```bash
python2 -m virtualenv venv
source venv/bin/activate
pip install -r simulator/requirements.txt
cd simulator/bin
```

Then drop the `docker compose run --rm sim` prefix from the commands above.
