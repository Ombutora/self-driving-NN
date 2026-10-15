# NeuroEvolution Self-Driving Lab

A learning project where a population of 2D cars learns to drive a winding track by **neuroevolution**, using [Pygame](https://www.pygame.org/) for the simulation and [NEAT-Python](https://neat-python.readthedocs.io/) to evolve the neural networks.

No car is ever told how to drive. Each generation, every car is controlled by its own small neural network; the ones that get furthest around the track are selected to breed the next generation, and mutations gradually grow better (and bigger) networks.

> This is a 2D toy simulation for learning, not a real-world autonomous-driving system.

## How it works

| Part | Details |
|---|---|
| **Inputs (8)** | Seven distance sensors fanned across the front of the car (±66°), each reading 0–1 of a 145 px range, plus the car's current speed. |
| **Outputs (2)** | Steering (−1 left … +1 right) and throttle. |
| **Fitness** | Distance travelled forward along the track centerline, plus a small bonus for each frame survived. Driving backwards is penalised. |
| **Episode end** | A car is removed when it leaves the road or after 45 simulated seconds. The generation ends when every car is out. |
| **Evolution** | NEAT with a population of 60. Networks start fully connected with no hidden nodes, and mutations add or remove nodes and connections over time. Species protect new structures so they have time to improve. |

### The track
The track is a closed Catmull-Rom spline through the control points in `TRACK_POINTS` (top of `main.py`). It mixes long straights, S-bends and tight hairpins, so a car can't get round by holding one steering angle. It has to read its sensors and slow down for corners. Edit the points to design your own track.

### Realism tweaks
To keep driving from looking unnaturally perfect:
- **Steering lag:** the wheel moves only part of the way toward the requested angle each frame (`STEER_RESPONSE`, default `0.25`).
- **Sensor noise:** each sensor reading gets a small Gaussian error before it reaches the network (`SENSOR_NOISE`, default `0.02`).

Lower `STEER_RESPONSE` or raise `SENSOR_NOISE` to make driving harder.

### Performance
Sensors cast rays over a precomputed road mask (a boolean pixel grid) with numpy, so all 60 cars × 7 sensors are cheap each frame. The track image is rendered once and reused.

## Setup (Windows PowerShell)

```powershell
cd path\to\self-driving-NN\neural-driving
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal, then activate again.

On macOS/Linux use `python3 -m venv .venv` and `source .venv/bin/activate`.

## Controls

| Key | Action |
|---|---|
| `SPACE` | Pause / resume |
| `N` | Show / hide sensor rays |
| `S` | Save dashboard state (generation, history, best genome) |
| `ESC` | Quit |

The dashboard shows the current best network (green = inputs, yellow = outputs, blue = hidden; blue/red links = positive/negative weights), a fitness-history chart, and live stats.

## Outputs

Everything is written to `neural-driving/training/` (git-ignored):

| File | Contents |
|---|---|
| `neat-checkpoint-N` | Full NEAT population, saved every 5 generations and every 5 minutes |
| `best_genome.pkl` | The best genome found so far (a single network, not a resumable population) |
| `fitness_history.csv` / `.png` | Best and mean fitness per generation |

Resume training from a population checkpoint:

```powershell
python main.py --resume training/neat-checkpoint-5
```

## Project layout

```
neural-driving/
├── main.py            # simulation, cars, track, dashboard, training loop
├── neat_config.txt    # NEAT hyperparameters (population, mutation rates, ...)
└── requirements.txt   # pygame, neat-python, numpy, matplotlib
```

## Learning notes
1. Sensors provide the network inputs.
2. Network outputs control steering and throttle.
3. Fitness measures track progress plus survival.
4. Selection keeps useful genomes; mutations change network weights and structure.
5. NEAT keeps species apart to protect new network structures.
