# NeuroEvolution Self-Driving Lab

A learning project where a population of 2D cars learns to drive winding tracks by **neuroevolution**, using [Pygame](https://www.pygame.org/) for the simulation and [NEAT-Python](https://neat-python.readthedocs.io/) to evolve the neural networks.

No car is ever told how to drive. Each generation, every car is controlled by its own small neural network; the ones that drive furthest and fastest are selected to breed the next generation, and mutations gradually grow better (and bigger) networks.

> This is a 2D toy simulation for learning, not a real-world autonomous-driving system.

## How it works

| Part | Details |
|---|---|
| **Inputs (8)** | Seven distance sensors fanned across the front of the car (±66°), each reading 0–1 of a 145 px range, plus the car's current speed. |
| **Outputs (2)** | Steering (−1 left … +1 right) and throttle. |
| **Fitness** | Forward progress along the track centerline (1000 points per lap, so tracks of different lengths compare fairly), plus a small bonus for each frame survived. Driving backwards is penalised. Cars that finish get a **time bonus** for every frame left on the clock (see below). |
| **Episode end** | A car's run ends when it completes 3 laps, leaves the road, or runs out of time. The time limit is 1.5× what 3 laps at top speed would take, so it scales with track length. |
| **Generation** | Every genome drives the same 2 freshly generated random tracks one after the other. Its fitness is the average. |
| **Evolution** | NEAT with a population of 60. Networks start fully connected with no hidden nodes, and mutations add or remove nodes and connections over time. Species protect new structures so they have time to improve. |

### Training tracks vs. the test track
A network trained on a single track tends to **memorise** it rather than learn to drive. So:

- **Training** uses random tracks generated fresh every generation (`random_track()`). Random radii at jittered angles make a star-shaped loop, which is stretched to fill the play area, sometimes reversed so cars drive both clockwise and anticlockwise, and checked for overlaps, too-tight corners and sensible length.
- **Testing** uses a hand-made track (`TEST_TRACK_POINTS`) that training never sees, with long straights, S-bends and tight hairpins. After each generation the best genome drives it once without being drawn. **`best_genome.pkl` is the genome that scored best on this unseen track**, so it is the best *generaliser*, not the best memoriser.

All tracks are closed Catmull-Rom splines through control points. Edit `TEST_TRACK_POINTS` to design your own test, or `TRACKS_PER_GEN` to train on more tracks per generation (slower but more robust).

### Why the time bonus
Without a finish line, the best cars soon drive flat out for the whole 45 seconds and hit the maximum possible score, so evolution stops improving. In one 236-generation run, best fitness stopped rising at generation 21. Now a run ends after `TARGET_LAPS` (3) laps, and finishers earn `TIME_BONUS` (1.0) fitness per frame they had left. A car that finishes faster scores higher, so there's always room to improve.

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

The dashboard shows the current leader's network (green = inputs, yellow = outputs, blue = hidden; blue/red links = positive/negative weights), a fitness-history chart (green = best on training tracks, blue = best genome on the test track), and live stats including the current track and the latest test-track result.

## Watching the best driver

Replay the saved champion without training:

```powershell
python main.py --replay                           # best genome on the test track
python main.py --replay --track random            # a brand-new random track every run
python main.py --replay --track 4242              # a specific random track (same number = same track)
python main.py --replay path\to\some_genome.pkl   # any other saved genome
```

A single car drives with its sensors shown. The panel shows the current lap, time, speed and best 3-lap time, and the run restarts whenever the car finishes or crashes. Controls: `SPACE` pause, `N` sensors, `ESC` quit.

## Outputs

Everything is written to `neural-driving/training/` (git-ignored):

| File | Contents |
|---|---|
| `neat-checkpoint-N` | Full NEAT population, saved every 5 generations and every 5 minutes |
| `best_genome.pkl` | The genome with the best test-track score so far (a single network, not a resumable population) |
| `fitness_history.csv` / `.png` | Per generation: best and mean fitness on the training tracks, plus the best genome's test-track fitness and result (time, or where it crashed) |

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
3. Fitness measures track progress plus survival, with a bonus for finishing fast.
4. Selection keeps useful genomes; mutations change network weights and structure.
5. NEAT keeps species apart to protect new network structures.
6. Testing on a track the cars never trained on shows whether they learned to *drive* or just memorised a route.
