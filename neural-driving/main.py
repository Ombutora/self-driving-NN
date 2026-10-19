"""Neuroevolution self-driving cars using Pygame + NEAT-Python.
Run: python main.py
"""
from __future__ import annotations
import math, os, pickle, csv, time, argparse, random
from pathlib import Path
import pygame
import neat
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'training'
OUT.mkdir(exist_ok=True)
W, H = 1100, 760
MAP_W = 800                 # tracks live left of the dashboard panels
FPS = 60
ROAD_WIDTH = 105
# Hand-made test track: never used for training, only to check how well evolved drivers generalise.
TEST_TRACK_POINTS = [(330,80),(560,75),(720,105),(770,215),(700,305),(570,315),(525,400),(620,470),(750,520),(755,630),
                     (620,690),(440,670),(345,585),(230,650),(110,610),(80,475),(170,385),(320,360),(330,270),(230,240),(170,175),(220,95)]
TRACKS_PER_GEN = 2          # random training tracks each genome drives per generation
SENSOR_ANGLES = [-1.15, -0.65, -0.25, 0.0, 0.25, 0.65, 1.15]
SENSOR_RANGE = 145
SENSOR_NOISE = 0.02
STEER_RESPONSE = 0.25
CAR_LENGTH, CAR_WIDTH = 22, 13
MAX_SPEED = 3.8
TARGET_LAPS = 3             # a run ends successfully after this many laps
LAP_POINTS = 1000           # fitness for one full lap, so scores compare across track lengths
TIME_BONUS = 1.0            # fitness per frame left on the clock when a car finishes

BG = (17, 21, 29); ROAD = (48, 55, 66); EDGE = (225, 230, 237)
WHITE = (240, 243, 248); MUTED = (155, 166, 182); GREEN = (70, 220, 145)
RED = (255, 95, 105); BLUE = (90, 175, 255); YELLOW = (255, 210, 90)
RAY_STEPS = np.arange(0, SENSOR_RANGE + 1, 3)


def centerline(points, spacing=2.2):
    # Catmull-Rom spline through the control points, resampled to roughly even spacing.
    P = np.array(points, float); k = len(P); t = np.linspace(0, 1, 200, endpoint=False)[:, None]
    pts = np.vstack([0.5*(2*P[i] + (P[(i+1)%k]-P[i-1])*t + (2*P[i-1]-5*P[i]+4*P[(i+1)%k]-P[(i+2)%k])*t**2
                          + (3*P[i]-P[i-1]-3*P[(i+1)%k]+P[(i+2)%k])*t**3) for i in range(k)])
    closed = np.vstack([pts, pts[:1]])
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(closed, axis=0).T))])
    u = np.linspace(0, s[-1], int(s[-1]/spacing), endpoint=False)
    return np.column_stack((np.interp(u, s, closed[:, 0]), np.interp(u, s, closed[:, 1])))


class Track:
    def __init__(self, points, name):
        self.name = name
        self.centerline = centerline(points)
        self.length = len(self.centerline) * 2.2
        # Time limit scales with track length: 3 laps at top speed, plus 50% slack.
        self.max_frames = int(TARGET_LAPS * self.length / MAX_SPEED * 1.5)
        # Boolean (W, H) grid: True where the sensors see road.
        surf = pygame.Surface((W, H)); surf.fill((0, 0, 0))
        for p in self.centerline: pygame.draw.circle(surf, (255, 255, 255), p, ROAD_WIDTH/2 - 2)
        self.mask = pygame.surfarray.array_red(surf) > 0
        self._surface = None

    def nearest(self, x, y):
        # Centerline samples are dense enough for this small real-time simulation.
        d2 = (self.centerline[:, 0]-x)**2 + (self.centerline[:, 1]-y)**2
        idx = int(np.argmin(d2))
        return idx, math.sqrt(float(d2[idx]))

    def ray_distances(self, x, y, angle):
        # March all sensor rays at once over the precomputed road mask; off-screen counts as off-road.
        angles = angle + np.array(SENSOR_ANGLES)[:, None]
        px = (x + np.cos(angles)*RAY_STEPS).astype(int); py = (y + np.sin(angles)*RAY_STEPS).astype(int)
        inside = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        on_road = np.zeros_like(inside); on_road[inside] = self.mask[px[inside], py[inside]]
        hit = ~on_road
        dist = np.where(hit.any(axis=1), RAY_STEPS[hit.argmax(axis=1)], SENSOR_RANGE) / SENSOR_RANGE
        return dist.tolist()

    def draw(self, screen):
        # Rendered once: edge-coloured band with the road painted over it, leaving a thin border.
        if self._surface is None:
            self._surface = pygame.Surface((W, H)); self._surface.fill(BG)
            for p in self.centerline: pygame.draw.circle(self._surface, EDGE, p, ROAD_WIDTH/2 + 2)
            for p in self.centerline: pygame.draw.circle(self._surface, ROAD, p, ROAD_WIDTH/2)
            # Start/finish stripe across the road at the first centerline sample.
            (sx, sy), (nx, ny) = self.centerline[0], self.centerline[1]
            d = math.hypot(nx-sx, ny-sy); px, py = -(ny-sy)/d, (nx-sx)/d
            half = ROAD_WIDTH/2 - 5
            pygame.draw.line(self._surface, YELLOW, (sx-px*half, sy-py*half), (sx+px*half, sy+py*half), 4)
        screen.blit(self._surface, (0, 0))


def track_problem(cl):
    # Returns why a generated centerline is unusable, or None if it is a valid track.
    pad = ROAD_WIDTH/2 + 6
    if cl[:, 0].min() < pad or cl[:, 0].max() > MAP_W-pad or cl[:, 1].min() < pad or cl[:, 1].max() > H-pad: return 'bounds'
    if not 2200 < len(cl)*2.2 < 3600: return 'length'
    d1 = np.roll(cl, -1, 0) - np.roll(cl, 1, 0); d2 = np.roll(cl, -1, 0) - 2*cl + np.roll(cl, 1, 0)
    curvature = np.abs(d1[:, 0]*d2[:, 1] - d1[:, 1]*d2[:, 0]) / np.hypot(*d1.T)**3 * 4
    if 1/curvature.max() < 24: return 'too tight'
    # Separate parts of the loop must not touch: compare every sample with every 4th, skipping near neighbours.
    j = np.arange(0, len(cl), 4)
    dist = np.hypot(cl[:, None, 0]-cl[None, j, 0], cl[:, None, 1]-cl[None, j, 1])
    gap = np.abs(np.arange(len(cl))[:, None] - j[None, :]); gap = np.minimum(gap, len(cl)-gap)
    dist[gap < 120] = np.inf
    if dist.min() < ROAD_WIDTH + 20: return 'overlap'
    return None


def random_track(seed):
    # Star-shaped loop: random radii at jittered angles, stretched to fill the play area.
    rng = random.Random(seed)
    while True:
        n = rng.randint(10, 16)
        angles = [2*math.pi*(i + rng.uniform(-0.3, 0.3))/n for i in range(n)]
        # Built at roughly screen scale so the spline is sampled densely enough to measure.
        pts = np.array([(math.cos(a)*r, math.sin(a)*r) for a, r in ((a, rng.uniform(110, 370)) for a in angles)])
        if rng.random() < 0.5: pts = pts[::-1]          # drive both clockwise and anticlockwise
        cl = centerline(pts); lo, hi = cl.min(0), cl.max(0); pad = ROAD_WIDTH/2 + 10
        pts[:, 0] = pad + (pts[:, 0]-lo[0]) / (hi[0]-lo[0]) * (MAP_W - 2*pad)
        pts[:, 1] = pad + (pts[:, 1]-lo[1]) / (hi[1]-lo[1]) * (H - 2*pad)
        if track_problem(centerline(pts)) is None:
            return Track(pts, f'random #{seed}')


class Car:
    def __init__(self, genome_id, genome, net, track):
        self.genome_id, self.genome, self.net, self.track = genome_id, genome, net, track
        (self.x, self.y), (nx, ny) = track.centerline[0], track.centerline[1]
        self.angle = math.atan2(ny-self.y, nx-self.x)
        self.speed = 1.8
        self.steer = 0.0
        self.alive = True
        self.age = 0
        self.last_idx, _ = track.nearest(self.x, self.y)
        self.progress_total = 0.0
        self.sensors = []
        self.fitness = 0.0
        self.finished = False
        self.color = (80 + (genome_id*37)%150, 100 + (genome_id*53)%130, 150 + (genome_id*19)%100)

    def corners(self):
        points=[]
        for dx,dy in [(-CAR_LENGTH/2,-CAR_WIDTH/2),(CAR_LENGTH/2,-CAR_WIDTH/2),(CAR_LENGTH/2,CAR_WIDTH/2),(-CAR_LENGTH/2,CAR_WIDTH/2)]:
            points.append((self.x + dx*math.cos(self.angle)-dy*math.sin(self.angle), self.y + dx*math.sin(self.angle)+dy*math.cos(self.angle)))
        return points

    def update(self):
        if not self.alive: return
        samples = len(self.track.centerline)
        idx, offset = self.track.nearest(self.x, self.y)
        if offset > ROAD_WIDTH/2 - 7:
            self.alive = False
            return
        # Signed progress on the closed loop; ignore small backwards jitter.
        delta = idx - self.last_idx
        if delta < -samples/2: delta += samples
        elif delta > samples/2: delta -= samples
        self.progress_total += max(-2, min(4, delta))
        self.last_idx = idx
        self.sensors = self.track.ray_distances(self.x, self.y, self.angle)
        # Noisy readings so the network can't rely on perfectly clean distances.
        noisy = [min(1.0, max(0.0, s + random.gauss(0, SENSOR_NOISE))) for s in self.sensors]
        outputs = self.net.activate(noisy + [self.speed/MAX_SPEED])
        steering = max(-1.0, min(1.0, outputs[0]))
        throttle = outputs[1] if len(outputs)>1 else 0.5
        # The wheel turns gradually toward the requested angle instead of snapping to it.
        self.steer += (steering - self.steer) * STEER_RESPONSE
        self.angle += self.steer * 0.065
        self.speed = max(1.25, min(MAX_SPEED, self.speed + (throttle-0.35)*0.10))
        self.x += math.cos(self.angle)*self.speed
        self.y += math.sin(self.angle)*self.speed
        self.age += 1
        self.fitness = max(self.fitness, LAP_POINTS*self.progress_total/samples + self.age*0.01)
        if self.progress_total >= TARGET_LAPS*samples:
            # Finished: reward the frames left on the clock so faster drivers keep scoring higher.
            self.finished = True; self.alive = False
            self.fitness += TIME_BONUS*(self.track.max_frames - self.age)
        elif self.age >= self.track.max_frames: self.alive = False

    @property
    def laps(self):
        return max(0, int(self.progress_total // len(self.track.centerline)))

    def result(self):
        # Short human-readable outcome of a finished run.
        if self.finished: return f'{self.age/60:.1f}s'
        return f'{"crash" if self.age < self.track.max_frames else "timeout"} lap {self.laps+1}'

    def draw(self, screen, show_sensors=False):
        if not self.alive: return
        pygame.draw.polygon(screen, self.color, self.corners())
        nose=(self.x+math.cos(self.angle)*CAR_LENGTH/2, self.y+math.sin(self.angle)*CAR_LENGTH/2)
        pygame.draw.circle(screen, WHITE, (int(nose[0]),int(nose[1])), 2)
        if show_sensors:
            for a, val in zip(SENSOR_ANGLES, self.sensors):
                end=(self.x+math.cos(self.angle+a)*val*SENSOR_RANGE, self.y+math.sin(self.angle+a)*val*SENSOR_RANGE)
                pygame.draw.line(screen, GREEN, (self.x,self.y), end, 1)
                pygame.draw.circle(screen, YELLOW, (int(end[0]),int(end[1])), 2)


def draw_network(screen, genome, config, rect, font):
    x0,y0,w,h=rect
    pygame.draw.rect(screen, (24,29,39), rect, border_radius=8)
    pygame.draw.rect(screen, (55,64,78), rect, 1, border_radius=8)
    title=font.render('BEST GENOME • NETWORK', True, WHITE); screen.blit(title,(x0+12,y0+10))
    input_keys=list(config.genome_config.input_keys); output_keys=list(config.genome_config.output_keys)
    nodes=set(input_keys+output_keys+list(genome.nodes.keys()))
    enabled=[c for c in genome.connections.values() if c.enabled]
    for c in enabled: nodes.update(c.key)
    hidden=[n for n in nodes if n not in input_keys and n not in output_keys]
    layers=[input_keys, sorted(hidden), output_keys]
    usable_top=y0+42; usable_h=h-56
    positions={}
    for li,layer in enumerate(layers):
        if not layer: continue
        xx=x0+25+(w-50)*li/2
        for j,node in enumerate(layer):
            yy=usable_top+usable_h*(j+1)/(len(layer)+1)
            positions[node]=(xx,yy)
    for c in enabled:
        if c.key[0] in positions and c.key[1] in positions:
            a,b=positions[c.key[0]],positions[c.key[1]]
            col=BLUE if c.weight>=0 else RED
            pygame.draw.line(screen,col,a,b,max(1,min(3,int(abs(c.weight)*1.3))))
    for node,pos in positions.items():
        col=GREEN if node in input_keys else (YELLOW if node in output_keys else BLUE)
        pygame.draw.circle(screen,col,(int(pos[0]),int(pos[1])),7)
        pygame.draw.circle(screen,(15,20,28),(int(pos[0]),int(pos[1])),3)


def save_stats(history):
    with open(OUT/'fitness_history.csv','w',newline='') as f:
        writer=csv.writer(f); writer.writerow(['generation','best_fitness','mean_fitness','test_fitness','test_result']); writer.writerows(history)
    if history:
        plt.figure(figsize=(8,3.2)); xs=[r[0] for r in history]
        plt.plot(xs,[r[1] for r in history],label='Best (training tracks)'); plt.plot(xs,[r[2] for r in history],label='Mean (training tracks)')
        plt.plot(xs,[r[3] for r in history],label='Best genome on test track')
        plt.xlabel('Generation'); plt.ylabel('Fitness'); plt.grid(True,alpha=.25); plt.legend(); plt.tight_layout()
        plt.savefig(OUT/'fitness_history.png',dpi=130); plt.close()


def load_history(path):
    # Older history files have no test columns; fall back to the training best for those rows.
    with open(path,newline='') as f: rows=list(csv.reader(f))[1:]
    return [(int(r[0]),float(r[1]),float(r[2]),float(r[3]) if len(r)>3 else float(r[1]),r[4] if len(r)>4 else '') for r in rows]


def handle_events(app, allow_save=True):
    for event in pygame.event.get():
        if event.type==pygame.QUIT: app.running=False
        elif event.type==pygame.KEYDOWN:
            if event.key==pygame.K_ESCAPE: app.running=False
            elif event.key==pygame.K_SPACE: app.paused=not app.paused
            elif event.key==pygame.K_s and allow_save: app.save_checkpoint()
            elif event.key==pygame.K_n: app.show_sensors=not app.show_sensors


def drive(car):
    # Run one car to the end without drawing (used for the test-track check).
    while car.alive: car.update()
    return car


def eval_genomes(genomes, config, app):
    app.generation += 1
    nets={gid: neat.nn.FeedForwardNetwork.create(genome,config) for gid,genome in genomes}
    totals={gid: 0.0 for gid,_ in genomes}; finished=[]
    # Every genome drives the same fresh random tracks, so it can't just memorise one layout.
    for i in range(TRACKS_PER_GEN):
        app.track=random_track(app.rng.randrange(10**6)); app.track_info=f'{i+1}/{TRACKS_PER_GEN} {app.track.name}'
        cars=[Car(gid,genome,nets[gid],app.track) for gid,genome in genomes]
        while any(c.alive for c in cars) and app.running:
            app.clock.tick(FPS)
            handle_events(app)
            if not app.paused:
                for c in cars: c.update()
            app.draw(cars,config)
        if not app.running: raise KeyboardInterrupt
        for c in cars: totals[c.genome_id]+=c.fitness
        finished+=[c.age/60 for c in cars if c.finished]
    for gid,genome in genomes: genome.fitness=totals[gid]/TRACKS_PER_GEN
    best_gid,best=max(genomes,key=lambda g:g[1].fitness)
    mean=float(np.mean([g.fitness for _,g in genomes]))
    # Generalisation check on the hand-made track that training never sees.
    test=drive(Car(best_gid,best,nets[best_gid],app.test_track))
    app.last_test=test.result()
    if app.best_genome is None or test.fitness>app.best_test:
        app.best_test=test.fitness; app.best_test_result=test.result(); app.best_genome=best
        with open(OUT/'best_genome.pkl','wb') as f: pickle.dump(best,f)
    app.history.append((app.generation,float(best.fitness),mean,float(test.fitness),test.result()))
    save_stats(app.history)
    print(f"Generation {app.generation:4d} | best {best.fitness:7.1f} | mean {mean:7.1f} | finished {len(finished):3d}/{len(genomes)*TRACKS_PER_GEN}"
          f" | test track {test.fitness:7.1f} ({test.result()})")


class App:
    def __init__(self, config_path):
        pygame.init(); pygame.display.set_caption('NeuroEvolution • Self-Driving Lab')
        self.screen=pygame.display.set_mode((W,H)); self.clock=pygame.time.Clock(); self.font=pygame.font.SysFont('consolas',16); self.big=pygame.font.SysFont('consolas',22,bold=True)
        self.running=True; self.paused=False; self.show_sensors=False; self.generation=0; self.best_genome=None; self.history=[]
        self.config_path=config_path; self.replay=False; self.rng=random.Random()
        self.test_track=Track(TEST_TRACK_POINTS,'test track'); self.track=self.test_track; self.track_info=''
        self.best_test=-math.inf; self.best_test_result='-'; self.last_test='-'; self.best_time=None

    def save_checkpoint(self):
        with open(OUT/'neat-checkpoint.pkl','wb') as f: pickle.dump({'generation':self.generation,'history':self.history,'best_test':self.best_test,'best_genome':self.best_genome},f)
        print('Saved dashboard checkpoint data. NEAT population checkpoints are saved separately by neat.Checkpointer.')

    def draw(self,cars,config):
        self.screen.fill(BG); self.track.draw(self.screen)
        for c in cars: c.draw(self.screen,self.show_sensors and c.alive)
        panel=pygame.Rect(W-285,515,265,220); pygame.draw.rect(self.screen,(24,29,39),panel,border_radius=10); pygame.draw.rect(self.screen,(55,64,78),panel,1,border_radius=10)
        if self.replay:
            car=cars[0]; best_time=f'{self.best_time:.1f}s' if self.best_time is not None else '-'
            lines=[('REPLAY • BEST GENOME',WHITE),(f'Track: {self.track.name}',WHITE),(f'Run: {self.generation}',WHITE),(f'Lap: {min(car.laps+1,TARGET_LAPS)}/{TARGET_LAPS}',GREEN),(f'Time: {car.age/60:.1f}s',YELLOW),(f'Speed: {car.speed:.2f}',BLUE),(f'Best {TARGET_LAPS}-lap time: {best_time}',BLUE),('SPACE pause  |  N sensors',MUTED),('ESC quit',MUTED)]
        else:
            leader=max(cars,key=lambda c:c.fitness).fitness if cars else 0
            lines=[('NEUROEVOLUTION LAB',WHITE),(f'Generation: {self.generation}',WHITE),(f'Track {self.track_info}',WHITE),(f'Cars alive: {sum(c.alive for c in cars)}/{len(cars)}',GREEN),(f'Leader: {leader:.0f}',YELLOW),(f'Last test: {self.last_test}',BLUE),(f'Best test: {self.best_test_result}',BLUE),('SPACE pause  |  S save',MUTED),('N sensors  |  ESC quit',MUTED)]
        for i,(txt,col) in enumerate(lines): self.screen.blit(self.font.render(txt,True,col),(panel.x+12,panel.y+10+i*22))
        # Show the live leader's network while training, otherwise the saved best.
        genome=cars[0].genome if self.replay else (max(cars,key=lambda c:c.fitness).genome if cars else self.best_genome)
        if genome is not None: draw_network(self.screen,genome,config,(W-285,18,265,290),self.font)
        # Compact fitness plot: best on training tracks (green) and on the unseen test track (blue).
        plot=pygame.Rect(W-285,322,265,180); pygame.draw.rect(self.screen,(24,29,39),plot,border_radius=8); pygame.draw.rect(self.screen,(55,64,78),plot,1,border_radius=8)
        self.screen.blit(self.font.render('FITNESS HISTORY',True,WHITE),(plot.x+12,plot.y+10))
        if self.history:
            series=[([r[1] for r in self.history],GREEN),([r[3] for r in self.history],BLUE)]
            allv=series[0][0]+series[1][0]; lo=min(allv); hi=max(allv); span=max(1,hi-lo)
            for vals,col in series:
                pts=[(int(plot.x+12+(plot.w-24)*(i/max(1,len(vals)-1))), int(plot.y+plot.h-18-(plot.h-52)*(v-lo)/span)) for i,v in enumerate(vals)]
                if len(pts)>1: pygame.draw.lines(self.screen,col,False,pts,2)
            label=f'test {max(series[1][0]):.0f}'
            self.screen.blit(self.font.render(label,True,BLUE),(plot.right-12-self.font.size(label)[0],plot.y+10))
        if self.paused: self.screen.blit(self.big.render('PAUSED',True,YELLOW),(W//2-45,20))
        pygame.display.flip()


def replay(genome_path, config, track_choice):
    # Watch a saved genome drive on its own; the run restarts whenever the car finishes or crashes.
    with open(genome_path,'rb') as f: genome=pickle.load(f)
    app=App(ROOT/'neat_config.txt'); app.replay=True; app.show_sensors=True; app.best_genome=genome
    net=neat.nn.FeedForwardNetwork.create(genome,config)
    history_path=Path(genome_path).parent/'fitness_history.csv'
    if history_path.exists(): app.history=load_history(history_path)
    print(f'Replaying {genome_path} on {track_choice} track(s)')
    while app.running:
        app.generation += 1
        if track_choice=='random': app.track=random_track(app.rng.randrange(10**6)); app.best_time=None
        elif track_choice!='test': app.track=random_track(int(track_choice))
        car=Car(0,genome,net,app.track); car.color=YELLOW
        while car.alive and app.running:
            app.clock.tick(FPS)
            handle_events(app, allow_save=False)
            if not app.paused: car.update()
            app.draw([car],config)
        if not app.running: break
        if car.finished: app.best_time=min(app.best_time or car.age/60, car.age/60)
        print(f'Run {app.generation} on {app.track.name}: {"finished " + str(TARGET_LAPS) + " laps in " if car.finished else ""}{car.result()}')
    pygame.quit()


def build_config():
    config_path=ROOT/'neat_config.txt'
    if not config_path.exists(): raise FileNotFoundError(f'Missing config file: {config_path}')
    return neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, str(config_path))


def main():
    parser=argparse.ArgumentParser(description='Neuroevolution self-driving simulation')
    parser.add_argument('--resume', type=str, default=None, help='Path to a NEAT checkpoint file in training/')
    parser.add_argument('--replay', type=str, nargs='?', const=str(OUT/'best_genome.pkl'), default=None,
                        help='Watch a saved genome drive (default: training/best_genome.pkl)')
    parser.add_argument('--track', type=str, default='test',
                        help="Replay track: 'test' (default), 'random' (new track every run) or a number for a specific random track")
    args=parser.parse_args()
    if args.track not in ('test','random') and not args.track.isdigit(): parser.error("--track must be 'test', 'random' or a number")
    config=build_config()
    if args.replay: return replay(args.replay, config, args.track)
    app=App(ROOT/'neat_config.txt')
    if args.resume:
        pop=neat.Checkpointer.restore_checkpoint(args.resume)
        print(f'Resumed NEAT population from {args.resume}')
    else:
        pop=neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True)); pop.add_reporter(neat.StatisticsReporter())
    pop.add_reporter(neat.Checkpointer(generation_interval=5, time_interval_seconds=300, filename_prefix=str(OUT/'neat-checkpoint-')))
    try: pop.run(lambda genomes,cfg: eval_genomes(genomes,cfg,app), 1000)
    except KeyboardInterrupt: print('\nSimulation stopped. Checkpoints and training outputs are in ./training')
    finally:
        save_stats(app.history); pygame.quit()

if __name__=='__main__': main()
