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
FPS = 60
POPULATION = 60
ROAD_WIDTH = 105
# Control points of the closed track (screen coords); a Catmull-Rom spline runs through them.
TRACK_POINTS = [(330,80),(560,75),(720,105),(770,215),(700,305),(570,315),(525,400),(620,470),(750,520),(755,630),
                (620,690),(440,670),(345,585),(230,650),(110,610),(80,475),(170,385),(320,360),(330,270),(230,240),(170,175),(220,95)]
SENSOR_ANGLES = [-1.15, -0.65, -0.25, 0.0, 0.25, 0.65, 1.15]
SENSOR_RANGE = 145
SENSOR_NOISE = 0.02
STEER_RESPONSE = 0.25
CAR_LENGTH, CAR_WIDTH = 22, 13
MAX_SPEED = 3.8

BG = (17, 21, 29); ROAD = (48, 55, 66); EDGE = (225, 230, 237)
WHITE = (240, 243, 248); MUTED = (155, 166, 182); GREEN = (70, 220, 145)
RED = (255, 95, 105); BLUE = (90, 175, 255); YELLOW = (255, 210, 90)


def centerline(points=TRACK_POINTS, spacing=2.2):
    # Catmull-Rom spline through the control points, resampled to roughly even spacing.
    P = np.array(points, float); k = len(P); t = np.linspace(0, 1, 200, endpoint=False)[:, None]
    pts = np.vstack([0.5*(2*P[i] + (P[(i+1)%k]-P[i-1])*t + (2*P[i-1]-5*P[i]+4*P[(i+1)%k]-P[(i+2)%k])*t**2
                          + (3*P[i]-P[i-1]-3*P[(i+1)%k]+P[(i+2)%k])*t**3) for i in range(k)])
    closed = np.vstack([pts, pts[:1]])
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(closed, axis=0).T))])
    u = np.linspace(0, s[-1], int(s[-1]/spacing), endpoint=False)
    return np.column_stack((np.interp(u, s, closed[:, 0]), np.interp(u, s, closed[:, 1])))
CENTERLINE = centerline()


def build_road_mask(radius):
    # Boolean (W, H) grid: True where a point is within `radius` of the centerline.
    surf = pygame.Surface((W, H)); surf.fill((0, 0, 0))
    for p in CENTERLINE: pygame.draw.circle(surf, (255, 255, 255), p, radius)
    return pygame.surfarray.array_red(surf) > 0
SENSOR_MASK = build_road_mask(ROAD_WIDTH/2 - 2)
RAY_STEPS = np.arange(0, SENSOR_RANGE + 1, 3)


def nearest_track(x, y):
    # Centerline samples are dense enough for this small real-time simulation.
    d2 = (CENTERLINE[:, 0]-x)**2 + (CENTERLINE[:, 1]-y)**2
    idx = int(np.argmin(d2))
    return idx, math.sqrt(float(d2[idx]))


def ray_distances(x, y, angle):
    # March all sensor rays at once over the precomputed road mask; off-screen counts as off-road.
    angles = angle + np.array(SENSOR_ANGLES)[:, None]
    px = (x + np.cos(angles)*RAY_STEPS).astype(int); py = (y + np.sin(angles)*RAY_STEPS).astype(int)
    inside = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    on_road = np.zeros_like(inside); on_road[inside] = SENSOR_MASK[px[inside], py[inside]]
    hit = ~on_road
    dist = np.where(hit.any(axis=1), RAY_STEPS[hit.argmax(axis=1)], SENSOR_RANGE) / SENSOR_RANGE
    return dist.tolist()


class Car:
    def __init__(self, genome_id, genome, net):
        self.genome_id, self.genome, self.net = genome_id, genome, net
        (self.x, self.y), (nx, ny) = CENTERLINE[0], CENTERLINE[1]
        self.angle = math.atan2(ny-self.y, nx-self.x)
        self.speed = 1.8
        self.steer = 0.0
        self.alive = True
        self.age = 0
        self.max_progress = 0
        self.last_idx, _ = nearest_track(self.x, self.y)
        self.progress_total = 0.0
        self.sensors = []
        self.fitness = 0.0
        self.color = (80 + (genome_id*37)%150, 100 + (genome_id*53)%130, 150 + (genome_id*19)%100)

    def corners(self):
        points=[]
        for dx,dy in [(-CAR_LENGTH/2,-CAR_WIDTH/2),(CAR_LENGTH/2,-CAR_WIDTH/2),(CAR_LENGTH/2,CAR_WIDTH/2),(-CAR_LENGTH/2,CAR_WIDTH/2)]:
            points.append((self.x + dx*math.cos(self.angle)-dy*math.sin(self.angle), self.y + dx*math.sin(self.angle)+dy*math.cos(self.angle)))
        return points

    def update(self):
        if not self.alive: return
        idx, offset = nearest_track(self.x, self.y)
        if offset > ROAD_WIDTH/2 - 7:
            self.alive = False
            self.fitness = max(self.fitness, self.progress_total + self.age*0.01)
            return
        # Signed progress on the closed loop; ignore small backwards jitter.
        delta = idx - self.last_idx
        if delta < -len(CENTERLINE)/2: delta += len(CENTERLINE)
        elif delta > len(CENTERLINE)/2: delta -= len(CENTERLINE)
        self.progress_total += max(-2, min(4, delta))
        self.last_idx = idx
        self.max_progress = max(self.max_progress, self.progress_total)
        self.sensors = ray_distances(self.x, self.y, self.angle)
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
        self.fitness = max(self.fitness, self.progress_total + self.age*0.01)
        if self.age > 60*45: self.alive = False

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


_TRACK_SURFACE = None


def draw_track(screen):
    # Rendered once: edge-coloured band with the road painted over it, leaving a thin border.
    global _TRACK_SURFACE
    if _TRACK_SURFACE is None:
        _TRACK_SURFACE = pygame.Surface((W, H)); _TRACK_SURFACE.fill(BG)
        for p in CENTERLINE: pygame.draw.circle(_TRACK_SURFACE, EDGE, p, ROAD_WIDTH/2 + 2)
        for p in CENTERLINE: pygame.draw.circle(_TRACK_SURFACE, ROAD, p, ROAD_WIDTH/2)
        # Start/finish stripe across the road at the first centerline sample.
        (sx, sy), (nx, ny) = CENTERLINE[0], CENTERLINE[1]
        d = math.hypot(nx-sx, ny-sy); px, py = -(ny-sy)/d, (nx-sx)/d
        half = ROAD_WIDTH/2 - 5
        pygame.draw.line(_TRACK_SURFACE, YELLOW, (sx-px*half, sy-py*half), (sx+px*half, sy+py*half), 4)
    screen.blit(_TRACK_SURFACE, (0, 0))


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
        writer=csv.writer(f); writer.writerow(['generation','best_fitness','mean_fitness']); writer.writerows(history)
    if history:
        plt.figure(figsize=(8,3.2)); xs=[r[0] for r in history]
        plt.plot(xs,[r[1] for r in history],label='Best fitness'); plt.plot(xs,[r[2] for r in history],label='Mean fitness')
        plt.xlabel('Generation'); plt.ylabel('Fitness'); plt.grid(True,alpha=.25); plt.legend(); plt.tight_layout()
        plt.savefig(OUT/'fitness_history.png',dpi=130); plt.close()


def eval_genomes(genomes, config, app):
    cars=[]
    for gid, genome in genomes:
        genome.fitness=0.0
        cars.append(Car(gid,genome,neat.nn.FeedForwardNetwork.create(genome,config)))
    app.cars=cars
    app.generation += 1
    app.generation_best=0.0
    while any(c.alive for c in cars) and app.running:
        app.clock.tick(FPS)
        for event in pygame.event.get():
            if event.type==pygame.QUIT: app.running=False
            elif event.type==pygame.KEYDOWN:
                if event.key==pygame.K_ESCAPE: app.running=False
                elif event.key==pygame.K_SPACE: app.paused=not app.paused
                elif event.key==pygame.K_s: app.save_checkpoint()
                elif event.key==pygame.K_n: app.show_sensors=not app.show_sensors
        if app.paused:
            app.draw(cars, config); continue
        for c in cars: c.update()
        for c in cars: c.genome.fitness=max(c.genome.fitness,c.fitness)
        best=max(cars,key=lambda c:c.fitness)
        app.generation_best=max(app.generation_best,best.fitness)
        if app.best_genome is None or best.fitness>=app.best_ever:
            app.best_genome=best.genome; app.best_ever=best.fitness
        app.draw(cars,config)
    if cars:
        best=max(cars,key=lambda c:c.fitness)
        mean=float(np.mean([c.fitness for c in cars]))
        app.history.append((app.generation,float(best.fitness),mean))
        with open(OUT/'best_genome.pkl','wb') as f: pickle.dump(app.best_genome,f)
        save_stats(app.history)
        print(f"Generation {app.generation:4d} | best {best.fitness:8.1f} | mean {mean:8.1f} | alive {sum(c.alive for c in cars)}")
    if not app.running: raise KeyboardInterrupt


class App:
    def __init__(self, config_path):
        pygame.init(); pygame.display.set_caption('NeuroEvolution • Self-Driving Lab')
        self.screen=pygame.display.set_mode((W,H)); self.clock=pygame.time.Clock(); self.font=pygame.font.SysFont('consolas',16); self.big=pygame.font.SysFont('consolas',22,bold=True)
        self.running=True; self.paused=False; self.show_sensors=False; self.generation=0; self.generation_best=0; self.best_ever=0; self.best_genome=None; self.cars=[]; self.history=[]
        self.config_path=config_path

    def save_checkpoint(self):
        with open(OUT/'neat-checkpoint.pkl','wb') as f: pickle.dump({'generation':self.generation,'history':self.history,'best_ever':self.best_ever,'best_genome':self.best_genome},f)
        print('Saved dashboard checkpoint data. NEAT population checkpoints are saved separately by neat.Checkpointer.')

    def draw(self,cars,config):
        self.screen.fill(BG); draw_track(self.screen)
        for c in cars: c.draw(self.screen,self.show_sensors and c.alive)
        panel=pygame.Rect(W-285,515,265,170); pygame.draw.rect(self.screen,(24,29,39),panel,border_radius=10); pygame.draw.rect(self.screen,(55,64,78),panel,1,border_radius=10)
        lines=[('NEUROEVOLUTION LAB',WHITE), (f'Generation: {self.generation}',WHITE),(f'Cars alive: {sum(c.alive for c in cars)}/{len(cars)}',GREEN),(f'Generation best: {self.generation_best:.1f}',YELLOW),(f'Best ever: {self.best_ever:.1f}',BLUE),('SPACE pause  |  S save',MUTED),('N sensors  |  ESC quit',MUTED)]
        for i,(txt,col) in enumerate(lines): self.screen.blit(self.font.render(txt,True,col),(panel.x+12,panel.y+10+i*22))
        genome=self.best_genome
        if cars:
            livebest=max(cars,key=lambda c:c.fitness)
            if genome is None or livebest.fitness>=self.generation_best: genome=livebest.genome
        if genome is not None: draw_network(self.screen,genome,config,(W-285,18,265,290),self.font)
        # Compact fitness plot from saved history.
        plot=pygame.Rect(W-285,322,265,180); pygame.draw.rect(self.screen,(24,29,39),plot,border_radius=8); pygame.draw.rect(self.screen,(55,64,78),plot,1,border_radius=8)
        self.screen.blit(self.font.render('FITNESS HISTORY',True,WHITE),(plot.x+12,plot.y+10))
        if self.history:
            vals=[r[1] for r in self.history]; lo=min(vals); hi=max(vals); span=max(1,hi-lo)
            pts=[]
            for i,v in enumerate(vals):
                px=plot.x+12+(plot.w-24)*(i/max(1,len(vals)-1)); py=plot.y+plot.h-18-(plot.h-52)*(v-lo)/span; pts.append((int(px),int(py)))
            if len(pts)>1: pygame.draw.lines(self.screen,GREEN,False,pts,2)
            self.screen.blit(self.font.render(f'best {hi:.1f}',True,GREEN),(plot.x+12,plot.y+32))
        self.screen.blit(self.font.render('Fitness rewards forward progress and survival.',True,MUTED),(18,H-28))
        if self.paused: self.screen.blit(self.big.render('PAUSED',True,YELLOW),(W//2-45,20))
        pygame.display.flip()


def build_config():
    config_path=ROOT/'neat_config.txt'
    if not config_path.exists(): raise FileNotFoundError(f'Missing config file: {config_path}')
    return neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, str(config_path))


def main():
    parser=argparse.ArgumentParser(description='Neuroevolution self-driving simulation')
    parser.add_argument('--resume', type=str, default=None, help='Path to a NEAT checkpoint file in training/')
    args=parser.parse_args()
    config=build_config(); app=App(ROOT/'neat_config.txt')
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
