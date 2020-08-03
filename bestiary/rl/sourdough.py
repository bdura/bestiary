from typing import Optional

import gym
import numpy as np
from loguru import logger
from sklearn.neighbors import KDTree
from tqdm import tqdm

from bestiary.rl.utils import select_along, stratified_index, MAX_INT


class FoodSource(object):

    def __init__(self, size: int = 100, energy: float = 1, bandwidth: float = 2., speed: float = 100.):
        self.size_ = size
        self.energy = energy
        self.bandwidth = bandwidth

        self.alpha = np.random.uniform(0, np.pi)
        self.rho = .8 * size

        self.omega = speed / self.rho

    def position(self):
        return np.array([np.cos(self.alpha), np.sin(self.alpha)]) * self.rho

    def move(self):
        self.alpha += np.random.uniform(self.omega / 2, self.omega)

    def food(self, position, center=None):
        if center is None:
            center = self.position()
        center = center.reshape(1, -1)
        return self.energy * np.exp(- (np.linalg.norm(position - center, axis=1) / self.bandwidth) ** 2)


class Ecoli(object):

    def __init__(self, size: int = 10, n_max: int = 1000, energy_levels: int = 2,
                 max_food: float = .5, food_levels: int = 3, depletion_rate: float = .05,
                 mutation_rate: float = .1, bandwidth: float = 5., speed: float = .5,
                 record: bool = False, seed: Optional[int] = None):

        np.random.seed(seed)

        self.size_ = size

        self.depletion_rate = depletion_rate
        self.mutation_rate = mutation_rate

        self.position = np.random.uniform(-size, size, size=(n_max, 2))
        self.direction = np.random.uniform(0, 2 * np.pi, size=n_max)

        self.energy = np.ones(n_max) * .5

        self.epsilon = 0

        # s = (E, F, dF)
        self.q = np.random.randn(n_max, energy_levels, food_levels, 3, 2)

        self.energy_levels_ = np.linspace(0, .9, energy_levels)
        self.food_levels_ = np.linspace(0, max_food, food_levels)

        self.max_food = max_food

        self.level_ = np.zeros(n_max)

        self.alive = self.energy > 0

        self.food_source = FoodSource(size=size, energy=max_food * 1.1, bandwidth=bandwidth, speed=speed)

        self.history_ = []
        self.record = record

    @property
    def dead(self):
        return self.alive.sum() == 0

    def run(self, indices):

        if indices.sum() == 0:
            return

        alpha = self.direction[indices]
        u = np.array([np.cos(alpha), np.sin(alpha)]).T

        self.position[indices] += u * np.random.normal(1., 1e-1, size=len(u)).reshape(-1, 1)

    def tumble(self, indices):

        if indices.sum() == 0:
            return

        self.direction[indices] = np.random.uniform(0, 2 * np.pi, size=indices.sum())

    def deplete(self):
        self.energy -= self.depletion_rate
        self.alive = self.energy > 0

        self.level_[~self.alive] = 0

    def mutate(self):
        self.q[self.alive] += np.random.normal(0, self.mutation_rate, size=self.q[self.alive].shape)

    def conjugate(self):

        if self.alive.sum() <= 1:
            return []

        position = self.position[self.alive]
        q = self.q[self.alive]

        tree = KDTree(position)

        d, i = tree.query(position, k=2)
        distance, nn = d[:, 1], i[:, 1]

        conjugate = distance < 0.2
        # conjugate = np.logical_and(conjugate, np.random.rand(len(conjugate)) > .5)
        conjugate_from = nn[conjugate]

        q[conjugate] += np.random.rand(*q[conjugate].shape) * .1 * (q[conjugate_from] - q[conjugate])

        return stratified_index(self.alive, conjugate)

    def mitose(self):

        mitose = self.energy > 1
        dead = np.argsort(self.energy)[:mitose.sum()]
        dead = np.isin(np.arange(len(self.position)), dead)

        self.energy[mitose] /= 2
        self.energy[dead] = self.energy[mitose]

        self.q[dead] = self.q[mitose]
        self.level_[dead] = self.level_[mitose]

        self.position[dead] = self.position[mitose]
        self.direction[dead] = np.random.uniform(0, 2 * np.pi, size=dead.sum())

        self.alive = np.logical_or(self.alive, dead)

        n = mitose.sum()
        if n > 0:
            logger.info(f"{n} bacteria just mitosed")

        return mitose[self.alive]

    def discretise(self, energy, food):

        e = np.digitize(energy, self.energy_levels_) - 1
        f = np.digitize(food, self.food_levels_) - 1

        df = np.clip(f - self.level_[self.alive], -1, 1).astype(np.int) + 1
        self.level_[self.alive] = f

        return e, f, df

    def observe(self, food):
        energy = self.energy[self.alive]
        e, f, df = self.discretise(energy, food)

        return e, f, df

    def get_q(self, e, f, df):
        q = self.q[self.alive]

        q = select_along(q, e)
        q = select_along(q, f)
        q = select_along(q, df)

        return q

    def select_actions(self, e, f, df):
        q = self.get_q(e, f, df)

        actions = q.argmax(axis=1)

        if self.epsilon == 0:
            return actions

        choice = np.random.choice([0, 1], size=len(actions), p=[self.epsilon, 1 - self.epsilon])
        actions = choice * actions + (1 - choice) * np.random.choice(2, size=actions.shape)

        return actions

    def act(self, actions):

        run = stratified_index(self.alive, actions == 0)
        tumble = stratified_index(self.alive, actions == 1)

        self.run(run)
        self.tumble(tumble)

    def step(self, act=True):

        self.food_source.move()
        food = self.food_source.food(self.position)

        self.energy[self.alive] += food[self.alive]
        self.mitose()
        self.conjugate()
        self.deplete()

        self.energy = np.ones_like(self.energy) * .5

        observations = self.observe(food[self.alive])

        if act and self.alive.sum() > 0:
            self.mutate()
            actions = self.select_actions(*observations)
            self.act(actions)

        if self.record:
            record = dict(
                alive=int(self.alive.sum()),
                energy=self.energy[self.alive],
                q=self.q[self.alive],
                position=self.position[self.alive],
                direction=self.direction[self.alive],
                food=self.food_source.position(),
            )

            self.history_.append(record)

        return observations

    def simulate(self, generations=1000, verbose=False):
        iterator = range(generations)

        if verbose:
            iterator = tqdm(iterator, ascii=True, ncols=100)

        for _ in iterator:
            self.step()

            if self.alive.sum() == 0:
                break


class SourdoughEnvironment(gym.Env):

    def render(self, mode='human'):
        pass

    def __init__(self, size: int = 100, energy_levels: int = 2, max_food: float = .5,
                 food_levels: int = 3, depletion_rate: float = .05,
                 mutation_rate: float = .1, bandwidth: float = 5., speed: float = .5,
                 max_steps: int = 200, seed=None):

        super(SourdoughEnvironment, self).__init__()

        np.random.seed(seed)

        # Gym specs
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.MultiDiscrete([energy_levels, food_levels, 3])

        self.kwargs = dict(size=size, n_max=1, energy_levels=energy_levels,
                           max_food=max_food, food_levels=food_levels, depletion_rate=depletion_rate,
                           mutation_rate=mutation_rate, bandwidth=bandwidth, speed=speed)

        self.ecoli = None
        self.done_ = True

        self.max_steps = max_steps
        self.steps = 0

    @staticmethod
    def unpack_observation(observations):
        e, f, df = observations
        return e[0], f[0], df[0]

    @staticmethod
    def pack_action(action):
        return np.array([action])

    def observe(self):
        obs = self.ecoli.step(act=False)
        if self.ecoli.dead:
            return None
        return self.unpack_observation(obs)

    def reset(self, difficulty=1.):
        self.ecoli = Ecoli(**self.kwargs, seed=np.random.randint(MAX_INT))
        self.ecoli.position = np.random.normal(
            self.ecoli.food_source.position(),
            self.ecoli.food_source.bandwidth * difficulty,
        ).reshape(1, 2)

        self.done_ = False
        self.steps = 0

        return self.observe()

    def step(self, action):

        self.ecoli.act(self.pack_action(action))
        obs = self.observe()

        self.steps += 1

        self.done_ = self.ecoli.dead or (0 < self.max_steps < self.steps)
        reward = 1

        if self.done_:
            obs = None
            reward = 0

        return obs, reward, self.done_, {}


class Bacterium(object):

    def __init__(self, lr: float = 1e-1, gamma: float = .99, genome: Optional[np.ndarray] = None,
                 q: Optional[np.ndarray] = None, energy_levels: int = 2,
                 food_levels: int = 3, depletion_rate: float = .05,
                 mutation_rate: float = .1):

        self.kwargs_ = locals()
        self.kwargs_.pop("genome")
        self.kwargs_.pop("q")

        if genome is None:
            # Energy, near-death, fitness, abundance, gregarious, reproduction
            genome = np.random.randn(5)

        self.genome = genome

        if q is None:
            # Energy, near-death, fitness, abundance, gregarious, reproduction
            q = np.random.randn(energy_levels, food_levels, 3, 2)

        self.q = q

        self.gamma = gamma
        self.lr = lr

        self.mutation_rate = mutation_rate
        self.depletion_rate = depletion_rate

    def conjugate(self, other):
        self.q += np.random.rand(*other.q.shape) * .1 * (other.q - self.q)

    def mutate(self):
        self.q += np.random.normal(0, self.mutation_rate, size=self.q.shape)


class RewardEcoli(Ecoli):

    def __init__(self, n_max: int = 1000, lr: float = 1e-1, gamma: float = .99, epsilon=.05, *args, **kwargs):

        super(RewardEcoli, self).__init__(*args, n_max=n_max, **kwargs)

        # Food, energy, near-death, fitness, abundance, gregarious, reproduction
        self.genome = np.random.randn(n_max, 7)

        self.gamma = gamma
        self.lr = lr

        self.e_ = np.zeros(n_max, dtype=np.int)
        self.f_ = np.zeros(n_max, dtype=np.int)
        self.df_ = np.zeros(n_max, dtype=np.int)
        self.a_ = np.zeros(n_max, dtype=np.int)

        self.learning = False

        self.epsilon = epsilon

    @staticmethod
    def near_death(energy):
        return energy < 5e-2

    @staticmethod
    def fitness(energy):
        return energy > .9

    def abundance(self, food):
        return food > .9 * self.max_food

    @staticmethod
    def gregarious(conjugation):
        return conjugation

    @staticmethod
    def reproduction(mitose):
        return mitose

    def reward(self, food, conjugation, mitose):
        energy = self.energy[self.alive]

        r = np.stack([
            food,
            energy,
            1 * self.near_death(energy),
            1 * self.fitness(energy),
            1 * self.abundance(food),
            1 * self.gregarious(conjugation),
            1 * self.reproduction(mitose),
            # np.ones(len(energy)),
        ], axis=1)

        logger.debug(r.shape)

        return np.sum(self.genome[self.alive] * r, axis=1)

    def mutate(self):
        self.genome += np.random.normal(0, self.mutation_rate, size=self.genome.shape)

    def mitose(self):

        mitose = self.energy > 1
        dead = np.argsort(self.energy)[:mitose.sum()]
        dead = np.isin(np.arange(len(self.position)), dead)

        self.energy[mitose] /= 2
        self.energy[dead] = self.energy[mitose]

        self.q[dead] = self.q[mitose]
        self.level_[dead] = self.level_[mitose]

        self.e_[dead] = self.e_[mitose]
        self.f_[dead] = self.f_[mitose]
        self.df_[dead] = self.df_[mitose]
        self.a_[dead] = self.a_[mitose]

        self.position[dead] = self.position[mitose]
        self.direction[dead] = np.random.uniform(0, 2 * np.pi, size=dead.sum())

        self.alive = np.logical_or(self.alive, dead)

        n = mitose.sum()
        if n > 0:
            logger.info(f"{n} bacteria just mitosed")

        return mitose

    def learn(self, new_state, reward, actions):
        students = np.arange(len(self.alive))[self.alive]

        energy_, food_, food_gradient_ = new_state

        if self.learning:

            s = self.e_[students], self.f_[students], self.df_[students]

            for s, e, f, df, a, e_, f_, df_, r in zip(students, *s, self.a_, *new_state, reward):
                self.q[s, e, f, df, a] += self.lr * (
                        (r + self.gamma * self.q[s, e_, f_, df_].max())
                        - self.q[s, e, f, df, a]
                )

        else:
            self.learning = True

        self.e_[self.alive] = energy_
        self.f_[self.alive] = food_
        self.df_[self.alive] = food_gradient_
        self.a_[self.alive] = actions

    def step(self, **kwargs):

        self.food_source.move()
        food = self.food_source.food(self.position)

        self.energy[self.alive] += food[self.alive]
        mitose = self.mitose()
        conjugation = self.conjugate()
        self.deplete()

        if self.alive.sum() > 0:
            new_states = self.observe(food[self.alive])
            actions = self.select_actions(*new_states)
            reward = self.reward(food[self.alive], conjugation[self.alive], mitose[self.alive])

            self.learn(new_states, reward, actions)

            self.act(actions)
            self.mutate()

        if self.record:
            record = dict(
                alive=int(self.alive.sum()),
                energy=self.energy[self.alive],
                q=self.q[self.alive],
                genome=self.genome[self.alive],
                position=self.position[self.alive],
                direction=self.direction[self.alive],
                food=self.food_source.position(),
            )

            self.history_.append(record)

    def simulate(self, generations=1000, verbose=False):
        iterator = range(generations)

        if verbose:
            iterator = tqdm(iterator, ascii=True, ncols=100)

        for _ in iterator:
            self.step()

            if self.alive.sum() == 0:
                break
