class CustomEnvironment(ParallelEnv):
    def __init__(self):
        self.escape_y = None
        self.escape_x = None
        self.guard_y = None
        self.guard_x = None
        self.prisoner_y = None
        self.prisoner_x = None
        self.timestep = None
        self.possible_agents = ["prisoner", "guard"]

    def reset(self, seed=None, return_info=False, options=None):
        self.agents = copy(self.possible_agents)
        self.timestep = 0

        self.prisoner_x = 0
        self.prisoner_y = 0

        self.guard_x = 7
        self.guard_y = 7

        self.escape_x = random.randint(2, 5)
        self.escape_y = random.randint(2, 5)

        observations = {
            a: (
                self.prisoner_x + 7 * self.prisoner_y,
                self.guard_x + 7 * self.guard_y,
                self.escape_x + 7 * self.escape_y,
            )
            for a in self.agents
        }
        return observations

    def step(self, actions):
        # Execute actions
        prisoner_action = actions["prisoner"]
        guard_action = actions["guard"]

        if prisoner_action == 0 and self.prisoner_x > 0:
            self.prisoner_x -= 1
        elif prisoner_action == 1 and self.prisoner_x < 6:
            self.prisoner_x += 1
        elif prisoner_action == 2 and self.prisoner_y > 0:
            self.prisoner_y -= 1
        elif prisoner_action == 3 and self.prisoner_y < 6:
            self.prisoner_y += 1

        if guard_action == 0 and self.guard_x > 0:
            self.guard_x -= 1
        elif guard_action == 1 and self.guard_x < 6:
            self.guard_x += 1
        elif guard_action == 2 and self.guard_y > 0:
            self.guard_y -= 1
        elif guard_action == 3 and self.guard_y < 6:
            self.guard_y += 1

        # Check termination conditions
        terminations = {a: False for a in self.agents}
        rewards = {a: 0 for a in self.agents}
        if self.prisoner_x == self.guard_x and self.prisoner_y == self.guard_y:
            rewards = {"prisoner": -1, "guard": 1}
            terminations = {a: True for a in self.agents}

        elif self.prisoner_x == self.escape_x and self.prisoner_y == self.escape_y:
            rewards = {"prisoner": 1, "guard": -1}
            terminations = {a: True for a in self.agents}

        # Check truncation conditions (overwrites termination conditions)
        truncations = {a: False for a in self.agents}
        if self.timestep > 100:
            rewards = {"prisoner": 0, "guard": 0}
            truncations = {"prisoner": True, "guard": True}
        self.timestep += 1

        # Get observations
        observations = {
            a: (
                self.prisoner_x + 7 * self.prisoner_y,
                self.guard_x + 7 * self.guard_y,
                self.escape_x + 7 * self.escape_y,
            )
            for a in self.agents
        }

        # Get dummy infos (not used in this example)
        infos = {a: {} for a in self.agents}

        return observations, rewards, terminations, truncations, infos

    def render(self):
        grid = np.zeros((7, 7))
        grid[self.prisoner_y, self.prisoner_x] = "P"
        grid[self.guard_y, self.guard_x] = "G"
        grid[self.escape_y, self.escape_x] = "E"
        print(f"{grid} \n")

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return MultiDiscrete([7 * 7 - 1] * 3)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(4)




class parallel_env(ParallelEnv, EzPickle):
    metadata = {'render_modes': ['ansi'], "name": "PlayerEnv-Multi-v0"}

    def __init__(self, n_agents: int = 20, new_step_api: bool = True) -> None:
        EzPickle.__init__(
            self,
            n_agents,
            new_step_api
        )

        self._episode_ended = False
        self.n_agents = n_agents

        self.possible_agents = [
            f"player_{idx}" for idx in range(n_agents)]

        self.agents = self.possible_agents[:]

        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        self.observation_spaces = spaces.Dict(
            {agent: spaces.Box(shape=(len(self.agents),),
                               dtype=np.float64, low=0.0, high=1.0) for agent in self.possible_agents}
        )

        self.action_spaces = spaces.Dict(
            {agent: spaces.Discrete(4) for agent in self.possible_agents}
        )
        self.current_step = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def __calculate_observation(self, agent_id: int) -> np.ndarray:
        return self.observation_space(agent_id).sample()

    def __calculate_observations(self) -> np.ndarray:
        observations = {
            agent: self.__calculate_observation(
                agent_id=agent)
            for agent in self.agents
        }
        return observations

    def observe(self, agent):
        return self.__calculate_observation(agent_id=agent)

    def step(self, actions):
        if self._episode_ended:
            return self.reset()
        observations = self.__calculate_observations()
        rewards = random.sample(range(100), self.n_agents)
        self.current_step += 1
        self._episode_ended = self.current_step >= 100
        infos = {agent: {} for agent in self.agents}
        dones = {agent: self._episode_ended for agent in self.agents}
        rewards = {
            self.agents[i]: rewards[i]
            for i in range(len(self.agents))
        }
        if self._episode_ended:
            self.agents = {}  # To satisfy `set(par_env.agents) == live_agents`
        return observations, rewards, dones, infos

    def reset(self,
              seed: Optional[int] = None,
              return_info: bool = False,
              options: Optional[dict] = None,):
        self.agents = self.possible_agents[:]
        self._episode_ended = False
        self.current_step = 0
        observations = self.__calculate_observations()
        return observations

    def render(self, mode="human"):
        # TODO: IMPLEMENT
        print("TO BE IMPLEMENTED")

    def close(self):
        pass
