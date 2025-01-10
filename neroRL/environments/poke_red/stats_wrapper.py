from collections import defaultdict
from dataclasses import dataclass
from enum import Enum

import numpy as np
from gymnasium import Env

from neroRL.environments.poke_red.events_2 import filtered_event_names
from neroRL.environments.poke_red.items import Items
from neroRL.environments.poke_red.map_data_2 import map_locations
from neroRL.environments.poke_red.moves import Moves
from neroRL.environments.poke_red.red_gym_env_v2 import RedGymEnv
from neroRL.environments.poke_red.pokedex import Pokedex, PokedexOrder

event_flags_start = 0xD747
event_flags_end = 0xD887
MAP_N_ADDRESS = 0xD35E


class WildEncounterResult(Enum):
    WIN = 0
    LOSE = 1
    CAUGHT = 2
    ESCAPED = 3

    def __repr__(self):
        return self.name


@dataclass
class WildEncounter:
    species: PokedexOrder
    level: int
    result: WildEncounterResult


class StatsWrapper(Env):
    def __init__(self, env: RedGymEnv):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.max_steps = env.max_steps

        self.env.pyboy.hook_register(
            None, "PlayerCanExecuteMove", self.increment_move_hook, None
        )
        self.env.pyboy.hook_register(
            None, "AnimateHealingMachine", self.pokecenter_hook, None
        )
        self.env.pyboy.hook_register(
            None, "RedsHouse1FMomText.heal", self.pokecenter_hook, None
        )
        self.env.pyboy.hook_register(None, "UseItem_", self.chose_item_hook, None)
        self.env.pyboy.hook_register(
            None, "FaintEnemyPokemon.wild_win", self.record_wild_win_hook, None
        )
        self.env.pyboy.hook_register(None, "HandlePlayerBlackOut", self.blackout_hook, None)
        self.env.pyboy.hook_register(
            None, "ItemUseBall.captured", self.catch_pokemon_hook, None
        )
        self.env.pyboy.hook_register(
            None, "TryRunningFromBattle.canEscape", self.escaped_battle_hook, None
        )

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset()
        self.init_stats_fields(obs["events"])
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.update_stats(obs["events"])
        if done or truncated:
            info = self.get_info()
        return obs, reward, done, truncated, info

    def render(self):
        return self.env.render()

    def init_stats_fields(self, event_obs):
        self.party_size = 1
        self.total_heal = 0
        self.num_heals = 0
        self.died_count = 0
        self.party_levels = np.asarray([-1 for _ in range(6)])
        self.events_sum = 0
        self.max_opponent_level = 0
        self.seen_coords = 0
        self.current_location = self.env.read_m(MAP_N_ADDRESS)
        self.location_first_visit_steps = {loc: -1 for loc in map_locations.keys()}
        self.location_frequency = {loc: 0 for loc in map_locations.keys()}
        self.location_steps_spent = {loc: 0 for loc in map_locations.keys()}
        self.current_events = event_obs
        self.events_steps = {name: -1 for name in filtered_event_names}
        self.caught_species = np.zeros(152, dtype=np.uint8)
        self.move_usage = defaultdict(int)
        self.pokecenter_count = 0
        self.pokecenter_location_count = defaultdict(int)
        self.item_usage = defaultdict(int)
        self.wild_encounters: list[WildEncounter] = []

    def update_stats(self, event_obs):
        self.party_size = self.env.party_size
        self.total_heal = self.env.total_healing_rew
        self.num_heals = self.env.num_heals
        self.seen_coords = len(self.env.seen_coords)
        self.max_opponent_level = self.env.update_max_op_level(opp_base_level=0)
        self.died_count = self.env.died_count
        self.update_party_levels()
        self.update_location_stats()
        self.update_event_stats(event_obs)
        self.update_pokedex()
        self.update_time_played()

    def update_party_levels(self):
        for i in range(
            self.env.pyboy.memory[self.env.pyboy.symbol_lookup("wPartyCount")[1]]
        ):
            self.party_levels[i] = self.env.pyboy.memory[
                self.env.pyboy.symbol_lookup(f"wPartyMon{i+1}Level")[1]
            ]

    def update_location_stats(self):
        new_location = self.env.read_m(MAP_N_ADDRESS)
        # Steps needed to reach this location
        if self.location_first_visit_steps[new_location] == -1:
            self.location_first_visit_steps[new_location] = self.env.step_count
        # Number of times this location was visited
        if new_location != self.current_location:
            self.location_frequency[new_location] += 1
            self.current_location = new_location
        # Number of steps that were spent in this location
        elif new_location == self.current_location:
            self.location_steps_spent[new_location] += 1

    def update_event_stats(self, event_obs):
        # check if self.current_events is equal to event_obs
        # if not, find the index that is different and update the steps
        comparison = self.current_events == event_obs
        if np.all(comparison):
            return
        changed_ids = np.where(comparison == False)[0]
        for i in changed_ids:
            self.events_steps[filtered_event_names[i]] = self.env.step_count
            self.events_sum += 1
        self.current_events = event_obs

    def update_pokedex(self):
        # TODO: Make a hook
        _, wPokedexOwned = self.env.pyboy.symbol_lookup("wPokedexOwned")
        _, wPokedexOwnedEnd = self.env.pyboy.symbol_lookup("wPokedexOwnedEnd")

        caught_mem = self.env.pyboy.memory[wPokedexOwned:wPokedexOwnedEnd]
        self.caught_species = np.unpackbits(
            np.array(caught_mem, dtype=np.uint8), bitorder="little"
        )
    
    def update_time_played(self):
        hours = self.env.pyboy.memory[self.env.pyboy.symbol_lookup("wPlayTimeHours")[1]]
        minutes = self.env.pyboy.memory[self.env.pyboy.symbol_lookup("wPlayTimeMinutes")[1]]
        self.seconds_played = hours * 3600 + minutes * 60
        self.seconds_played += self.env.pyboy.memory[self.env.pyboy.symbol_lookup("wPlayTimeSeconds")[1]]

    def increment_move_hook(self, *args, **kwargs):
        _, wPlayerSelectedMove = self.env.pyboy.symbol_lookup("wPlayerSelectedMove")
        self.move_usage[
            Moves(self.env.pyboy.memory[wPlayerSelectedMove]).name.lower()
        ] += 1

    def pokecenter_hook(self, *args, **kwargs):
        self.pokecenter_count += 1
        map_location = self.env.read_m(MAP_N_ADDRESS)
        self.pokecenter_location_count[map_location] += 1

    def chose_item_hook(self, *args, **kwargs):
        _, wCurItem = self.env.pyboy.symbol_lookup("wCurItem")
        self.item_usage[Items(self.env.pyboy.memory[wCurItem]).name.lower()] += 1

    def record_battle(self, result: WildEncounterResult):
        _, wEnemyMon = self.env.pyboy.symbol_lookup("wEnemyMon")
        _, wEnemyMon1Level = self.env.pyboy.symbol_lookup("wCurEnemyLevel")
        self.wild_encounters.append(
            WildEncounter(
                species=PokedexOrder(self.env.pyboy.memory[wEnemyMon]),
                level=self.env.pyboy.memory[wEnemyMon1Level],
                result=result,
            )
        )

    def record_wild_win_hook(self, *args, **kwargs):
        self.record_battle(WildEncounterResult.WIN)

    def blackout_hook(self, *args, **kwargs):
        _, wIsInBattle = self.env.pyboy.symbol_lookup("wIsInBattle")
        # lost battle == -1
        # no battle == 0
        # wild battle == 1
        # trainer battle == 2
        if self.env.pyboy.memory[wIsInBattle] == 1:
            self.record_battle(WildEncounterResult.LOSE)

    def catch_pokemon_hook(self, *args, **kwargs):
        self.record_battle(WildEncounterResult.CAUGHT)

    def escaped_battle_hook(self, *args, **kwargs):
        self.record_battle(WildEncounterResult.ESCAPED)

    def get_info(self):
        info = {
            "seconds_played": self.seconds_played,
            "party_size": self.party_size,
            "party_levels": self.party_levels,
            "caught_species": {
                Pokedex(pokemon_id + 1).name
                for pokemon_id, caught in enumerate(self.caught_species)
                if caught
            },
            "total_heal": self.total_heal,
            "num_heals": self.num_heals,
            "died_count": self.died_count,
            "seen_coords": self.seen_coords,
            "max_opponent_level": self.max_opponent_level,
            "events_sum": self.events_sum,
            "events_steps": self.events_steps,
            "move_usage": self.move_usage,
            "pokecenter_count": sum(self.pokecenter_location_count.values()),
            "pokecenter_location_count": self.pokecenter_location_count,
            "item_usage": self.item_usage,
            "location_first_visit_steps": self.location_first_visit_steps,
            "location_frequency": self.location_frequency,
            "location_steps_spent": self.location_steps_spent,
            "wild_encounters": self.wild_encounters,
        }
        return info
