"""Provide minecraft flat functionality."""

from __future__ import annotations
from typing import Any


import logging
import threading

import elements
import embodied
import numpy as np

np.float = float  # pyright: ignore[reportAttributeAccessIssue]
np.int = int  # pyright: ignore[reportAttributeAccessIssue]
np.bool = bool  # pyright: ignore[reportAttributeAccessIssue]

from minerl.herobraine.env_spec import EnvSpec  # pyright: ignore[reportMissingImports]
from minerl.herobraine.hero import handler  # pyright: ignore[reportMissingImports]
from minerl.herobraine.hero import handlers  # pyright: ignore[reportMissingImports]
from minerl.herobraine.hero import mc  # pyright: ignore[reportMissingImports]
from minerl.herobraine.hero.mc import INVERSE_KEYMAP  # pyright: ignore[reportMissingImports]  # fmt: skip


class Wood(embodied.Wrapper):
    """Represent wood."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the wood.

        Args:
            args: Positional arguments forwarded to the wrapped callable.
            kwargs: Keyword arguments forwarded to the wrapped callable.
        """
        actions = BASIC_ACTIONS
        self.rewards = [
            CollectReward("log", repeated=1),
            HealthReward(),
        ]
        length = kwargs.pop("length", 36000)
        env = MinecraftBase(actions, *args, **kwargs)
        env = embodied.wrappers.TimeLimit(env, length)
        super().__init__(env)

    def step(self, action: Any) -> Any:
        """Advance state.

        Args:
            action: Action value.

        Returns:
            Result of the operation.
        """
        obs = self.env.step(action)
        reward = sum([fn(obs, self.env.inventory) for fn in self.rewards])
        obs["reward"] = np.float32(reward)
        return obs


class Climb(embodied.Wrapper):
    """Represent climb."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the climb.

        Args:
            args: Positional arguments forwarded to the wrapped callable.
            kwargs: Keyword arguments forwarded to the wrapped callable.
        """
        actions = BASIC_ACTIONS
        length = kwargs.pop("length", 36000)
        env = MinecraftBase(actions, *args, **kwargs)
        env = embodied.wrappers.TimeLimit(env, length)
        super().__init__(env)
        self._previous = None
        self._health_reward = HealthReward()

    def step(self, action: Any) -> Any:
        """Advance state.

        Args:
            action: Action value.

        Returns:
            Result of the operation.
        """
        obs = self.env.step(action)
        x, y, z = obs["log/player_pos"]
        height = np.float32(y)
        if obs["is_first"]:
            self._previous = height
        reward = (
            height - self._previous
        ) + self._health_reward(  # pyright: ignore[reportOperatorIssue]
            obs
        )
        obs["reward"] = np.float32(reward)
        self._previous = height
        return obs


class Diamond(embodied.Wrapper):
    """Represent diamond."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the diamond.

        Args:
            args: Positional arguments forwarded to the wrapped callable.
            kwargs: Keyword arguments forwarded to the wrapped callable.
        """
        actions = {
            **BASIC_ACTIONS,
            "craft_planks": dict(craft="planks"),
            "craft_stick": dict(craft="stick"),
            "craft_crafting_table": dict(craft="crafting_table"),
            "place_crafting_table": dict(place="crafting_table"),
            "craft_wooden_pickaxe": dict(nearbyCraft="wooden_pickaxe"),
            "craft_stone_pickaxe": dict(nearbyCraft="stone_pickaxe"),
            "craft_iron_pickaxe": dict(nearbyCraft="iron_pickaxe"),
            "equip_stone_pickaxe": dict(equip="stone_pickaxe"),
            "equip_wooden_pickaxe": dict(equip="wooden_pickaxe"),
            "equip_iron_pickaxe": dict(equip="iron_pickaxe"),
            "craft_furnace": dict(nearbyCraft="furnace"),
            "place_furnace": dict(place="furnace"),
            "smelt_iron_ingot": dict(nearbySmelt="iron_ingot"),
        }
        self.rewards = [
            CollectReward("log", once=1),
            CollectReward("planks", once=1),
            CollectReward("stick", once=1),
            CollectReward("crafting_table", once=1),
            CollectReward("wooden_pickaxe", once=1),
            CollectReward("cobblestone", once=1),
            CollectReward("stone_pickaxe", once=1),
            CollectReward("iron_ore", once=1),
            CollectReward("furnace", once=1),
            CollectReward("iron_ingot", once=1),
            CollectReward("iron_pickaxe", once=1),
            CollectReward("diamond", once=1),
            HealthReward(),
        ]
        length = kwargs.pop("length", 36000)
        env = MinecraftBase(actions, *args, **kwargs)
        env = embodied.wrappers.TimeLimit(env, length)
        super().__init__(env)

    def step(self, action: Any) -> Any:
        """Advance state.

        Args:
            action: Action value.

        Returns:
            Result of the operation.
        """
        obs = self.env.step(action)
        reward = sum([fn(obs, self.env.inventory) for fn in self.rewards])
        obs["reward"] = np.float32(reward)
        return obs


BASIC_ACTIONS = {
    "noop": dict(),
    "attack": dict(attack=1),
    "turn_up": dict(camera=(-15, 0)),
    "turn_down": dict(camera=(15, 0)),
    "turn_left": dict(camera=(0, -15)),
    "turn_right": dict(camera=(0, 15)),
    "forward": dict(forward=1),
    "back": dict(back=1),
    "left": dict(left=1),
    "right": dict(right=1),
    "jump": dict(jump=1, forward=1),
    "place_dirt": dict(place="dirt"),
}


class CollectReward:
    """Represent collect reward."""

    def __init__(self, item: Any, once: int = 0, repeated: int = 0) -> None:
        """Initialize the collect reward.

        Args:
            item: Item value.
            once: Once value.
            repeated: Repeated value.
        """
        self.item = item
        self.once = once
        self.repeated = repeated
        self.previous = 0
        self.maximum = 0

    def __call__(self, obs: Any, inventory: Any) -> Any:
        """Apply the collect reward.

        Args:
            obs: Obs to process.
            inventory: Inventory to process.

        Returns:
            Result produced by the operation.
        """
        current = inventory[self.item]
        if obs["is_first"]:
            self.previous = current
            self.maximum = current
            return 0
        reward = self.repeated * max(0, current - self.previous)
        if self.maximum == 0 and current > 0:
            reward += self.once
        self.previous = current
        self.maximum = max(self.maximum, current)
        return reward


class HealthReward:
    """Represent health reward."""

    def __init__(self, scale: float = 0.01) -> None:
        """Initialize the health reward.

        Args:
            scale: Scale value.
        """
        self.scale = scale
        self.previous = None

    def __call__(self, obs: Any, inventory: Any | None = None) -> Any:
        """Apply the health reward.

        Args:
            obs: Obs to process.
            inventory: Inventory to process.

        Returns:
            Result produced by the operation.
        """
        health = obs["health"]
        if obs["is_first"]:
            self.previous = health
            return 0
        reward = self.scale * (health - self.previous)
        self.previous = health
        return np.float32(reward)


class MinecraftBase(embodied.Env):
    """Represent minecraft base."""

    LOCK = threading.Lock()
    NOOP = dict(
        camera=(0, 0),
        forward=0,
        back=0,
        left=0,
        right=0,
        attack=0,
        sprint=0,
        jump=0,
        sneak=0,
        craft="none",
        nearbyCraft="none",
        nearbySmelt="none",
        place="none",
        equip="none",
    )

    def __init__(
        self,
        actions: Any,
        repeat: int = 1,
        size: tuple[Any, ...] = (64, 64),
        break_speed: float = 100.0,
        gamma: float = 10.0,
        sticky_attack: int = 30,
        sticky_jump: int = 10,
        pitch_limit: tuple[Any, ...] = (-60, 60),
        log_inv_keys: tuple[Any, ...] = ("log", "cobblestone", "iron_ingot", "diamond"),
        logs: bool = False,
    ) -> None:
        """Initialize the minecraft base.

        Args:
            actions: Actions value.
            repeat: Repeat value.
            size: Requested number of elements.
            break_speed: Break speed value.
            gamma: Gamma value.
            sticky_attack: Sticky attack value.
            sticky_jump: Sticky jump value.
            pitch_limit: Pitch limit value.
            log_inv_keys: Log inv keys value.
            logs: Logs value.
        """
        if logs:
            logging.basicConfig(level=logging.DEBUG)
        self._repeat = repeat
        self._size = size
        if break_speed != 1.0:
            sticky_attack = 0

        # Make env
        with self.LOCK:
            self._gymenv = MineRLEnv(
                size, break_speed  # pyright: ignore[reportArgumentType]
            ).make()
        from . import from_gym

        self._env = from_gym.FromGym(self._gymenv)
        self._inventory = {}

        # Observations
        self._inv_keys = [
            k
            for k in self._env.obs_space
            if k.startswith("inventory/")
            if k != "inventory/log2"
        ]
        self._inv_log_keys = [f"inventory/{k}" for k in log_inv_keys]
        assert all(k in self._inv_keys for k in self._inv_log_keys), (
            self._inv_keys,
            self._inv_log_keys,
        )
        self._step = 0
        self._max_inventory = None
        self._equip_enum = self._gymenv.observation_space["equipped_items"]["mainhand"][
            "type"
        ].values.tolist()
        self._obs_space = self.obs_space

        # Actions
        actions = self._insert_defaults(actions)
        self._action_names = tuple(actions.keys())
        self._action_values = tuple(actions.values())
        message = f"Minecraft action space ({len(self._action_values)}):"
        print(message, ", ".join(self._action_names))
        self._sticky_attack_length = sticky_attack
        self._sticky_attack_counter = 0
        self._sticky_jump_length = sticky_jump
        self._sticky_jump_counter = 0
        self._pitch_limit = pitch_limit
        self._pitch = 0

    @property
    def obs_space(self) -> dict[Any, Any]:
        """Handle observation space.

        Returns:
            Result of the operation.
        """
        return {
            "image": elements.Space(np.uint8, self._size + (3,)),
            "inventory": elements.Space(np.float32, len(self._inv_keys), 0),
            "inventory_max": elements.Space(np.float32, len(self._inv_keys), 0),
            "equipped": elements.Space(np.float32, len(self._equip_enum), 0, 1),
            "reward": elements.Space(np.float32),
            "health": elements.Space(np.float32),
            "hunger": elements.Space(np.float32),
            "breath": elements.Space(np.float32),
            "is_first": elements.Space(bool),
            "is_last": elements.Space(bool),
            "is_terminal": elements.Space(bool),
            **{f"log/{k}": elements.Space(np.int64) for k in self._inv_log_keys},
            # 'log/player_pos': elements.Space(np.float32, 3),
        }

    @property
    def act_space(self) -> dict[Any, Any]:
        """Handle act space.

        Returns:
            Result of the operation.
        """
        return {
            "action": elements.Space(np.int32, (), 0, len(self._action_values)),
            "reset": elements.Space(bool),
        }

    def step(self, action: Any) -> Any:
        """Advance state.

        Args:
            action: Action value.

        Returns:
            Result of the operation.
        """
        action = action.copy()
        index = action.pop("action")
        action.update(self._action_values[index])
        action = self._action(action)
        if action["reset"]:
            obs = self._reset()
        else:
            following = self.NOOP.copy()
            for key in ("attack", "forward", "back", "left", "right"):
                following[key] = action[key]
            for act in [action] + ([following] * (self._repeat - 1)):
                obs = self._env.step(act)
                if self._env.info and "error" in self._env.info:
                    obs = self._reset()
                    break
        obs = self._obs(obs)
        self._step += 1
        assert "pov" not in obs, list(obs.keys())
        return obs

    @property
    def inventory(self) -> Any:
        """Handle inventory.

        Returns:
            Result of the operation.
        """
        return self._inventory

    def _reset(self) -> Any:
        with self.LOCK:
            obs = self._env.step({"reset": True})
        self._step = 0
        self._max_inventory = None
        self._sticky_attack_counter = 0
        self._sticky_jump_counter = 0
        self._pitch = 0
        self._inventory = {}
        return obs

    def _obs(self, obs: Any) -> Any:
        obs["inventory/log"] += obs.pop("inventory/log2")
        self._inventory = {
            k.split("/", 1)[1]: obs[k] for k in self._inv_keys if k != "inventory/air"
        }
        inventory = np.array([obs[k] for k in self._inv_keys], np.float32)
        if self._max_inventory is None:
            self._max_inventory = inventory
        else:
            self._max_inventory = np.maximum(self._max_inventory, inventory)
        index = self._equip_enum.index(obs["equipped_items/mainhand/type"])
        equipped = np.zeros(len(self._equip_enum), np.float32)
        equipped[index] = 1.0
        # player_x = obs['location_stats/xpos']
        # player_y = obs['location_stats/ypos']
        # player_z = obs['location_stats/zpos']
        obs = {
            "image": obs["pov"],
            "inventory": inventory,
            "inventory_max": self._max_inventory.copy(),
            "equipped": equipped,
            "health": np.float32(obs["life_stats/life"] / 20),
            "hunger": np.float32(obs["life_stats/food"] / 20),
            "breath": np.float32(obs["life_stats/air"] / 300),
            "reward": np.float32(0.0),
            "is_first": obs["is_first"],
            "is_last": obs["is_last"],
            "is_terminal": obs["is_terminal"],
            **{f"log/{k}": np.int64(obs[k]) for k in self._inv_log_keys},
            # 'log/player_pos': np.array([player_x, player_y, player_z], np.float32),
        }
        for key, value in obs.items():
            space = self._obs_space[key]
            if not isinstance(value, np.ndarray):
                value = np.array(value)
            assert value in space, (key, value, value.dtype, value.shape, space)
        return obs

    def _action(self, action: Any) -> Any:
        if self._sticky_attack_length:
            if action["attack"]:
                self._sticky_attack_counter = self._sticky_attack_length
            if self._sticky_attack_counter > 0:
                action["attack"] = 1
                action["jump"] = 0
                self._sticky_attack_counter -= 1
        if self._sticky_jump_length:
            if action["jump"]:
                self._sticky_jump_counter = self._sticky_jump_length
            if self._sticky_jump_counter > 0:
                action["jump"] = 1
                action["forward"] = 1
                self._sticky_jump_counter -= 1
        if self._pitch_limit and action["camera"][0]:
            lo, hi = self._pitch_limit
            if not (lo <= self._pitch + action["camera"][0] <= hi):
                action["camera"] = (0, action["camera"][1])
            self._pitch += action["camera"][0]
        return action

    def _insert_defaults(self, actions: Any) -> Any:
        actions = {name: action.copy() for name, action in actions.items()}
        for key, default in self.NOOP.items():
            for action in actions.values():
                if key not in action:
                    action[key] = default
        return actions


class MineRLEnv(EnvSpec):
    """Represent mine rlenv."""

    def __init__(
        self, resolution: tuple[Any, ...] = (64, 64), break_speed: int = 50
    ) -> None:
        """Initialize the mine rlenv.

        Args:
            resolution: Resolution value.
            break_speed: Break speed value.
        """
        self.resolution = resolution
        self.break_speed = break_speed
        super().__init__(name="MineRLEnv-v1")

    def create_agent_start(self) -> list[Any]:
        """Create agent start.

        Returns:
            Result of the operation.
        """
        return [BreakSpeedMultiplier(self.break_speed)]

    def create_agent_handlers(self) -> list[Any]:
        """Create agent handlers.

        Returns:
            Result of the operation.
        """
        return []

    def create_server_world_generators(self) -> list[Any]:
        """Create server world generators.

        Returns:
            Result of the operation.
        """
        return [handlers.DefaultWorldGenerator(force_reset=True)]

    def create_server_quit_producers(self) -> list[Any]:
        """Create server quit producers.

        Returns:
            Result of the operation.
        """
        return [handlers.ServerQuitWhenAnyAgentFinishes()]

    def create_server_initial_conditions(self) -> list[Any]:
        """Create server initial conditions.

        Returns:
            Result of the operation.
        """
        return [
            handlers.TimeInitialCondition(allow_passage_of_time=True, start_time=0),
            handlers.SpawningInitialCondition(allow_spawning=True),
        ]

    def create_observables(self) -> list[Any]:
        """Create observables.

        Returns:
            Result of the operation.
        """
        return [
            handlers.POVObservation(self.resolution),
            handlers.FlatInventoryObservation(mc.ALL_ITEMS),
            handlers.EquippedItemObservation(
                mc.ALL_ITEMS, _default="air", _other="other"
            ),
            handlers.ObservationFromCurrentLocation(),
            handlers.ObservationFromLifeStats(),
        ]

    def create_actionables(self) -> list[Any]:
        """Create actionables.

        Returns:
            Result of the operation.
        """
        kw = dict(_other="none", _default="none")
        return [
            handlers.KeybasedCommandAction("forward", INVERSE_KEYMAP["forward"]),
            handlers.KeybasedCommandAction("back", INVERSE_KEYMAP["back"]),
            handlers.KeybasedCommandAction("left", INVERSE_KEYMAP["left"]),
            handlers.KeybasedCommandAction("right", INVERSE_KEYMAP["right"]),
            handlers.KeybasedCommandAction("jump", INVERSE_KEYMAP["jump"]),
            handlers.KeybasedCommandAction("sneak", INVERSE_KEYMAP["sneak"]),
            handlers.KeybasedCommandAction("attack", INVERSE_KEYMAP["attack"]),
            handlers.CameraAction(),
            handlers.PlaceBlock(["none"] + mc.ALL_ITEMS, **kw),
            handlers.EquipAction(["none"] + mc.ALL_ITEMS, **kw),
            handlers.CraftAction(["none"] + mc.ALL_ITEMS, **kw),
            handlers.CraftNearbyAction(["none"] + mc.ALL_ITEMS, **kw),
            handlers.SmeltItemNearby(["none"] + mc.ALL_ITEMS, **kw),
        ]

    def is_from_folder(self, folder: Any) -> bool:
        """Return whether from folder.

        Args:
            folder: Folder value.

        Returns:
            Whether from folder.
        """
        return folder == "none"

    def get_docstring(self) -> str:
        """Return docstring.

        Returns:
            Result of the operation.
        """
        return ""

    def determine_success_from_rewards(self, rewards: Any) -> bool:
        """Handle determine success from rewards.

        Args:
            rewards: Rewards value.

        Returns:
            Result of the operation.
        """
        return True

    def create_rewardables(self) -> list[Any]:
        """Create rewardables.

        Returns:
            Result of the operation.
        """
        return []

    def create_server_decorators(self) -> list[Any]:
        """Create server decorators.

        Returns:
            Result of the operation.
        """
        return []

    def create_mission_handlers(self) -> list[Any]:
        """Create mission handlers.

        Returns:
            Result of the operation.
        """
        return []

    def create_monitors(self) -> list[Any]:
        """Create monitors.

        Returns:
            Result of the operation.
        """
        return []


class BreakSpeedMultiplier(handler.Handler):
    """Represent break speed multiplier."""

    def __init__(self, multiplier: float = 1.0) -> None:
        """Initialize the break speed multiplier.

        Args:
            multiplier: Multiplier value.
        """
        self.multiplier = multiplier

    def to_string(self) -> Any:
        """Handle to string.

        Returns:
            Result of the operation.
        """
        return f"break_speed({self.multiplier})"

    def xml_template(self) -> str:
        """Handle xml template.

        Returns:
            Result of the operation.
        """
        return "<BreakSpeedMultiplier>{{multiplier}}</BreakSpeedMultiplier>"
