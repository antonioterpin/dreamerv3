"""Provide loconav quadruped functionality."""

import os

from dm_control import composer  # pyright: ignore[reportMissingImports]
from dm_control import mjcf  # pyright: ignore[reportMissingImports]
from dm_control.composer.observation import observable  # pyright: ignore[reportMissingImports]  # fmt: skip
from dm_control.locomotion.walkers import base  # pyright: ignore[reportMissingImports]
from dm_control.locomotion.walkers import legacy_base  # pyright: ignore[reportMissingImports]  # fmt: skip
from dm_control.mujoco.wrapper import mjbindings  # pyright: ignore[reportMissingImports]  # fmt: skip
import numpy as np

enums = mjbindings.enums
mjlib = mjbindings.mjlib


class Quadruped(legacy_base.Walker):
    """Represent quadruped."""

    def _build(self, name="walker", initializer=None):
        super()._build(initializer=initializer)
        self._mjcf_root = mjcf.from_path(
            os.path.join(os.path.dirname(__file__), "loconav_quadruped.xml")
        )
        if name:
            self._mjcf_root.model = name
        self._prev_action = np.zeros(self.action_spec.shape, self.action_spec.dtype)

    def initialize_episode(self, physics, random_state):
        """Handle initialize episode.

        Args:
            physics: Physics value.
            random_state: Random state value.
        """
        self._prev_action = np.zeros_like(self._prev_action)

    def apply_action(self, physics, action, random_state):
        """Apply action.

        Args:
            physics: Physics value.
            action: Action value.
            random_state: Random state value.
        """
        super().apply_action(physics, action, random_state)
        self._prev_action[:] = action

    def _build_observables(self):
        return QuadrupedObservables(self)

    @property
    def mjcf_model(self):
        """Handle mjcf model.

        Returns:
            Result of the operation.
        """
        return self._mjcf_root

    @property
    def upright_pose(self):
        """Handle upright pose.

        Returns:
            Result of the operation.
        """
        return base.WalkerPose()

    @composer.cached_property
    def actuators(self):
        """Handle actuators.

        Returns:
            Result of the operation.
        """
        return self._mjcf_root.find_all("actuator")

    @composer.cached_property
    def root_body(self):
        """Handle root body.

        Returns:
            Result of the operation.
        """
        return self._mjcf_root.find("body", "torso")

    @composer.cached_property
    def bodies(self):
        """Handle bodies.

        Returns:
            Result of the operation.
        """
        return tuple(self.mjcf_model.find_all("body"))

    @composer.cached_property
    def mocap_tracking_bodies(self):
        """Handle mocap tracking bodies.

        Returns:
            Result of the operation.
        """
        return tuple(self.mjcf_model.find_all("body"))

    @property
    def mocap_joints(self):
        """Handle mocap joints.

        Returns:
            Result of the operation.
        """
        return self.mjcf_model.find_all("joint")

    @property
    def _foot_bodies(self):
        return (
            self._mjcf_root.find("body", "toe_front_left"),
            self._mjcf_root.find("body", "toe_front_right"),
            self._mjcf_root.find("body", "toe_back_right"),
            self._mjcf_root.find("body", "toe_back_left"),
        )

    @composer.cached_property
    def end_effectors(self):
        """Handle end effectors.

        Returns:
            Result of the operation.
        """
        return self._foot_bodies

    @composer.cached_property
    def observable_joints(self):
        """Handle observable joints.

        Returns:
            Result of the operation.
        """
        return self._mjcf_root.find_all("joint")

    @composer.cached_property
    def egocentric_camera(self):
        """Handle egocentric camera.

        Returns:
            Result of the operation.
        """
        return self._mjcf_root.find("camera", "egocentric")

    def aliveness(self, physics):
        """Handle aliveness.

        Args:
            physics: Physics value.

        Returns:
            Result of the operation.
        """
        return (physics.bind(self.root_body).xmat[-1] - 1.0) / 2.0

    @composer.cached_property
    def ground_contact_geoms(self):
        """Handle ground contact geoms.

        Returns:
            Result of the operation.
        """
        foot_geoms = []
        for foot in self._foot_bodies:
            foot_geoms.extend(foot.find_all("geom"))
        return tuple(foot_geoms)

    @property
    def prev_action(self):
        """Handle prev action.

        Returns:
            Result of the operation.
        """
        return self._prev_action


class QuadrupedObservables(legacy_base.WalkerObservables):
    """Represent quadruped observables."""

    @composer.observable
    def actuator_activations(self):
        """Handle actuator activations.

        Returns:
            Result of the operation.
        """

        def actuator_activations_in_egocentric_frame(physics):
            """Handle actuator activations in egocentric frame.

            Args:
                physics: Physics value.

            Returns:
                Result of the operation.
            """
            return physics.data.act

        return observable.Generic(actuator_activations_in_egocentric_frame)

    @composer.observable
    def root_global_pos(self):
        """Handle root global pos.

        Returns:
            Result of the operation.
        """

        def root_pos(physics):
            """Handle root pos.

            Args:
                physics: Physics value.

            Returns:
                Result of the operation.
            """
            root_xpos, _ = self._entity.get_pose(physics)
            return np.reshape(root_xpos, -1)

        return observable.Generic(root_pos)

    @composer.observable
    def torso_global_pos(self):
        """Handle torso global pos.

        Returns:
            Result of the operation.
        """

        def torso_pos(physics):
            """Handle torso pos.

            Args:
                physics: Physics value.

            Returns:
                Result of the operation.
            """
            root_body = self._entity.root_body
            root_body_xpos = physics.bind(root_body).xpos
            return np.reshape(root_body_xpos, -1)

        return observable.Generic(torso_pos)

    @property
    def proprioception(self):
        """Handle proprioception.

        Returns:
            Result of the operation.
        """
        return [
            self.joints_pos,
            self.joints_vel,
            self.actuator_activations,
            self.sensors_accelerometer,
            self.sensors_gyro,
            self.sensors_velocimeter,
            self.sensors_force,
            self.sensors_torque,
            self.world_zaxis,
            self.root_global_pos,
            self.torso_global_pos,
        ] + self._collect_from_attachments("proprioception")
