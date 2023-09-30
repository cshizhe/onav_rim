#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from typing import Optional, Union

import numpy as np

import habitat_sim
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim


def action_to_one_hot(action: int) -> np.array:
    one_hot = np.zeros(len(HabitatSimActions), dtype=np.float32)
    one_hot[action] = 1
    return one_hot


class ShortestPathFollower:
    r"""Utility class for extracting the action on the shortest path to the
        goal.

    :param sim: HabitatSim instance.
    :param goal_radius: Distance between the agent and the goal for it to be
            considered successful.
    :param return_one_hot: If true, returns a one-hot encoding of the action
            (useful for training ML agents). If false, returns the
            SimulatorAction.
    :param stop_on_error: Return stop if the follower is unable to determine a
                          suitable action to take next.  If false, will raise
                          a habitat_sim.errors.GreedyFollowerError instead
    """

    def __init__(
        self,
        sim: HabitatSim,
        goal_radius: float,
        return_one_hot: bool = True,
        stop_on_error: bool = True,
    ):

        self._return_one_hot = return_one_hot
        self._sim = sim
        self._goal_radius = goal_radius
        self._follower: Optional[habitat_sim.GreedyGeodesicFollower] = None
        self._current_scene = None
        self._stop_on_error = stop_on_error

    def _build_follower(self):
        if self._current_scene != self._sim.habitat_config.SCENE:
            self._follower = self._sim.make_greedy_follower(
                0,
                self._goal_radius,
                stop_key=HabitatSimActions.STOP,
                forward_key=HabitatSimActions.MOVE_FORWARD,
                left_key=HabitatSimActions.TURN_LEFT,
                right_key=HabitatSimActions.TURN_RIGHT,
            )
            self._current_scene = self._sim.habitat_config.SCENE

    def _get_return_value(self, action) -> Union[int, np.array]:
        if self._return_one_hot:
            return action_to_one_hot(action)
        else:
            return action

    def get_next_action(
        self, goal_pos: np.array
    ) -> Optional[Union[int, np.array]]:
        """Returns the next action along the shortest path."""
        self._build_follower()
        assert self._follower is not None
        try:
            next_action = self._follower.next_action_along(goal_pos)
            # print('follower: {}'.format(next_action))
        except habitat_sim.errors.GreedyFollowerError as e:
            if self._stop_on_error:
                next_action = HabitatSimActions.STOP
            else:
                raise e

        return self._get_return_value(next_action)

    def find_path(
        self, goal_pos: np.ndarray
    ):
        self._build_follower()
        assert self._follower is not None
        try:
            actions = self._follower.find_path(goal_pos)
        except habitat_sim.errors.GreedyFollowerError as e:
            if self._stop_on_error:
                actions = None
            else:
                raise e
        return actions

    @property
    def mode(self):
        warnings.warn(".mode is depricated", DeprecationWarning)
        return ""

    @mode.setter
    def mode(self, new_mode: str):
        warnings.warn(".mode is depricated", DeprecationWarning)
