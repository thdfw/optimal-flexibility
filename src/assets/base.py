from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict


class State(BaseModel):
    model_config = ConfigDict(frozen=True)


class Action(BaseModel):
    model_config = ConfigDict(frozen=True)


class Params(BaseModel):
    horizon: int


S = TypeVar("S", bound=State)
A = TypeVar("A", bound=Action)
P = TypeVar("P", bound=Params)


class Model(ABC, Generic[S, A, P]):
    def __init__(self, params: P):
        self.params = params

    @abstractmethod
    def next_state(self, state: S, action: A) -> S:
        raise NotImplementedError


class Asset(ABC, Generic[S, A, P]):
    def __init__(self, params: P):
        self.params = params
        self.state_space: list[S] = self.get_state_space()
        self.action_space: list[A] = self.get_action_space()
        self.model: Model[S, A, P] = self.get_model()

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_state_space(self) -> list[S]:
        raise NotImplementedError

    @abstractmethod
    def get_action_space(self) -> list[A]:
        raise NotImplementedError

    @abstractmethod
    def get_model(self) -> Model[S, A, P]:
        raise NotImplementedError

    @abstractmethod
    def get_available_actions(self, state: S, time_step: int) -> list[A]:
        raise NotImplementedError

    @abstractmethod
    def next_state(self, state: S, action: A) -> S:
        raise NotImplementedError

    def closest_state(self, state: S) -> S:
        return min(self.state_space, key=lambda candidate: self.state_distance(state, candidate))

    @abstractmethod
    def state_distance(self, state1: S, state2: S) -> float:
        raise NotImplementedError

    @abstractmethod
    def cost(self, state: S, next_state: S, action: A, time_step: int) -> float:
        raise NotImplementedError
