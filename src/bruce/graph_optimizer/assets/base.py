import hashlib
import json
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, Self, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _resolve_bruce_git_commit(default: str = "Unknown") -> str:
    path = Path(__file__).resolve().parent
    for _ in range(8):
        if (path / ".git").is_dir():
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=path,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
            return default
        if path.parent == path:
            break
        path = path.parent
    return default


class State(BaseModel):
    model_config = ConfigDict(frozen=True)

    index: int = -1
    """Position in the asset's ``state_space``; matches transition matrix column. -1 if not in the space."""


class Action(BaseModel):
    model_config = ConfigDict(frozen=True)

    index: int
    """Position in the asset's ``action_space``; matches transition matrix row."""


class Params(BaseModel):
    """
    horizon: number of optimization steps.
    start_unix_s: start time of the first step in Unix seconds.
    site_id: site's unique identifier.
    timestep_duration_hours: duration of each step in hours (one entry per step).
    bruce_git_commit: git commit of the bruce package.

    Forecast arrays must align with horizon (energies/rates for that step's interval).
    """
    horizon: int
    start_unix_s: int
    site_id: str
    timestep_duration_hours: list[float] = Field(default_factory=list)
    bruce_git_commit: str = Field(default_factory=_resolve_bruce_git_commit)

    @model_validator(mode="after")
    def _normalize_timestep_duration_hours(self) -> Self:
        if not self.timestep_duration_hours:
            self.timestep_duration_hours = [1.0] * self.horizon
        elif len(self.timestep_duration_hours) != self.horizon:
            raise ValueError(
                f"timestep_duration_hours length ({len(self.timestep_duration_hours)}) "
                f"must equal horizon ({self.horizon})"
            )
        if any(dt <= 0 for dt in self.timestep_duration_hours):
            raise ValueError("each timestep_duration_hours entry must be positive")
        return self

    def validate_bid_params_update(self, updated: Self) -> None:
        """Reject bid-time changes that would require rebuilding the graph."""


S = TypeVar("S", bound=State)
A = TypeVar("A", bound=Action)
P = TypeVar("P", bound=Params)


class TransitionMatrixParams(BaseModel, ABC):
    model_config = ConfigDict(frozen=True)

    @classmethod
    @abstractmethod
    def from_asset_params(cls, params: Params) -> Self:
        raise NotImplementedError

    @property
    def hash(self) -> str:
        payload = json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()[:8]

    def cache_filename(self, asset_name: str) -> str:
        return f"{asset_name}_{self.hash}.matrix.npz"


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

    @abstractmethod
    def elec_used_kwh(self, state: S, action: A, time_step: int) -> float:
        raise NotImplementedError

    @abstractmethod
    def initial_state(self) -> S:
        raise NotImplementedError

    @abstractmethod
    def transition_matrix_params(self) -> TransitionMatrixParams:
        raise NotImplementedError

    def update_params(self, params: P) -> None:
        self.params = params
        self.model.params = params
        self.on_params_updated()

    def on_params_updated(self) -> None:
        pass

    def allow_transition(self, state: S, next_state: S, action: A, time_step: int) -> bool:
        return True
