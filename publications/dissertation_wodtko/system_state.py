
from dataclasses import dataclass, field

from subjective_logic import Opinion2d as Opinion
from enum import Enum

class Mode(Enum):
    MINIMAL_FEASIBLE = 1
    STATE_OF_HEALTH = 2

@dataclass
class SystemState:
    name: str = "NOTSET"

    mode: Mode = Mode.MINIMAL_FEASIBLE

    v_1: Opinion = field(default_factory=lambda : Opinion())

    concurrent_sa: Opinion = field(default_factory=lambda : Opinion())

    v_2: Opinion = field(default_factory=lambda : Opinion())

    v_3: Opinion = field(default_factory=lambda : Opinion())

    fusion: Opinion = field(default_factory=lambda : Opinion())
    planning: Opinion = field(default_factory=lambda : Opinion())

    def getPerceptionState(self):
        match self.mode:
            case Mode.MINIMAL_FEASIBLE:
                inter = self.v_1.comultiply(self.v_2)
                sa_fused = inter.wb_fuse(self.concurrent_sa)
                return sa_fused.comultiply(self.v_3)

            case Mode.STATE_OF_HEALTH:
                inter = self.v_1.cum_fuse(self.v_2)
                sa_fused = inter.wb_fuse(self.concurrent_sa)
                return sa_fused.cum_fuse(self.v_3)

    def getOverall(self):
        return self.getPerceptionState().multiply(self.fusion).multiply(self.planning)

    def interpolate(self, other, interp_fac: float):
        interp = SystemState()
        interp.name = self.name + "_interpolate_" + other.name

        interp.v_1 = self.v_1.interpolate(other.v_1, interp_fac)
        interp.v_2 = self.v_2.interpolate(other.v_2, interp_fac)
        interp.v_3 = self.v_3.interpolate(other.v_3, interp_fac)

        interp.concurrent_sa = self.concurrent_sa.interpolate(other.concurrent_sa, interp_fac)

        interp.fusion = self.fusion.interpolate(other.fusion, interp_fac)
        interp.planning = self.planning.interpolate(other.planning, interp_fac)

        return interp
