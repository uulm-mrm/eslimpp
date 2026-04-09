
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

    sensor_1: Opinion = field(default_factory=lambda : Opinion())
    processing_1: Opinion = field(default_factory=lambda : Opinion())

    concurrent_sa: Opinion = field(default_factory=lambda : Opinion())

    sensor_2: Opinion = field(default_factory=lambda : Opinion())
    processing_2: Opinion = field(default_factory=lambda : Opinion())

    sensor_3: Opinion = field(default_factory=lambda : Opinion())
    processing_3: Opinion = field(default_factory=lambda : Opinion())

    fusion: Opinion = field(default_factory=lambda : Opinion())
    planning: Opinion = field(default_factory=lambda : Opinion())

    def getProcessingState(self, id: int):
        if id == 1:
            return self.sensor_1.multiply(self.processing_1)
        elif id == 2:
            return self.sensor_2.multiply(self.processing_2)
        else:
            return self.sensor_3.multiply(self.processing_3)

    def getPerceptionState(self):
        match self.mode:
            case Mode.MINIMAL_FEASIBLE:
                inter = self.getProcessingState(1).comultiply(self.getProcessingState(2))
                sa_fused = inter.wb_fuse(self.concurrent_sa)
                return sa_fused.comultiply(self.getProcessingState(3))

            case Mode.STATE_OF_HEALTH:
                inter = self.getProcessingState(1).cum_fuse(self.getProcessingState(2))
                sa_fused = inter.wb_fuse(self.concurrent_sa)
                return sa_fused.cum_fuse(self.getProcessingState(3))

    def getOverall(self):
        return self.getPerceptionState().multiply(self.fusion).multiply(self.planning)

    def interpolate(self, other, interp_fac: float):
        interp = SystemState()
        interp.name = self.name + "_interpolate_" + other.name

        interp.sensor_1 = self.sensor_1.interpolate(other.sensor_1, interp_fac)
        interp.processing_1 = self.processing_1.interpolate(other.processing_1, interp_fac)
        interp.sensor_2 = self.sensor_2.interpolate(other.sensor_2, interp_fac)
        interp.processing_2 = self.processing_2.interpolate(other.processing_2, interp_fac)
        interp.sensor_3 = self.sensor_3.interpolate(other.sensor_3, interp_fac)
        interp.processing_3 = self.processing_3.interpolate(other.processing_3, interp_fac)

        interp.concurrent_sa = self.concurrent_sa.interpolate(other.concurrent_sa, interp_fac)

        interp.fusion = self.fusion.interpolate(other.fusion, interp_fac)
        interp.planning = self.planning.interpolate(other.planning, interp_fac)

        return interp
