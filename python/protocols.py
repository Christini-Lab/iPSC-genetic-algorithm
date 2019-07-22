"""Contains protocols to act in silico to probe cellular mechanics."""

import bisect
from typing import List, Union


class SingleActionPotentialProtocol:
    """Encapsulates state and behavior of a single action potential protocol."""

    def __init__(self, duration: float=1.8):
        self.duration = duration


class IrregularPacingProtocol:
    """Encapsulates state and behavior of a irregular pacing protocol.

    Attributes:
        duration: Duration of integration.
        stimulation_offsets: Each offset corresponds to the
            seconds after diastole begins that stimulation will
            occur. Cannot exceed `max_stim_interval_duration`, which is the
            time between beats when cell is pacing naturally.
    """

    # The start of a diastole must be below this voltage, in Vm.
    DIAS_THRESHOLD_VOLTAGE = -0.06

    # TODO find appropriate comment.
    EPSILON = .0001

    # Set to time between naturally occurring spontaneous beats.
    _MAX_STIM_INTERVAL = 1.55

    STIM_AMPLITUDE_AMPS = 7.5e-10
    STIM_DURATION_SECS = 0.005

    def __init__(self, duration: int, stimulation_offsets: List[float]) -> None:
        self.duration = duration
        self.stimulation_offsets = stimulation_offsets
        # TODO Consider renaming stimulation times.
        self.all_stimulation_times = []

    @property
    def stimulation_offsets(self):
        return self._stimulation_offsets

    @stimulation_offsets.setter
    def stimulation_offsets(self, offsets):
        for i in offsets:
            if i > self._MAX_STIM_INTERVAL:
                raise ValueError(
                    'Stimulation offsets from diastolic start cannot be '
                    'greater than `self.max_stim_interval_duration` because '
                    'the cell will have started to spontaneously beat.')
        self._stimulation_offsets = offsets

    def make_offset_generator(self):
        return (i for i in self._stimulation_offsets)


class VoltageClampStep:
    """A step in a voltage clamp protocol."""

    def __init__(self, voltage: float, duration: float) -> None:
        self.voltage = voltage
        self.duration = duration

    def __str__(self):
        return 'Voltage: {}, Duration: {}'.format(self.voltage, self.duration)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return (abs(self.voltage - other.voltage) < 0.001 and
                abs(self.duration - other.duration) < 0.001)


class VoltageClampProtocol:
    """Encapsulates state and behavior of a voltage clamp protocol."""

    HOLDING_STEP = VoltageClampStep(voltage=-0.08, duration=1)

    def __init__(self, steps: List[VoltageClampStep]=None) -> None:
        if steps:
            self.steps = [self.HOLDING_STEP] + steps
            self.voltage_change_endpoints = self.init_voltage_change_endpoints()
        else:
            self.steps = []

    def __eq__(self, other):

        if not isinstance(other, self.__class__):
            return False

        if len(other.steps) != len(self.steps):
            return False

        for i in range(len(other.steps)):
            if other.steps[i] != self.steps[i]:
                return False
        return True

    def __str__(self):
        return ' | '.join([i.__str__() for i in self.steps])

    def __repr__(self):
        return self.__str__()

    def init_voltage_change_endpoints(self) -> List[float]:
        """Initializes voltage change endpoints based on the steps provided.

        For example, if the steps provided are:
            VoltageClampStep(voltage=1, duration=1),
            VoltageClampStep(voltage=2, duration=0.5),
        the voltage change points would be at 1 second and 1.5 seconds.

        Returns:
            A list of voltage change endpoints.
        """

        voltage_change_endpoints = []
        cumulative_time = 0
        for i in self.steps:
            cumulative_time += i.duration
            voltage_change_endpoints.append(cumulative_time)
        return voltage_change_endpoints

    def get_voltage_at_time(self, time: float) -> float:
        """Gets the voltage based on provided steps for the specified time."""
        step_index = bisect.bisect_left(self.voltage_change_endpoints, time)
        if step_index != len(self.voltage_change_endpoints):
            return self.steps[step_index].voltage
        raise ValueError('End of voltage protocol.')

    def get_time_interval_for_voltage(self, voltage):
        """Retrieves the time interval when a particular voltage was active."""
        # Todo how should this handle the case when there are more than
        # Todo one voltages in the protocol?
        # TODO THIS IS A BUG. UNEXPECTED BEHAVIOR WHEN TWO STEPS HAVE SAME
        # TODO VOLTAGE
        for i in range(len(self.steps)):
            if self.steps[i].voltage == voltage:
                index = i
                break
        else:
            raise ValueError('Voltage specified is not in protocol.')

        if index == 0:
            return 0.0, self.voltage_change_endpoints[index]
        else:
            return (self.voltage_change_endpoints[index - 1],
                    self.voltage_change_endpoints[index])


PROTOCOL_TYPE = Union[
    SingleActionPotentialProtocol,
    IrregularPacingProtocol,
    VoltageClampProtocol,
]
