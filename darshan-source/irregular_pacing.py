"""Contains class representing an irregular pacing protocol."""


class IrregularPacingProtocol(object):
    """Encapsulates state and behavior of a irregular pacing protocol.

    Attributes:
        duration: The duration of integration.
        stimulation_offsets: A list of numbers. Each number corresponds to the
            seconds after diastole begins that stimulation will
            occur. Cannot exceed `max_stim_interval_duration`, which is the
            time between spontaneous beats.
        DIAS_THRESHOLD_VOLTAGE: The start of a diastole must be below this
            voltage, in Vm.
        MAX_STIM_INTERVAL: Time between naturally occurring spontaneous beats.
        STIM_AMPLITUDE_AMPS: Amplitude of induced stimulation, in amperes.
        STIM_DURATION: Duration for which to stimulate, in seconds.
    """

    DIAS_THRESHOLD_VOLTAGE = -0.06
    STIM_AMPLITUDE_AMPS = 7.5e-10
    _MAX_STIM_INTERVAL = 1.55
    _STIM_DURATION = 0.005

    stimulation_times = []
    diastole_starts = []

    def __init__(self, duration, stimulation_offsets):
        self.duration = duration
        self.stimulation_offsets = stimulation_offsets

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

    def add_stimulation_time(self, t):
        self.stimulation_times.append(t)

    def should_stimulate(self, t):
        for i in range(len(self.stimulation_times)):
            if abs(self.stimulation_times[i] - t) < self._STIM_DURATION:
                return True
        return False
