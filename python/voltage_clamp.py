"""Contains class representing an voltage clamp protocol."""

import bisect


class VoltageClampSteps:

    def __init__(self, voltage, duration):
        self.voltage = voltage
        self.duration = duration


class VoltageClampProtocol:
    """Encapsulates state and behavior of a voltage clamp protocol."""

    def __init__(self, steps):
        self.steps = steps
        self.voltage_change_endpoints = self.init_voltage_change_endpoints()

    def init_voltage_change_endpoints(self):
        voltage_change_endpoints = []
        cumulative_time = 0
        for i in self.steps:
            cumulative_time += i.duration
            voltage_change_endpoints.append(cumulative_time)
        return voltage_change_endpoints

    def get_voltage_at_time(self, time):
        step_index = bisect.bisect_left(self.voltage_change_endpoints, time)
        if step_index != len(self.voltage_change_endpoints):
            return self.steps[step_index].voltage
        raise ValueError('End of voltage protocol.')
