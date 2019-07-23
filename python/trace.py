"""Contains three classes containing information about a trace."""

from typing import List

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import protocols


class IrregularPacingInfo:
    """Contains information regarding irregular pacing.

    Attributes:
        peaks: Times when a AP reaches its peak.
        stimulations: Times when cell is stimulated.
        diastole_starts: Times when the diastolic period begins.
        apd_90_end_voltage: The voltage at next APD 90. Is set to -1 to indicate
            voltage has not yet been calculated.
        apd_90s: Times of APD 90s.
    """

    _STIMULATION_DURATION = 0.005
    _PEAK_DETECTION_THRESHOLD = 0.025
    _MIN_VOLT_DIFF = 0.00001
    _PEAK_MIN_DIS = 1.5
    AVG_AP_START_VOLTAGE = -0.075

    def __init__(self) -> None:
        self.peaks = []
        self.stimulations = []
        self.diastole_starts = []

        # Set to -1 to indicate it has not yet been set.
        self.apd_90_end_voltage = -1
        self.apd_90s = []

    def add_apd_90(self, apd_90: float) -> None:
        self.apd_90s.append(apd_90)
        self.apd_90_end_voltage = -1

    def should_stimulate(self, t: float) -> bool:
        """Checks whether stimulation should occur given a time point."""
        for i in range(len(self.stimulations)):
            distance_from_stimulation = t - self.stimulations[i]
            if 0 < distance_from_stimulation < self._STIMULATION_DURATION:
                return True
        return False

    def plot_stimulations(self, trace: 'Trace') -> None:
        stimulation_y_values = _find_trace_y_values(
            trace=trace,
            timings=self.stimulations)

        sti = plt.scatter(self.stimulations, stimulation_y_values, c='red')
        plt.legend((sti,), ('Stimulation',), loc='upper right')

    def plot_peaks_and_apd_ends(self, trace: 'Trace') -> None:
        peak_y_values = _find_trace_y_values(
            trace=trace,
            timings=self.peaks)
        apd_end_y_values = _find_trace_y_values(
            trace=trace,
            timings=self.apd_90s)

        peaks = plt.scatter(self.peaks, peak_y_values, c='red')
        apd_end = plt.scatter(
            self.apd_90s,
            apd_end_y_values,
            c='orange')
        plt.legend((peaks, apd_end), ('Peaks', 'APD 90'), loc='upper right')

    def detect_peak(self,
                    t: List[float],
                    y_voltage: float,
                    d_y_voltage: List[float]) -> bool:
        # Skip check on first few points.
        if len(t) < 2:
            return False

        if y_voltage < self._PEAK_DETECTION_THRESHOLD:
            return False
        if d_y_voltage[-1] <= 0 < d_y_voltage[-2]:
            # TODO edit so that successive peaks are discovered. Decrease peak
            # TODO mean distance.
            if not (self.peaks and t[-1] - self.peaks[-1] < self._PEAK_MIN_DIS):
                return True
        return False

    def detect_apd_90(self, y_voltage: float) -> bool:
        return self.apd_90_end_voltage != -1 and abs(
            self.apd_90_end_voltage - y_voltage) < 0.001


class Current:
    """Encapsulates a current at a single time step."""

    def __init__(self, name: str, value: float):
        self.name = name
        self.value = value

    def __str__(self):
        return '{}: {}'.format(self.name, self.value)

    def __repr__(self):
        return '{}: {}'.format(self.name, self.value)


class CurrentResponseInfo:
    """Contains info of currents in response to voltage clamp protocol.

    Attributes:
        protocol: Specifies voltage clamp protocol which created the current
            response.
        currents: A list of current timesteps.

    """

    def __init__(self, protocol: protocols.VoltageClampProtocol=None) -> None:
        self.protocol = protocol
        self.currents = []

    def calculate_current_contribution(
            self,
            timings: List[float],
            start_t: float,
            end_t: float,
            target_currents: List[str]=None) -> pd.DataFrame:
        """Calculates the contribution of each current over a period of time."""
        start_index = _find_closest_t_index(timings, start_t)
        end_index = _find_closest_t_index(timings, end_t)

        total_current_sum = 0
        individual_sums = dict()
        for i in range(start_index, end_index + 1, 1):
            for j in self.currents[i]:
                abs_j_value = abs(j.value)
                total_current_sum += abs_j_value

                # Only calculate current contributions of currents provided in
                # target currents (if they were provided).
                if target_currents and j.name not in target_currents:
                    continue

                if j.name in individual_sums:
                    individual_sums[j.name] += abs_j_value
                else:
                    individual_sums[j.name] = abs_j_value

        fraction_sums = dict()
        for name, value in individual_sums.items():
            fraction_sums[name] = value / total_current_sum

        percent_contributions = []
        parameter_names = []
        for key, val in fraction_sums.items():
            parameter_names.append(key)
            percent_contributions.append(val)

        df = pd.DataFrame(
            data={'Parameter': parameter_names,
                  'Max Percent Contribution': percent_contributions})
        df['Max Percent Contribution'] = pd.to_numeric(df['Max Percent Contribution'])
        return df

    def get_current_summed(self):
        current = []
        for i in self.currents:
            current.append(sum([j.value for j in i]))

        median_current = np.median(current)
        for i in range(len(current)):
            if abs(current[i] - median_current) > 5:
                current[i] = 0
        normalized = 2. * (current - np.min(current)) / np.ptp(current) - 1
        return normalized / 4

    def get_single_current_values(self, current_name: str) -> List[float]:
        # Find the index of the specified current.
        for i in range(len(self.currents[0])):
            if self.currents[0][i].name == current_name:
                current_index = i
                break
        else:
            raise ValueError('Current {} is not found.'.format(current_name))

        values = []
        for current in self.currents:
            values.append(current[current_index].value)
        return values


def _find_trace_y_values(trace, timings):
    """Given a trace, finds the y values of the timings provided."""
    y_values = []
    for i in timings:
        array = np.asarray(trace.t)
        index = _find_closest_t_index(array, i)
        y_values.append(trace.y[index])
    return y_values


def _find_closest_t_index(array, t):
    """Given an array, return the index with the value closest to t."""
    return (np.abs(np.array(array) - t)).argmin()


class Trace:
    """Represents a spontaneous or probed response from cell.

    Attributes:
        t: Timestamps of the response.
        y: The membrane voltage, in volts, at a point in time.
        pacing_info: Contains additional information about cell pacing. Will be
            None if no pacing has occurred.
        current_response_info: Contains information about individual currents
            in the cell. Will be set to None if the voltage clamp protocol was
            not used.
    """

    def __init__(self,
                 t: List[float],
                 y: List[float],
                 pacing_info: IrregularPacingInfo=None,
                 current_response_info: CurrentResponseInfo=None) -> None:
        self.t = t
        self.y = y
        self.pacing_info = pacing_info
        self.current_response_info = current_response_info

    def plot(self):
        plt.plot(self.t, self.y)

    def plot_with_currents(self):
        if not self.current_response_info:
            return ValueError('Trace does not have current info stored. Trace '
                              'was not generated with voltage clamp protocol.')

        voltage_line, = plt.plot(self.t, self.y, 'b', label='Voltage')
        current_line, = plt.plot(
            self.t,
            self.current_response_info.get_current_summed(),
            'r--',
            label='Current')
        plt.legend(handles=[voltage_line, current_line])

    def plot_only_currents(self, color):
        if not self.current_response_info:
            return ValueError('Trace does not have current info stored. Trace '
                              'was not generated with voltage clamp protocol.')

        plt.plot(self.t, self.current_response_info.get_current_summed(), color)
