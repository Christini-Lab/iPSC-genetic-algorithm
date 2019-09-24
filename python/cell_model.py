from math import log, sqrt
from typing import List

import numpy as np
from scipy import integrate

import ga_configs
import protocols
import trace


class CellModel:
    """An implementation a general cell model

    Attributes:
        default_parameters: A dict containing tunable parameters
        updated_parameters: A dict containing all parameters that are being
            tuned.
    """

    def __init__(self, y_initial=[], default_parameters=None, updated_parameters=None,
            no_ion_selective_dict=None):
        self.y_initial = y_initial
        self.default_parameters = default_parameters
        self.no_ion_selective = {}
        self.is_no_ion_selective = False

        if updated_parameters:
            self.default_parameters.update(updated_parameters)
        if no_ion_selective_dict:
            self.no_ion_selective = no_ion_selective_dict
            self.is_no_ion_selective = True

        self.t = []
        self.y_voltage = []
        self.d_y_voltage = []
        self.current_response_info = None
        self.full_y = []

    @property
    def no_ion_selective(self):
        return self.__no_ion_selective

    @no_ion_selective.setter
    def no_ion_selective(self, no_ion_selective):
        self.__no_ion_selective = no_ion_selective

    def generate_response(self, protocol, is_no_ion_selective=False):
        """Returns a trace based on the specified target objective.

        Args:
            protocol: An object of a specified protocol.

        Returns:
            A Trace object representing the change in membrane potential over
            time.
        """
        # Reset instance variables when model is run again.
        self.t = []
        self.y_voltage = []
        self.d_y_voltage = []
        self.full_y = []

        self.is_no_ion_selective = is_no_ion_selective

        if isinstance(protocol, protocols.SingleActionPotentialProtocol):
            return self.generate_single_AP_response(protocol)
        elif isinstance(protocol, protocols.IrregularPacingProtocol):
            return self.generate_irregular_pacing_response(protocol)
        elif isinstance(protocol, protocols.VoltageClampProtocol):
            return self.generate_VC_protocol_response(protocol)

    def generate_single_action_potential_function(self):
        def single_action_potential(t, y):
            return self.action_potential_diff_eq(t, y)

        return single_action_potential

    def generate_single_AP_response(self, protocol):
        """
        Args:
            protocol: An object of a specified protocol.

        Returns:
            A single action potential trace
        """
        self.current_response_info = trace.CurrentResponseInfo()
        try:
            solution = integrate.solve_ivp(
                self.generate_single_action_potential_function(),
                [0, protocol.duration],
                self.y_initial,
                method='BDF')
            self._set_data_without_error(solution, is_current_response=True)
        except ValueError:
            print('Model could not produce trace.')
            return None
        return trace.Trace(self.t,
                           self.y_voltage,
                           current_response_info=self.current_response_info)

    def generate_irregular_pacing_response(self, protocol):
        """
        Args:
            protocol: An irregular pacing protocol 
        Returns:
            A irregular pacing trace
        """
        pacing_info = trace.IrregularPacingInfo()
        try:
            solution = integrate.solve_ivp(self.generate_irregular_pacing_function(
                protocol, pacing_info), [0, protocol.duration],
                                self.y_initial,
                                method='BDF',
                                max_step=1e-3)
            self._set_data_without_error(solution)
        except ValueError:
            return None
        return trace.Trace(self.t, self.y_voltage, pacing_info=pacing_info)

    def generate_VC_protocol_response(self, protocol):
        """
        Args:
            protocol: A voltage clamp protocol
        Returns:
            A Trace object for a voltage clamp protocol
        """
        self.current_response_info = trace.CurrentResponseInfo(
            protocol=protocol)
        try:
            solution = integrate.solve_ivp(
                self.generate_voltage_clamp_function(protocol),
                [0, protocol.get_voltage_change_endpoints()[-1]],
                self.y_initial,
                method='BDF',
                max_step=1e-3)
            self._set_data_without_error(solution, is_current_response=True)
        except ValueError:
            return None
        return trace.Trace(self.t,
                           self.y_voltage,
                           current_response_info=self.current_response_info)

    def generate_irregular_pacing_function(self, protocol, pacing_info):
        offset_times = protocol.make_offset_generator()

        def irregular_pacing(t, y):
            d_y = self.action_potential_diff_eq(t, y)

            if pacing_info.detect_peak(self.t, y[0], self.d_y_voltage):
                pacing_info.peaks.append(t)
                voltage_diff = abs(pacing_info.AVG_AP_START_VOLTAGE - y[0])
                pacing_info.apd_90_end_voltage = y[0] - voltage_diff * 0.9

            if pacing_info.detect_apd_90(y[0]):
                try:
                    pacing_info.add_apd_90(t)
                    pacing_info.stimulations.append(t + next(offset_times))
                except StopIteration:
                    pass

            if pacing_info.should_stimulate(t):
                i_stimulation = protocol.STIM_AMPLITUDE_AMPS / self.cm_farad
            else:
                i_stimulation = 0.0

            d_y[0] += i_stimulation
            return d_y

        return irregular_pacing

    def generate_voltage_clamp_function(self, protocol):
        def voltage_clamp(t, y):
            y[0] = protocol.get_voltage_at_time(t)
            return self.action_potential_diff_eq(t, y)

        return voltage_clamp

    def _set_data_without_error(self, solution, is_current_response=False):
        """This method retroactively removes all unused steps from self.t,
           self.y_voltage, self.full_y, and self.d_y_voltage after integrator
           finishes.
           This method was made for use ONLY with the forllowing methods:
               - .generate_single_AP_response()
               - .generate_irregular_pacing_response()
               - .generate_VC_protocol_response()
           These methods call solve_ivp(), which iterates over the
           right hand side with a variable-sized timestep integrator.
           To track current and a couple other parameters, we write to
           the self.parameter list during each iteration, despite the fact
           that the BDF integrator may throw out an interation.
        """
        time_full = np.asarray(self.t)
        [un, indices] = np.unique(np.flip(time_full), return_index=True)
        new_indices = np.abs(indices - len(time_full))
        mask = np.invert(np.insert(np.diff(new_indices) < 0, [0], False))
        correct_indices = new_indices[mask] - 1

        self.t = np.asarray(self.t)[correct_indices].tolist()
        self.y_voltage = np.asarray(self.y_voltage)[correct_indices].tolist()
        self.full_y =  np.asarray(self.full_y)[correct_indices].tolist()
        self.d_y_voltage = \
            np.asarray(self.d_y_voltage)[correct_indices].tolist()

        correct_currents = trace.CurrentResponseInfo()
        for i in correct_indices:
            if is_current_response:
                correct_currents.currents.append(
                       self.current_response_info.currents[i])

        self.current_response_info.currents = correct_currents.currents

