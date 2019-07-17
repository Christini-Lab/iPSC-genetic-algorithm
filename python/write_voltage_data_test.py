import unittest
import protocols
import os
import csv
import numpy as np
import paci_2018
import numpy.testing as tst
import write_voltage_data


class TestWritingData(unittest.TestCase):
    def setUpClass():
        path_to_data = './data/paci_sap_baseline.csv'
        path_to_figure = './data/paci_sap_baseline.png' 
        try:
            os.remove(path_to_figure)
            os.remove(path_to_data)
        except:
            pass

        write_voltage_data.main()

    def test_file_exists(self):
        path_to_output = './data/paci_sap_baseline.csv'

        self.assertTrue(os.path.exists(path_to_output), \
                'output file does not exist')  

    def test_write_data(self):
        path_to_output = './data/paci_sap_baseline.csv'
        protocol = protocols.SingleActionPotentialProtocol()
        baseline_trace = paci_2018.generate_trace(protocol)

        write_voltage_data.save_data_to_file(baseline_trace)
        with open(path_to_output) as csv_file:
            csv_reader = csv.reader(csv_file)
            loaded_data = np.array(list(csv_reader)) 

        print(len(loaded_data))

        tst.assert_raises(AssertionError, tst.assert_array_equal, \
                baseline_trace.y, loaded_data[:,1], \
                'loaded data is equal to Paci')

    def test_load_and_plot_data(self):
        path_to_data = './data/paci_sap_baseline.csv'
        path_to_figure = './data/paci_sap_baseline.png' 
        try:
            os.remove(path_to_figure)
        except:
            print('no figure to clean up')
        
        write_voltage_data.load_and_plot_data(path_to_data)

        self.assertTrue(os.path.exists(path_to_figure), \
                'the figure was not created')
