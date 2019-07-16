import unittest
import protocols
import os
import csv
import numpy as np
import paci_2018
import numpy.testing as tst

class TestWritingData(unittest.TestCase):
    def test_file_exists(self):
        path_to_output = './data/paci_sap_baseline.csv'

        self.assertTrue(os.path.exists(path_to_output), \
                'output file does not exist')  

    def test_write_data(self):
        path_to_output = './data/paci_sap_baseline.csv'
        protocol = protocols.SingleActionPotentialProtocol()
        
        baseline_trace = paci_2018.generate_trace(protocol)
        with open(path_to_output) as csv_file:
            csv_reader = csv.reader(csv_file)
            loaded_data = np.array(list(csv_reader)) 

        tst.assert_raises(AssertionError, tst.assert_array_equal, \
                baseline_trace.y, loaded_data[:,1], \
                'loaded data is equal to Paci')

