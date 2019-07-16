import paci_2018
import protocols
import numpy as np
import protocols

def save_data_to_file(model_trace):
    model_data = np.array([model_trace.t, model_trace.y]) 
    
    np.savetxt('./data/paci_sap_baseline.csv', model_data, delimiter=',')

def main():
    protocol = protocols.SingleActionPotentialProtocol()
    save_data_to_file(paci_2018.generate_trace(protocol))


if __name__ == '__main__':
    main()

