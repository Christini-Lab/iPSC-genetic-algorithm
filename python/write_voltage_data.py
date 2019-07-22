import paci_2018
import numpy as np
import protocols
import matplotlib.pyplot as plt


def save_data_to_file(model_trace):
    model_data = np.transpose(np.array([
        np.asarray(model_trace.t), 
        np.asarray(model_trace.y)]))
    
    np.savetxt('./data/paci_sap_baseline.csv', model_data, delimiter=',')


def load_and_plot_data(path_to_output):
    trace_data = np.loadtxt(path_to_output, delimiter=',')

    plt.plot(trace_data[:,0], trace_data[:,1])
    plt.savefig('./data/paci_sap_baseline.png')


def main():
    protocol = protocols.SingleActionPotentialProtocol()
    save_data_to_file(paci_2018.generate_trace(protocol))
    load_and_plot_data('./data/paci_sap_baseline.csv')


if __name__ == '__main__':
    main()

