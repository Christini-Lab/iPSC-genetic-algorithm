from paci_2018 import PaciModel
from kernik import KernikModel
import protocols
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdb


def plot_voltage_currents():
    kernik_protocol = protocols.SingleActionPotentialProtocol(983)
    baseline_kernik = KernikModel()
    baseline_kernik.generate_response(kernik_protocol)

    coi = ['I_K1', 'I_Kr', 'I_Ks', 'I_CaL', 'I_Na', 'I_To']
    currents_df = get_current_data(baseline_kernik.current_response_info.currents, coi, baseline_kernik.t)

    fig = plt.figure(figsize=(10,8))
    plt.plot(baseline_kernik.t, baseline_kernik.y_voltage, label='Kernik')
    plt.axis([0, 983, -80, 40])
    ax = plt.gca()
    ax.tick_params(labelsize=18)
    ax.tick_params(axis='x', labelcolor='none')
    plt.ylabel('Voltage (mV)', fontsize=20)
    #plt.savefig('figures/voltage.svg')

    plt.show()

def get_current_data(currents, currents_of_interest, time):
    new_currents = []
    for temp_currents in currents:
        currents_of_interest
        new_currents.append([current.value for current in temp_currents if current.name in ['I_K1', 'I_Kr', 'I_Ks', 'I_CaL', 'I_Na', 'I_To']])
    
    new_currents = np.array(new_currents)
    currents_df = pd.DataFrame(new_currents, columns=currents_of_interest)
    currents_df['Time (ms)'] = np.array(time)
    fig = plt.figure(figsize=(10,8))
    axarr = fig.add_subplot(611)
    axes = currents_df.set_index('Time (ms)').plot(subplots=True, figsize=(8,6), ax=axarr, sharex=True, fontsize=18)
    plt.xticks(rotation=0)
    axes[5].set_xlabel('Time (ms)', fontsize=20)
    index = 0
    for ax in axes:
        ax.legend(loc=2)
        #ax.tick_params(axis='y', length=0, labelcolor='none')

    ax = fig.add_subplot(111, frameon=False)
    ax.tick_params(labelcolor='none', length=0, top='off', bottom='off', left='off', right='off')
    ax.set_ylabel('Currents', labelpad=20, fontsize=20)
    #plt.savefig('figures/currents.svg')
    return currents_df

plot_voltage_currents()



#paci_protocol = protocols.SingleActionPotentialProtocol(1.8)
#baseline_paci = PaciModel()
#baseline_paci.generate_response(paci_protocol)


#t_max_p = baseline_paci.t[np.array(baseline_paci.y_voltage).argmax()]*1000
#t_max_k = baseline_kernik.t[np.array(baseline_kernik.y_voltage).argmax()]
pdb.set_trace()

#plt.plot(np.array(baseline_paci.t)*1000-t_max_p, [v*1000 for v in baseline_paci.y_voltage], label='Paci')
#plt.plot(np.array(baseline_kernik.t)-t_max_k, baseline_kernik.y_voltage, label='Kernik')
#plt.xlabel('Time (ms)', fontsize=20)
#plt.ylabel('Voltage', fontsize=20)
#plt.legend()
#plt.show()
