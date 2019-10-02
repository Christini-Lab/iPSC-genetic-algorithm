from paci_2018 import PaciModel
from kernik import KernikModel
import protocols
import matplotlib.pyplot as plt
import numpy as np
import pdb


paci_protocol = protocols.SingleActionPotentialProtocol(1.8)
baseline_paci = PaciModel()
baseline_paci.generate_response(paci_protocol)

kernik_protocol = protocols.SingleActionPotentialProtocol(1800)
baseline_kernik = KernikModel()
baseline_kernik.generate_response(kernik_protocol)

plt.plot([t*1000 for t in baseline_paci.t], [v*1000 for v in baseline_paci.y_voltage], label='Paci')
plt.plot(baseline_kernik.t, baseline_kernik.y_voltage, label='Kernik')
plt.xlabel('Time (ms)', fontsize=20)
plt.ylabel('Voltage', fontsize=20)
plt.legend()

plt.show()

t_max_p = baseline_paci.t[np.array(baseline_paci.y_voltage).argmax()]*1000
t_max_k = baseline_kernik.t[np.array(baseline_kernik.y_voltage).argmax()]
pdb.set_trace()

plt.plot(np.array(baseline_paci.t)*1000-t_max_p, [v*1000 for v in baseline_paci.y_voltage], label='Paci')
plt.plot(np.array(baseline_kernik.t)-t_max_k, baseline_kernik.y_voltage, label='Kernik')
plt.xlabel('Time (ms)', fontsize=20)
plt.ylabel('Voltage', fontsize=20)
plt.legend()
plt.show()
