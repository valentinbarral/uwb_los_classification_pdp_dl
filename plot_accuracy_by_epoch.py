# MIT License

# Copyright (c) 2020 Group of Electronic Technology and Communications. University of A Coruna.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
import matplotlib.patches as mpatches
import matplotlib.lines as mlines


version = 'v5'

resultsAccuracyByEpoch = []
resultsValidationAccuracyByEpoch = []

modeToPlot = 0
pdp_size = 30
indexRep = 0

cir_energy_mode = 0 # 0:raw, 1:normalized
cir_energy_mode_label = ''
if cir_energy_mode==1:
    cir_energy_mode_label = '_normalized'

mode = ['cir', 'others+cir', 'pdp', 'others+pdp', 'cir[152]', 'others+cir[152]', 'others']

if (modeToPlot==2 or modeToPlot == 5):
    resultsAccuracyByEpoch.append(np.array(np.load('Results_'+version+'/accuracy_by_epoch_'+str(modeToPlot)+ '_pdp_'+ str(pdp_size) +cir_energy_mode_label+'.npy'))+timeMakePDP)
    resultsValidationAccuracyByEpoch.append(np.load('Results_'+version+'/validation_accuracy_by_epoch_'+str(modeToPlot)+ '_pdp_'+ str(pdp_size) +cir_energy_mode_label+'.npy'))
else:
    resultsAccuracyByEpoch.append(np.load('Results_'+version+'/accuracy_by_epoch_'+str(modeToPlot)+ cir_energy_mode_label+'.npy'))
    resultsValidationAccuracyByEpoch.append(np.load('Results_'+version+'/validation_accuracy_by_epoch_'+str(modeToPlot)+cir_energy_mode_label+'.npy'))


fig_accuracy_by_epoch = plt.figure()
plt.plot(resultsAccuracyByEpoch[0][indexRep])
plt.plot(resultsValidationAccuracyByEpoch[0][indexRep])
plt.title('Accuracy using ' + mode[modeToPlot])
plt.axes().minorticks_on()
plt.grid(b=True, which='both', color='k', linestyle='-', alpha=0.4)
plt.grid(b=True, which='minor', color='k', linestyle='--', alpha=0.2)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.tight_layout()

if (modeToPlot==2 or modeToPlot == 5):
    resultsAccuracyByEpoch.append(np.array(np.load('Results_'+version+'/accuracy_by_epoch_'+str(modeToPlot)+ '_pdp_'+ str(pdp_size) +cir_energy_mode_label+'.npy'))+timeMakePDP)
    plt.savefig('Results_'+version+'/fig_accuracy_by_epoch_' +str(modeToPlot)+ '_pdp_'+ str(pdp_size)+ cir_energy_mode_label +'.pdf', bbox_inches = 'tight', pad_inches = 0.02)
else:
    plt.savefig('Results_'+version+'/fig_accuracy_by_epoch_' +str(modeToPlot) + cir_energy_mode_label +'.pdf', bbox_inches = 'tight', pad_inches = 0.02)

plt.show()