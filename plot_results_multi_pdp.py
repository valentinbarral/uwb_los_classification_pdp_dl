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



resultsAccuracy = []
resultsExecutioinTime = []
resultsF1 = []
resultsPrecision = []
resultsRecall = []
version = 'v5'

cir_first_size = 152
pdpLFactors = [5,10,20,40]
pdpSizes = map(lambda x: int(cir_first_size/x), pdpLFactors)
useExtraFeatures = False
cir_energy_mode = 0 # 0:raw, 1:normalized



cir_energy_mode_label = ''
if cir_energy_mode==1:
    cir_energy_mode_label = '_normalized'

timeMakePDP= 14.0108*0.7


modesToLoad = [2]
modeLabel = 'pdp'
if (useExtraFeatures):
    modesToLoad = [5]
    modeLabel = 'pdp+extras'

indexPdp = -1

for pdp_size in pdpSizes:
    resultsAccuracy.append([])
    resultsExecutioinTime.append([])
    resultsF1.append([])
    resultsPrecision.append([])
    resultsRecall.append([])
    indexPdp +=1
    for mode in modesToLoad:
        resultsAccuracy[indexPdp].append(np.load('Results_'+version+'/accuracy_'+str(mode)+ '_pdp_'+ str(pdp_size) + cir_energy_mode_label+'.npy'))
        resultsExecutioinTime[indexPdp].append(np.array(np.load('Results_'+version+'/execution_'+str(mode)+ '_pdp_'+ str(pdp_size) + cir_energy_mode_label+'.npy'))+timeMakePDP)
        resultsF1[indexPdp].append(np.load('Results_'+version+'/f1_'+str(mode)+ '_pdp_'+ str(pdp_size) + cir_energy_mode_label+'.npy'))
        resultsPrecision[indexPdp].append(np.load('Results_'+version+'/precision_'+str(mode)+ '_pdp_'+ str(pdp_size) + cir_energy_mode_label+'.npy'))
        resultsRecall[indexPdp].append(np.load('Results_'+version+'/recall_'+str(mode)+ '_pdp_'+ str(pdp_size) + cir_energy_mode_label+'.npy'))



mode =  ['pdp', 'pdp_20', 'pdp_30' , 'others+pdp_10',  'others+pdp_20', 'others+pdp_30']
colors = ['blue', 'purple', 'pink', 'blue', 'purple', 'pink']
ind = np.arange(6) + 1

plt.rc('axes', axisbelow=True)
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=20)

# fig_accuracy = plt.figure()
# plt.axes().minorticks_on()
# plt.grid(b=True, which='both', color='k', linestyle='-', alpha=0.4)
# plt.grid(b=True, which='minor', color='k', linestyle='--', alpha=0.2)
# # plt.bar(mode,resultsAccuracy)
# bp=plt.boxplot(resultsAccuracy,showfliers=False, patch_artist=True, whis=[0,100],showmeans=True)
# for patch, color in zip(bp['boxes'], colors):
#     patch.set_facecolor(color)

mean_patch = mlines.Line2D([], [], color='green', marker='^',
                          markersize=10, label='Mean', linewidth=0)
median_patch = mlines.Line2D([], [], color='red', label='Median')
# plt.legend(handles=[mean_patch, median_patch])

# plt.title('Accuracy')
# plt.ylabel('Accuracy')
# plt.xticks(ind,mode)
# plt.ylim(0.70, 1.0)
# plt.tight_layout()

# plt.savefig("plot_accuracy.pdf")
# plt.show()


# fig_time = plt.figure()
# plt.axes().minorticks_on()
# plt.grid(b=True, which='both', color='k', linestyle='-', alpha=0.4)
# plt.grid(b=True, which='minor', color='k', linestyle='--', alpha=0.2)
# bp=plt.boxplot(resultsExecutioinTime,showfliers=False, patch_artist=True, whis=[0,100],showmeans=True)
# for patch, color in zip(bp['boxes'], colors):
#     patch.set_facecolor(color)
# plt.legend(handles=[mean_patch, median_patch])
# #plt.title('Execution time')
# for aMean in bp['means']:
#     plt.text(aMean._x+0.35, aMean._y -0.005, '%.3f' % aMean._y,
#          horizontalalignment='center',size=18)
# plt.ylabel('Time (s)')
# plt.xticks(ind,mode)
# plt.tight_layout()
# plt.savefig("plot_time.pdf", bbox_inches = 'tight',
#     pad_inches = 0.02)
# plt.show()


# fig_f1 = plt.figure()
# plt.axes().minorticks_on()
# plt.grid(b=True, which='both', color='k', linestyle='-', alpha=0.4)
# plt.grid(b=True, which='minor', color='k', linestyle='--', alpha=0.2)
# bp=plt.boxplot(resultsF1,showfliers=False, patch_artist=True, whis=[0,100],showmeans=True)
# for patch, color in zip(bp['boxes'], colors):
#     patch.set_facecolor(color)
# #plt.title('F1-Score')
# for aMean in bp['means']:
#     plt.text(aMean._x+0.35, aMean._y -0.005, '%.1f' % aMean._y, horizontalalignment='center',size=18)

# plt.legend(handles=[mean_patch, median_patch])
# plt.ylabel('F1-Score')
# plt.xticks(ind,mode)
# plt.ylim(0.70, 1.0)
# plt.tight_layout()
# plt.savefig("plot_f1score.pdf", bbox_inches = 'tight',
#     pad_inches = 0.02)
# plt.show()


# fig_precision = plt.figure()
# plt.axes().minorticks_on()
# plt.grid(b=True, which='both', color='k', linestyle='-', alpha=0.4)
# plt.grid(b=True, which='minor', color='k', linestyle='--', alpha=0.2)
# bp=plt.boxplot(resultsPrecision,showfliers=False, patch_artist=True, whis=[0,100],showmeans=True)
# for patch, color in zip(bp['boxes'], colors):
#     patch.set_facecolor(color)
# plt.title('Precision')
# plt.legend(handles=[mean_patch, median_patch])
# plt.ylabel('Precision')
# plt.ylim(0.70, 1.0)
# plt.xticks(ind,mode)
# plt.tight_layout()
# plt.savefig("plot_precision.pdf")
# plt.show()


# fig_recall = plt.figure()
# plt.axes().minorticks_on()
# plt.grid(b=True, which='both', color='k', linestyle='-', alpha=0.4)
# plt.grid(b=True, which='minor', color='k', linestyle='--', alpha=0.2)
# bp=plt.boxplot(resultsRecall,showfliers=False, patch_artist=True, whis=[0,100],showmeans=True)
# for patch, color in zip(bp['boxes'], colors):
#     patch.set_facecolor(color)
# plt.title('Recall')
# plt.legend(handles=[mean_patch, median_patch])
# plt.ylabel('Recall')
# plt.ylim(0.70, 1.0)
# plt.xticks(ind,mode)
# plt.tight_layout()
# plt.savefig("plot_recall.pdf")
# plt.show()


##########################
# PLOTS ONLY WITH CIR


pdpLabels = ['L=5','L=10', 'L=20', 'L=40']
pdpColors = ['blue', 'purple', 'pink', 'steelblue']

resultsAccuracyOnlycir = np.vstack(resultsAccuracy)
#resultsAccuracyOnlycir= resultsAccuracyOnlycir[:,:,0]
resultsF1Onlycir = np.vstack(resultsF1)
#resultsF1Onlycir[2,0]= 0.76
#resultsF1Onlycir[2,7]= 0.76
#resultsF1Onlycir = resultsF1Onlycir[:,:,0]
resultsExecutioinTimeOnlycir = np.vstack(resultsExecutioinTime)
#resultsExecutioinTimeOnlycir = resultsExecutioinTimeOnlycir[:,:,0]

indOnlyCir = np.arange(4) + 1


fig_f1_only_cir = plt.figure()
plt.axes().minorticks_on()
plt.grid(b=True, which='both', color='k', linestyle='-', alpha=0.4)
plt.grid(b=True, which='minor', color='k', linestyle='--', alpha=0.2)
# plt.bar(mode,resultsAccuracy)
bp=plt.boxplot(resultsF1Onlycir.tolist(),showfliers=True, patch_artist=True, whis=[0,100],showmeans=True)
for patch, color in zip(bp['boxes'], pdpColors):
    patch.set_facecolor(color)

mean_patch = mlines.Line2D([], [], color='green', marker='^',
                          markersize=10, label='Mean', linewidth=0)
median_patch = mlines.Line2D([], [], color='red', label='Median')
plt.legend(handles=[mean_patch, median_patch])

for aMean in bp['means']:
    plt.text(aMean._x+0.35, aMean._y -0.005, '%.3f' % aMean._y,
         horizontalalignment='center',size=18)

#plt.title('F1-Score')
plt.ylabel('F1-Score')
plt.xticks(indOnlyCir,pdpLabels)
plt.ylim(0, 1.0)
plt.tight_layout()
plt.savefig('fig_f1_multi_'+ modeLabel + cir_energy_mode_label+ '.pdf', bbox_inches = 'tight',
    pad_inches = 0.02)
plt.show()


fig_time_only_cir = plt.figure()
plt.axes().minorticks_on()
plt.grid(b=True, which='both', color='k', linestyle='-', alpha=0.4)
plt.grid(b=True, which='minor', color='k', linestyle='--', alpha=0.2)
# plt.bar(mode,resultsAccuracy)
bp=plt.boxplot(resultsExecutioinTimeOnlycir.tolist(),showfliers=True, patch_artist=True, whis=[0,100],showmeans=True)
for patch, color in zip(bp['boxes'], pdpColors):
    patch.set_facecolor(color)

mean_patch = mlines.Line2D([], [], color='green', marker='^',
                          markersize=10, label='Mean', linewidth=0)
median_patch = mlines.Line2D([], [], color='red', label='Median')
plt.legend(handles=[mean_patch, median_patch])

for aMean in bp['means']:
    plt.text(aMean._x+0.35, aMean._y -0.005, '%.1f' % aMean._y,
         horizontalalignment='center',size=18)

#plt.title('Training time')
plt.ylabel('Time (s)')
plt.xticks(indOnlyCir,pdpLabels)
plt.tight_layout()
plt.savefig('fig_time_multi_'+ modeLabel + cir_energy_mode_label+ '.pdf', bbox_inches = 'tight',
    pad_inches = 0.02)
plt.show()

# ##########################
# # PLOTS Others

# resultsAccuracyOther = np.array(resultsAccuracy)[[1,3,5],:]
# resultsF1Other = np.array(resultsF1)[[1,3,5],:]
# resultsExecutioinTimeOther = np.array(resultsExecutioinTime)[[1,3,5],:]
# modeOther = ['cir+extra', 'pdp+extra',  'cir[152]+extra']
# colorsOther = ['blue', 'purple', 'pink']
# indOther = np.arange(3) + 1


# fig_f1_other = plt.figure()
# plt.axes().minorticks_on()
# plt.grid(b=True, which='both', color='k', linestyle='-', alpha=0.4)
# plt.grid(b=True, which='minor', color='k', linestyle='--', alpha=0.2)
# # plt.bar(mode,resultsAccuracy)
# bp=plt.boxplot(resultsF1Other.tolist(),showfliers=False, patch_artist=True, whis=[0,100],showmeans=True)
# for patch, color in zip(bp['boxes'], colorsOther):
#     patch.set_facecolor(color)

# mean_patch = mlines.Line2D([], [], color='green', marker='^',
#                           markersize=10, label='Mean', linewidth=0)
# median_patch = mlines.Line2D([], [], color='red', label='Median')
# plt.legend(handles=[mean_patch, median_patch])

# for aMean in bp['means']:
#     plt.text(aMean._x+0.35, aMean._y -0.005, '%.3f' % aMean._y,
#          horizontalalignment='center',size=18)

# #plt.title('F1-Score')
# plt.ylabel('F1-Score')
# plt.xticks(indOther,modeOther)
# plt.ylim(0.70, 1.0)
# plt.tight_layout()
# plt.savefig("fig_f1_others.pdf", bbox_inches = 'tight',
#     pad_inches = 0.02)
# plt.show()


# fig_time_only_cir = plt.figure()
# plt.axes().minorticks_on()
# plt.grid(b=True, which='both', color='k', linestyle='-', alpha=0.4)
# plt.grid(b=True, which='minor', color='k', linestyle='--', alpha=0.2)
# # plt.bar(mode,resultsAccuracy)
# bp=plt.boxplot(resultsExecutioinTimeOther.tolist(),showfliers=False, patch_artist=True, whis=[0,100],showmeans=True)
# for patch, color in zip(bp['boxes'], colorsOther):
#     patch.set_facecolor(color)

# mean_patch = mlines.Line2D([], [], color='green', marker='^',
#                           markersize=10, label='Mean', linewidth=0)
# median_patch = mlines.Line2D([], [], color='red', label='Median')
# plt.legend(handles=[mean_patch, median_patch])

# for aMean in bp['means']:
#     plt.text(aMean._x+0.35, aMean._y -0.005, '%.1f' % aMean._y,
#          horizontalalignment='center',size=18)

# #plt.title('Training time')
# plt.ylabel('Time (s)')
# plt.xticks(indOther,modeOther)
# plt.tight_layout()
# plt.savefig("fig_time_others.pdf", bbox_inches = 'tight',
#     pad_inches = 0.02)
# plt.show()