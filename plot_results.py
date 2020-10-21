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

resultsAccuracy = []
resultsExecutioinTime = []
resultsF1 = []
resultsPrecision = []
resultsRecall = []

timeMakePDP= 14.0108*0.7

sequence = [0,3,2,1,4,5,6,7] 

plot_only_cir = True
plot_cir_and_extra = True

pdp_size = 30

cir_energy_mode = 1 # 0:raw, 1:normalized
cir_energy_mode_label = ''
if cir_energy_mode==1:
    cir_energy_mode_label = '_normalized'

# #NORMALIZED
# for mode in sequence:
#     # resultsAccuracy.append(np.load('Results/accuracy_'+str(mode)+'_normalized_energy'+'.npy'))
#     if (mode==2 or mode == 1):
#         resultsExecutioinTime.append(np.array(np.load('Results/execution_'+str(mode)+'_normalized_energy'+'.npy'))+timeMakePDP)
#     else:
#         resultsExecutioinTime.append(np.load('Results/execution_'+str(mode)+'_normalized_energy'+'.npy'))
    
#     resultsF1.append(np.load('Results/f1_'+str(mode)+'_normalized_energy'+'.npy'))
#     # resultsPrecision.append(np.load('Results/precision_'+str(mode)+'_normalized_energy'+'.npy'))
#     # resultsRecall.append(np.load('Results/recall_'+str(mode)+'_normalized_energy'+'.npy'))


sequence = [0,1,2,3,4,5] 
#NOT NORMALIZED
for mode in sequence:
    # resultsAccuracy.append(np.load('Results/accuracy_'+str(mode)+'.npy'))
    if (mode==2 or mode == 5):
        resultsExecutioinTime.append(np.array(np.load('Results_'+version+'/execution_'+str(mode)+ '_pdp_'+ str(pdp_size) +cir_energy_mode_label+'.npy'))+timeMakePDP)
        resultsF1.append(np.load('Results_'+version+'/f1_'+str(mode)+ '_pdp_'+ str(pdp_size) +cir_energy_mode_label+'.npy'))
    else:
        resultsExecutioinTime.append(np.load('Results_'+version+'/execution_'+str(mode)+ cir_energy_mode_label+'.npy'))
        resultsF1.append(np.load('Results_'+version+'/f1_'+str(mode)+cir_energy_mode_label+'.npy'))
        
    
    
    # resultsPrecision.append(np.load('Results/precision_'+str(mode)+'.npy'))
    # resultsRecall.append(np.load('Results/recall_'+str(mode)+'.npy'))


mode = ['cir', 'others+cir', 'pdp', 'others+pdp', 'cir[152]', 'others+cir[152]', 'others']
colors = ['blue', 'blue', 'purple', 'purple', 'pink', 'pink', 'steelblue']
ind = np.arange(7) + 1

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

if plot_only_cir:
    resultsSelection = [0,1,2]
    resultsF1Onlycir = np.array(resultsF1)[resultsSelection]
    resultsExecutioinTimeOnlycir = np.array(resultsExecutioinTime)[resultsSelection]
    modeOnlyCir = ['cir', 'cir[152]', 'pdp(L=5)']
    colorsOnlyCir = ['blue','steelblue', 'pink']
    indOnlyCir = np.arange(len(resultsSelection)) + 1


    fig_f1_only_cir = plt.figure()
    plt.axes().minorticks_on()
    plt.grid(b=True, which='both', color='k', linestyle='-', alpha=0.4)
    plt.grid(b=True, which='minor', color='k', linestyle='--', alpha=0.2)
    # plt.bar(mode,resultsAccuracy)
    bp=plt.boxplot(resultsF1Onlycir.tolist(),showfliers=True, patch_artist=True, whis=[0,100],showmeans=True)
    for patch, color in zip(bp['boxes'], colorsOnlyCir):
        patch.set_facecolor(color)

    mean_patch = mlines.Line2D([], [], color='green', marker='^',
                            markersize=10, label='Mean', linewidth=0)
    median_patch = mlines.Line2D([], [], color='red', label='Median')
    plt.legend(handles=[mean_patch, median_patch])

    for aMean in bp['means']:
        plt.text(aMean._x+0.31, aMean._y -0.038, '%.3f' % aMean._y,
            horizontalalignment='center',size=18)

    #plt.title('F1-Score')
    plt.ylabel('F1-Score')
    plt.xticks(indOnlyCir,modeOnlyCir)
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig('Results_'+version+'/fig_f1'  + cir_energy_mode_label +'.pdf', bbox_inches = 'tight',
        pad_inches = 0.02)
    plt.show()


    fig_time_only_cir = plt.figure()
    plt.axes().minorticks_on()
    plt.grid(b=True, which='both', color='k', linestyle='-', alpha=0.4)
    plt.grid(b=True, which='minor', color='k', linestyle='--', alpha=0.2)
    # plt.bar(mode,resultsAccuracy)
    bp=plt.boxplot(resultsExecutioinTimeOnlycir.tolist(),showfliers=True, patch_artist=True, whis=[0,100],showmeans=True)
    for patch, color in zip(bp['boxes'], colorsOnlyCir):
        patch.set_facecolor(color)

    mean_patch = mlines.Line2D([], [], color='green', marker='^',
                            markersize=10, label='Mean', linewidth=0)
    median_patch = mlines.Line2D([], [], color='red', label='Median')
    plt.legend(handles=[mean_patch, median_patch])

    for aMean in bp['means']:
        plt.text(aMean._x+0.35, aMean._y -0.001, '%.1f' % aMean._y,
            horizontalalignment='center',size=18)

    #plt.title('Training time')
    plt.ylabel('Time (s)')
    plt.xticks(indOnlyCir,modeOnlyCir)
    plt.tight_layout()
    plt.savefig('Results_'+version+'/fig_time'  + cir_energy_mode_label +'.pdf', bbox_inches = 'tight',
        pad_inches = 0.02)
    plt.show()

##########################
# PLOTS Others

if plot_cir_and_extra:
    resultsSelection = [3,4,5]
    resultsF1Other = np.array(resultsF1)[resultsSelection]
    resultsExecutioinTimeOther = np.array(resultsExecutioinTime)[resultsSelection]
    modeOther = ['cir\n+extra', 'cir[152]\n+extra', 'pdp(L=5)\n+extra']
    colorsOther = ['blue','steelblue', 'pink']
    indOther = np.arange(len(resultsSelection)) + 1


    fig_f1_other = plt.figure()
    plt.axes().minorticks_on()
    plt.grid(b=True, which='both', color='k', linestyle='-', alpha=0.4)
    plt.grid(b=True, which='minor', color='k', linestyle='--', alpha=0.2)
    # plt.bar(mode,resultsAccuracy)
    bp=plt.boxplot(resultsF1Other.tolist(),showfliers=True, patch_artist=True, whis=[0,100],showmeans=True)
    for patch, color in zip(bp['boxes'], colorsOther):
        patch.set_facecolor(color)

    mean_patch = mlines.Line2D([], [], color='green', marker='^',
                            markersize=10, label='Mean', linewidth=0)
    median_patch = mlines.Line2D([], [], color='red', label='Median')
    plt.legend(handles=[mean_patch, median_patch])

    for aMean in bp['means']:
        plt.text(aMean._x+0.30, aMean._y -0.032, '%.3f' % aMean._y,
            horizontalalignment='center',size=18)

    #plt.title('F1-Score')
    plt.ylabel('F1-Score')
    plt.xticks(indOther,modeOther)
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig('Results_'+version+'/fig_f1_others'  + cir_energy_mode_label +'.pdf', bbox_inches = 'tight',
        pad_inches = 0.02)
    plt.show()


    fig_time_only_cir = plt.figure()
    plt.axes().minorticks_on()
    plt.grid(b=True, which='both', color='k', linestyle='-', alpha=0.4)
    plt.grid(b=True, which='minor', color='k', linestyle='--', alpha=0.2)
    # plt.bar(mode,resultsAccuracy)
    bp=plt.boxplot(resultsExecutioinTimeOther.tolist(),showfliers=True, patch_artist=True, whis=[0,100],showmeans=True)
    for patch, color in zip(bp['boxes'], colorsOther):
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
    plt.xticks(indOther,modeOther)
    plt.tight_layout()
    plt.savefig('Results_'+version+'/fig_time_others'  + cir_energy_mode_label +'.pdf', bbox_inches = 'tight',
        pad_inches = 0.02)
    plt.show()