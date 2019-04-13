import sys
import os

def evaluate(folder, epoch_max, epoch_step):
    # epoch_max, epoch_step
    os.system("python3.6 export_accuracy_recall.py "+str(folder)+" "+str(epoch_max)+" "+str(epoch_step))

    #os.system("python3.6 plot_contacts_per_length.py /usr/data/cvpr_shared/biology/discr_rna/Data/ComparativeRNA/DP/CompAll/comparative.dp dp 0 300")

    # epoch_max, epoch_step, normalize, extension
    os.system("python3.6 plot_histogram.py  "+str(folder)+" "+str(epoch_max)+" "+str(epoch_step)+ " True")

    # filename, extension
    os.system("python3.6 plot_quality_from_csv.py  "+str(folder)+"/accuracy.csv")
    os.system("python3.6 plot_quality_from_csv.py "+str(folder)+"/precision.csv")
    os.system("python3.6 plot_quality_from_csv.py "+str(folder)+"/recall.csv")

    # statsfile, epoch_max, extension
    os.system("python3.6 plot_statistics.py  "+str(folder)+"/stats.npy")

    # sequence, epoch_max, epoch_step, thresh_stepsize, extension
    #os.system("python3.6 plot_roc.py /usr/data/cvpr_shared/biology/discr_rna/training/17-8 2 26 1")

    #os.system("python3.6 plot_targets.py /usr/data/cvpr_shared/biology/discr_rna/training/17-8/")

if __name__ == '__main__':
    folder = str(sys.argv[1])
    epoch_max = int(sys.argv[2])
    epoch_step = int(sys.argv[3])
    evaluate(folder, epoch_max, epoch_step)
    #eg /usr/data/cvpr_shared/biology/discr_rna/training/17-8