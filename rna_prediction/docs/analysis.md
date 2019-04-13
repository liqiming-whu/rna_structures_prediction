# Data analysis

The scripts in this folder can be used to analyze input data, network
performance and training outputs. Most scripts should be executed in the output
folder generated during a training run.


## Export accuracy and recall

Usage: `export_accuracy_recall.py directory epoch_max epoch_step [threshold]`

Exports the network's accuracy, precision and recall on the validation data
during one training run, one entry per epoch and sequence.
The data is stored in CSV files.  
To be used on the output folder of a network run.

* `directory`: The folder to apply to the script to.  
This should be the output folder generated during a training run.

* `epoch_max`: The epoch until which the statistics are supposed to be exported.

* `epoch_step`: Sets the distance between epochs to be output.  
Should be exactly the distance between epochs in the `predictions` subfolder.

* `[threshold]`: The threshold to binary classify the network output.
Values greater than `threshold` are considered positive,
smaller values are considered negative.  
Default: `0.5`


## Plot contacts per length

Usage: `plot_contacts_per_length.py filenam data_format [minlen] [maxlen] [extension]`

Creates a scatter plot of sequence length vs number of contacts  
from a data source in *Dot Parentheses* and *BpSeq* format.

* `filename`: The file of the database to be evaluated

* `data_format`: Decides whether to read *Dot Parentheses* or *BpSeq* or *JSON* format.  
Options: `dp` for *Dot Parentheses*, `bp` for *BpSeq*, `json` for *JSON*

* `[minlen]`: The minimum sequence length to be considered in the plot

* `[maxlen]`: The maximum sequence length to be considered in the plot

* `[extension]`: The extension for the generated image.
Supports the default *matplotlib* formats like `svg` or `png`.  
Default: `svg`


## Plot histogram

Usage: `plot_histogram.py directory epoch_max epoch_step [normalize] [extension]`

Plots a histogram of the network's positive and negative predictions
on the validation data during one training run, one plot per epoch.  
To be used on the output folder of a network run.

* `directory`: The folder to apply to the script to.  
This should be the output folder generated during a training run.

* `epoch_max`: The epoch until the histograms are to be plotted.

* `epoch_step`: Sets the distance between epochs to be plotted.  
Should be exactly the distance between epochs in the `predictions` subfolder.

* `[normalize]`: If set to True, the areas under both the positive and negative
graph are normalized to 1.
This is necessary with bigger sequence lengths because the number of negatives
exhibits quadratic growth relative to the number of positives.  
Options: `True`, `False`  
Default: `False`

* `[extension]`: The extension for the generated image.
Supports the default *matplotlib* formats like `svg` or `png`.  
Default: `svg`


## Plot per sequence statistics

Usage: `plot_per_sequence_statistics.py directory sequence epoch_max epoch_step [threshold] [extension]`

Plots accuracy, precision and recall for one sequence per epoch in a line graph.  
To be used on the output folder of a network run.

* `directory`: The folder to apply to the script to.  
This should be the output folder generated during a training run.

* `sequence`: The sequence for which the statistics are to be generated.  
Sequence ids match the zero based index of the sequence in `targets.json`.

* `epoch_max`: The epoch until the histograms are to be plotted.

* `epoch_step`: Sets the distance between epochs to be plotted.  
Should be exactly the distance between epochs in the `predictions` subfolder.

* `[threshold]`: The threshold to binary classify the network output.
Values greater than `threshold` are considered positive,
smaller values are considered negative.  
Default: `0.5`

* `[extension]`: The extension for the generated image.
Supports the default *matplotlib* formats like `svg` or `png`.  
Default: `svg`


## Plot predictions

Usage: `plot_predictions.py directory sequence epoch_max epoch_step [extension]`

Plots the output of the network as a contact map for a specific sequence,
one plot per epoch.  
To be used on the output folder of a network run.

* `directory`: The folder to apply to the script to.  
This should be the output folder generated during a training run.

* `sequence`: The sequence for which the statistics are to be generated.  
Sequence ids match the zero based index of the sequence in `targets.json`.

* `epoch_max`: The epoch until the histograms are to be plotted.

* `epoch_step`: Sets the distance between epochs to be plotted.  
Should be exactly the distance between epochs in the `predictions` subfolder.

* `[extension]`: The extension for the generated image.
Supports the default *matplotlib* formats like `svg` or `png`.  
Default: `svg`


## Plot quality from CSV

Usage: `plot_quality_from_csv.py filename [filter] [filter_length] [extension]`

Creates a scatter plot for sequence length vs quality measures,
one plot per epoch.  
To be used on the CSVs generated by `export_accuracy_recall.py`.

* `filename`: The csv file to be evaluated.

* `filter`: The filter to be used to smooth the point plot. 
Default: `'None'`

* `filter_length`: The length of the filter to be used.
Default: `51`

* `[extension]`: The extension for the generated image.
Supports the default *matplotlib* formats like `svg` or `png`.  
Default: `svg`


## Plot statistics

Usage: `plot_statistics.py statistics_file [epoch_max] [extension]`

Plots the network's overall loss, as well as accuracy, precision, recall and f1score
on the validation data in a line graph.  
To be used on the `stats.npy` file in the output folder of a network run.

* `statistics_file`: The `stats.npy` file to be evaluated.  
This file can be found in the folder generated by the network during training.

* `[epoch_max]`: The epoch until which the statistics should to be plotted.  
Default: `0`

* `[extension]`: The extension for the generated image.
Supports the default *matplotlib* formats like `svg` or `png`.  
Default: `svg`


## Plot targets

Usage: `plot_targets.py directory [extension]`

Plots the contact map for all (!) targets.  
To be used on the output folder of a network run.

* `directory`: The folder to apply to the script to.  
This should be the output folder generated during a training run.

* `[extension]`: The extension for the generated image.
Supports the default *matplotlib* formats like `svg` or `png`.  
Default: `svg`


## Plot ROC

Usage: `plot_roc.py directory sequence epoch_max epoch_step [extension]`

Plots the ROC curve of the network for a specific sequence, one plot per epoch.  
To be used on the output folder of a network run.

* `directory`: The folder to apply to the script to.  
This should be the output folder generated during a training run.

* `sequence`: The sequence for which the plots are to be generated.

* `epoch_max`: The epoch until the ROC curves are to be plotted.

* `epoch_step`: Sets the distance between epochs to be plotted.  
Should be exactly the distance between epochs in the `predictions` subfolder.

* `[extension]`: The extension for the generated image.
Supports the default *matplotlib* formats like `svg` or `png`.  
Default: `svg`

## Do automatic evaluation

Usage: `eval_script_ext.py directory epoch_max epoch_step`

Apply several of the mentioned evaluation scripts to the output folder of a training run.

* `directory`: The folder to apply to the script to.  
This should be the output folder generated during a training run.

* `epoch_max`: The epoch until evaluation should continue.

* `epoch_step`: Sets the distance between epochs to be evaluated.  
Should be exactly the distance between epochs in the `predictions` subfolder.

## Plot Arcplot

Usage: `matlab arcplot/plot_arcplots.m`
You might have to change paths in this matlab file to your data folders.

The code was taken from the Sükösd et al. paper, see data readme.
Plot Arcplots to compare the results of your architecture to the native structure. 
Only in ct file format. Conversion from bp/dp to ct can be done via the `RNA2Dconverter-master`, see `parsing` doc.