# Input parsing

The script in this folder can be used to transform a secondary structure
database in *Dot Parentheses* or *BpSeq* format to a JSON representation
that can be used by our training script. Furthermore, the dataset can be split
into training, validation and test data.


## Prepare data

Usage: `prepare_data.py path data_format [validation_ratio] [test_ratio] [path_similar]`

Used to transform a secondary structure database into our JSON representation.
If path and path_similar share similar RNA structures, they are removed from the path dataset.
Careful, code was taken from the Wilmott et al. paper (code no longer online) and is currently very expensive, needs to be parallelized!
Splits the resulting data into a training, validation and test set.

Note that for validation, an explicit validation set is not needed anymore (`validation_ratio` = 0) 
and the validation ratio can be set during training with the `val-ratio` argument.

Currently supports the *Dot Parentheses* and *BpSeq* formats.  
The following three JSON files are generated in the same directory as the
input file:  
`<path>-training.json`, `<path>-validation.json` and `<path>-test.json`.

* `path`: The database in *Dot Parentheses* or *BpSeq* format

* `data_format`: Decides whether to read *Dot Parentheses* or *BpSeq* format.  
Options: `dp` for *Dot Parentheses*, `bp` for *BpSeq*

* `[validation_ratio]`: The fraction of the database to be used for validation.  
Should be a decimal number between 0 and 1.  
Default: `0.2`

* `[test_ratio]`: The fraction of the database to be used for testing.  
Should be a decimal number between 0 and 1.  
Default: `0.2`

* `[path_similar]`: The database of possibly similar sequences in *Dot Parentheses* or *BpSeq* format  
Default: `'None'`

## RNA Converter

Usage: `python main.py first_seq`

The `RNA2Dconverter-master` is located in the root workspace folder. 
It was taken from the github profile `https://github.com/BiPUJ/RNA2Dconverter`. 
It is not in english, but usage can be derived from its `README.md`.

The program then allows to specify, in which format we want to convert first_seq to.
If it does not work for some files (since several sites have different conventions even for the same format)
,one can sometimes comment out certain lines in the code -> just debug in.

