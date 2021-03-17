# Data format

The module requires two types of input:

* Raw data files in mzML format.
* One feature table file in .csv format (multiple if peaks are not aligned).

> *Update, form version 0.7 aligned peaks are also supported with XCMS. See Feature table export from XCMS for more information.*

## Raw data in mzML

Most vendor formats can be converted to mzML format using [msconvert from proteowizard](http://proteowizard.sourceforge.net/).

## Feature table file format

The feature table file format can be easily created from any peak detection pipeline. Below, we give instruction on how to create this csv file from mzMine2 and XCMS.

> *File format differs between tools, please follow the instruction corresponding to the tool that you use to generate the feature table.*

### mzMine csv file structure

MzMine generates wide table format when exporting the features. Every row represents a different peak found in one or more samples.
The first two columns are sample independent and should contain the consensus m/z  and retention time of the peak across the samples. Every other column is sample specific, each sample requiring 8 columns with different information (m/z, RT, RT start, RT end, height, area, m/z min, m/z max). Expected column names are given below. Every sample present in the feature table should match the list of raw files present in the raw data folder.

Here is the list of headers for the first 10 columns as if we were importing a dataset containing one sample file named sample1.mzML.

* row m/z 
* row retention time 
* sample1.mzML Peak m/z 
* sample1.mzML Peak RT 
* sample1.mzML Peak RT start
* sample1.mzML Peak RT end 
* sample1.mzML Peak height 
* sample1.mzML Peak area
* sample1.mzML Peak m/z min 
* sample1.mzML Peak m/z max 

An example files are available in the data folder of NeatMS github repository, example of a single file containing aligned peaks is available [here](https://github.com/bihealth/NeatMS/blob/master/data/test_data/aligned_features.csv). An example of unaligned feature tables is given [here](https://github.com/bihealth/NeatMS/blob/master/data/test_data/unaligned_features). More details on how to generate them is given below.

### Feature table export from mzMine 2.0

The feature table can be exported before or after aligning peaks across samples. If exported before, one `.csv` file per sample should be created with the same names as the raw files, they should be all stored together in single folder. If exported after alignment, one `.csv` file is required under the name of your choice. The feature table file structure remains the same regardless of the alignment.

> *Important for aligned peaks: When importing the raw files in mzMine, the tool will attempt to detect and remove a common prefix from the raw file names (if it exists), make sure to disable it as the file names in the exported feature table will not match the actual raw file names anymore (you can also rename the raw files).*

After selecting the aligned peak list on the right side panel in mzMine2, you can access the export panel through `Peak list methods` > `Export/Import` > `Export to CSV file`.

First, make sure the `Field separator` is `,`. 

Select the following elements in the `Export common elements` panel:

* Export row m/z
* Export row retention time

Select the following in the `Export data file elements` panel:

* Peak m/z 
* Peak RT 
* Peak RT start
* Peak RT end 
* Peak height 
* Peak area 
* Peak m/z min 
* Peak m/z max

Make sure to unselect any other element and click OK. Your feature table file is now ready.

MZmine2 also provides filtering options, we recommend to not filter the peaks at this stage as it can be done in NeatMS, but doing it now will not impact on NeatMS usage.

### XCMS csv file structure

As mentioned above, XCMS file structure is different from mzMine. XCMS feature table is expected to be in long format. Only one .csv file is required even if peaks are not aligned. It requires the following information which is available by default in the `XCMSnExp` **R** object after peak detection.

Expected columns:

* mz
* mzmin
* mzmax
* rt
* rtmin
* rtmax
* into
* intb
* maxo
* sn
* sample
* sample_name
* feature_id (only required if peaks have been aligned)

> *For more details on the information contained in each variable, please refer to XCMS documentation.*

### Feature table export from XCMS (unaligned peaks)

The feature table can be reconstructed in many different ways using **R**, here is snippet that uses `dplyr` to generate the desired format. The main task is to bring the filenames into the dataframe and store in the `sample_name` column.

```
# This code assumes that the xdata variable corresponds 
# to the XCMSnExp object that contains the detected peaks 

# Load dplyr (required for left_join())
library(dplyr)

# Create the peak dataframe
feature_dataframe <- as.data.frame(chromPeaks(xdata))

# Retrieve the sample names and store it as a dataframe
sample_names_df <- as.data.frame(sampleNames(xdata))

# Rename the unique column "sample_name"
colnames(sample_names_df) <- c("sample_name")

# Generate the correct sample ids for matching purposes
# XCMS sampleNames() function returns sample names ordered by their ids
sample_names_df$sample <- seq.int(nrow(sample_names_df))

# Attach the sample names to the main dataframe by matching ids (sample column)
feature_dataframe <- left_join(feature_dataframe,sample_names_df, by="sample")

# Export the data as csv. 
# Note: Set row.names to FALSE as NeatMS does not need them
file_path <- "path/to/the/unaligned_feature_table.csv"
write.csv(feature_dataframe,file_path, row.names = FALSE)

```

> *When using your own code to reconstruct the dataframe, make sure to respect the correct order of the samples by matching the correct sample id to the correct sample name*
 
 
### Feature table export from XCMS (aligned peaks)

Here we can use the same code as above to get the peak specific information, but we will add the alignment (and grouping) information to the dataframe. This obviously assumes that you have aligned your peaks across samples and/or grouped peaks within samples. 

```
# The first part of the code is the same as for unaligned peaks, you can jump to the feature information addition 

# This code assumes that the xdata variable corresponds 
# to the XCMSnExp object that contains the detected peaks 

# Load dplyr (required for left_join())
library(dplyr)

# Create the peak dataframe
feature_dataframe <- as.data.frame(chromPeaks(xdata))

# Retrieve the sample names and store it as a dataframe
sample_names_df <- as.data.frame(sampleNames(xdata))

# Rename the unique column "sample_name"
colnames(sample_names_df) <- c("sample_name")

# Generate the correct sample ids for matching purposes
# XCMS sampleNames() function returns sample names ordered by their ids
sample_names_df$sample <- seq.int(nrow(sample_names_df))

# Attach the sample names to the main dataframe by matching ids (sample column)
feature_dataframe <- left_join(feature_dataframe,sample_names_df, by="sample")

### Feature information addition ###

# Here we will bring the feature alignment information stored in the XCMSnExp object to the dataframe that we have already created

featuresDef <- featureDefinitions(xdata)
featuresDef_df = data.frame(featuresDef)

# Only keep the information we need (column named 'peakidx')
# Get the index of the peakidx column
column_index <- which(colnames(featuresDef_df)=="peakidx")
features_df = data.frame(featuresDef_df[,column_index])
# Rename the column
peak_colummn_name <- colnames(features_df)
features_df = rename(features_df, "peak_id"=peak_colummn_name)

features_df <- cbind(feature_id= row.names(features_df),features_df)

# We'll use data.table for the next step
require(data.table)

# Get all the peak_id for each feature_id
features_df <- data.table(features_df)
features_df = features_df[, list(peak_id = unlist(peak_id)), by=feature_id]

# Bring the feature_id to the original peak dataframe
feature_dataframe = cbind(peak_id= row.names(feature_dataframe),feature_dataframe)
feature_dataframe$peak_id = as.character(feature_dataframe$peak_id)
feature_dataframe = left_join(feature_dataframe, features_df, by="peak_id")

# Note: The dataframe contains an extra column called peak_id, but this won't affect NeatMS and will simply be ignored (as would any other column not present in the list above).

# Export the data as csv. 
# Note: Set row.names to FALSE as NeatMS does not need them
file_path <- "path/to/the/aligned_feature_table.csv"
write.csv(feature_dataframe,file_path, row.names = FALSE)

```
