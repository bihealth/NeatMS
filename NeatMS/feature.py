import pandas as pd
import numpy as np
import pathlib
import logging

logger = logging.getLogger(__name__)

from .peak import Peak

class Feature():

    """
    Representation of an MS feature.

    Can contains several peaks that are grouped together from the same sample (adduct, isotopic peak, fragment...),
    mz and RT attributes correspond to the monoisotopic peak.
    Attached to a feature collection object that contain features from other samples aligned with this instance.  
    """

    # Feature counter to automatically assign feature ids
    feature_counter = 0
    feature_dict = dict()

    # This class method returns the feature instance from a feature id
    # This is to avoid iteration through all instances of features to find a specific one
    @classmethod
    def get_feature(cls, secondary_id):
        return cls.feature_dict[secondary_id]

    def __init__(self, mz, RT, sample=None, secondary_id=None, feature_collection=None):
        self.id = Feature.feature_counter
        self.secondary_id = secondary_id
        self.mz = mz
        self.RT = RT
        self.sample = sample
        self.peak_list = []
        self.feature_collection = feature_collection
        Feature.feature_counter += 1
        if secondary_id != None:
            Feature.feature_dict[self.secondary_id] = self


class FeatureCollection():

    """
    Representation of features from several samples aligned together

    Only contain a list of feature.
    """

    # Feature collection counter to automatically assign feature collection ids
    feature_collection_counter = 0

    def __init__(self):
         self.id = FeatureCollection.feature_collection_counter
         self.feature_list = []
         FeatureCollection.feature_collection_counter += 1


    def get_monoisotopic_peak_number(self, prediction_class, use_annotation):
        peak_count = 0
        for feature in self.feature_list:
            for peak in feature.peak_list:
                if peak.valid:
                    if (peak.monoisotopic is None) | (peak.monoisotopic is True):
                        if use_annotation:
                            if peak.annotation:
                                if peak.annotation.label in prediction_class:
                                    peak_count += 1
                            else:
                                if peak.prediction.label in prediction_class:
                                    peak_count += 1
                        else:
                            if peak.prediction.label in prediction_class:
                                peak_count += 1
        return peak_count


    def export_data(self, sample_list, export_classes, use_annotation, export_properties, long_format):
        # ["rt", "mz", "peak_mz", "peak_rt", "peak_rt_start", "peak_rt_end", "peak_mz_min", "peak_mz_max", "label", "height", "area", "area_bc", "sn"]
        if long_format:
            feature_data_list = []
            feature_id = self.id
            for feature in self.feature_list:
                sample_str = feature.sample.name
                for peak in feature.peak_list:
                    if peak.valid:
                        if use_annotation and peak.annotation:
                            label = peak.annotation.label
                        else:
                            label = peak.prediction.label
                        # If label in export_classes we start adding info to list
                        if label in export_classes:
                            data_list = []
                            data_list.append(feature_id)
                            data_list.append(sample_str)
                            if "mz" in export_properties:
                                data_list.append(feature.mz)
                            if "rt" in export_properties:
                                data_list.append(feature.RT)
                            for export_property in export_properties:
                                if export_property == "peak_mz":
                                    data_list.append(peak.mz)
                                if export_property == "peak_rt":
                                    data_list.append(peak.RT)
                                if export_property == "peak_rt_start":
                                    data_list.append(peak.rt_start)
                                if export_property == "peak_rt_end":
                                    data_list.append(peak.rt_end)
                                if export_property == "peak_mz_min":
                                    data_list.append(peak.mz_min)
                                if export_property == "peak_mz_max":
                                    data_list.append(peak.mz_max)
                                if export_property == "label":
                                    data_list.append(label)
                                if export_property == "height":
                                    data_list.append(peak.height)
                                if export_property == "area":
                                    data_list.append(peak.area)
                                if export_property == "area_bc":
                                    data_list.append(peak.area_baseline_corrected)
                                if export_property == "sn":
                                    data_list.append(peak.sn)
                            feature_data_list.append(data_list)
            return feature_data_list
        else:
            data_dictionnary = dict()
        if "mz" in export_properties:
            data_dictionnary["m/z"] = self.feature_list[0].mz
        if "rt" in export_properties:
            data_dictionnary["retention time"] = self.feature_list[0].RT
        for feature in self.feature_list:
            if feature.sample in sample_list:
                sample_str = feature.sample.name
                for peak in feature.peak_list:
                    if peak.valid:
                        if use_annotation and peak.annotation:
                            label = peak.annotation.label
                        else:
                            label = peak.prediction.label
                        if label in export_classes:
                            if (peak.monoisotopic is None) | (peak.monoisotopic is True):
                                for export_property in export_properties:
                                    if export_property == "peak_mz":
                                        entry_str = ' '.join([sample_str,'Peak m/z'])
                                        data_dictionnary[entry_str] = peak.mz
                                    if export_property == "peak_rt":
                                        entry_str = ' '.join([sample_str,'Peak RT'])
                                        data_dictionnary[entry_str] = peak.RT
                                    if export_property == "peak_rt_start":
                                        entry_str = ' '.join([sample_str,'Peak RT start'])
                                        data_dictionnary[entry_str] = peak.rt_start
                                    if export_property == "peak_rt_end":
                                        entry_str = ' '.join([sample_str,'Peak RT end'])
                                        data_dictionnary[entry_str] = peak.rt_end
                                    if export_property == "peak_mz_min":
                                        entry_str = ' '.join([sample_str,'Peak m/z min'])
                                        data_dictionnary[entry_str] = peak.mz_min
                                    if export_property == "peak_mz_max":
                                        entry_str = ' '.join([sample_str,'Peak m/z max'])
                                        data_dictionnary[entry_str] = peak.mz_max
                                    if export_property == "label":
                                        entry_str = ' '.join([sample_str,'Peak label'])
                                        data_dictionnary[entry_str] = label
                                    if export_property == "height":
                                        entry_str = ' '.join([sample_str,'Peak height'])
                                        data_dictionnary[entry_str] = peak.height
                                    if export_property == "area":
                                        entry_str = ' '.join([sample_str,'Peak area'])
                                        data_dictionnary[entry_str] = peak.area
        return data_dictionnary


class FeatureTable():

    """
    Representation of the feature extraction performed on the dataset

    Contains the feature table input file and enable its loading.
    Automate the creation of features and features object.

    Every feature table input format requires a specific class deirved 
    from this general FeatureTable class with dedicated loading functions.
    """

    # Feature table counter to automatically assign feature table ids
    feature_table_counter = 0

    def __init__(self, feature_table_path=None, origin=None):
        self.id = FeatureTable.feature_table_counter
        self.feature_table_path = feature_table_path
        self.feature_table = None
        self.column_map = None
        self.annotation_table = None
        self.feature_collection_list = []
        self.origin = origin
        FeatureTable.feature_table_counter += 1



class MzmineFeatureTable(FeatureTable):

    """
    Inherits from FeatureTable.

    Mzmine specific feature table object.
    Enable reading mzmine aligned and unaligned feature tables. 
    """

    def __init__(self, feature_table_path=None, origin=None):
        logger.info('Feature table format: MZmine')
        super().__init__(feature_table_path, origin)


    def load_feature_table(self):
        logger.info('Loading feature table and converting format')
        feature_table_file_folder = pathlib.Path(self.feature_table_path)
        if self.feature_table_path == None:
            logger.error('Feature table path missing')
            return None
        # If the path is a directory (Unaligned peaks, one table per sample)
        elif feature_table_file_folder.is_dir():
            # List all .csv files
            feature_table_files = [file for file in feature_table_file_folder.iterdir() if file.suffix.lower() in ['.csv']]
            # Create an empty dataframe
            feature_table = pd.DataFrame()
            # Iterate through all files (samples)
            for feature_table_file in feature_table_files:
                # Read the sample specific feature table
                sample_feature_table = pd.read_csv(feature_table_file)
                # Join the table to the main dataframe
                feature_table = feature_table.append(sample_feature_table, sort=False)
        # Else simply read the file
        else:
            feature_table = pd.read_csv(self.feature_table_path)
        # Reindex the dataframe
        feature_table = feature_table.reset_index(drop=True)
        # Drop extra empty columns sometimes created by mzmine
        feature_table = feature_table.dropna(how='all', axis=1)
        # Replace all 0 values with NA
        feature_table = feature_table.replace(0, np.nan)
        # Add feature index column (Important: perform after NA replacement)
        feature_table['Feature_ID'] = feature_table.index
        # Create list with the new column names so we can change format (wide to long)
        new_column_names = ['row m/z','row retention time']
        test_list = ['end', 'max', 'min', 'start'] 
        for column_name in feature_table.columns[2:-1]:
            res = any(ele in column_name for ele in test_list)
            split_column_name = column_name.split(' ')
            split_column_name.append(split_column_name.pop(0))
            if res:
                ele = split_column_name[2]
                split_column_name.remove(ele)
                split_column_name.insert(1,ele)
            new_column_name = str(' ').join(split_column_name)
            new_column_names.append(new_column_name)
        new_column_names.append('Feature_ID')
        # Change column names
        feature_table.columns = new_column_names
        # Convert dataframe from wide to long
        feature_table = pd.wide_to_long(feature_table, ["Peak m/z", "Peak RT", "Peak start RT", "Peak end RT","Peak height","Peak area","Peak min m/z","Peak max m/z"], i="Feature_ID", j="sample", sep=" ", suffix='.+')
        # Remove all empty entries
        feature_table = feature_table.dropna(how='all', axis=0, subset=["Peak m/z", "Peak RT", "Peak start RT", "Peak end RT","Peak height","Peak area","Peak min m/z","Peak max m/z"])
        self.feature_table = feature_table
        return feature_table


    def create_column_map(self, samples):
        # Useless since we convert input to long format 
        return None


    def load_features(self, sample_list):
        sample_map = dict()
        for sample in sample_list:
            sample_map[sample.file_name] = sample

        peak_number = self.feature_table.shape[0]
        feature_number = self.feature_table.index.max()[0]
        i = 0

        logger.info('Loading %d features and %d peaks ', feature_number, peak_number)

        df_group = self.feature_table.groupby('Feature_ID')
        for name, group in df_group:
            feature_collection = FeatureCollection()
            for row_index, row in group.iterrows():
                sample = sample_map[row_index[1]]
                mz = row['row m/z']
                RT = row['row retention time']
                feature = Feature(mz, RT, sample, feature_collection=feature_collection)
                peak_dict = dict(
                    sample = sample,
                    RT = row['Peak RT'], 
                    mz = row['Peak m/z'],
                    rt_start = row['Peak start RT'],
                    rt_end = row['Peak end RT'],
                    mz_min = row['Peak min m/z'],
                    mz_max = row['Peak max m/z'],
                    height = row['Peak height'],
                    area = row['Peak area'],
                )
                # Create peak
                new_peak = Peak(sample, peak_dict['RT'], peak_dict['mz'], peak_dict['rt_start'], peak_dict['rt_end'], peak_dict['mz_min'], 
                    peak_dict['mz_max'], peak_dict['height'], peak_dict['area'])
                # Add feature to the peak (one peak is attached to a unique feature)
                new_peak.feature = feature
                # Add the peak to the sample
                sample.peak_list.append(new_peak)
                # Add the peak to the feature
                feature.peak_list.append(new_peak)
                # Add the feature to the sample
                sample.feature_list.append(feature)
                # Add the feature to the feature collection
                feature_collection.feature_list.append(feature)
            # Add the feature to the feature list (all peaks belonging to this feature have now been created)
            self.feature_collection_list.append(feature_collection)
        logger.info('Feature table loaded with success')
        return None


class PeakonlyFeatureTable(FeatureTable):

    """
    Inherits from FeatureTable.

    Peakonly specific feature table object.
    Enable reading peakonly unaligned feature tables.
    Aligned features not supported as peakonly does not support alignment 
    """

    def __init__(self, feature_table_path=None, origin=None):
        logger.info('Feature table format: Peakonly')
        super().__init__(feature_table_path, origin)


    def load_feature_table(self):
        feature_table_file_folder = pathlib.Path(self.feature_table_path)
        if self.feature_table_path == None:
            logger.error('Feature table path missing')
            return None
        # If the path is a directory (Unaligned peaks, one table per sample)
        elif feature_table_file_folder.is_dir():
            # List all .csv files
            feature_table_files = [file for file in feature_table_file_folder.iterdir() if file.suffix.lower() in ['.csv']]
            # Create an empty dataframe
            feature_table = pd.DataFrame()
            # Iterate through all files (samples)
            for feature_table_file in feature_table_files:
                # Read the sample specific feature table
                sample_feature_table = pd.read_csv(feature_table_file)
                # Reconstruct the name of the corresponding raw file
                raw_file_name = feature_table_file.stem + '.mzML'
                # Modify the table: Create an rt column
                sample_feature_table['rt'] = sample_feature_table[['rtmin', 'rtmax']].mean(axis=1)
                # Keep only the column required (solves formating issues when some Unnamed extra column is created)
                sample_feature_table = sample_feature_table[['mz','rt','mzmin','mzmax','rtmin', 'rtmax','intensity']]
                # Add sample name to all columns except mz and rt
                sample_feature_table.columns = ['mz','rt',raw_file_name+' Peak mzmin',raw_file_name+' Peak mzmax',raw_file_name+' Peak rtmin',raw_file_name+' Peak rtmax',raw_file_name+' Peak area']
                # Join the table to the main dataframe
                feature_table = feature_table.append(sample_feature_table, sort=False)
            self.feature_table = feature_table.fillna(0)
            return feature_table
        else:
            logger.error('Provide path to directory containing single sample feature tables')
            return None


    def create_column_map(self, samples):
        feature_samples = []
        for column in self.feature_table.columns[2:]:
            feature_samples.append(column.split('.')[0])

        column_mapping = dict()
        for sample in samples:
            column_mapping[sample] = dict()

        for column in self.feature_table.columns[2:]:
            # if 'Peak m/z' in column and column.split(' ')[-1] == 'm/z':
            #     column_mapping[column.split('.')[0]]['mz'] = column
            # elif 'Peak RT' in column and column.split(' ')[-1] == 'RT':
            #     column_mapping[column.split('.')[0]]['RT'] = column
            if 'Peak rtmin' in column:
                column_mapping[column.split('.')[0]]['rt_start'] = column
            elif 'Peak rtmax' in column:
                column_mapping[column.split('.')[0]]['rt_end'] = column
            elif 'Peak mzmin' in column:
                column_mapping[column.split('.')[0]]['mz_min'] = column
            elif 'Peak mzmax' in column:
                column_mapping[column.split('.')[0]]['mz_max'] = column
            elif 'Peak area' in column:
                column_mapping[column.split('.')[0]]['area'] = column
        self.column_map = column_mapping
        return column_mapping


    def load_features(self, sample_list):
        feature_number = self.feature_table.shape[0]
        i = 0
        # Iterate through all features in the feature table
        for index, row in self.feature_table.iterrows():
            i += 1
            # Extract feature specific information (mz and RT, RT is converted in seconds)
            mz = row['mz']
            RT = row['rt'] 
            feature_collection = FeatureCollection()
            # Iterate through all samples in the experiment
            for sample in sample_list:
                # Extract peak specific data for individual sample
                peak_dict = dict(
                    sample = sample,
                    RT = RT, 
                    mz = mz,
                    rt_start = row[self.column_map[sample.name]['rt_start']],
                    rt_end = row[self.column_map[sample.name]['rt_end']],
                    mz_min = row[self.column_map[sample.name]['mz_min']],
                    mz_max = row[self.column_map[sample.name]['mz_max']],
                    height = None,
                    area = row[self.column_map[sample.name]['area']]
                )
                # We do not save the peak if it does not exist in this sample (i.e. rt_end == 0 in table)
                if peak_dict['rt_end'] != 0:
                    # Create feature (No secondary id)
                    feature = Feature(mz, RT, sample, feature_collection=feature_collection)
                    # Create peak
                    new_peak = Peak(sample, peak_dict['RT'], peak_dict['mz'], peak_dict['rt_start'], peak_dict['rt_end'], peak_dict['mz_min'], 
                        peak_dict['mz_max'], peak_dict['height'], peak_dict['area'])
                    # Add feature to the peak (one peak is attached to a unique feature)
                    new_peak.feature = feature
                    # Add the peak to the sample
                    sample.peak_list.append(new_peak)
                    # Add the peak to the feature
                    feature.peak_list.append(new_peak)
                    # Add the feature to the sample
                    sample.feature_list.append(feature)
                    # Add the feature to the feature collection (Only one feature per feature collection in this case)
                    feature_collection.feature_list.append(feature)
            # Add the feature to the feature list (all peaks belonging to this feature have now been created)
            self.feature_collection_list.append(feature_collection)
        return None


class XcmsFeatureTable(FeatureTable):

    """
    Inherits from FeatureTable.

    XCMS specific feature table object.
    Enable reading XCMS unaligned feature tables.
    Aligned features not yet supported for XCMS
    """

    def __init__(self, feature_table_path=None, origin=None):
        logger.info('Feature table format: XCMS')
        super().__init__(feature_table_path, origin)


    def load_feature_table(self):
        feature_table_file_folder = pathlib.Path(self.feature_table_path)
        if self.feature_table_path == None:
            logger.error('Feature table path missing')
            return None
        # If the path is a directory (Unaligned peaks, one table per sample)
        elif feature_table_file_folder.is_dir():
            logger.error('Only .csv file is supported as XCMS feature table, directory given')
            return None
        else:
            feature_table = pd.read_csv(self.feature_table_path)
            self.feature_table = feature_table
            return feature_table


    def create_column_map(self, samples):
        # XCMS input is long format, we only need to map the sample names
        return None


    def load_features(self, sample_list):
        sample_map = dict()
        for sample in sample_list:
            sample_map[sample.file_name] = sample

        peak_number = self.feature_table.shape[0]
        feature_number = self.feature_table.index.max()
        i = 0

        logger.info('Loading %d features and %d peaks ', feature_number, peak_number)
        # Iterate through all features in the feature table
        for index, row in self.feature_table.iterrows():
            i += 1
            # Extract feature specific information (mz and RT, RT is converted in seconds)
            mz = row['mz']
            RT = row['rt'] / 60
            # No alignment, one feature per feature collection
            feature_collection = FeatureCollection()
            # Create feature (No secondary id)
            sample = sample_map[row['sample_name']]
            feature = Feature(mz, RT, sample, feature_collection=feature_collection)
            peak_dict = dict(
                sample = sample,
                RT = RT, 
                mz = mz,
                rt_start = row['rtmin'] / 60,
                rt_end = row['rtmax'] / 60,
                mz_min = row['mzmin'],
                mz_max = row['mzmax'],
                height = row['maxo'],
                area = row['into'],
                area_bc = row['intb'],
                sn = row['sn']
            )      
            # Create peak
            new_peak = Peak(sample, peak_dict['RT'], peak_dict['mz'], peak_dict['rt_start'], peak_dict['rt_end'], peak_dict['mz_min'], 
                peak_dict['mz_max'], peak_dict['height'], peak_dict['area'], area_baseline_corrected=peak_dict['area_bc'], sn=peak_dict['sn'])
            # Add feature to the peak (one peak is attached to a unique feature)
            new_peak.feature = feature
            # Add the peak to the sample
            sample.peak_list.append(new_peak)
            # Add the peak to the feature
            feature.peak_list.append(new_peak)
            # Add the feature to the sample
            sample.feature_list.append(feature)
            # Add the feature to the feature collection (Only one feature per feature collection in this case)
            feature_collection.feature_list.append(feature)
            # Add the feature to the feature list (all peaks belonging to this feature have now been created)
            self.feature_collection_list.append(feature_collection)
        logger.info('Feature table loaded with success')
        return None
