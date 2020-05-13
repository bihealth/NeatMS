import pandas as pd
import pathlib

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


    def export_data(self, sample_list, export_classes, use_annotation, export_properties):
        # ["rt", "mz", "peak_mz", "peak_rt", "peak_rt_start", "peak_rt_end", "peak_mz_min", "peak_mz_max", "label", "height", "area"]
        data_dictionnary = dict()
        if "mz" in export_properties:
            data_dictionnary["m/z"] = self.feature_list[0].mz
        if "rt" in export_properties:
            data_dictionnary["retention time"] = self.feature_list[0].RT
        for feature in self.feature_list:
            sample_str = feature.sample.name
            for peak in feature.peak_list:
                if peak.valid:
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
                                data_dictionnary[entry_str] = peak.prediction.label
                                if use_annotation and peak.annotation:
                                    data_dictionnary[entry_str] = peak.annotation.label
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
    Automate the creation of features and features object to the experiment.
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


    def load_feature_table(self):
        feature_table_file_folder = pathlib.Path(self.feature_table_path)
        if self.feature_table_path == None:
            print("Please, provdide path to a feature table")
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
                # feature_table = pd.concat([feature_table, sample_feature_table], axis=1, sort=False)
            self.feature_table = feature_table.fillna(0)
            return feature_table
        else:
            feature_table = pd.read_csv(self.feature_table_path)
            self.feature_table = feature_table
            return feature_table


    def create_column_map(self, samples):
        feature_samples = []
        for column in self.feature_table.columns[2:]:
            feature_samples.append(column.split('.')[0])

        column_mapping = dict()
        for sample in samples:
            column_mapping[sample] = dict()

        for column in self.feature_table.columns[2:]:
            if 'Peak m/z' in column and column.split(' ')[-1] == 'm/z':
                column_mapping[column.split('.')[0]]['mz'] = column
            elif 'Peak RT' in column and column.split(' ')[-1] == 'RT':
                column_mapping[column.split('.')[0]]['RT'] = column
            elif 'Peak RT start' in column:
                column_mapping[column.split('.')[0]]['rt_start'] = column
            elif 'Peak RT end' in column:
                column_mapping[column.split('.')[0]]['rt_end'] = column
            elif 'Peak m/z min' in column:
                column_mapping[column.split('.')[0]]['mz_min'] = column
            elif 'Peak m/z max' in column:
                column_mapping[column.split('.')[0]]['mz_max'] = column
            elif 'Peak height' in column:
                column_mapping[column.split('.')[0]]['height'] = column
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
            mz = row['row m/z']
            RT = row['row retention time'] 
            feature_collection = FeatureCollection()
            # Iterate through all samples in the experiment
            for sample in sample_list:
                # Extract peak specific data for individual sample
                peak_dict = dict(
                    sample = sample,
                    RT = row[self.column_map[sample.name]['RT']], 
                    mz = row[self.column_map[sample.name]['mz']],
                    rt_start = row[self.column_map[sample.name]['rt_start']],
                    rt_end = row[self.column_map[sample.name]['rt_end']],
                    mz_min = row[self.column_map[sample.name]['mz_min']],
                    mz_max = row[self.column_map[sample.name]['mz_max']],
                    height = row[self.column_map[sample.name]['height']],
                    area = row[self.column_map[sample.name]['area']]
                )
                # We do not save the peak if it does not exist in this sample (i.e. RT == 0 in table)
                if peak_dict['RT'] != 0:
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

