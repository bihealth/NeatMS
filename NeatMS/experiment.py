import pandas as pd
import pickle
import pathlib
import logging

logger = logging.getLogger(__name__)

from .data import RawData, DataReader, PymzmlDataReader
from .sample import Sample
from .feature import Feature, FeatureCollection, FeatureTable, MzmineFeatureTable, PeakonlyFeatureTable, XcmsFeatureTable
from .annotation import AnnotationTable

class Experiment():
    """
    Simple representation of MS experiment.

    Contain the data and meta-data relative to an experiment, 
    no metadata specific to the experiment design is stored (such as biological groups).
    The location and list of the raw data files and feature file are stored within this class.
    The raw data (spectra, chromatogram and features) is stored in objects of type Sample and 
    FeatureTable directly accessible through their respective lists.

    """

    def __init__(self, raw_file_folder_path, feature_table_path, input_format, labels=['High_quality','Low_quality','Noise'], load_MS1=False, name="NeatMS_experiment"):
        self.name = name
        self.raw_file_folder = self.get_file_folder(raw_file_folder_path)
        self.raw_files = self.get_raw_files()
        self.feature_tables = []
        self.data_reader = self.set_data_reader(feature_table_path)
        self.samples = self.load_samples(load_MS1)
        self.input_format = input_format
        self.load_feature_table(feature_table_path, labels)


    def set_data_reader(self, feature_table_path):
        feature_table_file_folder = pathlib.Path(feature_table_path)
        # Test the feature table file format 
        if feature_table_file_folder.is_dir() or feature_table_file_folder.suffix.lower() == '.csv':
            # If the file format is .csv, use pymzml backend file reader
            # First check that raw files are indeed in .mzml format
            raw_file_type = set([file.suffix.lower() for file in self.raw_file_folder.iterdir()])
            # Temporary fix: sometimes an empty suffix is present (probably due to temporary files) we discard it for now
            # TODO: Create stable fix
            raw_file_type.discard('')
            if len(raw_file_type) > 1:
                logger.error('The raw file folder contains several file formats, only .mzml format is accepted')
                return None
            if list(raw_file_type)[0] != '.mzml':
                logger.error('Required format is .mzml. Provided format: %s',list(raw_file_type)[0])
                return None
            data_reader = PymzmlDataReader()
            logger.info('Data reader backend: %s', 'pymzml')
            return data_reader
        else:
            # Use the openms backend file reader
            # TODO: Feature table from openms needs to be define
            data_reader = OpenmsDataReader()
            logger.info('Data reader backend: %s', 'pyopnems')
            return data_reader


    def get_file_names(self):
        return [file.name for file in self.samples]
        
    
    def get_file_folder(self, file_folder_path):
        if file_folder_path != None:
            file_folder = pathlib.Path(file_folder_path)
            return file_folder
        return None
    

    def get_raw_files(self):
        if self.raw_file_folder != None:
            return [file for file in self.raw_file_folder.iterdir() if file.suffix.lower() in ['.mzxml','.mzml','.mzdata']]
        return None


    def load_samples(self, load_data=False):
        if self.raw_file_folder == None:
            logger.error('raw_file_folder_path: %s does not exist', self.raw_file_folder)
            return None
        if len(self.raw_files) == 0:
            logger.error('Raw folder is empty')
            return None
        file_count = 0
        samples = []
        for file in self.raw_files:
            file_count += 1
            logger.info('Loading file %d / %d', file_count, len(self.raw_files))
            sample = Sample(file, self.data_reader)
            if load_data:
                sample.get_MS1()
            samples.append(sample)
        self.samples = samples
        return samples


    # This contain code duplicate, need to be factorised
    def load_feature_table(self, feature_table_path, labels=['High_quality','Low_quality','Noise']):
        feature_table_file_folder = pathlib.Path(feature_table_path)
        # Test if the path is a folder (several files = unaligned features)
        if feature_table_file_folder.is_dir():
            # Test if provided files are all in .csv format 
            feature_table_file_type = set([file.suffix.lower() for file in feature_table_file_folder.iterdir()])
            if len(feature_table_file_type) > 1:
                logger.error('The feature table file folder contains several file formats, only .csv format is accepted')
                return None
            if list(feature_table_file_type)[0] != '.csv':
                logger.error('Required format is .csv. Provided format: %s',list(feature_table_file_type)[0])
                return None
            if self.input_format.lower() == 'mzmine':
                feature_table = MzmineFeatureTable(feature_table_path)
            elif self.input_format.lower() == 'peakonly':
                feature_table = PeakonlyFeatureTable(feature_table_path)
            elif self.input_format.lower() == 'xcms':
                logger.error('XCMS input should be a .csv file, directory given')
            self.feature_tables.append(feature_table)
            feature_table.load_feature_table()
            sample_names = self.get_file_names()
            feature_table.create_column_map(sample_names)
            feature_table.load_features(self.samples)
            annotation_table = AnnotationTable(feature_table, labels)
            feature_table.annotation_table = annotation_table
            return feature_table
        # Else test if feature table file format is csv (one file = aligned features) 
        elif feature_table_file_folder.suffix.lower() == '.csv':
            logger.info('Loading feature table: %s',feature_table_file_folder.resolve())
            # Default feature table loading
            if self.input_format.lower() == 'mzmine':
                feature_table = MzmineFeatureTable(feature_table_path)
            elif self.input_format.lower() == 'peakonly':
                logger.error('Aligned features from peakonly are not supported')
            elif self.input_format.lower() == 'xcms':
                feature_table = XcmsFeatureTable(feature_table_path)
            self.feature_tables.append(feature_table)
            feature_table.load_feature_table()
            sample_names = self.get_file_names()
            feature_table.create_column_map(sample_names)
            feature_table.load_features(self.samples)
            annotation_table = AnnotationTable(feature_table, labels)
            feature_table.annotation_table = annotation_table
            return feature_table
        else:
            # TODO: add support for openms .featurexml or .consensusfeaturexml file format 
            logger.error('Only .csv file format is currently supported for the feature table')

    def get_columns(self, export_properties):
        # Need to respect the export properties order (4 first colunms, the rest follow the input)
        columns = ['feature ID', 'sample']
        if "mz" in export_properties:
            columns.append('m/z')
        if "rt" in export_properties:
            columns.append('retention time')
        # Quick hack: TODO remove for loop
        for export_property in export_properties:
            if export_property == "peak_mz":
                columns.append("peak_mz")
            if export_property == "peak_rt":
                columns.append("peak_rt")
            if export_property == "peak_rt_start":
                columns.append("peak_rt_start")
            if export_property == "peak_rt_end":
                columns.append("peak_rt_end")
            if export_property == "peak_mz_min":
                columns.append("peak_mz_min")
            if export_property == "peak_mz_max":
                columns.append("peak_mz_max")
            if export_property == "label":
                columns.append("label")
            if export_property == "height":
                columns.append("height")
            if export_property == "area":
                columns.append("area")
            if export_property == "area_bc":
                columns.append("area_bc")
            if export_property == "sn":
                columns.append("sn")
        return columns


    def export_to_dataframe(self, export_classes = ["High_quality", "Low_quality","Noise"], min_group_classes = ["High_quality", "Low_quality","Noise"],min_group_size = 0, exclude = [], use_annotation = False, export_properties = ["rt", "mz", "height","area","label"], long_format = True):
        total_sample_number = len(self.samples)
        # Set the minimum feature size (number of aligned peaks) for the feature to be kept
        if (0 < min_group_size <= 1):
            min_group_size = min_group_size * total_sample_number
        # Adjust the number to 1 if "min_group_size == 0", most conservative option
        # if min_group_size == 0:
        #     min_group_size = 1
        # If the exclusion list is not empty
        if len(exclude) > 0:
            sample_list = []
            # Iterate through all the samples
            for sample in self.samples:
                # If the sample object or the sample name is not in the exclusion list, keep the sample (add to the list)
                if (sample not in exclude) & (sample.name not in exclude):
                    sample_list.append(sample)
        else:
            sample_list = self.samples
        feature_collection_list = self.feature_tables[0].feature_collection_list
        export_feature_collection_list = []
        for feature_collection in feature_collection_list:
            # First filter to exclude features present in fewer samples than "min_group_size"
            if len(feature_collection.feature_list) >= min_group_size:
                # Get the number of peak predicted (annotated) with the target classes
                peak_count = feature_collection.get_monoisotopic_peak_number(prediction_class=min_group_classes, use_annotation=use_annotation)
                if (peak_count > 0) and (peak_count >= min_group_size):
                    # Add the feature collection to the export list
                    export_feature_collection_list.append(feature_collection)
        export_data = []
        for feature_collection in export_feature_collection_list:
            feature_collection_data = feature_collection.export_data(sample_list = sample_list, export_classes = export_classes, use_annotation = use_annotation, export_properties = export_properties, long_format = long_format)
            if long_format:
                export_data = export_data + feature_collection_data
            else:
                export_data.append(feature_collection_data)

        if long_format:
            columns = self.get_columns(export_properties)
            dataframe = pd.DataFrame(export_data, columns = columns) 
        else:
            dataframe = pd.DataFrame(export_data)

        return dataframe


    def export_csv(self, filename, index=True, na_rep='', export_classes = ["High_quality", "Low_quality", "Noise"], min_group_classes = ["High_quality", "Low_quality", "Noise"], min_group_size = 0, exclude = [], use_annotation = False, export_properties = ["rt", "mz", "height","area","label"]):
        dataframe = self.export_to_dataframe(export_classes, min_group_classes, min_group_size, exclude, use_annotation, export_properties)
        dataframe.to_csv(filename, index=index, na_rep=na_rep)
        file_path = pathlib.Path(filename)
        logger.info('Exporting data to %s',file_path.resolve())


    def save(self):
        file_name = self.name + '.pkl'
        with open(file_name, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

