from .data import RawData

class Sample():

    """
    Representation of a MS sample contain in an experiment.

    Contains, peak and feature lists, raw_data object to store raw file and connect to appropriate backend reader. 
    """

    # Sample counter to automatically assign sample ids
    sample_counter = 0
    sample_dict = dict()

    # This class method returns the peak instance from a peak id
    # This is to avoid iteration through all instances of peaks to find a specific one
    @classmethod
    def get_sample(cls, name):
        return cls.sample_dict[name]
    
    def __init__(self, raw_file_path, data_reader):
        self.id = Sample.sample_counter
        self.raw_file = raw_file_path
        self.name = self.raw_file.stem
        self.file_name = self.raw_file.name
        self.peak_list = []
        self.feature_list = []
        self.raw_data = RawData(self, raw_file_path, data_reader)
        Sample.sample_counter += 1
        Sample.sample_dict[self.name] = self


    def get_MS1(self):
        return self.raw_data.get_MS1()


    def extract_chromatogram(self, rt_start, rt_end, mz_min, mz_max):
        return self.raw_data.extract_chromatogram(rt_start, rt_end, mz_min, mz_max)


    def create_interpolated_chromatograms(self, vals=120, margin=1, normalise=True, peak_list=None, min_scan_num=5):
        # By default, if no peak list is provided, all peaks are used 
        if peak_list is None:
            peak_list = self.peak_list

        self.raw_data.create_interpolated_chromatograms(vals, margin, normalise, peak_list, min_scan_num)


