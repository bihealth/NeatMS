import pymzml
import numpy as np

from .peak import Chromatogram, Peak

class RawData():

    """
    Representation of raw data file.

    Links sample class to its reader backend.
    """

    def __init__(self, sample, file_path, reader):
        self.sample = sample
        self.file = file_path
        self.reader = reader
        self.MS1 = None

    def get_MS1(self):
        if self.MS1 is None:
            self.MS1 = self.reader.get_MS1(self.file)
        return self.MS1


    def extract_chromatogram(self, rt_start, rt_end, mz_min, mz_max):
        if self.MS1 is None:
            chromatogram = self.reader.extract_chromatogram(self.file, rt_start, rt_end, mz_min, mz_max)
        else:
            # If The MS1 data has been extracted, then extract the chromatogram directly from the array (splited into two lines for comprehension purposes) 
            mz_indexes = np.where(np.logical_and(self.MS1[0] >= mz_min, self.MS1[0] <= mz_max) & np.logical_and(self.MS1[2] >= rt_start, self.MS1[2] <= rt_end))
            dataframe = self.MS1[:,mz_indexes]

            # Sum intensities (average mz) for entries with the same retention time
            all_rt = np.unique(dataframe[2])
            chromatogram = np.array([[rt, dataframe[1][dataframe[2] == rt].sum(), dataframe[0][dataframe[2] == rt].mean()] for rt in all_rt]).T

        return chromatogram


    def create_interpolated_chromatograms(self, vals, margin, normalise, peak_list, min_scan_num):
        # TODO: IF self.MS1 exist, iterate through the numpy array instead (method within this class, not the reader)
        self.reader.create_interpolated_chromatograms(self.file, vals, margin, normalise, peak_list, min_scan_num)


class DataReader():

    """
    Abstract class, general raw data file reader object
    """

    reader_counter = 0

    def __init__(self):
        self.id = DataReader.reader_counter
        DataReader.reader_counter += 1


class PymzmlDataReader(DataReader):

    """
    Inherits from DataReader.

    Helper class that uses pymzml to read mzML formated raw files.
    """

    def __init__(self):
        super().__init__()


    def get_MS1(self, file_path):
        mzml_file = str(file_path)
        run = pymzml.run.Reader(mzml_file)

        master_list = [[],[],[]]

        # Iterate through all the spectra
        for n, spec in enumerate(run):
            # Only extract MS 1 spectra
            if spec.ms_level == 1:
                rt = spec.scan_time_in_minutes()
                
                peaks = spec.peaks("centroided")
                rt_column = np.full(peaks.shape[0], rt)
                # Extract mz values and store them in the sample master list
                master_list[0].append(peaks[:,0])
                # Extract intensity values and store them in the sample master list
                master_list[1].append(peaks[:,1])
                # Add the reconstructed rt values to the sample master list (list of single rt value)
                master_list[2].append(rt_column)

        # Restructure the array
        mz_values = np.concatenate(master_list[0], axis=0, out=None)
        intensities = np.concatenate(master_list[1], axis=0, out=None)
        rt_values = np.concatenate(master_list[2], axis=0, out=None)
        master_array = np.array([mz_values,intensities,rt_values])

        # Return the MS1 array
        return master_array


    def extract_chromatogram(self, file_path, rt_start, rt_end, mz_min, mz_max):
        mzml_file = str(file_path)
        run = pymzml.run.Reader(mzml_file)

        # Extracting signal within the given rt and mz window
        chromatogram = np.array([[p[0], p[1], s.scan_time_in_minutes()] for s in run
                            if s.ms_level == 1 and s.scan_time_in_minutes() >= rt_start and s.scan_time_in_minutes() <= rt_end for p in s.peaks("centroided")
                            if p[0] >= mz_min and p[0] <= mz_max ]).T

        # One more step is required as a single rt can contain several mz values, we need to find and sum those values
        # Extract all unique rt values
        all_rt = np.unique(chromatogram[2])
        # Sum Intensities for rt values that are not unique (Equivalent of pandas groupby but using numpy arrays)
        # Average mz for rt values that are not unique (mz values are kept for consistensy but not used)
        # We also transpose the resulting array to get the correct dimensions 
        final_dataframe = np.array([[rt, chromatogram[1][chromatogram[2] == rt].sum(), chromatogram[0][chromatogram[2] == rt].mean()] for rt in all_rt]).T
        return final_dataframe


    def create_interpolated_chromatograms(self, file_path, vals, margin, normalise, peak_list, min_scan_num):
        mzml_file = str(file_path)
        run = pymzml.run.Reader(mzml_file)

        rt_list = []
        for i in range(len(peak_list)):
            peak = peak_list[i]
            peak.dataframe = [[],[],[]]
            rt_start, rt_end = peak.get_window_margin(margin)
            rt_list.append([rt_start,rt_end])

        # Extract MS1 data spectrum by spectrum and reconstruct peaks chromatogram on the way
        # The master list will store the full sample MS1 data (mz, intensity, retention time)
        master_list = [[],[],[]]
        # Iterate through all the spectra
        for n, spec in enumerate(run):
            # Only extract MS1 spectra
            if spec.ms_level == 1:
                rt = spec.scan_time_in_minutes()
                
                peaks = spec.peaks("centroided")
                rt_column = np.full(peaks.shape[0], rt)
                # Extract mz values and store them in the sample master list
                master_list[0].append(peaks[:,0])
                # Extract intensity values and store them in the sample master list
                master_list[1].append(peaks[:,1])
                # Add the reconstructed rt values to the sample master list (list of single rt value)
                master_list[2].append(rt_column)
            
                # Checking peak ranges
                # If this spectrum falls into a peak, we extract the corresponding data to create peak chromatogram 
                peak_index = 0
                # Iterate through our list of peak start and end values 
                for start_stop in rt_list:
                    # If the current spectrum retention time falls within a peak window
                    if start_stop[0] <= rt <= start_stop[1]:
                        # Get the corresponding peak objects (everything is stored in ordered lists so the list index can be used to find corresponding objects/values in different lists)
                        peak = peak_list[peak_index]
                        # Get the mz window of the peak
                        mz_min = peak.mz_min
                        mz_max = peak.mz_max
                        # Extract the indexes of all mz values from the spectrum falling within the peak mz window
                        mz_indexes = np.where(np.logical_and(peaks[:,0] >= mz_min, peaks[:,0] <= mz_max))
                        # If the number of values within the window > 0
                        if mz_indexes[0].size != 0:

                            peak_rt_array = np.full(mz_indexes[0].size, rt)
                            peak.dataframe[0].append(peaks[:,0][mz_indexes])
                            peak.dataframe[1].append(peaks[:,1][mz_indexes])
                            peak.dataframe[2].append(peak_rt_array)
                    peak_index += 1


        # Iterate through all the peaks to restructure the chromatogram dataframe that we just created
        for i in range(len(peak_list)):
            peak = peak_list[i]
            if len(peak.dataframe[0]) <= 1:
                peak.valid = False
                # Set peak.dataframe to None
                peak.dataframe = None
            if peak.valid != False:
                peak_rt_values = np.concatenate(peak.dataframe[2], axis=0, out=None)
                unique_rt = np.unique(peak_rt_values)
                # Extract the number of scan in the peak for filtering peaks with too few scans
                peak.scan_number = np.where(np.logical_and(unique_rt >= peak.rt_start, unique_rt <= peak.rt_end))[0].size
                # Test if the dataframe actually contains values (it can sometime happen that a reported peak is empty and therefore not valid)
                # Set valid to True/False -> meaning that it can or cannot be used by the NN for classification
                if (len(unique_rt) <= 1) or peak.scan_number < min_scan_num:
                    peak.valid = False
                    # Set peak.dataframe to None
                    peak.dataframe = None
                else:
                    peak.valid = True
                    # Concatenate the array for each dimension so we have one array per dimension
                    peak_mz_values = np.concatenate(peak.dataframe[0], axis=0, out=None)
                    peak_intensities = np.concatenate(peak.dataframe[1], axis=0, out=None)
                    
                    # Create the matrix of the chromatogram
                    temp_dataframe = np.array([peak_mz_values,peak_intensities,peak_rt_values])
                    # One more step is required as a single rt can contain several mz values, we need to find and sum those values
                    # Extract all unique rt values
                    all_rt = np.unique(temp_dataframe[2])
                    # Sum Intensities for rt values that are not unique (Equivalent of pandas groupby but using numpy arrays)
                    # We also transpose the resulting array to get the correct dimensions 
                    final_dataframe = np.array([[rt, temp_dataframe[1][temp_dataframe[2] == rt].sum()] for rt in all_rt]).T
                    
                    # Interpolation of the chromatogram to have the correct matrice size and peak window information for neural network feeding
                    # Remove the baseline (very crude approach of removing the minimum value)
                    intensity_array = final_dataframe[1] - np.amin(final_dataframe[1])
                    # Get peak retention time start and end (including margin)
                    rt_start, rt_end = peak.get_window_margin(margin)
                    # Prepare our evenly spaced numbers over the retention time interval for interpoaltion
                    xvals = np.linspace(rt_start, rt_end, vals)

                    # Interpolate intensity values, anything outside is set to 0
                    yinterp = np.interp(xvals, final_dataframe[0], intensity_array, left=0, right=0)
                    # We scale the interpolated intensity values between 0 and 1 (Important)
                    yinterp = np.divide(yinterp, yinterp.max())
                    # Create our array representing the peak window (1 = within the rt window, 0 = outside the rt window)
                    window = [1 if (rt >= peak.rt_start) and (rt <= peak.rt_end) else 0 for rt in xvals]
                    # Create our final matrix
                    interpolated_chromatogram = np.array([yinterp,window])
                    # Create a chromatogram object to store this matrix
                    chromatogram_object = Chromatogram(peak, vals, margin, interpolated_chromatogram)
                    # Add it to the list of chromatograms available for this peak (for dev only to test different margin and matrix sizes)
                    peak.formatted_chromatograms.append(chromatogram_object)
                    # Set peak.dataframe to None
                    peak.dataframe = None

        # Restructure the array
        # mz_values = np.concatenate(master_list[0], axis=0, out=None)
        # intensities = np.concatenate(master_list[1], axis=0, out=None)
        # rt_values = np.concatenate(master_list[2], axis=0, out=None)
        # master_array = np.array([mz_values,intensities,rt_values])

        # Return the MS1 array
        # return master_array

        return None


class OpenmsDataReader(DataReader):

    """
    Inherits from DataReader.

    Helper class that uses pyopenms to read mzML, mzxml and mzdata formated raw files.

    This class is currently not in use, we'll be released with the next version
    """

    def __init__(self):
        super().__init__()


    def get_openms_file_type(self, suffix):
        import pyopenms
        # Define file format and return corresponding pyopenms file object
        if suffix.lower() == '.mzxml':
            return pyopenms.MzXMLFile()
        elif suffix.lower() == '.mzml':
            return pyopenms.MzMLFile()
        elif suffix.lower() == '.mzdata':
            return pyopenms.MzDataFile()
        else:
            print('Data format is not supported!!')


    def load_openms_exp(self, raw_file):
        import pyopenms
        # Create a pyopenms file object
        sample = self.get_openms_file_type(raw_file.suffix)
        # Create pyopenms experiment
        exp = pyopenms.MSExperiment()

        # Create file options
        options = pyopenms.PeakFileOptions()
        # Set MS level
        options.setMSLevels([1])
        # Set option
        sample.setOptions(options)
        # Load the file with options
        sample.load(str(self.raw_file), exp)
        # Return pyopenms experiment object
        return exp


    def get_MS1(self, file_path):
        import pyopenms
        raw_file = file_path

        # Load raw data using pyopenms backend
        exp = self.load_openms_exp(raw_file)
        # Extract MS1 data spectrum by spectrum
        spectra = exp.getSpectra()
        # The master list will store the full sample MS1 data (mz, intensity, retention time)
        master_list = [[],[],[]]
        # Iterate through the spectra
        for i in range(len(spectra)):

            # Get the retention time value of this spectrum
            spectrum = sample.data.getSpectrum(i)
            rt = spectrum.getRT()
            # Get all the peaks present in this spectrum
            peaks = spectrum.get_peaks() 

            # Create a retention time list of the same length
            rt_column = np.full(peaks[0].shape, rt)
            # Extract mz values and store them in the sample master list
            master_list[0].append(peaks[0])
            # Extract intensity values and store them in the sample master list
            master_list[1].append(peaks[1])
            # Add the reconstructed rt values to the sample master list (list of single rt value)
            master_list[2].append(rt_column)
            
        mz_values = np.concatenate(master_list[0], axis=0, out=None)
        intensities = np.concatenate(master_list[1], axis=0, out=None)
        rt_values = np.concatenate(master_list[2], axis=0, out=None)
        master_array_openms = np.array([mz_values,intensities,rt_values])

        # Return the MS1 array
        return master_array
