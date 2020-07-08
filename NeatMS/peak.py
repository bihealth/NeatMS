import numpy as np

class Chromatogram():

    """
    Representation of a chromatographic peak

    Contains the related peak object, the size and margin of the chromatogram and its dataframe as a numpy array.
    """

    def __init__(self, peak, size, margin, dataframe):
        self.peak = peak
        self.size = size
        self.margin = margin
        self.dataframe = dataframe



class Peak():

    """
    Representation of a chromatographic peak.

    Contains the related feature and sample it belongs to, as well as all peak characteristics.
    """

    # Peak counter to automatically assign peak ids
    peak_counter = 0
    peak_dict = dict()

    # This class method returns the peak instance from a peak id
    # This is to avoid iteration through all instances of peaks to find a specific one
    @classmethod
    def get_peak(cls, id):
        return cls.peak_dict[id]


    def __init__(self, sample, RT, mz, rt_start, rt_end, mz_min, mz_max, height, area, feature = None, monoisotopic= None, valid=None, area_baseline_corrected=None, sn=None):
        self.id = Peak.peak_counter
        self.sample = sample
        self.RT = RT
        self.mz = mz
        self.rt_start = rt_start
        self.rt_end = rt_end
        self.mz_min = mz_min
        self.mz_max = mz_max
        self.height = height
        self.area = area
        self.feature = feature
        self.annotation = None
        self.scan_number = None
        self.monoisotopic = monoisotopic
        self.valid = valid
        self.area_baseline_corrected = area_baseline_corrected
        self.sn = sn
        self.formatted_chromatograms = []
        Peak.peak_dict[self.id] = self
        Peak.peak_counter += 1            


    def get_height(self):
        if self.height == None:
            # Deprecated, will be removed from next version on
            self.height = max(self.get_chromatogram_from_openms()['I'])
        return self.height


    def get_window_margin(self, margin=None):
        # Need to define a margin window around this to see what is around the peak
        if (margin != None) and (margin != 0):
            window_size = self.rt_end - self.rt_start
            rt_start = self.rt_start - (window_size*margin)
            rt_end = self.rt_end + (window_size*margin)
        else:
            rt_start = self.rt_start
            rt_end = self.rt_end
        return rt_start, rt_end


    def get_scan_number(self):
        if scan_number == None:
            chromatogram = self.sample.MS1[(self.sample.MS1['mz'] >= self.mz_min) & 
                              (self.sample.MS1['mz'] <= self.mz_max) & 
                              (self.sample.MS1['rt'] >= self.rt_start) & 
                              (self.sample.MS1['rt'] <= self.rt_end)]
            # Return the number of unique RT values which correspond to the number of scans 
            return chromatogram.rt.nunique()
        else:
            return self.scan_number


    def set_scan_number(self):
        if self.scan_number == None:
            chromatogram = self.sample.MS1[(self.sample.MS1['mz'] >= self.mz_min) & 
                              (self.sample.MS1['mz'] <= self.mz_max) & 
                              (self.sample.MS1['rt'] >= self.rt_start) & 
                              (self.sample.MS1['rt'] <= self.rt_end)]
            # Set scan number as the number of unique RT values which correspond to the number of scans 
            self.scan_number = chromatogram.rt.nunique()


    def get_chromatogram(self, margin=None):
        rt_start, rt_end = self.get_window_margin(margin)
        # We extract the values inside the peak window (mz and RT window)
        chromatogram = self.sample.extract_chromatogram(rt_start, rt_end, self.mz_min, self.mz_max)
        # Containt [rt, intensity, mz]
        return chromatogram


    def interpolate_chromatogram(self, vals, margin=1, normalise=True):
        # Get the full chromatogram with margin
        chromatogram = self.get_chromatogram(margin)
        # Remove the baseline
        intensity_array = chromatogram[1] - np.amin(chromatogram[1])
        # Get peak retention time start and end (including margin)
        rt_start, rt_end = self.get_window_margin(margin)
        # Prepare our evenly spaced numbers over the retention time interval for interpoaltion
        xvals = np.linspace(rt_start, rt_end, vals)
        # Interpolate intensity values, anything outside is set to 0
        yinterp = np.interp(xvals, chromatogram[0], intensity_array, left=0, right=0)
        # We scale the interpolated intensity values between 0 and 1 (Important)
        if normalise:
            yinterp = np.divide(yinterp, yinterp.max())
        # Create our array representing the peak window (1 = within the rt window, 0 = outside the rt window)
        window = [1 if (rt >= self.rt_start) and (rt <= self.rt_end) else 0 for rt in xvals]
        # Create our final matrix
        interpolated_chromatogram = np.array([yinterp,window])
        # Create a chromatogram object to store this matrix
        chromatogram_object = Chromatogram(self, vals, margin, interpolated_chromatogram)
        # Add it to the list of chromatograms available for this peak (for dev only to test different margin and matrix sizes)
        self.formatted_chromatograms.append(chromatogram_object)
        return interpolated_chromatogram


    def get_formatted_chromatogram(self, vals, margin):
        for chromatogram in self.formatted_chromatograms:
            if (chromatogram.size == vals) and (chromatogram.margin == margin):
                return chromatogram.dataframe
        chromatogram = self.interpolate_chromatogram(vals, margin, True)
        return chromatogram


    def formatted_chromatogram_exists(self, vals, margin):
        for chromatogram in self.formatted_chromatograms:
            if (chromatogram.size == vals) and (chromatogram.margin == margin):
                return True
        return False

