import numpy as np
import pandas as pd
import pathlib
import logging
import random

logger = logging.getLogger(__name__)

class NN_handler():

    """
    This class contains code for neural network manipulation. 
    Neural network, Model creation or loading, batch creation, model training and hyperparameter optimisation (threshold)
    Connection to the data is made through the experiment class
    """

    def __init__(self, experiment, matrice_size=120, margin=1, table_index=0, exclude=[], min_scan_num=5):
        self.experiment = experiment
        self.annotation_table = experiment.feature_tables[table_index].annotation_table
        self.labels = self.get_labels(exclude)
        self.label_list = sorted(self.labels)
        self.matrice_size = matrice_size
        self.margin = margin
        self.min_scan_num = min_scan_num


    def get_labels(self, exclude=[]):
        self.labels = [annotation.label for annotation in self.annotation_table.annotations if (annotation.label not in exclude)]  
        return self.labels


    def create_batches(self, validation_split=0.1, normalise_class=False, merge_classes=[]):
        from sklearn.preprocessing import LabelEncoder
        from keras.utils import to_categorical
        # Get all the peaks that have been annotated
        all_peaks = [peak for annotation in self.annotation_table.annotations for peak in annotation.peaks if (annotation.label in self.labels)]
        # Randomise them as they come by order of annotation
        random.shuffle(all_peaks)

        # If merge classes is on, redefine the label list
        if len(merge_classes) > 0:
            self.label_list = sorted(list(set([label if label not in merge_classes[0] else merge_classes[1] for label in self.labels])))

        # This is to get an equal number of training point (peak) for all different classes
        # If this is not used, the sets are unbalanced towards noise peaks
        if normalise_class:
            # initialise smallest class to a very high value
            smallest_class = 1000000
            if len(merge_classes) > 0:
                merged_class_size = 0
                for annotation in self.annotation_table.annotations:
                    if annotation.label in self.labels:
                        if annotation.label in merge_classes[0]:
                            merged_class_size += len(annotation.peaks)
                        else:
                            class_size = len(annotation.peaks)
                            smallest_class = class_size if class_size < smallest_class else smallest_class
                smallest_class = merged_class_size if merged_class_size < smallest_class else smallest_class
            else:
                for annotation in self.annotation_table.annotations:
                    if annotation.label in self.labels:
                        class_size = len(annotation.peaks)
                        smallest_class = class_size if class_size < smallest_class else smallest_class

            peaks = []
            class_count = [0] * len(self.label_list)
            for peak in all_peaks:
                if (len(merge_classes) > 0) and (peak.annotation.label in merge_classes[0]):
                    peak_class = merge_classes[1]
                    if class_count[self.label_list.index(merge_classes[1])] < smallest_class:
                        peaks.append(peak)
                        class_count[self.label_list.index(merge_classes[1])] += 1
                else:
                    if class_count[self.label_list.index(peak.annotation.label)] < smallest_class:
                        peaks.append(peak)
                        class_count[self.label_list.index(peak.annotation.label)] += 1

        else:
            peaks = all_peaks

        # Randomise again to avoid that one label only is stacked at the end of the list
        random.shuffle(peaks)

        # Create placeholder lists to store the data
        data = []
        labels = []
        ids = []
        num_data = []

        max_height = 0
        max_rt_window = 0
        max_mz_window = 0
        max_scan_number = 0

        # Hotfix to accelerate batch creation by extracting chromatograms using using sample module
        # Check if chromatograms do not already exist
        create_chromatograms = False
        for peak in peaks:
            if not peak.formatted_chromatogram_exists(self.matrice_size, self.margin):
                create_chromatograms = True
                break

        # Get all samples and its associated peaks present in the current peak list and store them
        if create_chromatograms:
            sample_peak_dict = dict()
            for peak in peaks: 
                if peak.sample in sample_peak_dict:
                    sample_peak_dict[peak.sample].append(peak)
                else:
                    sample_peak_dict[peak.sample] = [peak]

            for sample_key in sample_peak_dict:
                sample_key.create_interpolated_chromatograms(self.matrice_size, self.margin, peak_list=sample_peak_dict[sample_key], min_scan_num=self.min_scan_num)
        # End hotfix

        valid_peaks = []
        for peak in peaks:

            if peak.valid:
                valid_peaks.append(peak)
                # Get the interpolated, baseline corrected and normalized peak
                # dataframe = peak.interpolate_chromatogram(self.matrice_size,  self.margin, True)
                dataframe = peak.get_formatted_chromatogram(self.matrice_size, self.margin)

                # Extract only the Intensity and window value
                data.append([dataframe[0], dataframe[1]])
                # If merge classes is on, replace the classes to merge by the unique class given
                if len(merge_classes) > 0:
                    if peak.annotation.label in merge_classes[0]:
                        labels.append(merge_classes[1])
                    else:
                        labels.append(peak.annotation.label)
                else:
                    labels.append(peak.annotation.label)
                ids.append(peak.id)

        # Create an encoder and fit the label list 
        label_encoder = LabelEncoder()
        label_encoder.fit(self.label_list)

        # Find the index to create training/test split
        split_index = int(round(validation_split*len(valid_peaks)))

        # Create test data
        test_data = data[:split_index]
        test_labels = labels[:split_index]
        test_ids = ids[:split_index]

        # Create validation data
        val_data = data[split_index:split_index*2]
        val_labels = labels[split_index:split_index*2]
        val_ids = ids[split_index:split_index*2]

        # Create training data
        training_data = data[split_index*2:]
        training_labels = labels[split_index*2:]
        training_ids = ids[split_index*2:]
        

        self.test_batch = dict(
            ids = test_ids,
            data = np.asarray(test_data),
            batch_label = 'test_batch',
            labels = label_encoder.transform(np.asarray(test_labels)),
            label_str = test_labels)

        self.training_batch = dict(
            ids = training_ids,
            data = np.asarray(training_data),
            batch_label = 'training_batch',
            labels = label_encoder.transform(np.asarray(training_labels)),
            label_str  = training_labels)

        self.validation_batch = dict(
            ids = val_ids,
            data = np.asarray(val_data),
            batch_label = 'validation_batch',
            labels = label_encoder.transform(np.asarray(val_labels)),
            label_str = val_labels)

        # Format the data to feed the CNN
        self.training_batch_size = len(self.training_batch['ids'])
        self.test_batch_size = len(self.test_batch['ids'])
        self.validation_batch_size = len(self.validation_batch['ids'])

        # Reshape all batches of data
        self.X_train = self.training_batch['data'].reshape(self.training_batch_size,2,self.matrice_size,1).astype('float32')
        self.X_test = self.test_batch['data'].reshape(self.test_batch_size,2,self.matrice_size,1).astype('float32')
        self.X_validation = self.validation_batch['data'].reshape(self.validation_batch_size,2,self.matrice_size,1).astype('float32')

        # One hot encoding of labels
        self.y_train = to_categorical(self.training_batch['labels'])
        self.y_test = to_categorical(self.test_batch['labels'])
        self.y_validation = to_categorical(self.validation_batch['labels'])


    def create_model(self, lr=0.00001, optimizer='Adam', model=None):
        from keras.models import Model, load_model
        from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Input, concatenate
        from keras import regularizers
        from keras.optimizers import SGD, Adam     

        # if model then load existing trained model
        if model:
            # This is not right until we also save the training and testing sets
            model_path = pathlib.Path(model)
            logger.info('Using existing model: %s',model_path.resolve())
            self.class_model = load_model(model)

        # else, create a new model
        else:
            number_of_classes = len(self.label_list)

            inputs_A = Input(shape=(2,self.matrice_size,1))
            output_1A = Conv2D(32, kernel_size=5, activation='relu', padding='same', name='conv2d_1')(inputs_A)
            output_2A = MaxPooling2D(pool_size=(1, 2))(output_1A)
            output_3A = Conv2D(64, kernel_size=3, activation='relu', padding='same', name='conv2d_2')(output_2A)

            output_4A = Flatten(name='flatten_1')(output_3A)

            output_5A = Dense(128, kernel_regularizer=regularizers.l2(0.001), activation='relu', name='dense_1')(output_4A)
            output_6A = Dropout(0.5, name='dropout_1')(output_5A)
            prediction_class = Dense(number_of_classes, activation='softmax', name='dense_prediction')(output_6A)
            self.class_model = Model(inputs=inputs_A, outputs=prediction_class)            

            if optimizer != 'Adam':
                opt = SGD(lr=lr)
            else:
                opt = Adam(lr=lr)
            logger.info('Compiling model')
            self.class_model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics=['accuracy','mae'])
        return None


    def get_model_summary(self):
        self.class_model.summary()


    def train_model(self, epochs=1000):
        self.class_model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), epochs=epochs)            


    def get_threshold(self, label='High_quality'):
        prob_df = self.get_true_vs_false_positive_df(label)
        prob_df['diff'] = prob_df['True'] - prob_df['False']
        threshold = prob_df['Probablity_threshold'][prob_df['diff'].idxmax()]
        return threshold


    def predict_peaks(self, threshold, samples=[], label='High_quality'):
        # Get the index of the label monitored
        probability_index = self.label_list.index(label)
        # If the threshold is not set manually
        # Get the otpimum threshold for label assignment
        # Make sure the sample list is not empty
        if not samples:
            samples = self.experiment.samples
        logger.info('Predicting peaks from {} samples'.format(len(samples)))
        # Create batches of data, one batch of data corresponds to all peaks in one sample
        for sample in samples:
            # Temporary fix to remove all existing formatted chromatogram created during training as they are pandas dataframes and not numpy arrays
            # TODO: rewrite the training part to use numpy arrays so this fix is not required anymore
            for peak in sample.peak_list:
                peak.formatted_chromatograms = []
                # Hotfix to add a prediction attribute to the peak object
                # TODO: Add it to the class definition
                peak.prediction = None

            # Generate all peak dataframe for the specific sample
            logger.info("Extracting and formatting peak chromatograms for sample: %s",sample.name)
            sample.create_interpolated_chromatograms(self.matrice_size, self.margin, min_scan_num=self.min_scan_num)
            data = []
            # Create the batch by adding all formatted chromatograms in one list
            valid_peak_list = []
            for peak in sample.peak_list:
                # Only add the peak if it is valid
                if peak.valid:
                    dataframe = peak.get_formatted_chromatogram(self.matrice_size, self.margin)
                    data.append([dataframe[0],dataframe[1]])
                    valid_peak_list.append(peak)

            # Transform the data batch (list) to a numpy array
            batch = np.asarray(data)
            # Reshape the batch so it can be fed to the Neural Network
            batch = batch.reshape(batch.shape[0],2,self.matrice_size,1).astype('float32')
            # Make prediction for the batch (all peaks present in this sample)
            logger.info('Predicting %d peaks',batch.shape[0])
            batch_prediction =  self.class_model.predict(batch)
            # Iterate through peaks to assign the predicted label
            for i in range(len(valid_peak_list)):
                # Test if the probability of the monitored label is above the threshold
                if batch_prediction[i][probability_index] >= threshold:
                    # If so set this label as the predicted peak label
                    self.annotation_table.set_peak_predicted_label(valid_peak_list[i], label)
                # Else, get the label associated with the prediction and assign it to the peak
                else:
                    # Get the index of the highest probability
                    predicted_arg = np.argmax(batch_prediction[i])
                    # Get the corresponding label
                    predicted_label = self.label_list[predicted_arg]
                    # Set this label as the predicted peak label
                    self.annotation_table.set_peak_predicted_label(valid_peak_list[i], predicted_label)


    def get_true_vs_false_positive_df(self, label='High_quality'):
        # Get the index of the label monitored
        probability_index = self.label_list.index(label)

        low_quality_index = self.label_list.index('Low_quality')
        # Create a dataframe to store all predicted values
        prob_array = np.arange(0,1.0,0.01)
        true_prediction = np.zeros(len(prob_array))
        false_prediction = np.zeros(len(prob_array))
        false_low = np.zeros(len(prob_array))
        false_noise = np.zeros(len(prob_array))

        prob_df = pd.DataFrame(list(zip(prob_array, true_prediction, false_prediction, false_low, false_noise)), 
                               columns =['Probablity_threshold', 'True', 'False', 'False_low', 'False_noise'])

        true_prediction = 0
        false_prediction = 0

        # Iterate through the test data and predict probability of belonging to the given label
        for i in range(len(self.X_validation)+1):
            j = i-1
            if j >= 0:
                prediciton_one_hot = self.class_model.predict(self.X_validation[j:i])

                # Get the real label of the peak
                real_arg = np.argmax(self.y_validation[j])
                # Get the predicted probability for the monitored label
                prediction_prob = prediciton_one_hot[0][probability_index]
                # If the label of the peak == the label monitor we have a true peak
                if real_arg == probability_index:
                    prob_df.loc[prob_df['Probablity_threshold'] <= prediction_prob, 'True'] += 1
                # Else this is a noise peak
                elif real_arg == low_quality_index:
                    prob_df.loc[prob_df['Probablity_threshold'] <= prediction_prob, 'False_low'] += 1
                    prob_df.loc[prob_df['Probablity_threshold'] <= prediction_prob, 'False'] += 1
                else:
                    prob_df.loc[prob_df['Probablity_threshold'] <= prediction_prob, 'False_noise'] += 1
                    prob_df.loc[prob_df['Probablity_threshold'] <= prediction_prob, 'False'] += 1

        prob_df['True'] = prob_df['True']/prob_df['True'].max()
        prob_df['False'] = prob_df['False']/prob_df['False'].max()
        prob_df['False_low'] = prob_df['False_low']/prob_df['False_low'].max()
        prob_df['False_noise'] = prob_df['False_noise']/prob_df['False_noise'].max()


        return prob_df

