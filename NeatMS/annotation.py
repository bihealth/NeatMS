import random

class Annotation():

    """
    Representation of a label


    Contains a list of peaks which have been assigned a specific label
    """

    # Annotation counter to automatically assign annotation ids
    annotation_counter = 0

    def __init__(self, label):
        self.id = Annotation.annotation_counter
        self.label = label
        self.peaks = []
        Annotation.annotation_counter += 1


class AnnotationTable():

    """
    Keep track of peaks that have been labelled.

    Contains ordered list labelled peaks and unlabelled peaks.
    Facilitate labelling of new peaks. 
    """

    # Annotation table counter to automatically assign annotation table ids
    annotation_table_counter = 0

    def __init__(self, feature_table=None, labels=[]):
        self.id = AnnotationTable.annotation_table_counter
        self.feature_table = feature_table
        self.labelled_peaks = []
        self.unlabelled_peaks = self.create_unlabelled_peak_list()
        self.annotations = self.create_annotation_objects(labels)
        AnnotationTable.annotation_table_counter += 1

    def create_annotation_objects(self, labels):
        annotations = []
        for label in labels:
            annotation = Annotation(label)
            annotations.append(annotation)
        return annotations

    def create_unlabelled_peak_list(self, monoisotopic_only=False):
        if self.feature_table:
            # unlabelled_peaks = [peak for feature in self.feature_table.feature_list for peak in feature.peak_list]
            unlabelled_peaks = []
            for feature_collection in self.feature_table.feature_collection_list:
                for feature in feature_collection.feature_list:
                    for peak in feature.peak_list:
                        if monoisotopic_only: 
                            if peak.monoisotopic:
                                unlabelled_peaks.append(peak)
                        else:
                            unlabelled_peaks.append(peak)
            self.unlabelled_peaks = unlabelled_peaks
            return unlabelled_peaks
        else:
            print("No feature table yet attached to the annotation table")
            return None

    def get_peak_to_label(self):
        random_peak = random.choice(self.unlabelled_peaks)
        return random_peak

    def get_annotation(self, label):
        for annotation in self.annotations:
            if annotation.label == label:
                return annotation
        print("No annotation with the label exist!")
        return None

    def set_peak_label(self, peak, label):
        # If the peak already has an annotation
        if peak.annotation:
            # Remove the peak from the peak list in the annotation object
            peak.annotation.peaks.remove(peak)
            # We do not remove the annotation from the peak as it will be replaced by the new one
        annotation = self.get_annotation(label)
        peak.annotation = annotation
        annotation.peaks.append(peak)
        # If the peak is not already in the labelled peak list
        if peak not in self.labelled_peaks:
            # Add the peak
            self.labelled_peaks.append(peak)
        # If the peak is still in the unlabelled peak list
        if peak in self.unlabelled_peaks:
            # Remove the peak
            self.unlabelled_peaks.remove(peak) 


    def set_peak_predicted_label(self, peak, label):
        # We do not touch peak annotation here, only prediction
        # If the peak already has a prediction
        if peak.prediction:
            # Remove the peak from the peak list in the annotation (prediction) object
            peak.preditcion.peaks.remove(peak)
            # We do not remove the prediction from the peak as it will be replaced by the new one
        # Get the new annotation object using the label 
        prediction = self.get_annotation(label)
        # Assign it to the peak as prediction
        peak.prediction = prediction
        # Add the peak to the annotation object peak list
        prediction.peaks.append(peak)


class AnnotationTool():

    """
    The annotation tool only points to an annotation table. 
    The tool can access the already labelled peaks using the ordered list in the annotation table.
    New peaks to annotate are always selected randomly from the unlabelled peak list.

    This class implements the jupyter notebook integrated interface for peak labelling.

    Extra external libraries are required to launch the annotation tool. See documentation for  more details. 
    """

    def __init__(self, experiment, margin=1, review=[], table_index=0):


        self.annotation_table = experiment.feature_tables[table_index].annotation_table
        # Retention time window margins
        self.margin = margin
        # List of classes to review, if the list is not empty, review mode is considered 'ON'
        self.review = review

    def launch_annotation_tool(self):
        from jupyter_dash import JupyterDash

        import dash
        import dash_core_components as dcc
        import dash_html_components as html
        import plotly.graph_objs as go
        # Initialise with the first peak to annotate
        # Every variable has to be assigned to self to be accessible by the annotation tool due to function closure
        # Get a random peak to annotate
        if not self.review:
            self.peak = self.annotation_table.get_peak_to_label()
            # Set a variable to track the "previous" button click (go through the labelled peak list backwards)
            # When the buttin next is clicked:
            # If previous == 0, the peak to labelled will be randomly selected from the unlabelled peak list
            # If previous < 0, we return the corresponding peak in the labelled peak list
            # Previous is initialised to 0 as the first peak is randomly selected from the unlabelled peak list
            self.previous = 0
        else:
            # When reviewing peaks, we set previous to the first peak in the labelled peak list
            self.previous = -len(self.annotation_table.labelled_peaks)
            review_peak = False
            # We increment previous until we find a peak that is labelled with a class in the review list 
            while review_peak == False:
                # If the peak label is in the review list
                if self.annotation_table.labelled_peaks[self.previous].annotation.label in self.review:
                    # Assign this peak to the current peak 
                    self.peak = self.annotation_table.labelled_peaks[self.previous]
                    # Set review_peak to True to get out of the loop
                    review_peak = True
                else:
                    # Otherwise we increment previous to check the next peak
                    self.previous += 1

        # Get the list of possible labels regardless of review mode
        self.labels = [annotation.label for annotation in self.annotation_table.annotations]
        self.label_options = []
        for label in self.labels:
            self.label_options.append(dict(label=label.replace('_',' '), value=label))

        # This is currently not supported when Dash is embedded in Jupyter notebook
        external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

        # app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
        app = JupyterDash('NeatMS_annotation_tool')

        app.layout = html.Div(children=[

            dcc.Graph(
                id='peak-graph',
            ),
            html.Div([
                html.Div([
                    dcc.RadioItems(
                        id='input-class',
                        options=self.label_options,
                        value='Noise',
                        labelStyle={'display': 'inline-block', 'text-align': 'justify', 'margin-right': '2%'}
                    ),
                ], style={'margin-bottom':'2%','textAlign': 'center'}),
                html.Div([
                    html.Button('Previous', id='btn-previous', n_clicks_timestamp=0),
                    html.Button('Next', id='btn-next', n_clicks_timestamp=0),
                ], style={'margin-bottom':'2%','textAlign': 'center'}),
                html.Div([
                    dcc.Slider(
                        id='slider-updatemargin',
                        min=0,
                        max=1,
                        step=0.05,
                        value=self.margin,
                        marks={
                            0: {'label': '0'},
                            0.25: {'label': '0.25'},
                            0.50: {'label': '0.50'},
                            0.75: {'label': '0.75'},
                            1: {'label': '1'}
                        },
                        updatemode='drag'
                    ),
                    html.Div(id='updatemargin-output-container', style={'margin-top': 20}) 
                ], style={'width': '100%', 'textAlign': 'center', 'justify-content': 'center'}),
            ], style={'textAlign': 'center'})

        ])

        def create_line_plot(chromatogram, boundaries):
            return {
                'data': [
                        go.Scatter(x=chromatogram[0], y=chromatogram[1], mode='lines')
                    ],
                'layout': dict(
                        title= str('Peak %i - m/z: %.4f (%.4f-%.4f)' % (boundaries['id'], boundaries['mz'], boundaries['mz_min'], boundaries['mz_max'])),
                        shapes = [
                            dict(
                                type = 'line',
                                x0 = boundaries['rt_start'],
                                y0 = 0,
                                x1 = boundaries['rt_start'],
                                y1 = boundaries['height'],
                                line = dict(
                                    color = 'rgb(0, 0, 0)',
                                    width = 2
                                )
                            ),
                            dict(
                                type = 'line',
                                x0 = boundaries['rt_end'],
                                y0 = 0,
                                x1 = boundaries['rt_end'],
                                y1 = boundaries['height'],
                                line = dict(
                                    color = 'rgb(0, 0, 0)',
                                    width = 2
                                )
                            )
                        ],
                        xaxis = dict(
                            title = 'Retention time (min)'
                        ),
                        yaxis = dict(
                             title = 'Intensity'
                        )
                )

            }

        @app.callback(dash.dependencies.Output('updatemargin-output-container', 'children'),
              [dash.dependencies.Input('slider-updatemargin', 'value')])
        def display_value(value):
            return 'Retention time window margin: {:0.0f}%'.format(value*100)

        @app.callback(
            dash.dependencies.Output('peak-graph', 'figure'),
                    [dash.dependencies.Input('btn-next', 'n_clicks_timestamp'),
                    dash.dependencies.Input('btn-previous', 'n_clicks_timestamp'),
                    dash.dependencies.Input('slider-updatemargin', 'value')],
                    [dash.dependencies.State('input-class', 'value')])
        def updatePeak(next_peak, previous_peak, margin_value, class_value):
            # If the margin value has changed, no button has been click, only the slider was changed
            if margin_value != self.margin:
                # We only update the margin value
                self.margin = margin_value
            else:
                # If next or previous button has been clicked
                if next_peak > 0 or previous_peak > 0:
                    # If next button has been clicked
                    if next_peak > previous_peak:
                        # If we are looking at peaks already annotated 
                        if self.previous < -1:
                            # Change annotation of the previous peak
                            self.annotation_table.set_peak_label(self.peak, class_value)
                            # If review mode is 'OFF'
                            if not self.review:
                                # Increment the previous counter
                                self.previous += 1
                                # Get the next peak in the list
                                self.peak = self.annotation_table.labelled_peaks[self.previous]
                            # If review mode is 'ON'
                            else:
                                review_peak = False
                                self.previous += 1
                                # We increment previous until we find a peak that is labelled with a class in the review list 
                                while review_peak == False:
                                    # If the peak label is in the review list
                                    if self.annotation_table.labelled_peaks[self.previous].annotation.label in self.review:
                                        # Assign this peak to the current peak 
                                        self.peak = self.annotation_table.labelled_peaks[self.previous]
                                        # Set review_peak to True to get out of the loop
                                        review_peak = True
                                    else:
                                        # Otherwise we increment previous to check the next peak
                                        self.previous += 1
                        else:
                            # Change annotation of the previous peak (Noise for now)
                            self.annotation_table.set_peak_label(self.peak, class_value)
                            # If we are looking at unlabelled peaks, make sure the previous counter is 0
                            self.previous = 0
                            # Get a random peak
                            self.peak = self.annotation_table.get_peak_to_label()
                    # If previous button was clicked
                    else :
                        # Change annotation of the previous peak (Noise for now)
                        self.annotation_table.set_peak_label(self.peak, class_value)
                        # If first click on previous, decrement by 2 as we just added the new labelled peak as the last peak in the list
                        if self.previous == 0:
                            self.previous -= 2
                        # Else we just decrement by 1
                        else:
                            # Decrement the previous counter 
                            self.previous -= 1
                        # If review mode is 'OFF'
                        if not self.review:
                            # Get the already annotated peak from the list
                            self.peak = self.annotation_table.labelled_peaks[self.previous]
                        # If review mode is 'ON'
                        else:
                            # Decrement of previous already applied above
                            review_peak = False
                            # We decrement previous until we find a peak that is labelled with a class in the review list 
                            while review_peak == False:
                                # If the peak label is in the review list
                                if self.annotation_table.labelled_peaks[self.previous].annotation.label in self.review:
                                    # Assign this peak to the current peak 
                                    self.peak = self.annotation_table.labelled_peaks[self.previous]
                                    # Set review_peak to True to get out of the loop
                                    review_peak = True
                                else:
                                    # Otherwise we increment previous to check the next peak
                                    self.previous -= 1
                else:
                    # Initialisation, no button has been yet clicked, returning the initial random peak
                    self.peak = self.peak
            chromatogram = self.peak.get_chromatogram(self.margin)
            boundaries = dict(
                id = self.peak.id,
                rt_start = self.peak.rt_start,
                rt_end = self.peak.rt_end,
                height = self.peak.get_height(),
                mz_min = self.peak.mz_min,
                mz_max = self.peak.mz_max,
                mz = self.peak.mz
                )
            return create_line_plot(chromatogram, boundaries)

        @app.callback(
            dash.dependencies.Output('input-class', 'value'),
                    [dash.dependencies.Input('peak-graph', 'figure')])
        def updatePeak(data):
            if self.peak.annotation != None:
                class_value = self.peak.annotation.label
            else:
                class_value = 'Noise'
            return class_value

        # return app
        app.run_server(mode='inline')

