import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import operator
import pandas as pd
import numpy as np
from functools import reduce
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split

from keras.layers import (
    Conv1D,
    Dense,
    Dropout,
    # Input,
    Concatenate,
    GlobalMaxPooling1D,
    MaxPooling1D,
    Flatten,
)
from keras.models import Model, Sequential

# Since we're adding callbacks to elements that don't exist in the app.layout,
# Dash will raise an exception to warn us that we might be
# doing something wrong.
# In this case, we're adding the elements through a callback, so we can ignore
# the exception.
app = dash.Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

index_page = html.Div([
    html.H1(
        children='Past & Future Employment By County',
        style = {
            'width': '400px',
            'fontSize' : '120px',
            'textAlign' : 'left',
            'padding-left' : '5px',
            'display': 'inline-block'
            }
    ),
    html.Br(),
    dcc.Link('Go to Past Employment', href='/page-1'),
    html.Br(),
    dcc.Link('Go to Future Employment', href='/page-2'),
])

############### DATA FOR EMPLOYMENT TIME SERIES BY COUNTY
# Step 2. Read file
us_employment = pd.read_csv(
    "C:/Users/oefel/Desktop/R_project/OECD/EconomicTracker-main/EconomicTracker-main/data/Employment - County - Daily.csv"
)
# social capital
us_social_indices = pd.read_csv("capturing-dataset.tsv", sep="\t").rename(
    {"fips_n": "countyfips"}, axis=1
)

# Step 3. Data prep
# index covid cases to start at date after covid vaccine is rolled out
us_employment["date"] = pd.to_datetime(us_employment[["year", "month", "day"]])
# index mobility to start at date after covid vaccine is rolled out
us_employment = us_employment[us_employment["date"] > "2020-12-18"]
# drop nan
us_employment = us_employment[
    ~(us_employment["emp_incbelowmed"] == ".")
]

# merge data
merged_data = us_employment.merge(
    us_social_indices[["countyfips", "countyname"]], on=["countyfips"]
)
# convert emp_incbelowmed to numeric
# us_employment["countyfips"] = us_employment["countyfips"].astype(str)
merged_data["emp_incbelowmed"] = merged_data["emp_incbelowmed"].astype(float)
df = merged_data.pivot(index="date", columns="countyname", values="emp_incbelowmed")

page_1_layout = html.Div([
    html.H1('Past Employment By County'),
    html.P([
    html.Label("Choose a county"),
        dcc.Dropdown(
            id='page-1-dropdown',
            options=[{'label' : i, 'value' : i} for i in sorted(df.columns)],
            value='Travis County, Texas'
    )], style = {'width': '400px',
                                    'fontSize' : '20px',
                                    'padding-left' : '100px',
                                    'display': 'inline-block',
                                    'white-space': 'nowrap', 
                                    'text-overflow': 'ellipsis'}),
    # adding a plot
    dcc.Graph(id = 'plot1'),
    # html.Div(id='page-1-content'),
    html.Br(),
    dcc.Link('Go to Page 2', href='/page-2'),
    html.Br(),
    dcc.Link('Go back to home', href='/'),
])

@app.callback(dash.dependencies.Output('plot1', 'figure'),[dash.dependencies.Input('page-1-dropdown', 'value')])
def page_1_dropdown(value):
    # updating the plot
    fig = go.Figure([go.Bar(x=df.index, y=df[value])])
    return fig

############### DATA FOR PREDICTING EMPLOYMENT RECOVERY
# Step 2. Read file
X_countyname = pd.read_csv(
    "C:/Users/oefel/Desktop/timeseries_classification/X_countyname.csv"
)

dropdown_counties = []
for county in X_countyname["countyname"].unique():
    subset = X_countyname[X_countyname["countyname"] == county]
    if (
        subset[["new_case_rate", "gps_away_from_home", "spend_all"]].to_numpy().shape[0]
        == 86
    ):
        dropdown_counties.append(county)

# Step 2. Read file
y_countyname = pd.read_csv(
    "C:/Users/oefel/Desktop/timeseries_classification/y_countyname.csv"
)

page_2_layout = html.Div([
    html.H1('Predicting Employment by County'),
    html.P([
    html.Label("Choose a county"),
        dcc.Dropdown(
            id='page-2-dropdown',
            options=[{'label' : i, 'value' : i} for i in sorted(dropdown_counties)],
            value='Travis County, Texas')
    ], style = {'width': '400px',
                'fontSize' : '20px',
                'padding-left' : '100px',
                'display': 'inline-block',
                'white-space': 'nowrap', 
                'text-overflow': 'ellipsis'}),
    # adding a plot
    dcc.Graph(id = 'plot2'),
    # html.Div(id='page-2-content'),
    html.Br(),
    dcc.Link('Go to Page 1', href='/page-1'),
    html.Br(),
    dcc.Link('Go back to home', href='/')
])

@app.callback(dash.dependencies.Output('plot2', 'figure'),[dash.dependencies.Input('page-2-dropdown', 'value')])
def page_2_radios(value):
    input = value
        # input = str(input)
    # take out selected
    X_minus_selected = X_countyname[X_countyname["countyname"] != input]

    # ensure all timeseries are all of the same length(86 days)
    check_len_timeseries = []
    counties = []

    for county in X_minus_selected["countyname"].unique():
        subset = X_minus_selected[X_minus_selected["countyname"] == county]
        if (
            subset[["new_case_rate", "gps_away_from_home", "spend_all"]]
            .to_numpy()
            .shape[0]
            == 86
        ):
            check_len_timeseries.append(
                subset[["new_case_rate", "gps_away_from_home", "spend_all"]].to_numpy()
            )
            counties.append(county)
    # create X
    X = (
        np.concatenate(check_len_timeseries, axis=0)
        .reshape(len(counties), 86, 3)
        .astype(np.float32)
    )  # sample, timesteps, features
    y_minus_selected = y_countyname[y_countyname["countyname"] != input]
    # create y
    total_perc_change = []

    for county in counties:
        subset = y_minus_selected[y_minus_selected["countyname"] == county]
        total_perc_change.append(
            reduce(lambda x, y: x + y + x * y, subset["emp_incbelowmed"], 1)
        )

    y = np.array(total_perc_change).astype(np.float32)
    # split to train, test, split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    # add selected back to test data
    X_selected = X_countyname[X_countyname["countyname"] == input][
        ["new_case_rate", "gps_away_from_home", "spend_all"]
    ].to_numpy()
    X_selected.reshape(1, 86, 3)
    # add selected back to test
    X_test = np.vstack((X_test, X_selected.reshape(1, 86, 3)))
    y_subset = y_countyname[y_countyname["countyname"] == input]
    y_selected = reduce(lambda x, y: x + y + x * y, y_subset["emp_incbelowmed"], 1)
    y_selected = np.array(y_selected).astype(np.float32)
    y_test = np.append(y_test, y_selected)

    n_timesteps, n_features, n_outputs = (
        X_train.shape[1],
        X_train.shape[2],
        1,
    )

    def create_model(verbose=0, epochs=10, batchsize=32):
        model = Sequential()
        # solves what is called the vanishing gradient problem whereby
        # the neural network would not be able to feed back important gradient information
        # from the output layer back to the input layer
        model.add(
            Conv1D(
                filters=10,
                kernel_size=1,
                activation="relu",  # popular with regression neural nets
                input_shape=(n_timesteps, n_features),
            )
        )
        model.add(Conv1D(filters=10, kernel_size=1, activation="relu"))
        model.add(Dropout(0.5))
        # model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(50, activation="relu"))  # try activation='elu' ???
        # Initializers define the way to set the initial random weights of Keras layers.
        model.add(Dense(n_outputs, kernel_initializer="normal", activation="linear"))
        return model

    # create the model
    model = create_model()
    model.compile(
        loss="mean_absolute_error", optimizer="adam", metrics=["mean_absolute_error"]
    )
    model.fit(X_train, y_train)
    weights = model.get_weights()
    ## create single item model
    single_item_model = create_model(batchsize=1)
    single_item_model.set_weights(weights)
    single_item_model.compile(
        loss="mean_absolute_error", optimizer="adam", metrics=["mean_absolute_error"]
    )

    prediction = single_item_model.predict(
        np.array(X_test[-1], ndmin=3)
    )  # pass counties.index('countyname')
    # print(prediction[0])
    # prediction = -.2784392

    # updating the plot
    dates = df.index.append(pd.Index([pd.to_datetime("2021-10-24")])).values
    employment = df[input].append(pd.Series(prediction[0])).values
    fig = go.Figure([go.Bar(x=dates, y=employment)])
    return fig


# Update the index
@app.callback(dash.dependencies.Output('page-content', 'children'),[dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/page-1':
        return page_1_layout
    elif pathname == '/page-2':
        return page_2_layout
    else:
        return index_page
    # You could also return a 404 "URL not found" page here


if __name__ == '__main__':
    app.run_server(debug=True)