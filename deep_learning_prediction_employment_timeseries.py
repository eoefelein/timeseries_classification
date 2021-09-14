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
# https://towardsdatascience.com/a-gentle-invitation-to-interactive-visualization-with-dash-a200427ccce9

from keras.layers import (
    Conv1D,
    Dense,
    Dropout,
    Input,
    Concatenate,
    GlobalMaxPooling1D,
    MaxPooling1D,
    Flatten,
)
from keras.models import Model, Sequential

# Step 1. Launch the application
app = dash.Dash()

# Step 2. Read file
X_countyname = pd.read_csv(
    "C:/Users/oefel/Desktop/timeseries_classification/X_countyname.csv"
)
# take out selected
# X_countyname = X_countyname[X_countyname["countyname"] != input]

# ensure all timeseries are all of the same length(86 days)
# check_len_timeseries = []
dropdown_counties = []

for county in X_countyname["countyname"].unique():
    subset = X_countyname[X_countyname["countyname"] == county]
    if (
        subset[["new_case_rate", "gps_away_from_home", "spend_all"]].to_numpy().shape[0]
        == 86
    ):
        # check_len_timeseries.append(
        #     subset[["new_case_rate", "gps_away_from_home", "spend_all"]].to_numpy()
        # )
        dropdown_counties.append(county)

# # create X
# X = (
#     np.concatenate(check_len_timeseries, axis=0).reshape(1309, 86, 3).astype(np.float32)
# )  # sample, timesteps, features

# Step 2. Read file
y_countyname = pd.read_csv(
    "C:/Users/oefel/Desktop/timeseries_classification/y_countyname.csv"
)

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
us_employment = us_employment[us_employment["date"] > "2020-05-01"]
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

# dropdown options
features = dropdown_counties
opts = [{'label' : i, 'value' : i} for i in features]

# Step 5. Add callback functions
@app.callback(Output('plot', 'figure'),[Input('opt', 'value')])
def update_figure(input, X_countyname, y_countyname):
    selected = input
    # take out selected
    X_minus_selected = X_countyname[X_countyname["countyname"] != selected]

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
    y_minus_selected = y_countyname[y_countyname["countyname"] != selected]
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
    X_selected = X_countyname[X_countyname["countyname"] == selected][
        ["new_case_rate", "gps_away_from_home", "spend_all"]
    ].to_numpy()
    # add selected back to test
    X_test = np.vstack((X_test, X_selected.reshape(1, 86, 3)))
    y_subset = y_countyname[y_countyname["countyname"] == selected]
    y_selected = reduce(lambda x, y: x + y + x * y, y_subset["emp_incbelowmed"], 1)
    y_selected = np.array(y_selected).astype(np.float32)
    y_test = np.append(y_test, y_selected)

    n_timesteps, n_features, n_outputs = (
        X_train.shape[1],
        X_train.shape[2],
        1,
    )

    def create_model(verbose=0, epochs=10):
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
        model.add(Dense(50, activation="elu"))  # try activation='elu' ???
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
    single_item_model = create_model()
    single_item_model.set_weights(weights)
    single_item_model.compile(
        loss="mean_absolute_error", optimizer="adam", metrics=["mean_absolute_error"]
    )

    prediction = single_item_model.predict(
        np.array(X_test[-1], ndmin=3)
    )  # pass counties.index('countyname')
    print(prediction[0])

    # updating the plot
    dates = df.index.append(pd.Series(pd.to_datetime("2021-10-24"))).reset_index(drop=True)
    employment = df[input].append(pd.Series(prediction[0])).reset_index(drop=True)
    fig = go.Figure([go.Bar(x=dates, y=employment)])
    # fig = px.bar(df, x=df.index, y=df[input])
    return fig

# index = counties.index(input)
# single_item_model.predict(np.array(X[index], ndmin=3)) # pass counties.index('countyname')

# Step 6. Add the server clause
if __name__ == "__main__":
    app.run_server()