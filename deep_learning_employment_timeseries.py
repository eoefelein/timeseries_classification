import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import pandas as pd
import plotly.graph_objects as go

# Step 1. Launch the application
app = dash.Dash()

# Step 2. Read file
us_employment = pd.read_csv(
    "OECD/EconomicTracker-main/EconomicTracker-main/data/Employment - County - Daily.csv"
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
print(df)

# dropdown options
features = df.columns
opts = [{'label' : i, 'value' : i} for i in features]

# Step 3. Create a plotly figure
# trace_1 = go.Scatter(
#     x=df.index, y=df.columns, line=dict(width=2, color="rgb(229, 151, 50)")
# )
# fig = px.line(df, x=df.index, y=df.columns)
# fig.update_layout(template="plotly_dark")
# fig.show()

# Step 4. Create a Dash layout
app.layout = html.Div([
                # a header and a paragraph
                html.Div([
                    html.H1("Employment per U.S. County"),
                    html.P("Dash is so interesting!!")],
                    style = {'padding' : '50px' ,'backgroundColor' : '#3aaab2'}),
                # dropdown
                html.P([
                    html.Label("Choose a county"),
                    dcc.Dropdown(id = 'opt', options = opts,
                                value = opts[0])
                        ], style = {'width': '400px',
                                    'fontSize' : '20px',
                                    'padding-left' : '100px',
                                    'display': 'inline-block'}),
                # adding a plot
                dcc.Graph(id = 'plot')
])

# Step 5. Add callback functions
@app.callback(Output('plot', 'figure'),[Input('opt', 'value')])
def update_figure(input):
    # updating the plot
    fig = go.Figure([go.Bar(x=df.index, y=df[input])])
    # fig = px.bar(df, x=df.index, y=df[input])
    return fig

# Step 6. Add the server clause
if __name__ == "__main__":
    app.run_server()