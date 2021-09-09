import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import pandas as pd
import plotly.express as px

# Step 1. Launch the application
app = dash.Dash()

# Step 2. Read file
# covid case rate
us_covid_cases = pd.read_csv("C:/Users/oefel/Desktop/R_project/OECD/us_covid_recovery.csv").drop(

    [
        "Unnamed: 0",
    ],
    axis=1,
)

# Step 3. Data prep
# index covid cases to start at date after covid vaccine is rolled out
us_covid_cases = us_covid_cases[us_covid_cases["date"] > "2020-12-18"]
us_covid_cases["countyfips"] = us_covid_cases["countyfips"].astype(str)
df = us_covid_cases.pivot(index="date", columns="countyfips", values="new_case_count")

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
                    html.H1("This is my first dashboard"),
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
    fig = px.line(df, x=df.index, y=df[input])
    return fig

# Step 6. Add the server clause
if __name__ == "__main__":
    app.run_server()