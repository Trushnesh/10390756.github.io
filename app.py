import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
import plotly.plotly as py
from dash.dependencies import Input, Output

mapbox_access_token = "pk.eyJ1IjoicHJpeWF0aGFyc2FuIiwiYSI6ImNqbGRyMGQ5YTBhcmkzcXF6YWZldnVvZXoifQ.sN7gyyHTIq1BSfHQRBZdHA"

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

df = pd.read_csv(
    'D:/Trushnesh/DBS_Study_Material/Visualisation/database.csv')

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

map_list = ["Scatter Plots on Mapbox", "Choropleth Maps", "Scatter Plots on Maps", "Bubble Maps", "Lines on Maps"]
app.layout = html.Div([
    html.H1("Bubble chart by using plotly package", style={"textAlign": "center"}),
    html.Div([
        html.Div([
            dcc.Dropdown(id="map-type",
                         options=[
                             {'label': i, 'value': i} for i in map_list],
                         value='Scatter Plots on Mapbox', className="six columns")
        ], className="row",
            style={"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "80%",
                   "padding-top": 10}),

            dcc.Graph(id="my-graph"),], className="row"),

], className="container")

@app.callback(
    dash.dependencies.Output('my-graph', 'figure'),
    [dash.dependencies.Input('map-type', 'value')]
    )
def update_graph(type):
    events_df = df.drop([3378, 7512, 20650])

    events_df["Year"] = [(each.split("/")[2]) for each in events_df.iloc[:, 0]]

    data = events_df
    year_list = list(data['Year'].unique())
    YearType = []

    for i in year_list:
        val = data[data['Year'] == i]
        YearType.append(len(val))

    dfType = pd.DataFrame({'year_list': year_list, 'Count_Type': YearType})
    new_index = (dfType['year_list'].sort_values(ascending=True)).index.values
    sorted_data = dfType.reindex(new_index)

    magnitude = []
    for i in sorted_data.year_list:
        x = data[data['Year'] == i]
        data_magnitude = sum(x.Magnitude) / len(x.Magnitude)
        magnitude.append(data_magnitude)
    sorted_data["Magnitude"] = magnitude

    depth = []
    for i in sorted_data.year_list:
        x = data[data['Year'] == i]
        data_depth = sum(x.Depth) / len(x.Depth)
        depth.append(data_depth)
    sorted_data["Depth"] = depth
    # bubble chart visualization
    bubble_color = [each for each in sorted_data.Count_Type]
    bubble = [
        {
            'y': sorted_data.Magnitude,
            'x': sorted_data.Depth,
            'mode': 'markers',
            'marker': {
                'color': bubble_color,
                'size': sorted_data['Count_Type'],
                'sizemode': 'area',
                'sizeref': (2. * max(sorted_data['Count_Type']) / (40. ** 2)),
                'showscale': True},
            'text': sorted_data.year_list
        }
    ]

    layout2 = go.Layout(
        xaxis=dict(title='Average Depth of Each Year'),
        yaxis=dict(title='Average Magnitude of Each Year'))


    return {
        'data': bubble,
        'layout': layout2
    }

server = app.server

if __name__ == '__main__':
    app.run_server(debug=True)





