from flask import Flask, request, send_from_directory
from dash.dependencies import Input, Output, State
from scipy.stats.stats import pearsonr
import flask
import dash
import dash_table
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import plotly
import requests
import re
import os
import base64
import io
import logging

logging.basicConfig(level=logging.INFO)
server = Flask(__name__)
main_app = dash.Dash(server=server, routes_pathname_prefix='/main/')
main_app.title = 'Feature Analyzer'
main_app.secret_key = 'HsuanweiyangFeatureAnalyzerKey'

upload_data_card = dbc.Card(
    [
        dbc.CardHeader('Upload your data'),
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col(dcc.Upload(id='upload_button', children=[html.Button('Upload')], multiple=False)),
                        dbc.Col(html.Div(id='upload_status'))
                    ]
                ),
                dbc.Row(
                    [
                        html.Details(
                            [
                                html.Summary('Expend Upload Info'),
                                dash_table.DataTable(
                                    id='upload_table',
                                    style_table={'height': '30vh', 'overflowY': 'scroll'}
                                )
                            ], open=False, id='upload_table_expend'
                        )
                    ]
                ),

            ]
        )
    ]
)

feature_info_card = dbc.Card(
    [
        dbc.CardHeader('Feature Information'),
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dcc.Dropdown(id='file_select', placeholder='Select File'),
                        dbc.Button(id='file_select_button', children='Select')
                    ]
                ),
                dbc.Row(
                    [
                        html.Details(
                            [
                                html.Summary('Expend Details'),
                                dash_table.DataTable(
                                    id='file_info_table', style_table={'height': '30vh', 'overflowY': 'scroll'},
                                    row_selectable="multi", sort_action="native"
                                )
                            ], open=False, id='file_info_table_expend'
                        )
                    ]
                ),
                dbc.Row(
                    [
                        html.Details(
                            [
                                html.Summary('Expend Graph'),
                                dcc.Graph(
                                    id='feature_graph'
                                )
                            ], open=False, id='feature_graph_expend'
                        )
                    ]
                ),
                dbc.Row(
                    [
                        html.Details(
                            [
                                html.Summary('Expend Correlation Table'),
                                dcc.Checklist(id='target_select', labelStyle={'display': 'inline-block'}),
                                dbc.Button(id='target_select_button', children='Submit'),
                                dash_table.DataTable(
                                    id='feature_correlation_table', style_table={'height': '30vh', 'overflowY': 'scroll'},
                                    sort_action="native", export_headers='display', export_format='csv'
                                )
                            ], open=False, id='feature_correlation_table_expend'
                        )
                    ]
                ),
            ]
        )
    ]
)


main_app.layout = html.Div(
    [upload_data_card, feature_info_card]
)


@main_app.callback(
    [Output('upload_status', 'children'), Output('upload_table', 'columns'), Output('upload_table', 'data')],
    [Input('upload_button', 'contents')],
    [State('upload_button', 'filename')]
)
def upload_function(contents, filename):
    upload_msg = '上傳成功:{}'.format(filename)
    content_type, content_str = contents.split(',')
    decode_str = base64.b64decode(content_str)
    df = pd.read_csv(io.StringIO(decode_str.decode('big5')))
    buffer = io.StringIO()
    df.info(buf=buffer)
    df_info_output = buffer.getvalue()

    def extract_info():
        t = df_info_output.split('\n')
        row = 0
        start = None
        end = None
        column_name = ['Feature', 'Count', 'Null', 'Type']
        while row < len(t):
            if re.match("^Data columns", t[row]):
                start = row + 1
            elif re.match('^dtypes', t[row]):
                end = row
            row += 1
        extract_feature = []
        for f in t[start:end]:
            tmp_f = f.split()
            tmp_f[:-3] = [''.join(tmp_f[:-3])]
            f_dict = dict(zip(column_name, tmp_f))
            extract_feature.append(f_dict)
        extract_df = pd.DataFrame(extract_feature)
        return extract_df
    df_info = extract_info()
    os.makedirs('database', exist_ok=True)
    df.to_pickle(os.path.join('database', filename.split('.')[0]))
    return upload_msg, [{"name": col, "id": col} for col in df_info.columns], df_info.to_dict('records')


@main_app.callback(
    Output('file_select', 'options'),
    [Input('upload_status', 'children')]
)
def update_files(upload_status):
    files = os.listdir('database')
    return [{'label': k, 'value': k} for k in files]


feature_data_df = None
feature_list = None


@main_app.callback(
    Output('target_select', 'options'),
    [Input('file_info_table', 'data')]
)
def update_features(info_data):
    return [{'label': n, 'value': n} for n in feature_list]


@main_app.callback(
    [Output('file_info_table', 'columns'), Output('file_info_table', 'data'), Output('file_info_table_expend', 'open')],
    [Input('file_select_button', 'n_clicks')],
    [State('file_select', 'value')]
)
def get_file_info(n_clicks, file_name):
    if n_clicks:
        df = pd.read_pickle(os.path.join('database', file_name))
        df_stat = df.describe().T
        df_stat.insert(0, 'Feature', df_stat.index)
        df_stat = df_stat.round(2)
        global feature_data_df
        global feature_list
        feature_data_df = df.sort_index()
        feature_list = list(df_stat.index)
    return [{"name": col, "id": col} for col in df_stat.columns], df_stat.to_dict('records'), True


@main_app.callback(
    [Output('feature_graph', 'figure'), Output('feature_graph_expend', 'open')],
    [Input('file_select_button', 'n_clicks'), Input('file_info_table', 'selected_rows')],
    [State('file_select', 'value')]
)
def plot_feature(n_clicks, selected_row_idx, file_name):
    if n_clicks > 0:
        selected_features = [feature_list[i] for i in selected_row_idx]
        data_list = []
        for each_feature in selected_features:
            data_list.append(
                dict(
                    x=list(feature_data_df.index), y=feature_data_df[each_feature].values, name=each_feature
                )
            )
        feature_figure = dict(
            data=data_list,
            layout=dict(
                title=file_name, showlegend=True
            )
        )
        return feature_figure, True


@main_app.callback(
    [Output('feature_correlation_table', 'columns'), Output('feature_correlation_table', 'data'),
     Output('feature_correlation_table_expend', 'open')],
    [Input('target_select_button', 'n_clicks')],
    [State('target_select', 'value')]
)
def calculate_correlation(n_clicks, targets):
    if n_clicks > 0:
        correlation_dicts = []
        for target in targets:
            target_dict = {'Target': target}
            for feature in np.setdiff1d(feature_list, targets):
                corr_coefficient = pearsonr(feature_data_df[target], feature_data_df[feature].values)[0]
                target_dict[feature] = round(corr_coefficient, 2)
            correlation_dicts.append(target_dict)
    return [{'name': k, 'id': k} for k in correlation_dicts[0].keys()], correlation_dicts, True

if __name__ == '__main__':
    main_app.run_server(host='0.0.0.0', port=1111)