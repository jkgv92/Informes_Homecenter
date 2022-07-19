#!/usr/bin/env python
# coding: utf-8

# # Proyecto Monitoreo Homecenter - Informe Mensual
# 
# Junio 2022

# In[ ]:





# ¡Hola!, te presentamos el informe correspondiente a tus consumos del mes de junio de 2022. A continuación vas a encontrar un resumen de los consumos realizados de forma acumulada. Para esto encontrarás una serie de gráficas diseñadas para dar un vistazo a los consumos por sede. Finalmente, encontrarás un informe detallado para cada sede.
# 
# 

# ## Definitions
# 

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from Ubidots_library_acruz import Ubidots

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from datetime import datetime

import os
from dotenv import dotenv_values
config = dotenv_values(".env")

import requests
import json

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter
import seaborn as sns


from sys import exit

import os

import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
import plotly.offline as py
from plotly.subplots import make_subplots

pio.renderers.default = "notebook"

import config as cfg
import Report_library as Report


# ## Configuration

# In[2]:


USE_CELSIA_PALETTE = cfg.USE_CELSIA_PALETTE
lst_primary_palette = cfg.lst_primary_pallete
lst_secondary_palette = cfg.lst_secondary_palette
primary_palette = sns.color_palette(lst_primary_palette)
secondary_palette = sns.color_palette(lst_secondary_palette)
lst_big_colors = lst_primary_palette + lst_secondary_palette
big_palette = sns.color_palette(lst_big_colors)
CELSIA_FONT = cfg.CELSIA_FONT
SCATTERGEO_MAX_MARKER_AREA = cfg.SCATTERGEO_MAX_MARKER_AREA

ALLOWED_DATE_OVERLAP = cfg.ALLOWED_DATE_OVERLAP
device_group_label = cfg.DEVICE_GROUP_LABEL
wide_figure_size = cfg.WIDE_FIGURE_SIZE

LST_NIGHTTIME_HOURS = cfg.lst_nighttime_hours
confidence_interval = cfg.CONFIDENCE_INTERVAL
dct_dow = cfg.dct_dow

ALL_VARIABLE_LABELS = cfg.ALL_VARIABLE_LABELS
BLACKLISTED_VARIABLE_LABELS = cfg.BLACKLISTED_VARIABLE_LABELS
ACTIVE_ENERGY_LABELS = cfg.ACTIVE_ENERGY_LABELS
ACTIVE_POWER_LABELS = cfg.ACTIVE_POWER_LABELS
ACTIVE_ENERGY_LUMP_LABEL = cfg.ACTIVE_ENERGY_LUMP_LABEL
ACTIVE_POWER_LUMP_LABEL = cfg.ACTIVE_POWER_LUMP_LABEL
TOTAL_ACTIVE_ENERGY_LABEL = cfg.TOTAL_ACTIVE_ENERGY_LABEL
TOTAL_ACTIVE_POWER_LABEL = cfg.TOTAL_ACTIVE_POWER_LABEL
TOTAL_REACTIVE_ENERGY_LABEL = cfg.TOTAL_REACTIVE_ENERGY_LABEL

SUB_STR = cfg.SUB_STR

DATA_FREQUENCY = cfg.DATA_FREQUENCY
DATE_INTERVALS_TO_DISCARD = cfg.DATE_INTERVALS_TO_DISCARD

# If this isn't done before plotting with Seaborn
# the first plot won't be the right size
plt.figure()
plt.show()
sns.set(rc={'figure.figsize': wide_figure_size})
sns.set(font=CELSIA_FONT)
plt.close()

# Create the colormap for all heatmaps
colorlist=[big_palette[1], big_palette[0], big_palette[4]]
newcmp = LinearSegmentedColormap.from_list('my_cmap', colors=colorlist, N=256)


## POI level configuration
use_pickled_data = False
PICKLED_DATA_FILENAME = 'parsed_response_acumulado.pkl'

# Ubidots data parameters
# If not needed set = None
LST_DEVICE_ID_TO_REQUEST = None

# Specify the date interval to fetch data from
# the format must be: 'YYYY-MM-DD'

# The monthly report doesn't yet include
# any baseline data
BASELINE_DATE_INTERVAL = {
    'start': '2022-04-01',
    'end': '2022-04-01'
}

STUDY_DATE_INTERVAL = {
    'start': '2022-06-01',
    'end': '2022-06-30'
}


# ## Data loading

# In[3]:


# check for intersecting study and baseline intervals before making any requests
Report.check_intervals(BASELINE_DATE_INTERVAL, STUDY_DATE_INTERVAL, ALLOWED_DATE_OVERLAP)

dct_available_devices = Report.get_available_devices(device_group_label)
if (LST_DEVICE_ID_TO_REQUEST==None):
    LST_DEVICE_ID_TO_REQUEST = list(dct_available_devices.keys())

df_vars = Report.get_available_variables(LST_DEVICE_ID_TO_REQUEST)
df_vars['device_label'] = df_vars['device_id'].map(dct_available_devices)
is_blacklisted = df_vars['variable_label'].isin(BLACKLISTED_VARIABLE_LABELS)
df_vars_to_request = df_vars[~is_blacklisted].reset_index(drop=True)
df_vars_req_wide = df_vars_to_request.pivot(index='variable_label', columns='device_label', values='variable_id')

DCT_VAR_ID_TO_LABEL = dict(zip(list(df_vars['variable_id']), list(df_vars['variable_label'])))
SET_LABELS_TO_REQUEST = set(df_vars_to_request['variable_label'].sort_values())
VAR_IDS_TO_REQUEST = list(df_vars_to_request['variable_id'])




df = None
if (use_pickled_data is True):
    df = pd.read_pickle(PICKLED_DATA_FILENAME)
else:
    # A user might select baseline and study intervals
    # which are sufficiently far apart that fetching
    # the data in between is very inefficient.
    # So it's best to make a request per interval.
    response_bl = Report.make_request(
        VAR_IDS_TO_REQUEST, 
        BASELINE_DATE_INTERVAL, 
    )

    response_st = Report.make_request(
        VAR_IDS_TO_REQUEST, 
        STUDY_DATE_INTERVAL, 
    )
    
    df = Report.parse_response(response_bl, response_st, DCT_VAR_ID_TO_LABEL)
    pd.to_pickle(df, PICKLED_DATA_FILENAME)

# TODO: add pickling to this
df_coords = Report.get_coords_dataframe(LST_DEVICE_ID_TO_REQUEST)

Report.show_response_contents(df)


# ## Preprocessing

# In[4]:


# Discard first entry before cleaning as it might not belong to interval
# sort by datetime to guarantee chronological order when dropping rows
df.sort_values(by=['datetime', 'device', 'variable'], inplace=True)
df = Report.subset_drop_first_n_rows(df, subset_a='device', subset_b='variable', n_rows=1)

if len(DATE_INTERVALS_TO_DISCARD)>0:
    df = Report.subset_discard_date_intervals(df, DATE_INTERVALS_TO_DISCARD)

# TODO: fix this line. As a temp measure we are disallowing negative active
# power and active energy. This is not actually valid for some systems
# which may contain solar panels, for instance.
is_ea_or_pa = (df['variable'].isin(ACTIVE_ENERGY_LABELS + ACTIVE_POWER_LABELS))
is_negative = (df['value'] < 0)
df.loc[(is_ea_or_pa & is_negative), 'value'] = np.nan

df = Report.double_subset_rolling_clean(
    df, 
    subset_a='device', 
    subset_b='variable', 
    clean_on='value'
)

# plotting requires day of week and hour of day labels
df = Report.apply_datetime_transformations(df)

df_bl, df_st = Report.split_into_baseline_and_study(df, BASELINE_DATE_INTERVAL, STUDY_DATE_INTERVAL)


# ## Resultados

# In[5]:


# It makes sense to group by month because
# the reports seem to be monthly (for now).
# But this will break if the study period spans
# less than one month or if the data is incomplete
# (it won't interpolate or extrapolate consumption).
df_st_total_ea = df_st[df_st["variable"]==TOTAL_ACTIVE_ENERGY_LABEL]
agg_func = {'value':np.sum, 'outlier':'sum'}
lst_cols = ['month', 'device', 'device_name']
df_st_total_ea_dev_monthly = df_st_total_ea.groupby(by=lst_cols).resample('1M').agg(agg_func).reset_index().set_index('month')

df_total_cons_st = df_st_total_ea_dev_monthly[['value','device_name']].sort_values(by='value', ascending=False)
df_total_cons_st['consumption_mwh'] = round(df_total_cons_st['value'].astype(float)/1000, 2)
df_total_cons_st.drop(['value'], axis=1, inplace=True)
df_total_cons_st.reset_index(drop=True, inplace=True)

fig = px.bar(
    df_total_cons_st,
    x="device_name",
    y="consumption_mwh",
    color_discrete_sequence=[lst_big_colors[0]],
    labels={'device_name':'Sede', 'consumption_mwh':'Consumo total [MWh]'},
    title="Consumo total de energía activa por sede y circuito [MWh]",
)

fig.update_layout(
    font_family=CELSIA_FONT,
    font_size=12,
    font_color=lst_big_colors[1],
    title_x=0.5,
    width=1250,
    height=550,
    # xaxis=dict(showgrid=False),
    # yaxis=dict(showgrid=False),
    template='plotly_white'
)

fig.update_xaxes(
    tickangle=-45
)

fig.show()


# En la figura anterior se puede observar un ranking de consumo por cada una de las sedes monitoreadas. El consumo es en MWh.

# In[6]:


fig = px.pie(
    df_total_cons_st, 
    values="consumption_mwh", 
    names='device_name', 
    hover_data=['consumption_mwh'],
    labels={'device_name':'Sede', 'consumption_mwh':'Consumo total [MWh]'},
    title="Consumo total de energía activa por sede [MWh]",
    # color_discrete_sequence=lst_big_colors+lst_big_colors,
)

fig.update_layout(
    font_family=CELSIA_FONT,
    font_size=12,
    font_color=lst_big_colors[1],
    title_x=0.5,
    width=750,
    height=550,
)

fig.update_traces(
    textposition='inside', 
    textinfo='percent', 
    insidetextorientation='radial'
)

fig.update(
    layout_showlegend=True
)

fig.show()


# In[7]:


df_st_ea = df_st[df_st['variable'].isin(ACTIVE_ENERGY_LABELS)]

df_st_ea_variable = df_st_ea.groupby(by='variable').agg({'value':np.sum}).copy()
df_st_ea_variable.sort_values(by='value', ascending=False, inplace=True)
df_st_ea_variable['consumption_mwh'] = round(df_st_ea_variable['value'].astype(float)/1000, 2)
df_st_ea_variable.drop(['value'], axis=1, inplace=True)
df_st_ea_variable.reset_index(inplace=True)


fig = px.pie(
    df_st_ea_variable, 
    values="consumption_mwh", 
    names='variable', 
    hover_data=['consumption_mwh'],
    labels={'variable':'Circuito', 'consumption_mwh':'Consumo total [MWh]'},
    title="Consumo total de energía activa por circuito [MWh]",
    # color_discrete_sequence=lst_big_colors+lst_big_colors,
)

fig.update_layout(
    font_family=CELSIA_FONT,
    font_size=12,
    font_color=lst_big_colors[1],
    title_x=0.5,
    width=750,
    height=550,
)

fig.update_traces(
    textposition='inside', 
    textinfo='percent', 
    insidetextorientation='radial'
)

fig.update(
    layout_showlegend=True
)

fig.show()


# En las dos figuras anteriores muestran el consumo total de energía activa por sede y por circuito respectivamente.

# In[8]:


# 3-Layer Sankey Diagram
# Define the data that goes into Sankey Diagram
df_st_ea = df_st[df_st['variable'].isin(ACTIVE_ENERGY_LABELS)]
layer_1_source_label = 'Consumo total [kWh]'

# L3: Total consumption by device and variable
df_sankey_layer_3 = df_st_ea.groupby(by=['variable', 'device_name']).agg({'value':np.sum}).reset_index()
df_sankey_layer_3.rename(columns={'variable':'target_label', 'device_name':'source_label'}, inplace=True)

# L2: Total consumption by device
df_sankey_layer_2 =  df_st_ea.groupby(by='device_name').agg({'value':np.sum}).reset_index()
df_sankey_layer_2.rename(columns={'consumption_mwh':'value', 'device_name':'target_label'}, inplace=True)
df_sankey_layer_2['source_label'] = layer_1_source_label
df_sankey_layer_2['value'] = df_sankey_layer_2['value'] # * 1000

# L1: Total consumption
total_cons = df_st_ea.groupby(by='device_name').agg({'value':np.sum}).reset_index()

# Pair nodes with node labels
lst_labels_layer_1 = [layer_1_source_label]
lst_labels_layer_2 = list(df_sankey_layer_2.sort_values(by='value', ascending=False)['target_label'].unique())
lst_labels_layer_3 = list(df_sankey_layer_3.sort_values(by='value', ascending=False)['target_label'].unique())

lst_labels = lst_labels_layer_1 + lst_labels_layer_2 + lst_labels_layer_3
dct_node_to_label = dict(zip(range(len(lst_labels)), lst_labels))


# Assemble Sankey Diagram connectivity matrix
# TODO: replace with recursive function:
n_nodes_layer_1 = len(lst_labels_layer_1)
n_nodes_layer_2 = len(lst_labels_layer_2) # number of devices
n_nodes_layer_3 = len(lst_labels_layer_3) # number of distinct variables (sans totalizers)

lst_combinations = []
for i in range(n_nodes_layer_1):
    for j in range(n_nodes_layer_2):
        j+=n_nodes_layer_1
        for k in range(n_nodes_layer_3):
            k+= (n_nodes_layer_1 + n_nodes_layer_2)
            lst_combinations.append([i,j,k])

np_col_stack = np.column_stack(lst_combinations)

df_nodes = pd.DataFrame(
    data=[
        list(np_col_stack[0]) + list(np_col_stack[1]),
        list(np_col_stack[1]) + list(np_col_stack[2])
    ]
).transpose()

df_nodes.rename(columns={0:'source', 1:'target'}, inplace=True)
df_nodes['source_label'] = df_nodes['source'].map(dct_node_to_label)
df_nodes['target_label'] = df_nodes['target'].map(dct_node_to_label)
df_nodes.drop_duplicates(inplace=True)
df_nodes.reset_index(drop=True, inplace=True)


# Combine into final dataframe
df_sankey = pd.merge(df_nodes, df_sankey_layer_2, on=['source_label', 'target_label'], how='outer')
df_sankey = pd.merge(df_sankey, df_sankey_layer_3, on=['source_label', 'target_label'], how='outer')
df_sankey.loc[df_sankey['value_x'].isna(), 'value_x'] = 0
df_sankey.loc[df_sankey['value_y'].isna(), 'value_y'] = 0
df_sankey['value'] = df_sankey['value_x'] + df_sankey['value_y']
df_sankey.drop(columns=['value_x', 'value_y'], inplace=True)


# Draw the Sankey Diagram
fig = go.Figure(
    data=[
        go.Sankey(
            node = {
                'pad': 15,
                'thickness': 15,
                'line': {'color': 'black', 'width': 0.5},
                'label': lst_labels_layer_1 + list(df_sankey['target_label']) # maybe "lst_labels_layer_1 + ..." fixes this?
            },
            link = {
                'source': list(df_sankey['source']),
                'target': list(df_sankey['target']),
                'value': list(df_sankey['value']),
            }
        )
    ]
)

fig.update_layout(
    font_family=CELSIA_FONT,
    font_size=12,
    font_color=lst_big_colors[1],
    title_text="Consumo total de energía activa por sede y circuito [kWh]",
    title_x=0.5,
    width=1250,
    height=550
)

fig.show()


# En la figura anterior se puede observar la contribución de cada sede al consumo de cada tipo de circuito, en kWh.

# In[9]:


lataxis_range, lonaxis_range = Report.center_colombia()
df_map = pd.merge(df_coords, df_total_cons_st)
marker_scaler = SCATTERGEO_MAX_MARKER_AREA / df_map["consumption_mwh"].max()

fig = go.Figure()

fig.add_trace(
    go.Scattergeo(
        lon = df_map["longitude"],
        lat = df_map["latitude"],
        text = df_map["device_name"],
        fillcolor = lst_big_colors[1],
        # hoverinfo=['device_name', 'consumption_mwh'],
        # hoverlabel={'device_name':'Sede', 'consumption_mwh':'Consumo total [MWh]'},
        marker = dict(
            size = df_map["consumption_mwh"] * marker_scaler,
            line_width=0.5,
            sizemode = 'area',
            color=lst_big_colors[1],
        )
    )
)

fig.update_layout(
    font_family=CELSIA_FONT,
    font_size=12,
    font_color=lst_big_colors[1],
    title_text="Consumo total de energía activa por sede [MWh]",
    # labels={'device_name':'Sede', 'consumption_mwh':'Consumo total [MWh]'},
    title_x=0.5,
    width=750,
    height=550,

    margin={"r":50,"t":50,"l":50,"b":50},
    geo = go.layout.Geo(
        resolution = 50,
        scope = 'south america',
        showframe = True,
        showcoastlines = True,
        landcolor = lst_big_colors[0],
        countrycolor = "white" ,
        coastlinecolor = "white",
        projection_type = 'mercator',
        # lataxis_range= [ -5.0,  13.0],
        # lonaxis_range= [-65.0, -85.0],
        lataxis_range = lataxis_range,
        lonaxis_range = lonaxis_range,
        projection_scale=20
    )
)

fig.show()


# Así mismo, en la figura anterior, se puede observar la distribución de consumo en el espacio, siendo cada punto una sede monitoreada, y su tamaño equivalente al consumo realizado.

# Te invitamos a revisar los informes detallados para cada sede en las siguientes páginas.
