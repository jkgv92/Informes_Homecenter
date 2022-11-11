#!/usr/bin/env python
# coding: utf-8

# # Sección General

# ## Preambulo oculto

# In[1]:


# %load_ext autoreload
# %autoreload 2
import warnings
warnings.filterwarnings("ignore")

# this cell enables project path relative imports
import sys
from pathlib import Path
path_base_r_string = r'D:\OneDrive - CELSIA S.A E.S.P'
path_base = Path(path_base_r_string)
project_path = path_base/'Proyectos'/'Eficiencia_Energetica'/'Homecenter'/'Informes_Homecenter'
sys.path.append(str(project_path))


# In[2]:


# import all your modules here
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "notebook"
pio.templates.default = "plotly_white"

from library_ubidots_v2 import Ubidots as ubi
from library_report_v2 import Configuration as repcfg
from library_report_v2 import Cleaning as cln
from library_report_v2 import Graphing as grp
from library_report_v2 import Processing as pro


# In[3]:


df_hvac = pd.read_excel(project_path/'data'/'homecenter_store_cooling_data.xlsx')

df_office = pd.read_pickle(project_path/'data'/'office_level_data.pkl')

var_to_type = {
    'pa-area-de-bots':'otros',
    'pa-area-de-corte':'otros',
    'pa-car-center':'otros',
    'pa-concesiones':'otros',
    'pa-equipos-de-climatizacion':'aa',
    'pa-equipos-verticales':'otros',
    'pa-iluminacion':'ilu',
    'pa-iluminacion-parqueaderos':'ilu',
    'pa-iluminacion-patio-constructor':'ilu',
    'pa-iluminacion-patio-contenedores':'ilu',
    'pa-iluminacion-principal':'ilu',
    'pa-oficinas-y-servicios':'otros',
    'pa-patio-constructor':'otros',
    'pa-tablero-sorter':'otros',
    'pa-talleres':'otros',

    'ea-area-de-bots':'otros',
    'ea-area-de-corte':'otros',
    'ea-car-center':'otros',
    'ea-concesion':'otros',
    'ea-concesiones':'otros',
    'ea-equipos-de-climatizacion':'aa',
    'ea-equipos-verticales':'otros',
    'ea-iluminacion':'ilu',
    'ea-iluminacion-parqueaderos':'ilu',
    'ea-iluminacion-patio-constructor':'ilu',
    'ea-iluminacion-patio-contenedores':'ilu',
    'ea-iluminacion-principal':'ilu',
    'ea-oficinas-y-servicios':'otros',
    'ea-patio-constructor':'otros',
    'ea-tablero-sorter':'otros',
    'ea-talleres':'otros'
}


df_office['type'] = df_office['variable'].map(var_to_type)

# remove aggregate of offices
df_office = df_office.query("device_name != 'HC - Corporativo'")

name_to_pretty = {
    'HC - Barranquilla':'Barranquilla',
    'HC - Bello':'Bello',
    'HC - Bucaramanga':'Bucaramanga',
    'HC - Cali norte':'Cali Norte',
    'HC - Cali sur':'Cali Sur',
    'HC - Calle 80':'Calle 80',
    'HC - Cedritos':'Cedritos',
    'HC - Funza':'Funza',
    'HC - La Popa':'La Popa',
    'HC - Palmira':'Palmira',
    'HC - San Fernando':'San Fernando',
    'HC - San Juan':'San Juan',
    'HC - Tintal':'Tintal',
}

df_office['pretty_name'] = df_office['device_name'].map(name_to_pretty)

# add datetime attributes
df_office = pro.datetime_attributes(df_office)


# get power per circuit
power_vars = set(df_office.loc[df_office.variable.str.startswith('pa-'), 'variable'])
power_vars.remove('pa-iluminacion')
df_office_pa = df_office[df_office.variable.isin(power_vars)].copy()
df_office_pa = cln.remove_outliers_by_zscore(df_office_pa, zscore=5)

# get consumption per circuit
energy_vars = set(df_office.loc[df_office.variable.str.startswith('ea-'), 'variable'])
energy_vars.remove('ea-total')
df_office_cons = df_office[df_office.variable.isin(energy_vars)].copy()

# get overall office consumption
df_office_total_cons = df_office.query("variable == 'ea-total'")
df_office_total_cons['type'] = 'total'


# ## 1. Introducción

# La norma ISO 50001 tiene como objetivo trazar una metodología que permita a las empresas la mejora de la eficiencia energética en sus procesos. Aunque en este informe se presentará el análisis relacionado al consumo actual de la energía basándose en la metodología de dicha norma, es importante el nivel de relacionamiento y compromiso de los diferentes involucrados dentro de Sodimac, ya que esta norma plantea una metodología de mejora y gestión continua para garantizar una debida aplicación, que consiste en planificar, realizar, comprobar y actuar, tal como se presenta en la figura a continuación: 

# | ![Diagrama_proceso_ISO_50001.jpg](http://www.comunidadism.es/wp-content/uploads/2018/10/imagen-1-600x533.png) |
# |:--:|
# | <b>Figura 1. Diagrama del proceso ISO 50001. Fuente: AENOR - Norma ISO 50001</b>|

# Donde la planificación hace referencia a la gestión y seguimiento energético, a través de indicadores de eficiencia energética, objetivos y planes de acción, de acuerdo con la política definida por la empresa. La realización busca la implementación de las medidas propuestas en la fase de planificación, la verificación valida el cumplimiento de indicadores, la política y los objetivos trazados, para finalmente en la acción desarrollar las actividades necesarias para mejorar la eficiencia energética y la herramienta de gestión.

# ## 2. Planeación

# Bajo la metodología de la ISO 50001, y como parte del proceso de planeación se seleccionaron algunas de las sedes Sodimac para realizar un balance energético asociado a la implementación de un sistema de gestión de energía. Celsia como asociado estratégico de este plan, apoyó con la instalación de un sistema de monitoreo integral en relación con los diferentes consumos eléctricos finales.  
# 
# Como objetivo principal de este proceso se busca la identificación de oportunidades de mejora en eficiencia energética, recordando que este proceso bajo la metodología ISO 50001 es un proceso cíclico de mejora continua.

# ## 3. Hacer (Apoyo - Operación)

# Desde el mes de junio se iniciaron a registrar mediciones de los consumos energéticos a partir de sistemas de monitoreo instalados en las diferentes sedes en donde Celsia ofrece a Sodimac la oportunidad de ver los consumos eléctricos a partir de un acceso web en donde podrá ver discriminadamente los usos finales de energía según lo definido en el proceso de planeación.
# 
# A partir de lo anterior se definirán las oportunidades de mejora en eficiencia como parte del modelo de gestión de energía para cada una de las sedes analizadas. 

# ## 4. Verificación

# ### 4.1 Indicador de Eficiencia Energética
# 
# A nivel global para normalizar los consumos de energía, uno de los indicadores más utilizados comparar el desempeño energético entre edificaciones es el Energy Use Intensity (EUI), que no es más que una relación entre el consumo eléctrico anual y el área construida.  
# 
#  
# 
# Normativas e Instituciones como Commercial Buildings Energy Consumption Survey (CBECS), Energy Star y American Society of Heating, Refrigerating and Air-Conditioning Engineers (ASHRAE), tienen definidas algunos EUI base en función del uso y del clima asociado a la localización del inmueble, las zonas climáticas (1A, 2A & 3A) son definidas bajo la metodología definida por ASHRAE.  
# 
#  
# 
# Para el caso Sodimac se presentan a continuación las tiendas analizadas agrupadas por clima y comparadas con el EUI base según normativa internacional:  

# | ![EUI_general](figures/EUI_general.png) |
# |:--:|
# | <b>Figura 2. Consumos por unidad de área y clima</b>|

# De este análisis preliminar se puede identificar que sedes como la Popa, Cali Norte, Bello y Bucaramanga son sedes que si bien exceden la línea base (EUI Base) definido por normativa, son sedes que también exceden el consumo de energía en más de un 10% de otras sedes en condiciones climatológicas similares. EUI Base 1A (88KWh/m2) , 2A (80KWh/m2) & 3A (60KWh/m2) 
# 
#  
# 
# Sin embargo, del anterior análisis es importante mencionar que no en todas las sedes se estaba monitoreando el 100% de las cargas, por lo que habrá algunas variaciones en el análisis frente al EUI estimado para cada sede en aquellas donde la cobertura de medición fue menor a 100%. Lo anterior hace que sedes como Barranquilla, San Fernando y Palmira puedan ser susceptibles de mejoras frente a este indicador.  

# In[4]:


df_mean_yearly_cons_kwh = (
    (
        pd.concat([df_office_cons, df_office_total_cons])
        # df_office_total_cons
        .groupby(by=['device','device_name','pretty_name', 'type'])['value']
        .resample('1D')
        .sum()
        .reset_index()
    )
    .groupby(by=['device','device_name','pretty_name', 'type'])['value']
    .mean()
    .reset_index()
    .sort_values(by='value', ascending=False)
    .reset_index(drop=True)
)

df_mean_yearly_cons_kwh['value'] = (
    df_mean_yearly_cons_kwh['value']
    .astype(float)
    * 365.25
)

df_kpi = pd.merge(
    df_mean_yearly_cons_kwh,
    df_hvac
).sort_values(by='value', ascending=False)


df_kpi['total_kwh_per_m2'] = (df_kpi['value'] / df_kpi['sv_area']).round(2)
df_kpi = df_kpi.sort_values(by='total_kwh_per_m2', ascending=False)

# to get the missing devices we must request all the devices from the account
df_devices = ubi.get_available_devices_v2(label=None, level='account', page_size=1000)
df_devices = df_devices[df_devices['device_name'].str.startswith('HC - ')]
df_devices = df_devices.query("device_name != 'HC - Corporativo'")
DEVICE_IDS_TO_REQUEST = list(df_devices['device_id'])
df_coords = ubi.get_gps_for_multiple_device_id(DEVICE_IDS_TO_REQUEST)

df_map = pd.merge(df_coords, df_kpi.query("type == 'total'"))

lataxis_range, lonaxis_range = grp.center_colombia()

fig = go.Figure()

marker_scaler = repcfg.SCATTERGEO_MAX_MARKER_AREA / df_map["total_kwh_per_m2"].max()

fig.add_trace(
    go.Scattergeo(
        lon = df_map["longitude"],
        lat = df_map["latitude"],
        text = df_map["device_name"],
        fillcolor = repcfg.FULL_PALETTE[0],
        # hoverinfo=['device_name', 'consumption_mwh'],
        # hoverlabel={'device_name':'Sede', 'consumption_mwh':'Consumo total [MWh]'},
        marker = dict(
            size = df_map["total_kwh_per_m2"] * marker_scaler,
            line_width=0.5,
            sizemode = 'area',
            color=repcfg.FULL_PALETTE[1],
        )
    )
)

fig.update_layout(
    title_text="Consumo anual de energía activa por unidad de área [kWh/año/m^2]",
    font_family=repcfg.CELSIA_FONT,
    font_size=repcfg.PLOTLY_TITLE_FONT_SIZE,
    font_color=repcfg.FULL_PALETTE[1],
    title_x=repcfg.PLOTLY_TITLE_X,
    width=repcfg.JBOOK_PLOTLY_WIDTH,
    height=repcfg.JBOOK_PLOTLY_HEIGHT,

    margin={"r":50,"t":50,"l":50,"b":50},
    geo = go.layout.Geo(
        resolution = 50,
        scope = 'south america',
        showframe = True,
        showcoastlines = True,
        landcolor = repcfg.FULL_PALETTE[0],
        countrycolor = "white" ,
        coastlinecolor = "white",
        projection_type = 'mercator',
        lataxis_range = lataxis_range,
        lonaxis_range = lonaxis_range,
        projection_scale=20
    )
)

fig.show()


# En la figura anterior se puede observar la distribución del indicador en el espacio, siendo cada punto una sede monitoreada, y su tamaño equivalente al indicador.

# ### 4.2 Consumo General

# A continuación está el consumo anual esperado por sede.

# In[5]:


df_daily_cons_by_device = (
    df_office_total_cons
    .groupby(['pretty_name'])['value']
    .resample('1D')
    .sum()
    .reset_index()
)

df_mean_daily_cons_by_device = (
    df_daily_cons_by_device
    .groupby(['pretty_name'])['value']
    .mean()
    .reset_index()
    .sort_values(by='value', ascending=False)
)

df_mean_yearly_cons_by_device = df_mean_daily_cons_by_device

df_mean_yearly_cons_by_device['value'] = (
    df_mean_daily_cons_by_device['value']
    .astype(float)
    / 1000000 * 365.25
).round(2)


# In[6]:


fig = px.bar(
    df_mean_yearly_cons_by_device,
    x="pretty_name",
    y="value",
    labels={'pretty_name':'Sede', 'value':'Consumo anual [GWh]'},
    title="Consumo anual esperado por sede [GWh]",
)

fig.update_layout(
    font_family=repcfg.CELSIA_FONT,
    font_size=repcfg.PLOTLY_TITLE_FONT_SIZE,
    font_color=repcfg.FULL_PALETTE[1],
    title_x=repcfg.PLOTLY_TITLE_X,
    width=repcfg.JBOOK_PLOTLY_WIDTH,
    height=repcfg.JBOOK_PLOTLY_HEIGHT
)

fig.update_traces(marker_color=grp.hex_to_rgb(repcfg.FULL_PALETTE[0]))

fig.update_xaxes(
    tickangle=-45
)

fig.show()


# ### 4.2 Consumos Representativos

# Los consumos más representativos para las sedes Sodimac se pueden clasificar de la siguiente manera:  
# 
# - Climas Cálidos - Templados (1A & 2A): Aire Acondicionado e Iluminación. 
# 
# - Climas Fríos (3A): Iluminación y Concesiones.
# 
# Las siguientes figuras dan una idea de los consumos más significativos y tu relación con las sedes.

# In[7]:


df_daily_cons_by_type = (
    df_office_cons
    .groupby(['type'])['value']
    .resample('1D')
    .sum()
    .reset_index()
)

df_mean_daily_cons_by_type = (
    df_daily_cons_by_type
    .groupby(['type'])['value']
    .mean()
    .reset_index()
)

df_mean_yearly_cons_by_type = df_mean_daily_cons_by_type

df_mean_yearly_cons_by_type['value'] = (
    df_mean_yearly_cons_by_type['value']
    .astype(float)
    / 1000000 * 365.25
).round(2)

fig = px.pie(
    df_mean_daily_cons_by_type, 
    values="value", 
    names='type', 
    hover_data=['value'],
    labels={'type':'Tipo', 'value':'Consumo anual [GWh]'},
    title=f"Consumo anual esperado por tipo de carga [GWh]",
    color_discrete_sequence=repcfg.FULL_PALETTE,
)

fig.update_layout(
    font_family=repcfg.CELSIA_FONT,
    font_size=repcfg.PLOTLY_TITLE_FONT_SIZE,
    font_color=repcfg.FULL_PALETTE[1],
    title_x=repcfg.PLOTLY_TITLE_X,
    width=repcfg.JBOOK_PLOTLY_WIDTH,
    height=repcfg.JBOOK_PLOTLY_HEIGHT
)

fig.update_traces(
    # textposition='inside', 
    textinfo='percent', 
    # insidetextorientation='radial'
)

fig.update(
    layout_showlegend=True
)

fig.show()


# In[8]:


df_mean_yearly_cons_kwh = (
    (
        pd.concat([df_office_cons, df_office_total_cons])
        # df_office_total_cons
        .groupby(by=['device','device_name','pretty_name', 'type'])['value']
        .resample('1D')
        .sum()
        .reset_index()
    )
    .groupby(by=['device','device_name','pretty_name', 'type'])['value']
    .mean()
    .reset_index()
    .sort_values(by='value', ascending=False)
    .reset_index(drop=True)
)

df_mean_yearly_cons_kwh['value'] = (
    df_mean_yearly_cons_kwh['value']
    .astype(float)
    * 365.25
)

df_kpi = pd.merge(
    df_mean_yearly_cons_kwh,
    df_hvac
).sort_values(by='value', ascending=False)


df_kpi['total_kwh_per_m2'] = (df_kpi['value'] / df_kpi['sv_area']).round(2)
df_kpi = df_kpi.sort_values(by='total_kwh_per_m2', ascending=False)


# In[9]:


fig = px.bar(
    df_kpi,
    x="pretty_name",
    y="total_kwh_per_m2",
    color='type',
    color_discrete_sequence=repcfg.FULL_PALETTE,
    barmode='group',
    labels={'pretty_name':'Sede', 'total_kwh_per_m2':'Consumo anual específico [kWh/año/m^2]'},
    title="Consumo anual de energía activa por unidad de área [kWh/año/m^2]",
)

fig.update_layout(
    font_family=repcfg.CELSIA_FONT,
    font_size=repcfg.PLOTLY_TITLE_FONT_SIZE,
    font_color=repcfg.FULL_PALETTE[1],
    title_x=repcfg.PLOTLY_TITLE_X,
    width=repcfg.JBOOK_PLOTLY_WIDTH,
    height=repcfg.JBOOK_PLOTLY_HEIGHT
)

fig.update_xaxes(
    tickangle=-45
)

fig.show()


# In[10]:


# 3-Layer Sankey Diagram
# Define the data that goes into Sankey Diagram
df_st_ea = df_office_cons
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
    title_text="Contribución al consumo total de energía activa",
    font_family=repcfg.CELSIA_FONT,
    font_size=repcfg.PLOTLY_TITLE_FONT_SIZE,
    font_color=repcfg.FULL_PALETTE[1],
    title_x=repcfg.PLOTLY_TITLE_X,
    width=repcfg.JBOOK_PLOTLY_WIDTH,
    height=repcfg.JBOOK_PLOTLY_HEIGHT
)

fig.show()


# Te invitamos a revisar los informes detallados para cada sede en la siguiente sección.
