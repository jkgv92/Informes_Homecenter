#!/usr/bin/env python
# coding: utf-8

# # Sección por sede

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

from library_report_v2 import Configuration as repcfg
from library_report_v2 import Cleaning as cln
from library_report_v2 import Graphing as grp
from library_report_v2 import Processing as pro


# ## Get data

# - Calle 80 tiene una variable con potencia predominantemente negativa
# - San Juan tiene un conflicto entre el api-label "san-juan---aires-acondicionados" y su nombre "San Juan - Iluminación Principal"
# - Tintal parece tener equipos mal etiquetados como "otros" (demasiado consumo de otros)

# ### Get office level data

# In[3]:


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


# ### Get circuit level data

# In[4]:


df_circuit = pd.read_pickle(project_path/'data'/"device_level_data.pkl")
df_tag_aa_ilu = pd.read_excel(project_path/"ubidots_device_map.xlsx", sheet_name='AA-ILU')

# for some reason 'value' dtypes aren't consistent...
df_circuit['value'] = pd.to_numeric(df_circuit['value'], errors='coerce')

# merge both tables
df_circuit = (
    pd.merge(
        df_circuit.reset_index(),
        df_tag_aa_ilu,
        how='inner'
    )
    .set_index('datetime')
)


dates_to_remove = {
    'cali-norte---torre-de-enfriamiento':[
        ['2022-01-01','2022-05-25']
    ],
    'cali-sur---aires-acondicionados':[
        ['2022-01-01','2022-05-25']
    ],
    'cedritos---iluminacion-patio-constructor':[
        ['2022-01-01','2022-05-20']
    ],
    
    'cali-norte---iluminacion-parqueaderos-2':[
        ['2022-01-01','2022-04-30']
    ],
    'cali-norte---iluminacion-principal-2':[
        ['2022-01-01','2022-05-25']
    ],
    'iluminacion---primer-piso':[ # cedritos
        ['2022-01-01','2022-12-30']
    ],
    'cedritos-iluminacion-principal':[
        ['2022-05-15','2022-12-30']
    ],
    'bello---iluminacion-patio-constructor':[
        ['2022-01-01','2022-05-25']
    ],
    'tintal---iluminacion-parqueaderos':[
        ['2022-01-01','2022-06-20']
    ],
    'baq---aire-acondicionado-piso-ventas':[
        ['2022-01-01','2022-05-20']
    ],
}

# We're interested in the active power but due to 
# the way the devices have been set up in Ubidots 
# (with mismatched labels), there isn't a 
# straightforward way to get the active power for 
# all devices (mixed labels). To get around this
# we request the active consumption, which
# (oddly enough) is named energia-activa, and we
# cumsum() it to get the cummulative active
# energy, which we then differentiate to get an
# estimate (moving average) of the active power.

# get (unified) cummulative energy from consumption
df_circuit_cons = df_circuit.query('variable == "energia-activa"').copy()
df_circuit_ea = cln.recover_energy_from_consumption(df_circuit_cons, new_varname='energia-activa-acumulada')
df_circuit_cons = None

# get power from energy
df_circuit_pa = cln.differentiate_single_variable(df_circuit_ea, 'potencia-activa-sintetica', remove_gap_data=True)
df_circuit_ea = None

# clean power data
df_circuit_pa = cln.remove_outliers_by_zscore(df_circuit_pa, zscore=5)
df_circuit_pa = cln.subset_discard_date_intervals(df_circuit_pa, dates_to_remove)
df_circuit_pa = pro.datetime_attributes(df_circuit_pa)
# df_circuit_pa = df_circuit_pa.query("value >= 0")

# San Juan has the labels swapped by mistake
is_san_juan_aa = (df_circuit_pa['device_name'] == 'San Juan - Aires Acondicionados')
df_circuit_pa.loc[is_san_juan_aa, 'device'] == 'san-juan---aires-acondicionados'

is_san_juan_ilu_ppal = (df_circuit_pa['device_name'] == 'San Juan - Iluminación Principal')
df_circuit_pa.loc[is_san_juan_ilu_ppal, 'device'] == 'san-juan---iluminacion-principal'


# ## Functions

# In[5]:


def typical_day(df_office_total_cons, device_name, office_name):
    df_by_office = df_office_total_cons.query("device_name == @device_name").copy()

    df_hourly_cons = (
        df_by_office
        .groupby(['device_name'])['value']
        .resample('1H')
        .sum()
        .reset_index()
    )

    df_hourly_cons['hour'] = df_hourly_cons['datetime'].dt.hour

    df_hourly_mean_power = (
        df_hourly_cons
        .groupby(['device_name','hour'])['value']
        .agg([pro.q_low,'mean',pro.q_high])
        .reset_index()
    ).round(2)

    if (len(df_hourly_mean_power) > 0):
        grp.plot_typical_day_by_hour(
            df_hourly_mean_power, 
            subset='device_name', 
            title=f"Día típico para la sede de {office_name}", 
            bl_label="Promedio", 
            bl_ci_label="Intervalo",
            legend=True, 
            include_ci=True, 
            fill_ci=True
        )

    return None


def hvac_typical_day_week(df_office_pa, device_name, office_name, fix_negatives=False):
    df_office_pa_device = df_office_pa.query("device_name == @device_name").copy()
    # df_office_pa_device = df_office_pa_device[df_office_pa_device.value > 0]

    df_day_by_hour = (
        df_office_pa_device
        .reset_index()
        .groupby(['device_name','type','variable','hour'])['value']
        .agg(['median','mean','std','min',pro.q_low,pro.q_high,'max','count'])
        .reset_index()
    )

    df_week_by_day = (
        df_office_pa_device
        .reset_index()
        .groupby(['device_name','type','variable','cont_dow'])['value']
        .agg(['median','mean','std','min',pro.q_low,pro.q_high,'max','count'])
        .reset_index()
    )

    if (fix_negatives is True):
        df_day_by_hour.loc[(df_day_by_hour.q_low < 0), 'q_low'] = 0
        df_week_by_day.loc[(df_week_by_day.q_low < 0), 'q_low'] = 0

    if (len(df_day_by_hour) > 0):
        grp.plot_typical_day_by_hour(
            df_day_by_hour.query("type == 'aa'"), 
            subset='variable', 
            title=f"Día típico para los equipos de climatización en la sede de {office_name}", 
            bl_label="Promedio de:", 
            bl_ci_label="Intervalo de:",
            legend=True, 
            include_ci=True, 
            fill_ci=True
        )
    
    if (len(df_week_by_day) > 0):
        grp.plot_typical_week_by_day(
            df_week_by_day.query("type == 'aa'"),
            subset='variable',  
            title=f"Semana típica para los equipos de climatización en la sede de {office_name}",
            bl_label="Promedio de:", 
            bl_ci_label="Intervalo de:",
            legend=True, 
            include_ci=True, 
            fill_ci=True
        )



def lighting_typical_day_week(df_office_pa, device_name, office_name, fix_negatives=False):
    df_office_pa_device = df_office_pa.query("device_name == @device_name").copy()
    # df_office_pa_device = df_office_pa_device.query("value > 0")
    # df_office_pa_device = df_office_pa_device[df_office_pa_device.value > 0]

    df_day_by_hour = (
        df_office_pa_device
        .reset_index()
        .groupby(['device_name','type','variable','hour'])['value']
        .agg(['median','mean','std','min',pro.q_low,pro.q_high,'max','count'])
        .reset_index()
    )

    df_week_by_day = (
        df_office_pa_device
        .reset_index()
        .groupby(['device_name','type','variable','cont_dow'])['value']
        .agg(['median','mean','std','min',pro.q_low,pro.q_high,'max','count'])
        .reset_index()
    )

    if (fix_negatives is True):
        df_day_by_hour.loc[(df_day_by_hour.q_low < 0), 'q_low'] = 0
        df_week_by_day.loc[(df_week_by_day.q_low < 0), 'q_low'] = 0

    grp.plot_typical_day_by_hour(
        df_day_by_hour.query("type == 'ilu'"), 
        subset='variable', 
        title=f"Día típico para la iluminación en la sede de {office_name}", 
        bl_label="Promedio de:", 
        bl_ci_label="Intervalo de:",
        legend=True, 
        include_ci=True, 
        fill_ci=True
    )

    grp.plot_typical_week_by_day(
        df_week_by_day.query("type == 'ilu'"),
        subset='variable',  
        title=f"Semana típica para la iluminación en la sede de {office_name}",
        bl_label="Promedio de:", 
        bl_ci_label="Intervalo de:",
        legend=True, 
        include_ci=True, 
        fill_ci=True
    )


def bar_plot_monthly_cons(df_office_total_cons, device_name, office_name):
    df_office_total_cons_device = df_office_total_cons.query("device_name == @device_name").copy()
    
    df_office_total_cons_device_monthly = (
        df_office_total_cons_device
        .groupby(['month'])['value']
        .sum()
        .reset_index()
    )

    df_office_total_cons_device_monthly['value'] = df_office_total_cons_device_monthly['value'].round(2)

    fig = px.bar(
        df_office_total_cons_device_monthly,
        x="month",
        y="value",
        labels={'month':'Mes', 'value':'Consumo [kWh]'},
        title=f"Consumo mensual de energía activa [kWh] en la sede de {office_name}",
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
    fig.show()


def bar_plot_daytype_cons(df_office_total_cons, device_name, office_name):
    df_by_office = df_office_total_cons.query("device_name == @device_name").copy()

    df_daily_cons = (
        df_by_office
        .groupby(['device_name','type_of_day'])['value']
        .resample('1D')
        .sum()
        .reset_index()
    )

    df_daily_cons = df_daily_cons.query("value > 0")

    df_daily_cons_by_daytype = (
        df_daily_cons
        .groupby(['device_name','type_of_day'])['value']
        .mean()
        .reset_index()
    ).round(2)


    fig = px.bar(
        df_daily_cons_by_daytype,
        x="type_of_day",
        y="value",
        labels={'type_of_day':'Tipo de día', 'value':'Consumo [kWh]'},
        title=f"Consumo diario promedio de energía activa [kWh] en la sede de {office_name}",
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
    fig.show()
    

def hvac_power_histogram_auto(df_circuit_pa, office_label, office_name, quantile_cutoff=0):
    is_selection = (
        (df_circuit_pa['office_label'] == office_label)
        & (df_circuit_pa['type'] == 'AA')
    )

    df_hist =  cln.remove_lower_quantile(df_circuit_pa[is_selection], quantile=quantile_cutoff)

    if (len(df_hist) > 0):
        fig = px.histogram(
            df_hist,
            x="value",
            color='device_name',
            color_discrete_sequence=repcfg.FULL_PALETTE,
            labels={'value':'Potencia activa [kW]', 'device_name':'Circuito'},
            title=f"Climatización: distribución de potencia activa en la sede de {office_name}",
        )

        fig.update_layout(
            font_family=repcfg.CELSIA_FONT,
            font_size=repcfg.PLOTLY_TITLE_FONT_SIZE,
            font_color=repcfg.FULL_PALETTE[1],
            title_x=repcfg.PLOTLY_TITLE_X,
            width=repcfg.JBOOK_PLOTLY_WIDTH,
            height=repcfg.JBOOK_PLOTLY_HEIGHT
        )

        fig.show()


def piechart_zone_cons(df_office_cons, device_name, office_name):
    df_office_cons_device = df_office_cons.query("device_name == @device_name").copy()

    df_daily_cons_by_type = (
        df_office_cons_device
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
    ).round(2)

    fig = px.pie(
        df_mean_daily_cons_by_type, 
        values="value", 
        names='type', 
        hover_data=['value'],
        labels={'type':'Tipo', 'value':'Consumo diario promedio [kWh]'},
        title=f"Consumo promedio diario de energía activa por tipo de carga para la sede de {office_name}",
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


# ## Clima cálido húmedo (1A)

# ### Barranquilla

# #### Indicadores de desempeño energético (IDE)

# ![Diagrama_proceso_ISO_50001.jpg](figures/EUI_1A.png)

# De este análisis se puede identificar que la sede está al margen de la línea base (EUI Base) definido por normativa (88KWh/m2), es decir que su consumo energético es lo que se espera para un Retail de estas características. Sin embargo, se debe tener en cuenta que la cobertura de medición está en el orden del 60% 

# In[6]:


device_name = 'HC - Barranquilla'
office_name = 'Barranquilla'
office_label = df_office.query("device_name == @device_name")['device'].mode()[0]


# In[7]:


bar_plot_monthly_cons(df_office_total_cons, device_name, office_name)


# El consumo mensual promedio de los últimos meses para la tienda ha sido de 220.000 kWh/mes 

# In[8]:


piechart_zone_cons(df_office_cons, device_name, office_name)


# En esta tienda más del 70% de la energía consumida está asociada a los consumos de Aire Acondicionado (HVAC), lo que supone el mayor foco de interés para temas de optimización de energética. Por otro lado, el segundo consumidor más relevante está asociado a la iluminación principal de la tienda (aprox. 21%). Estos dos usos finales representan el 92% de la energía consumida del proyecto.  

# In[9]:


typical_day(df_office_total_cons, device_name, office_name)


# Dado que los consumos promedios diarios se mantienen relativamente constantes se presenta una curva de consumo diaria asociada al consumo hora a hora de las instalaciones. Las regiones sombreadas representan un intervalo de confianza de 95% para la línea base. 

# In[10]:


bar_plot_daytype_cons(df_office_total_cons, device_name, office_name)


# Dependiendo del día de la semana, no hay gran variación entre diferenciar un día entre semana, a un sábado o un domingo. En promedio, se consumen unos 7000kWh/día. 

# #### Equipos de climatización

# In[11]:


hvac_typical_day_week(df_office_pa, device_name, office_name)


# In[12]:


hvac_power_histogram_auto(df_circuit_pa, office_label, office_name, quantile_cutoff=0.35)


# Análisis Consumo HVAC – Esta sede cuenta con un consumo promedio constante en horario diurno para el sistema de aire acondicionado de la tienda. Dicho lo anterior, se evidencian optimizaciones asociadas al control y regulación de demanda térmica sujeta a la tecnología disponible y flexibilidad de esta. La demanda pico del sistema de HVAC ronda los 560 KW lo que puede representar una demanda pico térmica de 280-330TR.(de 430TR Instaladas) 
# 
#  
# 
# Lo anterior abre la posibilidad a un reemplazo de equipos, considerando unas capacidades menores y optimizando setpoints de operación, así como la administración de aire exterior. 

# #### Iluminación

# In[13]:


lighting_typical_day_week(df_office_pa, device_name, office_name)


# Análisis Consumo Iluminación Tienda – Esta sede cuenta con un sistema de control horario escalonado según la hora del día a pesar de contar con secciones de teja traslucida en cubierta. Dicho lo anterior se ve la posibilidad de control que tenga en cuenta el aporte de la iluminación natural. La demanda pico del sistema de iluminación ronda los 93kW. 

# ### La Popa

# #### Indicadores de desempeño energético (IDE)

# ![Diagrama_proceso_ISO_50001.jpg](figures/EUI_1A.png)

# De este análisis se puede identificar que la sede está por encima de la línea base (EUI Base) definido por normativa (88KWh/m2), es decir que su consumo energético es superior a lo que se espera para un Retail de estas características. Sin embargo, se debe tener en cuenta que la cobertura de medición está en el orden del 70%. 

# In[14]:


device_name = 'HC - La Popa'
office_name = 'La Popa'
office_label = df_office.query("device_name == @device_name")['device'].mode()[0]


# In[15]:


bar_plot_monthly_cons(df_office_total_cons, device_name, office_name)


# El consumo mensual promedio de los últimos meses para la tienda ha sido de 160000 kWh/mes 

# In[16]:


piechart_zone_cons(df_office_cons, device_name, office_name)


# En esta tienda más del 92% de la energía consumida está asociada a los consumos de Aire Acondicionado (HVAC), lo que supone el mayor foco de interés para temas de optimización de energética. Por otro lado, el segundo consumidor más relevante está asociado a la iluminación principal del patio constructor (aprox. 2.6%). Estos dos usos finales representan el 94.6% de la energía consumida del proyecto.  

# In[17]:


typical_day(df_office_total_cons, device_name, office_name)


# Dado que los consumos promedios diarios se mantienen relativamente constantes se presenta una curva de consumo diaria asociada al consumo hora a hora de las instalaciones. Las regiones sombreadas representan un intervalo de confianza de 95% para la línea base. 

# In[18]:


bar_plot_daytype_cons(df_office_total_cons, device_name, office_name)


# Dependiendo del día de la semana, no hay gran variación entre diferenciar un día entre semana, a un sábado o un domingo. En promedio, se consumen unos 5400kWh/día. 

# #### Equipos de climatización

# In[19]:


hvac_typical_day_week(df_office_pa, device_name, office_name)


# In[20]:


hvac_power_histogram_auto(df_circuit_pa, office_label, office_name, quantile_cutoff=0.15)


# Análisis Consumo HVAC – Esta sede cuenta con un consumo promedio constante en horario diurno para el sistema de aire acondicionado de la tienda. Dicho lo anterior, se evidencian optimizaciones asociadas al control y regulación de demanda térmica sujeta a la tecnología disponible y flexibilidad de esta. La demanda que en el periodo de muestra  del sistema de HVAC ronda los 500 KW lo que puede representar una demanda pico térmica de 250-300TR (de 400TR.Instaladas) 
# 
# 
# Lo anterior abre la posibilidad a un reemplazo de equipos, considerando unas capacidades menores y optimizando setpoints de operación, así como la administración de aire exterior. 

# #### Iluminación

# In[21]:


lighting_typical_day_week(df_office_pa, device_name, office_name)


# Análisis Consumo Iluminación Tienda – Esta sede cuenta con un sistema de control horario escalonado según la hora del día a pesar de contar con secciones de teja traslucida en cubierta. Dicho lo anterior se ve la posibilidad de control que tenga en cuenta el aporte de la iluminación natural. La demanda pico del sistema de iluminación ronda los 5kW para la iluminación principal y 6KW para el patio constructor 

# ### San Fernando

# #### Indicadores de desempeño energético (IDE)

# ![Diagrama_proceso_ISO_50001.jpg](figures/EUI_1A.png)

# De este análisis se puede identificar que la sede está al margen de la línea base (EUI Base) definido por normativa (88KWh/m2), es decir que su consumo energético es lo que se espera para un Retail de estas características. Sin embargo, se debe tener en cuenta que la cobertura de medición está en el orden del 60% 

# In[22]:


device_name = 'HC - San Fernando'
office_name = 'San Fernando'
office_label = df_office.query("device_name == @device_name")['device'].mode()[0]


# In[23]:


bar_plot_monthly_cons(df_office_total_cons, device_name, office_name)


# El consumo mensual promedio de los últimos meses para la tienda ha sido de 135000 kWh/mes 

# In[24]:


piechart_zone_cons(df_office_cons, device_name, office_name)


# En esta tienda más del 68% de la energía consumida está asociada a los consumos de Aire Acondicionado (HVAC), lo que supone el mayor foco de interés para temas de optimización de energética. Por otro lado, el segundo consumidor más relevante está asociado a la iluminación (aprox. 24%). Estos usos finales representan el 93.2% de la energía consumida del proyecto.  

# In[25]:


typical_day(df_office_total_cons, device_name, office_name)


# Dado que los consumos promedios diarios se mantienen relativamente constantes se presenta una curva de consumo diaria asociada al consumo hora a hora de las instalaciones. Las regiones sombreadas representan un intervalo de confianza de 95% para la línea base. 

# In[26]:


bar_plot_daytype_cons(df_office_total_cons, device_name, office_name)


# Dependiendo del día de la semana, no hay gran variación entre diferenciar un día entre semana, a un sábado o un domingo. En promedio, se consumen unos 4452kWh/día. 

# #### Equipos de climatización

# In[27]:


hvac_typical_day_week(df_office_pa, device_name, office_name)


# In[28]:


hvac_power_histogram_auto(df_circuit_pa, office_label, office_name, quantile_cutoff=0.40)


# Análisis Consumo HVAC – Esta sede cuenta con un consumo promedio constante en horario diurno para el sistema de aire acondicionado de la tienda. Dicho lo anterior, se evidencian optimizaciones asociadas al control y regulación de demanda térmica sujeta a la tecnología disponible y flexibilidad de esta. La demanda pico del sistema de HVAC ronda los 350KW lo que puede representar una demanda pico térmica de 200-250TR. (350TR Instaladas) 
# 
#  
# 
#  
# 
# Lo anterior abre la posibilidad a un reemplazo de equipos, considerando unas capacidades menores y optimizando setpoints de operación, así como la administración de aire exterior. 

# #### Iluminación

# In[29]:


lighting_typical_day_week(df_office_pa, device_name, office_name)


# Análisis Consumo Iluminación Tienda – Esta sede cuenta con un sistema de control horario constante según la hora del día a pesar de contar con secciones de teja traslucida en cubierta. Dicho lo anterior se ve la posibilidad de control que tenga en cuenta el aporte de la iluminación natural. La demanda pico del sistema de iluminación ronda los 36kW para la iluminación principal y 25KW para el patio constructor 

# ### Cali Norte

# #### Indicadores de desempeño energético (IDE)

# ![Diagrama_proceso_ISO_50001.jpg](figures/EUI_1A.png)

# De este análisis se puede identificar que la sede está por encima de la línea base (EUI Base) definido por normativa (88KWh/m2), es decir que su consumo energético es mayor a lo que se espera para un Retail de estas características.  

# In[30]:


device_name = 'HC - Cali norte'
office_name = 'Cali Norte'
office_label = df_office.query("device_name == @device_name")['device'].mode()[0]


# In[31]:


bar_plot_monthly_cons(df_office_total_cons, device_name, office_name)


# El consumo mensual promedio de los últimos meses para la tienda ha sido de 300000 kWh/mes 

# In[32]:


piechart_zone_cons(df_office_cons, device_name, office_name)


# En esta tienda más del 86% de la energía consumida está asociada a los consumos de Aire Acondicionado (HVAC), lo que supone el mayor foco de interés para temas de optimización de energética. Por otro lado, el segundo consumidor más relevante está asociado a la iluminación principal del patio constructor (aprox.9%). Estos usos finales representan el 95% de la energía consumida del proyecto. 

# In[33]:


typical_day(df_office_total_cons, device_name, office_name)


# Dado que los consumos promedios diarios se mantienen relativamente constantes se presenta una curva de consumo diaria asociada al consumo hora a hora de las instalaciones. Las regiones sombreadas representan un intervalo de confianza de 95% para la línea base. 

# In[34]:


bar_plot_daytype_cons(df_office_total_cons, device_name, office_name)


# Dependiendo del día de la semana, no hay gran variación entre diferenciar un día entre semana, a un sábado o un domingo. En promedio, se consumen unos 10500kWh/día. 

# #### Equipos de climatización

# In[35]:


hvac_typical_day_week(df_office_pa, device_name, office_name)


# In[36]:


# Torre de enfriamiento appears to be a mislabel
df_cali_norte_pa_aa = df_circuit_pa.query("office_label == 'hc---cali-norte' & type == 'AA'").copy()

fix_label = {
    'Cali Norte - Torre de Enfriamiento':'Cali Norte - HVAC',
    'cali-norte---torre-de-enfriamiento':'cali-norte---hvac',
    'Cali Norte - ML Chiller':'Cali Norte - ML Chiller',
    'cali-norte---ml-chiller':'cali-norte---ml-chiller',
}

df_cali_norte_pa_aa['device_name'] = df_cali_norte_pa_aa['device_name'].map(fix_label)
df_cali_norte_pa_aa['device'] = df_cali_norte_pa_aa['device'].map(fix_label)

hvac_power_histogram_auto(df_cali_norte_pa_aa, office_label, office_name, quantile_cutoff=0.45)


# Análisis Consumo HVAC – Esta sede cuenta con un consumo promedio constante en horario diurno para el sistema de aire acondicionado de la tienda. Dicho lo anterior, se evidencian optimizaciones asociadas al control y regulación de demanda térmica sujeta a la tecnología disponible y flexibilidad de esta. La demanda pico del sistema de HVAC ronda los 300KW lo que puede representar una demanda pico térmica de 110-130TR. (500TR Instaladas) 
# 
#  
# 
# Lo anterior abre la posibilidad a un reemplazo de equipos, considerando unas capacidades menores y optimizando setpoints de operación, así como la administración de aire exterior. 

# #### Iluminación

# In[37]:


lighting_typical_day_week(df_office_pa, device_name, office_name)


# Análisis Consumo Iluminación Tienda – Esta sede cuenta con un sistema de control horario constante según la hora del día. La demanda pico del sistema de iluminación ronda los 33kW para la iluminación principal y 14KW para el patio constructor. 
# 
#  
# 
# Es importante resaltar que en horas de las noches (entre 0h a las 4am) se identifican hasta 13KW encendidos que pueden representar hasta unos 19MM COP anuales.  

# ### Cali Sur

# #### Indicadores de desempeño energético (IDE)

# ![Diagrama_proceso_ISO_50001.jpg](figures/EUI_1A.png)

# De este análisis se puede identificar que la sede está en el rango de la línea base (EUI Base) definido por normativa (88KWh/m2), es decir que su consumo energético es lo que se espera de este tipo de inmueble.  

# In[38]:


device_name = 'HC - Cali sur'
office_name = 'Cali Sur'
office_label = df_office.query("device_name == @device_name")['device'].mode()[0]


# In[39]:


bar_plot_monthly_cons(df_office_total_cons, device_name, office_name)


# El consumo mensual promedio de los últimos meses para la tienda ha sido de 210000 kWh/mes 

# In[40]:


piechart_zone_cons(df_office_cons, device_name, office_name)


# En esta tienda más del 66% de la energía consumida está asociada a los consumos de Aire Acondicionado (HVAC), lo que supone el mayor foco de interés para temas de optimización de energética. Por otro lado, el segundo consumidor más relevante está asociado a la iluminación (aprox. 28%). Estos usos finales representan el 95% de la energía consumida del proyecto.  

# In[41]:


typical_day(df_office_total_cons, device_name, office_name)


# Dado que los consumos promedios diarios se mantienen relativamente constantes se presenta una curva de consumo diaria asociada al consumo hora a hora de las instalaciones. Las regiones sombreadas representan un intervalo de confianza de 95% para la línea base. 

# In[42]:


bar_plot_daytype_cons(df_office_total_cons, device_name, office_name)


# Dependiendo del día de la semana, no hay gran variación entre diferenciar un día entre semana, a un sábado o un domingo. En promedio, se consumen unos 10500kWh/día. 

# #### Equipos de climatización

# In[43]:


hvac_typical_day_week(df_office_pa, device_name, office_name)


# In[44]:


hvac_power_histogram_auto(df_circuit_pa, office_label, office_name, quantile_cutoff=0.50)


# Análisis Consumo HVAC – Esta sede cuenta con un consumo promedio constante en horario diurno para el sistema de aire acondicionado de la tienda. Dicho lo anterior, se evidencian optimizaciones asociadas al control y regulación de demanda térmica sujeta a la tecnología disponible y flexibilidad de esta. La demanda pico del sistema de HVAC ronda los 300 KW lo que puede representar una demanda pico térmica de 250-300TR. (418 TR Instaladas) 
# 
#  
# 
# Lo anterior abre la posibilidad a un reemplazo de equipos, considerando unas capacidades menores y optimizando setpoints de operación, así como la administración de aire exterior. 

# #### Iluminación

# In[45]:


lighting_typical_day_week(df_office_pa, device_name, office_name)


# Análisis Consumo Iluminación Tienda – Esta sede cuenta con un sistema de control horario constante según la hora del día a pesar de contar con secciones de teja traslucida en cubierta. Dicho lo anterior se ve la posibilidad de control que tenga en cuenta el aporte de la iluminación natural. La demanda pico del sistema de iluminación ronda los 94kW para la iluminación principal y 0.6KW para el patio constructor 
# 
#  
# 
# Es importante resaltar que en horas de las noches (entre 10pm a las 4am) se identifican hasta 16KW encendidos que pueden representar hasta unos 24MM COP anuales.  

# ### Palmira

# #### Indicadores de desempeño energético (IDE)

# ![Diagrama_proceso_ISO_50001.jpg](figures/EUI_1A.png)

# De este análisis se puede identificar que la sede está en el rango de la línea base (EUI Base) definido por normativa (88KWh/m2), es decir que su consumo energético es lo que se espera de este tipo de inmueble. Sin embargo, se debe tener en cuenta que la cobertura de medición está en el orden del 80% 

# In[46]:


device_name = 'HC - Palmira'
office_name = 'Palmira'
office_label = df_office.query("device_name == @device_name")['device'].mode()[0]


# In[47]:


bar_plot_monthly_cons(df_office_total_cons, device_name, office_name)


# El consumo mensual promedio de los últimos meses para la tienda ha sido de 85000 kWh/mes 

# In[48]:


piechart_zone_cons(df_office_cons, device_name, office_name)


# En esta tienda más del 63% de la energía consumida está asociada a los consumos de Aire Acondicionado (HVAC), lo que supone el mayor foco de interés para temas de optimización de energética. Por otro lado, el segundo consumidor más relevante está asociado a la iluminación (aprox. 20%). Estos usos finales representan el 83% de la energía consumida del proyecto.  

# In[49]:


typical_day(df_office_total_cons, device_name, office_name)


# Dado que los consumos promedios diarios se mantienen relativamente constantes se presenta una curva de consumo diaria asociada al consumo hora a hora de las instalaciones. Las regiones sombreadas representan un intervalo de confianza de 95% para la línea base. 

# In[50]:


bar_plot_daytype_cons(df_office_total_cons, device_name, office_name)


# Dependiendo del día de la semana, no hay gran variación entre diferenciar un día entre semana, a un sábado o un domingo. En promedio, se consumen unos 10500kWh/día. 

# #### Equipos de climatización

# In[51]:


hvac_typical_day_week(df_office_pa, device_name, office_name)


# In[52]:


df_palmira_pa_aa = df_office.query("device == 'hc---palmira' & variable == 'pa-equipos-de-climatizacion'").copy()
df_palmira_pa_aa['office_label'] = 'hc---palmira'
df_palmira_pa_aa['type'] = 'AA'


# In[53]:


# no data
hvac_power_histogram_auto(df_palmira_pa_aa, office_label, office_name, quantile_cutoff=0.5)


# Análisis Consumo HVAC – Esta sede cuenta con un consumo promedio constante en horario diurno para el sistema de aire acondicionado de la tienda. Dicho lo anterior, se evidencian optimizaciones asociadas al control y regulación de demanda térmica sujeta a la tecnología disponible y flexibilidad de esta. La demanda pico del sistema de HVAC ronda los 280 KW lo que puede representar una demanda pico térmica de 120TR. (150TR Instaladas) 
# 
#  
# 
# Lo anterior abre la posibilidad a optimizar setpoints de operación, así como la administración de aire exterior. 

# #### Iluminación

# In[54]:


lighting_typical_day_week(df_office_pa, device_name, office_name)


# Análisis Consumo Iluminación Tienda – Esta sede cuenta con un sistema de control horario constante según la hora del día a pesar de contar con secciones de teja traslucida en cubierta. Dicho lo anterior se ve la posibilidad de control que tenga en cuenta el aporte de la iluminación natural. La demanda pico del sistema de iluminación ronda los 33kW para la iluminación principal y 3.8KW para el patio constructor 
# 
#  
# 
# Es importante resaltar que en horas de las noches (entre 10pm a las 4am) se identifican hasta 5KW encendidos que pueden representar hasta unos 7MM COP anuales.  

# ## Clima cálido seco-templado (2A)

# ### Bello

# #### Indicadores de desempeño energético (IDE)

# ![Diagrama_proceso_ISO_50001.jpg](figures/EUI_2A.png)

# De este análisis se puede identificar que la sede está por encima de la línea base (EUI Base) definido por normativa (80KWh/m2), es decir que su consumo energético es mayor a lo que se espera para un Retail de estas características. Sin embargo, se debe tener en cuenta que la cobertura de medición está en el orden del 70% 

# In[55]:


device_name = 'HC - Bello'
office_name = 'Bello'
office_label = df_office.query("device_name == @device_name")['device'].mode()[0]


# In[56]:


bar_plot_monthly_cons(df_office_total_cons, device_name, office_name)


# El consumo mensual promedio de los últimos meses para la tienda ha sido de 95000 kWh/mes 

# In[57]:


piechart_zone_cons(df_office_cons, device_name, office_name)


# En esta tienda más del 56% de la energía consumida está asociada a los consumos de Aire Acondicionado (HVAC), lo que supone el mayor foco de interés para temas de optimización de energética. Por otro lado, el segundo consumidor más relevante está asociado a la iluminación (aprox. 37%). Estos usos finales representan el 93% de la energía consumida del proyecto.  

# In[58]:


typical_day(df_office_total_cons, device_name, office_name)


# Dado que los consumos promedios diarios se mantienen relativamente constantes se presenta una curva de consumo diaria asociada al consumo hora a hora de las instalaciones. Las regiones sombreadas representan un intervalo de confianza de 95% para la línea base. 

# In[59]:


bar_plot_daytype_cons(df_office_total_cons, device_name, office_name)


# Dependiendo del día de la semana, no hay gran variación entre diferenciar un día entre semana, a un sábado o un domingo. En promedio, se consumen unos 3200kWh/día. 

# #### Equipos de climatización

# In[60]:


hvac_typical_day_week(df_office_pa, device_name, office_name)


# In[61]:


hvac_power_histogram_auto(df_circuit_pa, office_label, office_name, quantile_cutoff=0.5)


# Análisis Consumo HVAC – Esta sede cuenta con un consumo promedio constante en horario diurno para el sistema de aire acondicionado de la tienda. Dicho lo anterior, se evidencian optimizaciones asociadas al control y regulación de demanda térmica sujeta a la tecnología disponible y flexibilidad de esta. La demanda pico del sistema de HVAC ronda los 300 KW lo que puede representar una demanda pico térmica de 150-170TR. (260TR Instaladas) 
# 
#  
# 
# Lo anterior abre la posibilidad a un reemplazo de equipos, considerando unas capacidades menores y optimizando setpoints de operación, así como la administración de aire exterior. 

# #### Iluminación

# In[62]:


lighting_typical_day_week(df_office_pa, device_name, office_name)


# Análisis Consumo Iluminación Tienda – Esta sede cuenta con un sistema de control horario constante. La demanda pico del sistema de iluminación ronda los 44kW (28KW en promedio) para la iluminación principal y 16KW para el patio constructor. Sin embargo presenta diferentes variaciones a lo largo de un día normal.  
# 
#  
# 
# Es importante resaltar que en horas de las noches (entre 10pm a las 4am) se identifican hasta 5KW encendidos que pueden representar hasta unos 22MM COP anuales.  

# ### Bucaramanga

# #### Indicadores de desempeño energético (IDE)

# ![Diagrama_proceso_ISO_50001.jpg](figures/EUI_2A.png)

# De este análisis se puede identificar que la sede está por encima de la línea base (EUI Base) definido por normativa (80KWh/m2), es decir que su consumo energético es mayor a lo que se espera para un Retail de estas características.  

# In[63]:


device_name = 'HC - Bucaramanga'
office_name = 'Bucaramanga'
office_label = df_office.query("device_name == @device_name")['device'].mode()[0]


# In[64]:


bar_plot_monthly_cons(df_office_total_cons, device_name, office_name)


# El consumo mensual promedio de los últimos meses para la tienda ha sido de 205000 kWh/mes 
# 
#  

# In[65]:


piechart_zone_cons(df_office_cons, device_name, office_name)


# En esta tienda más del 76% de la energía consumida está asociada a los consumos de Aire Acondicionado (HVAC), lo que supone el mayor foco de interés para temas de optimización de energética. Por otro lado, el segundo consumidor más relevante está asociado a la iluminación (aprox. 14%). Estos usos finales representan el 90% de la energía consumida del proyecto.  

# In[66]:


typical_day(df_office_total_cons, device_name, office_name)


# In[67]:


bar_plot_daytype_cons(df_office_total_cons, device_name, office_name)


# Dependiendo del día de la semana, no hay gran variación entre diferenciar un día entre semana, a un sábado o un domingo. En promedio, se consumen unos 6800kWh/día. 

# #### Equipos de climatización

# In[68]:


hvac_typical_day_week(df_office_pa, device_name, office_name)


# In[69]:


df_circuit_bucaramanga = df_circuit.query("office_label == 'hc---bucaramanga'").copy()
df_bga_pa_aa = df_circuit_bucaramanga.query("type == 'AA'").query("variable == 'potencia-activa-calculada'")


# In[70]:


hvac_power_histogram_auto(df_bga_pa_aa, office_label, office_name, quantile_cutoff=0)


# Análisis Consumo HVAC – Esta sede cuenta con un consumo promedio constante en horario diurno para el sistema de aire acondicionado de la tienda. Dicho lo anterior, se evidencian optimizaciones asociadas al control y regulación de demanda térmica sujeta a la tecnología disponible y flexibilidad de esta. La demanda pico del sistema de HVAC ronda los 550 KW lo que puede representar una demanda pico térmica de 220TR. (262TR Instaladas) 
# 
#  
# 
# Lo anterior abre la posibilidad a optimizar setpoints de operación, así como la administración de aire exterior. 

# #### Iluminación

# In[71]:


lighting_typical_day_week(df_office_pa, device_name, office_name)


# Análisis Consumo Iluminación Tienda – Esta sede cuenta con un sistema de control horario variable de escalonamiento según la hora del día la hora del día. La demanda pico del sistema de iluminación ronda los 50KW (32kWen promedio) para la iluminación principal y 14KW para el patio constructor 

# ### San Juan

# #### Indicadores de desempeño energético (IDE)

# ![Diagrama_proceso_ISO_50001.jpg](figures/EUI_2A.png)

# De este análisis se puede identificar que la sede está en el rango de la línea base (EUI Base) definido por normativa (80KWh/m2), es decir que su consumo energético es lo que se espera para un Retail de estas características 

# In[72]:


device_name = 'HC - San Juan'
office_name = 'San Juan'
office_label = df_office.query("device_name == @device_name")['device'].mode()[0]


# In[73]:


bar_plot_monthly_cons(df_office_total_cons, device_name, office_name)


# El consumo mensual promedio de los últimos meses para la tienda ha sido de 210000 kWh/mes 

# In[74]:


piechart_zone_cons(df_office_cons, device_name, office_name)


# En esta tienda más del 44% de la energía consumida está asociada a los consumos de Aire Acondicionado (HVAC). Por otro lado, el segundo consumidor más relevante está asociado a la iluminación (aprox. 41%). Estos usos finales representan el 85% de la energía consumida del proyecto.  

# In[75]:


typical_day(df_office_total_cons, device_name, office_name)


# Dado que los consumos promedios diarios se mantienen relativamente constantes se presenta una curva de consumo diaria asociada al consumo hora a hora de las instalaciones. Las regiones sombreadas representan un intervalo de confianza de 95% para la línea base. 

# In[76]:


bar_plot_daytype_cons(df_office_total_cons, device_name, office_name)


# Dependiendo del día de la semana, no hay gran variación entre diferenciar un día entre semana, a un sábado o un domingo. En promedio, se consumen unos 10500kWh/día. 

# #### Equipos de climatización

# In[77]:


hvac_typical_day_week(df_office_pa, device_name, office_name)


# In[78]:


hvac_power_histogram_auto(df_circuit_pa, office_label, office_name, quantile_cutoff=0.55)


# Análisis Consumo HVAC – Esta sede cuenta con un consumo promedio constante en horario diurno para el sistema de aire acondicionado de la tienda. Dicho lo anterior, se evidencian optimizaciones asociadas al control y regulación de demanda térmica sujeta a la tecnología disponible y flexibilidad de esta. La demanda pico del sistema de HVAC ronda los 450 KW lo que puede representar una demanda pico térmica de 250-300TR. (450TR Instaladas) 
# 
#  
# 
# Lo anterior abre la posibilidad a un reemplazo de equipos, considerando unas capacidades menores y optimizando setpoints de operación, así como la administración de aire exterior. 

# #### Iluminación

# In[79]:


lighting_typical_day_week(df_office_pa, device_name, office_name)


# Análisis Consumo Iluminación Tienda – Esta sede cuenta con un sistema de control horario constante según la hora del día a pesar de contar con secciones de teja traslucida en cubierta. Dicho lo anterior se ve la posibilidad de control que tenga en cuenta el aporte de la iluminación natural. La demanda pico del sistema de iluminación ronda los 120kW para la iluminación principal y 100KW para el patio constructor 
# 
#  
# 
# Es importante resaltar que en horas de las noches (entre 10pm a las 4am) se identifican hasta 35KW encendidos que pueden representar hasta unos 40MM COP anuales.  

# ## Clima frío (3A)

# ### Calle 80

# #### Indicadores de desempeño energético (IDE)

# ![Diagrama_proceso_ISO_50001.jpg](figures/EUI_3A.png)

# De este análisis se puede identificar que la sede está por debajo de la línea base (EUI Base) definido por normativa (60KWh/m2), es decir que su consumo energético es menor a lo que se espera para un Retail de estas características. Sin embargo, se debe tener en cuenta que la cobertura de medición está en el orden del 65%, que puede cambiar la interpretación de este indicador.  

# In[80]:


device_name =  'HC - Calle 80'
office_name = 'Calle 80'
office_label = df_office.query("device_name == @device_name")['device'].mode()[0]


# In[81]:


bar_plot_monthly_cons(df_office_total_cons, device_name, office_name)


# El consumo mensual promedio de los últimos meses para la tienda ha sido de 108000 kWh/mes 

# In[82]:


piechart_zone_cons(df_office_cons, device_name, office_name)


# En esta tienda más del 64% de la energía consumida está asociada a los consumos de Iluminación lo que supone el mayor foco de interés para temas de optimización de energética. Por otro lado, el segundo consumidor más relevante agrupado en la categoría de otros está asociado principalmente a las concesiones.  

# In[83]:


# locally clean this plot (so as to not affect the global consumption calcs)
df_cons_calle_80 = df_office_total_cons.query("device == 'calle-80'")
df_cons_calle_80 = cln.remove_outliers_by_zscore(df_cons_calle_80, zscore=5)
typical_day(df_cons_calle_80, device_name, office_name)


# Dado que los consumos promedios diarios se mantienen relativamente constantes se presenta una curva de consumo diaria asociada al consumo hora a hora de las instalaciones. Las regiones sombreadas representan un intervalo de confianza de 95% para la línea base.

# In[84]:


bar_plot_daytype_cons(df_office_total_cons, device_name, office_name)


# Dependiendo del día de la semana, no hay gran variación entre diferenciar un día entre semana, a un sábado o un domingo. En promedio, se consumen unos 3200kWh/día. 

# #### Iluminación

# In[85]:


lighting_typical_day_week(df_office_pa, device_name, office_name)


# Análisis Consumo Iluminación Tienda – Esta sede cuenta con un sistema que es variable (intervalo considerablemente variable) con un comportamiento regular en el transcurso de los días a pesar de contar con secciones de teja traslucida en cubierta. Dicho lo anterior se ve la posibilidad de control que tenga en cuenta el aporte de la iluminación natural. La demanda pico del sistema de iluminación ronda los 82kW(Promedio de 60KW) para la iluminación principal y 30KW(Promedio de 20KW) para el patio constructor 
# 
#  
# 
# Es importante resaltar que en horas de las noches (entre 10pm a las 4am) se identifican hasta 30KW encendidos que pueden representar hasta unos 38MM COP anuales.  

# ### Cedritos

# #### Indicadores de desempeño energético (IDE)

# ![Diagrama_proceso_ISO_50001.jpg](figures/EUI_3A.png)

# De este análisis se puede identificar que la sede está por debajo de la línea base (EUI Base) definido por normativa (60KWh/m2), es decir que su consumo energético es menor a lo que se espera para un Retail de estas características. Sin embargo, se debe tener en cuenta que la cobertura de medición está en el orden del 65%, que puede cambiar la interpretación de este indicador.  

# In[86]:


device_name = 'HC - Cedritos'
office_name = 'Cedritos'
office_label = df_office.query("device_name == @device_name")['device'].mode()[0]


# In[87]:


bar_plot_monthly_cons(df_office_total_cons, device_name, office_name)


# El consumo mensual promedio de los últimos meses para la tienda ha sido de 55000 kWh/mes 

# In[88]:


piechart_zone_cons(df_office_cons, device_name, office_name)


# En esta tienda más del 72% de la energía consumida está asociada a los consumos de Iluminación lo que supone el mayor foco de interés para temas de optimización de energética. Por otro lado, el segundo consumidor más relevante agrupado en la categoría de otros está asociado principalmente a las cargas del concesiones (24%) 

# In[89]:


typical_day(df_office_total_cons, device_name, office_name)


# Dado que los consumos promedios diarios se mantienen relativamente constantes se presenta una curva de consumo diaria asociada al consumo hora a hora de las instalaciones. Las regiones sombreadas representan un intervalo de confianza de 95% para la línea base. 

# In[90]:


bar_plot_daytype_cons(df_office_total_cons, device_name, office_name)


# Dependiendo del día de la semana, no hay gran variación entre diferenciar un día entre semana, a un sábado o un domingo. En promedio, se consumen unos 1600kWh/día. 

# #### Iluminación

# In[91]:


lighting_typical_day_week(df_office_pa, device_name, office_name)


# Análisis Consumo Iluminación Tienda – Esta sede cuenta con un sistema que es variable (intervalo considerablemente variable) con un comportamiento regular en el transcurso de los días a pesar de contar con secciones de teja traslucida en cubierta. Dicho lo anterior se ve la posibilidad de control que tenga en cuenta el aporte de la iluminación natural. La demanda pico del sistema de iluminación ronda los 86kW(Promedio de 60KW) para la iluminación principal y 30KW(Promedio de 20KW) para el patio constructor. 
# 
#  
# 
# Es importante resaltar que en horas de las noches (entre 10pm a las 4am) se identifican hasta 7KW encendidos que pueden representar hasta unos 8MM COP anuales.  

# ### Funza

# #### Indicadores de desempeño energético (IDE)

# ![Diagrama_proceso_ISO_50001.jpg](figures/EUI_3A.png)

# De este análisis se puede identificar que la sede está por debajo de la línea base (EUI Base) definido por normativa (60KWh/m2), es decir que su consumo energético es menor a lo que se espera para un Retail de estas características. . Sin embargo, se debe tener en cuenta que la cobertura de medición está en el orden del 20%, que puede cambiar la interpretación de este indicador. 

# In[92]:


device_name = 'HC - Funza'
office_name = 'Funza'
office_label = df_office.query("device_name == @device_name")['device'].mode()[0]


# In[93]:


bar_plot_monthly_cons(df_office_total_cons, device_name, office_name)


# El consumo mensual promedio de los últimos meses para la tienda ha sido de 57000 kWh/mes 

# In[94]:


piechart_zone_cons(df_office_cons, device_name, office_name)


# En esta tienda más del 50% de la energía consumida está asociada a los consumos de Iluminación lo que supone el mayor foco de interés para temas de optimización de energética. Por otro lado, el segundo consumidor más relevante agrupado en la categoría de otros está asociado principalmente a las cargas del tablero sorter (40%).  

# In[95]:


typical_day(df_office_total_cons, device_name, office_name)


# Dado que los consumos promedios diarios se mantienen relativamente constantes se presenta una curva de consumo diaria asociada al consumo hora a hora de las instalaciones. Las regiones sombreadas representan un intervalo de confianza de 95% para la línea base. 

# In[96]:


bar_plot_daytype_cons(df_office_total_cons, device_name, office_name)


# Dependiendo del día de la semana, no hay gran variación entre diferenciar un día entre semana, a un sábado o un domingo. En promedio, se consumen unos 2000kWh/día. 

# #### Iluminación

# In[97]:


lighting_typical_day_week(df_office_pa, device_name, office_name)


# Análisis Consumo Iluminación Tienda – Esta sede cuenta con un sistema que es variable sin un aparente comportamiento regular a pesar de contar con secciones de teja traslucida en cubierta. Dicho lo anterior se ve la posibilidad de control que tenga en cuenta el aporte de la iluminación natural. La demanda pico del sistema de iluminación ronda los 35kW(Promedio de 23KW) siendo esta la mayor demanda en el sistema de iluminación.  
# 
# Es importante resaltar que en horas de las noches (entre 10pm a las 4am) se identifican hasta 31KW encendidos que pueden representar hasta unos 40MM COP anuales.  Sin embargo, deberá analizarse entendiendo el uso especial de esta sede como centro de distribución. 

# ### Tintal

# #### Indicadores de desempeño energético (IDE)

# ![Diagrama_proceso_ISO_50001.jpg](figures/EUI_3A.png)

# De este análisis se puede identificar que la sede está por debajo de la línea base (EUI Base) definido por normativa (60KWh/m2), es decir que su consumo energético es menor a lo que se espera para un Retail de estas características.  

# In[98]:


device_name = 'HC - Tintal'
office_name = 'Tintal'
office_label = df_office.query("device_name == @device_name")['device'].mode()[0]


# In[99]:


bar_plot_monthly_cons(df_office_total_cons, device_name, office_name)


# El consumo mensual promedio de los últimos meses para la tienda ha sido de 63000 kWh/mes 

# In[100]:


piechart_zone_cons(df_office_cons, device_name, office_name)


# En esta tienda más del 50% de la energía consumida está asociada a los consumos de Iluminación lo que supone el mayor foco de interés para temas de optimización de energética. Por otro lado, el segundo consumidor más relevante agrupado en la categoría de otros está asociado principalmente a Equipos verticales (18%), Oficinas (13%) y concesiones (13%).  

# In[101]:


typical_day(df_office_total_cons, device_name, office_name)


# Dado que los consumos promedios diarios se mantienen relativamente constantes se presenta una curva de consumo diaria asociada al consumo hora a hora de las instalaciones. Las regiones sombreadas representan un intervalo de confianza de 95% para la línea base. 

# In[102]:


bar_plot_daytype_cons(df_office_total_cons, device_name, office_name)


# Dependiendo del día de la semana, no hay gran variación entre diferenciar un día entre semana, a un sábado o un domingo. En promedio, se consumen unos 2000kWh/día. 

# #### Iluminación

# In[103]:


lighting_typical_day_week(df_office_pa, device_name, office_name, fix_negatives=True)


# Análisis Consumo Iluminación Tienda – Esta sede cuenta con un sistema que es variable (intervalo considerablemente variable) con un comportamiento regular en el transcurso de los días a pesar de contar con secciones de teja traslucida en cubierta. Dicho lo anterior se ve la posibilidad de control que tenga en cuenta el aporte de la iluminación natural. La demanda pico del sistema de iluminación ronda los 82kW(Promedio de 60KW) para la iluminación principal y 30KW(Promedio de 20KW) para el patio constructor 
# 
# 
# Es importante resaltar que en horas de las noches (entre 10pm a las 4am) se identifican hasta 8KW encendidos que pueden representar hasta unos 10MM COP anuales.  

# ## Mejoras - Medidas para la Eficiencia Energética

# ### Listado de medidas

# CELSIA como aliado estratégico en procesos de mejora continua y eficiencia energética, tiene toda la disposición de acompañar a Sodimac en la implementación de las estrategias de eficiencias energéticas identificadas. Algunos de las áreas en las que más nos destacamos son:  
# 
# 1. Apoyo en el cambio de tecnologías de HVAC, por las más eficientes del mercado en modalidades de gestión de activos (tu no inviertes, nosotros lo hacemos por ti) o venta directa.
# 
# 2. Plataforma Central de Monitorio y Gestión de Energéticos; que te permitirá no solo monitorear consumos (como lo haces ahora a través de nuestra solución) sino que podrás controlar sistemas como el de iluminación en beneficio de reducir consumos y todo dentro de la misma interfaz.
# 
# 3. Asesoramiento y Estudios adicionales de eficiencia (mediciones específicas) en aquellos consumos más representativos, que puedan darte la tranquilidad de que vas a realizar una inversión inteligente.
# 
# A continuación, te listamos las oportunidades de mejora que identificamos a través de la asesoría que hoy te estamos brindando:

# ![Diagrama_proceso_ISO_50001.jpg](figures/medidas_02.png)

# ### Matriz de implementación

# A continuación, podrás ver en cuales de tus sedes son aplicables las estrategias que se identificaron dentro de tus instalaciones:  

# ![Diagrama_proceso_ISO_50001.jpg](figures/recomendaciones.png)

# ¡En Celsia nos encanta acompañarte en la meta de ser más eficientes!

# ![alt text](https://www.celsia.com/wp-content/uploads/2021/11/Celsia-Horizonal-Eslogan_Jpg.jpg)
