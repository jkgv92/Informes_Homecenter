# Base configuration ----------------------------------------------------------

lst_primary_pallete = ['#d5752d', '#59595b']
lst_secondary_palette = ['#13a2e1', '#00be91', '#fff65e', '#003fa2', '#ca0045']
USE_CELSIA_PALETTE = True
# Really it's Helvetica but one has to install it first.
# Arial does the trick and is officially endorsed.
CELSIA_FONT = 'Arial'
SCATTERGEO_MAX_MARKER_AREA = 1000

# Ubidots API
API_URL = 'https://industrial.api.ubidots.com/api/v1.6/devices/'
LST_VAR_FIELDS = [
    "value.value",
    "variable.id",
    # "variable.name",
    "device.label",
    "device.name",
    "timestamp"
]

LST_HEADERS = [
    'value',
    'variable',
    # 'variable_name',
    'device',
    'device_name',
    'timestamp'
]


# Date and Time
DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S"
DATE_FORMAT = "%Y-%m-%d"
LOCAL_TIMEZONE = 'America/Bogota'

# Plotting parameters
CONFIDENCE_INTERVAL = 95


# General parameters
ALLOWED_DATE_OVERLAP = 0
DAYS_PER_MONTH = 365.25/12

dct_dow = {
    0: 'lunes',
    1: 'martes',
    2: 'miércoles',
    3: 'jueves',
    4: 'viernes',
    5: 'sábado',
    6: 'domingo',
}

lst_nighttime_hours = [0, 1, 2, 3, 4, 5, 19, 20, 21, 22, 23]
list_overnight_hours = [0, 1, 2, 3, 4, 5, 21, 22, 23]

# Ubidots data parameters
DEVICE_GROUP_LABEL = 'homecenter-sedes'
DATA_FREQUENCY = '15T'
USE_PICKLED_DATA = True
HOUR_FREQUENCY = '1h'

# Plotting parameters
golden_ratio = (1+(5)**0.5)/2
# try 17.362, 24.579, 21.19 for widths
width = 17.362
height = width*5.5/12.5
WIDE_FIGURE_SIZE = (width, height)
SAVE_FIGURES = False
SHOW_OPTIONAL_FIGURES = False

plotly_width = 800# 1250
plotly_height = plotly_width / golden_ratio # 550

# Cleaning parameters
CLEAN_DATA = False
VALIDATE_CLEANING = False

SHORT_WINDOW = '5h'
SHORT_CONFIDENCE_INTERVAL = 97.5

LONG_WINDOW = '5D'
LONG_CONFIDENCE_INTERVAL = 99

SMOOTHING_METHOD = 'mean'
SMOOTHING_WINDOW = '3h'

REFERENCE_POWER_PERCENTILE = 99

# General parameters
# General parameters
COP_PER_KWH = 692.29

ALL_VARIABLE_LABELS = (
    'ac-tension-l1',
    'ac-tension-l2',
    'ac-tension-l3',
    'ea-aires-acondicionados',
    'ea-area-de-bots',
    'ea-area-de-corte',
    'ea-car-center',
    'ea-concesion',
    'ea-corte-y-dimensionado',
    'ea-equipos-de-climatizacion',
    'ea-equipos-verticales',
    'ea-iluminacion-cuarto-tecnico',
    'ea-iluminacion-edificio-adm',
    'ea-iluminacion-parqueaderos',
    'ea-iluminacion-parqueaderos-1',
    'ea-iluminacion-patio-constructor',
    'ea-iluminacion-patio-contenedores',
    'ea-iluminacion-principal',
    'ea-locales-comerciales',
    'ea-malacate-2000',
    'ea-malacate-hidraulico',
    'ea-mini-splits',
    'ea-oficinas-y-servicios',
    'ea-parqueadero-cubierto',
    'ea-pinturas',
    'ea-puente-grua',
    # 'ea-sabados' # messes up energy calcs
    'ea-tablero-sorter',
    'ea-talleres',
    'ea-toma-tienda-1',
    'ea-toma-tienda-2',
    'ea-total',
    'pa-area-de-bots',
    'pa-area-de-corte',
    'pa-car-center',
    'pa-concesiones',
    'pa-equipos-de-climatizacion',
    'pa-equipos-verticales',
    'pa-iluminacion',
    'pa-iluminacion-parqueaderos',
    'pa-iluminacion-patio-constructor',
    'pa-iluminacion-patio-contenedores',
    'pa-iluminacion-principal',
    'pa-oficinas-y-servicios',
    'pa-patio-constructor',
    'pa-tablero-sorter',
    'pa-talleres'
)

EXCLUDED_VARIABLE_LABELS = (
    'ac-tension-l1',    # debugging response <500>
    'ac-tension-l2',    # debugging response <500>
    'ac-tension-l3',    # debugging response <500>
    'temperatura',      # debugging response <500>
    'alertas',
    'cobertura',
    'consumo-aire-por-tr',
    'consumo-domingos-y-festivos',
    'consumo-por-area',
    'consumo-promedio-dia',
    'consumo-promedio-por-dia',
    'consumo-sabado',
    'consumo-semana',
    'consumo-total-tienda',
    'cruda_prueba',
    'ea-sabados',
    'hora-def',
    'new-variable',
    'test',
    'tipo-de-dia',
    'tipo-dia',
    'tipo_dia_hora',
    'token',
    'ton-co2-acumulado',
    'total-mes'
)

# cannot include total energy (frontera)
ACTIVE_ENERGY_LABELS = (
    'ea-aires-acondicionados',
    'ea-equipos-de-climatizacion',
    'ea-area-de-corte',
    'ea-car-center',
    'ea-corte-y-dimensionado',
    'ea-mini-splits',
    'ea-tablero-sorter',
    'ea-concesion',
    'ea-pinturas',
    'ea-area-de-bots',
    'ea-talleres',
    'ea-oficinas-y-servicios',
    'ea-equipos-verticales',
    'ea-iluminacion-parqueaderos-1',
    'ea-toma-tienda-2',
    'ea-parqueadero-cubierto',
    'ea-iluminacion-edificio-adm',
    'ea-toma-tienda-1',
    'ea-malacate-2000',
    'ea-iluminacion-principal',
    'ea-puente-grua',
    'ea-iluminacion-cuarto-tecnico',
    'ea-iluminacion-patio-constructor',
    'ea-malacate-hidraulico',
    'ea-iluminacion-patio-contenedores',
    'ea-iluminacion-parqueaderos',
    'ea-locales-comerciales'
)
ACTIVE_POWER_LABELS = (
    'pa-iluminacion-patio-constructor',
    'pa-talleres',
    'pa-tablero-sorter',
    'pa-area-de-corte',
    'pa-patio-constructor',
    'pa-area-de-bots',
    'pa-equipos-de-climatizacion',
    'pa-iluminacion-parqueaderos',
    'pa-iluminacion-principal',
    'pa-oficinas-y-servicios',
    'pa-equipos-verticales',
    'pa-concesiones',
    'pa-iluminacion',
    'pa-car-center',
    'pa-iluminacion-patio-contenedores'
)
ACTIVE_ENERGY_LUMP_LABEL = 'ea-otros'
ACTIVE_POWER_LUMP_LABEL = 'pa-otros'
TOTAL_ACTIVE_ENERGY_LABEL = 'ea-total'
TOTAL_ACTIVE_POWER_LABEL = 'pa-total'
TOTAL_REACTIVE_ENERGY_LABEL = 'er-total'

SUB_STR = ('ea-', 'pa-', 'ac-')

DATE_INTERVALS_TO_DISCARD = {
    'hc---cali-norte':[
        ['2022-03-23','2022-05-24'], # power scaling errors
        ['2022-04-29 00:00:00','2022-04-29 11:59:59'] # huge consumption outlier
    ],
    'hc-cali-sur':[
        ['2022-03-10','2022-05-21'] # scaling errors
    ],
    'hc---bucaramanga':[
        ['2022-03-25','2022-04-02'] # scaling errors
    ],
    'hc---palmira':[
        ['2022-03-25 12:00:00','2022-03-25 14:00:00'] # huge consumption outlier
    ],
    'hc-funza':[

    ],
    'calle-80':[
        ['2022-06-16 10:00:00','2022-06-16 16:00:00'] # huge consumption outlier
    ],
    'cedritos':[
        ['2022-01-01', '2022-05-20'] # power scaling errors
    ],
    'hc-san-fernando':[
        ['2022-02-23', '2022-04-24'], # consumption errors
        ['2022-04-23', '2022-05-04'] # consumption errors
    ],
    'hc-la-popa':[
        ['2022-01-01', '2022-05-20'] # consumption errors (from missing data?)
    ],
    'hc-baq':[
        ['2022-02-17', '2022-05-20'] # consumption errors (from missing data?)
    ],
    'hc-tintal':[
        ['2022-02-03', '2022-02-26']
    ],
    'hc-bello':[
        ['2022-01-01', '2022-01-29'], # huge consumption outlier
        ['2022-05-11', '2022-05-22'] # missing data
    ],
    'hc-san-juan':[
        ['2022-01-01', '2022-02-01'], # power outliers
    ]
}

BASELINE_DATE_INTERVAL = {
    'start': "2022-01-01",
    'end': "2022-08-31"
}

STUDY_DATE_INTERVAL = {
    'start': "2022-09-01",
    'end': "2022-09-30"
}
