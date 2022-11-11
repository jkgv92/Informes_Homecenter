import pandas as pd
import numpy as np
import scipy.stats as stats

import plotly.io as pio
import plotly.graph_objects as go
from matplotlib import pyplot as plt


import config as cfg

pio.renderers.default = "notebook"
pio.templates.default = "plotly_white"

LST_BIG_COLORS = cfg.lst_primary_pallete + cfg.lst_secondary_palette


def differentiate_single_variable(df, new_var_name, remove_gap_data=False):
    lst_df = []
    for device in set(df['device']):
        df_sel = df.query("device == @device").copy()
        if (~series.index.is_monotonic_increasing):
            series = series.sort_index()

        hour_deltas = df_sel.index.to_series().diff() / np.timedelta64(1, 'h')
        interval_mean_power = df_sel['value'].diff() / hour_deltas

        df_mono = pd.DataFrame()
        if (remove_gap_data is True):
            data_rate = hour_deltas.mode()[0]
            is_indeterminate = (hour_deltas > data_rate)
            df_mono['value'] = interval_mean_power[~is_indeterminate]
        else:
            df_mono['value'] = interval_mean_power

        df_mono['variable'] = new_var_name
        df_mono['device'] = device

        lst_df.append(df_mono)

    return pd.concat(lst_df)


def differentiate_series(series, remove_gap_data=False):

    if (~series.index.is_monotonic_increasing):
        series = series.sort_index()

    hour_deltas = series.index.to_series().diff() / np.timedelta64(1, 'h')
    interval_mean_power = series.diff() / hour_deltas

    if (remove_gap_data is True):
        data_rate = hour_deltas.mode()[0]
        is_indeterminate = (hour_deltas > data_rate)
        return interval_mean_power[~is_indeterminate]
    else:
        return interval_mean_power


def linearly_interpolate_series(series, data_rate_in_minutes=None):
    # If a data rate isn't provided the method will infer it
    # but generally one doesn't interpolate until the data has
    # been cleaned, which implies the removal of data, so it
    # is more robust to compute the data rate before cleaning
    # (sans duplicates, naturally).
    if data_rate_in_minutes is None:
        data_rate_in_minutes = int(
            series
            .index
            .to_series()
            .diff()
            .mode()
            .astype('timedelta64[m]')
        )

    return series.resample(f"{data_rate_in_minutes}T").first().interpolate(method='slinear')


def remove_zscore_outliers(df, threshold):
    lst_df = []
    for device in set(df['device']):
        df_sel = df.query("device == @device").copy()
        if (~df_sel.index.is_monotonic_increasing):
            df_sel = df_sel.sort_values(by='datetime')

        z_scores = stats.zscore(df_sel['value'], nan_policy='omit')
        lst_df.append(df_sel[abs(z_scores) < threshold])

    return pd.concat(lst_df)


def consumption_from_cummulative_energy(df):
    lst_df = []
    for device in set(df['device']):
        # loop through each device's unique set of variables
        df_sel = df.query("device == @device").sort_index().copy()
        if (~df_sel.index.is_monotonic_increasing):
            df_sel = df_sel.sort_values(by='datetime')

        df_sel = df_sel[~df_sel.index.duplicated(keep='first')]

        df_mono = pd.DataFrame()
        df_mono['value'] = df_sel['value'].diff()
        df_mono['variable'] = 'active_consumption'
        df_mono['device'] = device
        df_mono['description'] = df_sel['description'][0]

        lst_df.append(df_mono)

    return pd.concat(lst_df)


# 2.5th Percentile
def q_low(x):
    return x.quantile(0.025)


# 97.5th Percentile
def q_high(x):
    return x.quantile(0.975)


def hex_to_rgb(hex, alpha):
    if (hex[0] == '#'):
        hex = hex[1:]
    rgb = []
    for i in (0, 2, 4):
        decimal = int(hex[i:i+2], 16)
        rgb.append(decimal)

    return 'rgba'+str(tuple(rgb + [alpha]))


def plot_typical_day_by_hour(df_subset, subset=None, title=None, legend=False, include_ci=False, fill_ci=True):
    idx = 0
    fig = go.Figure()
    for subset_period in set(df_subset[subset]):
        df_plot = df_subset[df_subset[subset] == subset_period]
        hex_color = LST_BIG_COLORS[idx % len(LST_BIG_COLORS)]
        idx += 1

        if (include_ci is True):
            fillcolor = hex_to_rgb(hex_color, 0.2),
            line_color = hex_to_rgb(hex_color, 0.0),
            if (fill_ci is False):
                fillcolor = hex_to_rgb(hex_color, 0.0),
                line_color = hex_to_rgb(hex_color, 0.5),

            fig.add_trace(go.Scatter(
                x=pd.concat([df_plot['hour'], df_plot['hour'][::-1]]),
                y=pd.concat([df_plot['q_high'], df_plot['q_low'][::-1]]),
                fill='toself',
                fillcolor=fillcolor,
                line_color=line_color,
                line=dict(dash='dash'),
                showlegend=legend,
                name=f"Intervalo para el periodo {subset_period}"
            ))

        fig.add_trace(go.Scatter(
            x=df_plot['hour'],
            y=df_plot['mean'],
            line_color=hex_to_rgb(hex_color, 0.75),
            name=f"Promedio para el periodo {subset_period}",
            showlegend=legend,
        ))

    fig.update_layout(
        title=title,
        font_family=cfg.CELSIA_FONT,
        font_size=12,
        font_color=LST_BIG_COLORS[1],
        title_x=0.5,
        width=1250,
        height=550,
        yaxis=dict(title_text="Potencia Activa [kW]"),
        xaxis=dict(
            title_text="Hora del día",
            tickmode='array',
            tickvals=list(range(0, 24)),
            # ticktext = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
        )
    )

    fig.update_traces(mode='lines')
    fig.update_xaxes(rangemode="tozero")
    fig.update_yaxes(rangemode="tozero")
    fig.show()


def plot_typical_week_by_day(df_subset, subset=None, title=None, include_ci=False, fill_ci=True):
    idx = 0

    fig = go.Figure()
    for subset_period in set(df_subset[subset]):
        df_plot = df_subset[df_subset[subset] == subset_period]
        hex_color = LST_BIG_COLORS[idx % len(LST_BIG_COLORS)]
        idx += 1

        if (include_ci is True):
            fillcolor = hex_to_rgb(hex_color, 0.2),
            line_color = hex_to_rgb(hex_color, 0.0),
            if (fill_ci is False):
                fillcolor = hex_to_rgb(hex_color, 0.0),
                line_color = hex_to_rgb(hex_color, 0.5),

            fig.add_trace(go.Scatter(
                x=pd.concat([df_plot['cont_dow'], df_plot['cont_dow'][::-1]]),
                y=pd.concat([df_plot['q_high'], df_plot['q_low'][::-1]]),
                fill='toself',
                fillcolor=fillcolor,
                line_color=line_color,
                line=dict(dash='dash'),
                showlegend=True,
                name=f"Intervalo para el periodo {subset_period}",
            ))

        fig.add_trace(go.Scatter(
            x=df_plot['cont_dow'],
            y=df_plot['mean'],
            line_color=hex_to_rgb(hex_color, 0.75),
            name=f"Promedio para el periodo {subset_period}",
            showlegend=True,
        ))

    fig.update_layout(
        title=title,
        font_family=cfg.CELSIA_FONT,
        font_size=12,
        font_color=LST_BIG_COLORS[1],
        title_x=0.5,
        width=1250,
        height=550,
        yaxis=dict(title_text="Potencia Activa [kW]"),
        xaxis=dict(
            title_text="Día de la semana",
            tickmode='array',
            tickvals=[0, 1, 2, 3, 4, 5, 6, 7],
            ticktext=['Lunes', 'Martes', 'Miércoles',
                      'Jueves', 'Viernes', 'Sábado', 'Domingo']
        )
    )

    fig.update_traces(mode='lines')
    fig.update_xaxes(rangemode="tozero")
    fig.update_yaxes(rangemode="tozero")
    fig.show()


def plot_typical_year_by_week(df_subset, subset=None, title=None, include_ci=False, fill_ci=True):
    idx = 0

    fig = go.Figure()
    for subset_period in set(df_subset[subset]):
        df_plot = df_subset[df_subset[subset] == subset_period]
        hex_color = LST_BIG_COLORS[idx % len(LST_BIG_COLORS)]
        idx += 1

        if (include_ci is True):
            fillcolor = hex_to_rgb(hex_color, 0.2),
            line_color = hex_to_rgb(hex_color, 0.0),
            if (fill_ci is False):
                fillcolor = hex_to_rgb(hex_color, 0.0),
                line_color = hex_to_rgb(hex_color, 0.5),

            fig.add_trace(go.Scatter(
                x=pd.concat([df_plot['week'], df_plot['week'][::-1]]),
                y=pd.concat([df_plot['q_high'], df_plot['q_low'][::-1]]),
                fill='toself',
                fillcolor=fillcolor,
                line_color=line_color,
                line=dict(dash='dash'),
                showlegend=True,
                name=f"Intervalo para el periodo {subset_period}"
            ))

        fig.add_trace(go.Scatter(
            x=df_plot['week'],
            y=df_plot['mean'],
            line_color=hex_to_rgb(hex_color, 0.75),
            name=f"Promedio para el periodo {subset_period}",
            showlegend=True,
        ))

    fig.update_layout(
        title=title,
        font_family=cfg.CELSIA_FONT,
        font_size=12,
        font_color=LST_BIG_COLORS[1],
        title_x=0.5,
        width=1250,
        height=550,
        yaxis=dict(title_text="Potencia Activa [kW]"),
        xaxis=dict(
            title_text="Semana del año",
            # tickmode='array',
            # tickvals=list(range(0, 52)),
        )
    )

    fig.update_traces(mode='lines')
    fig.update_xaxes(rangemode="tozero")
    fig.update_yaxes(rangemode="tozero")
    fig.show()


def plot_cummulative_energy(df_energy):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_energy.index,
        y=df_energy['value'],
        # line_color=hex_to_rgb(hex_color, 0.75),
        # name=f"Promedio para el {year}",
        # showlegend=True,
    ))
    fig.update_layout(
        title=f"Acumulado de Energía Activa",
        font_family=cfg.CELSIA_FONT,
        font_size=12,
        # font_color=colors[1],
        title_x=0.5,
        width=1250,
        height=550,
        yaxis=dict(title_text="Energía Acumulada [kWh]"),
        xaxis=dict(
            title_text="Fecha",
            # tickmode='array',
            # tickvals=list(range(0, 52)),
        )
    )
    fig.update_traces(mode='lines')
    fig.update_xaxes(rangemode="tozero")
    fig.update_yaxes(rangemode="tozero")
    fig.show()


def compare_power_vs_synthetic_power(s_raw, s_synth, device):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=s_raw.index,
        y=s_raw,
        # line_color=hex_to_rgb(hex_color, 0.75),
        name=f"Potencia Activa",
        showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=s_synth.index,
        y=s_synth,
        # line_color=hex_to_rgb(hex_color, 0.75),
        name=f"Potencia Activa Sintétitca",
        showlegend=True,
    ))
    fig.update_layout(
        title=f"Potencia Activa Medida vs Sintética para la carga {device}",
        font_family=cfg.CELSIA_FONT,
        font_size=12,
        # font_color=colors[1],
        title_x=0.5,
        width=1250,
        height=550,
        yaxis=dict(title_text="Potencia Activa [kW]"),
        xaxis=dict(
            title_text="Fecha",
            # tickmode='array',
            # tickvals=list(range(0, 52)),
        )
    )
    fig.update_traces(mode='lines')
    fig.update_xaxes(rangemode="tozero")
    fig.update_yaxes(rangemode="tozero")
    fig.show()


def plot_list_of_series(lst_series, title=None):
    fig = go.Figure()
    for series in lst_series:
        # fig.add_trace(go.Scatter(
        #     mode='lines',
        #     x=series.index,
        #     y=series,
        #     # line_color=hex_to_rgb(hex_color, 0.75),
        #     name=series.name,
        #     showlegend=True,
        # ))

        fig.add_trace(go.Scatter(
            mode='markers',
            x=series.index,
            y=series,
            # line_color=hex_to_rgb(hex_color, 0.75),
            name=series.name,
            showlegend=True,
        ))

    fig.update_layout(
        title=title,
        font_family=cfg.CELSIA_FONT,
        font_size=12,
        # font_color=colors[1],
        title_x=0.5,
        width=1250,
        height=550,
        # yaxis=dict(title_text="Potencia Activa [kW]"),
        xaxis=dict(
            title_text="Fecha",
            # tickmode='array',
            # tickvals=list(range(0, 52)),
        )
    )
    # fig.update_traces(mode='lines')
    fig.update_xaxes(rangemode="tozero")
    fig.update_yaxes(rangemode="tozero")
    fig.show()


def plot_zscores(series):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index,
        y=series,
        # line_color=hex_to_rgb(hex_color, 0.75),
        name=f"Z-Score",
        showlegend=True,
    ))

    fig.update_layout(
        title=f"Z-Score de potencia sintética",
        font_family=cfg.CELSIA_FONT,
        font_size=12,
        # font_color=colors[1],
        title_x=0.5,
        width=1250,
        height=550,
        yaxis=dict(title_text="Z-Score"),
        xaxis=dict(
            title_text="Fecha",
            # tickmode='array',
            # tickvals=list(range(0, 52)),
        )
    )
    fig.update_traces(mode='lines')
    fig.update_xaxes(rangemode="tozero")
    fig.update_yaxes(rangemode="tozero")
    fig.show()


def repair_energy_series(energy, tolerance=None, trust_dips=False):
    # Energy anomalies seem to fall into either local or global shifts.
    # Local shifts can occur in the energy and time axis, while global
    # shifts appear to only occur in the energy axis.

    # A meter change will manifest as a global negative shift in energy.
    # Meaning all energy thereafter is shifted down by the same amount.
    # It is characterised by not presenting with a positive shift of
    # similar magnitude.

    # A local shift in energy is characterised by having both a rising
    # and a falling edge. It is possible to find a start and an end to
    # the anomaly... and, because energy shouldn't decrease, any time
    # it does we know it is an anomal and we can search for a match.

    # If a match is found we can classify it as a local shift. If no
    # match is found it is a global shift. If it's a global shift with
    # a falling edge, it is a meter swap, otherwise it's a "boost".

    # If the local shift is short in duration, finding a match is as
    # simple as looking for flipped edges that are close in magnitude
    # and picking the nearest one (in time axis). If no match can be
    # found ahead of the dip, then we look behind it. It gets tricky
    # when the local shift is long because the energies of the rising
    # and falling edges can fall outside of our tolerance. Simply
    # relaxing the tolerance can leave to bad matches when the
    # anomalies are short, so it is necessary to compensate for the
    # generation during the anomalous period.

    # Time shifts are hard to pick out algorithmically. If they could
    # be detected reliably then it would be a matter of shifting the
    # timestamps until the dE/dt correlated well with the active power.

    # Once global and local anomalies have been removed, imputing the
    # missing data can be done using a combination of integrating the
    # active power, and simple interpolation. Linear interpolation is
    # straightforward, and if the data is missing at random and the gaps
    # are small, then the underlying PDF is approximately preserved.
    # It might be better to interpolate using a ML model that accounts
    # for seasonal behaviors like hour of day and day of week.
    # Just taking the hourly mean is probably a significantly better
    # way of interpolating than linear.This can then be scaled to match
    # the generation during the gap.

    if (~energy.index.is_monotonic_increasing):
        energy = energy.sort_index()

    initial_energy = energy[0]
    delta_e = energy.diff()
    energy_dips = delta_e[delta_e < 0]

    for timestamp_0, dip in energy_dips.iteritems():
        is_subsequent = (delta_e.index > timestamp_0)
        is_previous = (delta_e.index < timestamp_0)
        is_aprox_opposite = (abs(delta_e + dip) < tolerance * -dip)

        subsequent_candidates = delta_e[(is_subsequent & is_aprox_opposite)]
        previous_candidates = delta_e[(is_previous & is_aprox_opposite)]

        timestamp_1 = np.nan
        if (len(subsequent_candidates) > 0):
            # Is left edge. Pick the closest candidate
            # to the right to define the interval
            timestamp_1 = subsequent_candidates.index[0]
        elif (len(previous_candidates) > 0):
            # Is right edge. Pick the closest candidate
            # to the left and swap endpoints
            timestamp_1 = timestamp_0
            timestamp_0 = previous_candidates.index[-1]

        # If timestamp_1 remains NaN then no candidates
        # were found, and it is either a swap or a wall.
        # But because we are only checking for decreases
        # in cummulative energy, walls won't even enter
        # this loop. So it must be a meter swap.

        is_meter_swap = (timestamp_1 is np.nan)

        if is_meter_swap:
            is_after_swap = (energy.index >= timestamp_0)
            # add back the energy accumulated by the previous meter
            energy.loc[is_after_swap] = energy.loc[is_after_swap] - dip
        else:
            if (trust_dips is False):
                is_within_peak_or_valley = (
                    (energy.index >= timestamp_0)
                    & (energy.index < timestamp_1)
                )
                # remove inside peaks or valleys
                energy = energy[~is_within_peak_or_valley]
            else:
                # remove anomalous deltas then cumsum to recover the series
                delta_e = delta_e.drop(index=[timestamp_0, timestamp_1])

    if (trust_dips is True):
        # add back initial energy to recover starting point
        delta_e[0] = initial_energy
        return delta_e.cumsum()
    else:
        return energy


def compare_baseline_day_by_hour(df_bl, df_st, title=None, include_ci=False, fill_ci=True):
    fig = go.Figure()
    if (include_ci is True):
        fillcolor = hex_to_rgb(LST_BIG_COLORS[1], 0.2)
        line_color = hex_to_rgb(LST_BIG_COLORS[1], 0)
        line_style = None
        if (fill_ci is False):
            fillcolor = hex_to_rgb(LST_BIG_COLORS[1], 0)
            line_color = hex_to_rgb(LST_BIG_COLORS[1], 0.5)
            line_style = dict(dash='dash')

        # plot confidence interval
        fig.add_trace(go.Scatter(
            x=pd.concat([df_bl['hour'], df_bl['hour'][::-1]]),
            y=pd.concat([df_bl['q_high'], df_bl['q_low'][::-1]]),
            fill='toself',
            fillcolor=fillcolor,
            line_color=line_color,
            line=line_style,
            showlegend=True,
            name=f"Intervalo de confianza"
        ))

    # plot mean curve
    fig.add_trace(go.Scatter(
        x=df_bl['hour'],
        y=df_bl['mean'],
        line_color=hex_to_rgb(LST_BIG_COLORS[1], 0.75),
        name=f"Promedio histórico (3 meses)",
        showlegend=True,
    ))

    # plot mean curve
    fig.add_trace(go.Scatter(
        x=df_st['hour'],
        y=df_st['mean'],
        line_color=hex_to_rgb(LST_BIG_COLORS[0], 0.75),
        name=f"Promedio actual (últimas 4 semanas)",
        showlegend=True,
    ))

    fig.update_layout(
        title=title,
        font_family=cfg.CELSIA_FONT,
        font_size=12,
        font_color=LST_BIG_COLORS[1],
        title_x=0.5,
        width=1250,
        height=550,
        yaxis=dict(title_text="Potencia Activa [kW]"),
        xaxis=dict(
            title_text="Hora del día",
            tickmode='array',
            tickvals=list(range(0, 24)),
            # ticktext = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
        )
    )

    fig.update_traces(mode='lines')
    fig.update_xaxes(rangemode="tozero")
    fig.update_yaxes(rangemode="tozero")
    fig.show()


def compare_baseline_week_by_day(df_bl, df_st, title=None, include_ci=False, fill_ci=True):
    fig = go.Figure()
    if (include_ci is True):
        fillcolor = hex_to_rgb(LST_BIG_COLORS[1], 0.2)
        line_color = hex_to_rgb(LST_BIG_COLORS[1], 0)
        line_style = None
        if (fill_ci is False):
            fillcolor = hex_to_rgb(LST_BIG_COLORS[1], 0)
            line_color = hex_to_rgb(LST_BIG_COLORS[1], 0.5)
            line_style = dict(dash='dash')

        fig.add_trace(go.Scatter(
            x=pd.concat([df_bl['cont_dow'], df_bl['cont_dow'][::-1]]),
            y=pd.concat([df_bl['q_high'], df_bl['q_low'][::-1]]),
            fill='toself',
            fillcolor=fillcolor,
            line_color=line_color,
            line=line_style,
            showlegend=True,
            name=f"Intervalo de confianza"
        ))

    # plot mean curve
    fig.add_trace(go.Scatter(
        x=df_bl['cont_dow'],
        y=df_bl['mean'],
        line_color=hex_to_rgb(LST_BIG_COLORS[1], 0.75),
        name=f"Promedio histórico (3 meses)",
        showlegend=True,
    ))

    # plot mean curve
    fig.add_trace(go.Scatter(
        x=df_st['cont_dow'],
        y=df_st['mean'],
        line_color=hex_to_rgb(LST_BIG_COLORS[0], 0.75),
        name=f"Promedio actual (últimas 4 semanas)",
        showlegend=True,
    ))

    fig.update_layout(
        title=title,
        font_family=cfg.CELSIA_FONT,
        font_size=12,
        font_color=LST_BIG_COLORS[1],
        title_x=0.5,
        width=1250,
        height=550,
        yaxis=dict(title_text="Potencia Activa [kW]"),
        xaxis=dict(
            title_text="Día de la semana",
            tickmode='array',
            tickvals=[0, 1, 2, 3, 4, 5, 6, 7],
            ticktext=['Lunes', 'Martes', 'Miércoles',
                      'Jueves', 'Viernes', 'Sábado', 'Domingo']
        )
    )

    fig.update_traces(mode='lines')
    fig.update_xaxes(rangemode="tozero")
    fig.update_yaxes(rangemode="tozero")
    fig.show()


def repair_monotonic_increasing_variable(df, max_iter=None, tolerance=None, trust_gaps=False):
    lst_df = []
    for device in set(df['device']):
        df_device = df.query("device == @device")
        for variable in set(df_device['variable']):
            df_device_variable = df_device.query("variable == @variable")

            # store a temporary copy to write over
            repaired_variable = df_device_variable['value'].copy()

            i = 0
            while ((i < max_iter) & (repaired_variable.diff() < 0).any()):
                i += 1
                repaired_variable = repair_energy_series(
                    repaired_variable,
                    tolerance=tolerance,
                    trust_dips=trust_gaps
                )

            df_device_variable_repaired = pd.merge(
                repaired_variable.to_frame(),
                df_device_variable.drop(columns='value'),
                left_index=True,
                right_index=True,
                how='left'
            )

        lst_df.append(df_device_variable_repaired)

    return pd.concat(lst_df)


def recover_energy_from_consumption(df):
    lst_df = []
    for device in set(df['device']):
        df_device = df.query("device == @device")
        for variable in set(df_device['variable']):
            df_device_variable = df_device.query("variable == @variable")

            if (~df_device_variable.index.is_monotonic_increasing):
                df_device_variable = df_device_variable.sort_index()

            df_device_variable['value'] = df_device_variable['value'].cumsum()

            lst_df.append(df_device_variable)

    return pd.concat(lst_df)
