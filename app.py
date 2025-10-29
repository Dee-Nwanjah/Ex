import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests

# Page config
st.set_page_config(
    page_title="COVID-19 Data Explorer",
    page_icon="🦠",
    layout="wide"
)

# Title and description
st.title("🦠 COVID-19 Global Data Explorer")
st.markdown("Interactive dashboard for exploring COVID-19 trends worldwide")

# Function to download COVID data
@st.cache_data # Cache the data loading
def download_covid_data():
    """Download latest COVID-19 data from reliable sources"""
    url = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"
    try:
        # Added Streamlit spinner for user feedback
        with st.spinner("Downloading COVID-19 data..."):
            df = pd.read_csv(url)
        st.success(f"✅ Successfully downloaded {len(df)} records")
        return df
    except Exception as e:
        st.error(f"❌ Error downloading data: {e}")
        # Fallback: create sample data for demonstration
        return create_sample_covid_data()

# Function to create sample COVID data if download fails (copying from previous cell)
def create_sample_covid_data():
    """Create sample COVID data if download fails"""
    st.warning("Creating sample data for demonstration...")
    countries = ['United States', 'United Kingdom', 'Germany', 'France', 'Italy', 'Spain', 'Canada', 'Australia']
    dates = pd.date_date_range('2020-01-01', '2023-12-31', freq='D')
    data = []
    for country in countries:
        for date in dates:
            days_since_start = (date - dates[0]).days
            base_cases = max(0, np.random.poisson(100) * (1 + 0.01 * days_since_start))
            data.append({
                'location': country,
                'date': date,
                'new_cases': max(0, base_cases + np.random.normal(0, 50)),
                'total_cases': None,
                'new_deaths': max(0, np.random.poisson(2)),
                'total_deaths': None,
                'population': np.random.randint(10000000, 350000000)
            })
    df = pd.DataFrame(data)
    for country in countries:
        mask = df['location'] == country
        df.loc[mask, 'total_cases'] = df.loc[mask, 'new_cases'].cumsum()
        df.loc[mask, 'total_deaths'] = df.loc[mask, 'new_deaths'].cumsum()
    return df

# Function to clean and prepare the data (copying from previous cell)
@st.cache_data # Cache the data cleaning
def clean_covid_data(df):
    """Clean and prepare COVID data for analysis."""
    with st.spinner("Cleaning and preparing data..."):
        # Filter for countries only (using the exclude_locations list from the notebook)
        exclude_locations = [
            'World', 'Europe', 'Asia', 'North America', 'South America', 'Africa', 'Oceania',
            'European Union', 'High income', 'Upper middle income', 'Lower middle income', 'Low income',
            'International', 'Wallis and Futuna', 'Timor', 'Tokelau', 'Tuvalu', 'Pitcairn',
            'Niue', 'Nauru', 'Northern Mariana Islands', 'Micronesia (country)', 'Kiribati',
            'Guinea-Bissau', 'Gibraltar', 'Faeroe Islands', 'Eritrea', 'Curacao',
            'Cook Islands', 'Christmas Island', 'Bouvet Island', 'British Virgin Islands',
            'Bonaire Sint Eustatius and Saba', 'Bermuda', 'Aruba', 'Anguilla', 'American Samoa',
            'Andorra', 'Alderney', 'Guernsey', 'Jersey', 'Isle of Man', 'Liechtenstein',
            'Monaco', 'San Marino', 'Vatican', 'Saint Helena', 'Falkland Islands',
            'South Georgia and the South Sandwich Islands', 'Cayman Islands', 'Turks and Caicos Islands',
            'Montserrat', 'Saint Barthelemy', 'Saint Pierre and Miquelon', 'Sint Maarten (Dutch part)',
            'Sint Eustatius', 'Saba', 'Greenland', 'French Guiana', 'Guadeloupe', 'Martinique',
            'Mayotte', 'Reunion', 'Saint Martin (French part)', 'French Polynesia',
            'New Caledonia', 'Norfolk Island', 'Palau', 'Samoa', 'Solomon Islands',
            'Tonga', 'Vanuatu', 'Western Sahara', 'North Korea'
        ]
        df_clean = df[~df['location'].isin(exclude_locations)].copy()
        df_clean['date'] = pd.to_datetime(df_clean['date'])
        df_clean = df_clean.sort_values(['location', 'date']).reset_index(drop=True)

        numeric_cols_to_convert = ['total_cases', 'new_cases', 'total_deaths', 'new_deaths',
                                   'population', 'new_cases_smoothed', 'new_deaths_smoothed',
                                   'total_cases_per_million', 'new_cases_per_million',
                                   'total_deaths_per_million', 'new_deaths_per_million',
                                   'reproduction_rate', 'icu_patients', 'hosp_patients',
                                   'total_vaccinations', 'people_vaccinated',
                                   'people_fully_vaccinated', 'total_boosters', 'new_vaccinations',
                                   'stringency_index', 'population_density', 'median_age',
                                   'aged_65_older', 'aged_70_older', 'gdp_per_capita',
                                   'extreme_poverty', 'cardiovasc_death_rate', 'diabetes_prevalence',
                                   'female_smokers', 'male_smokers', 'handwashing_facilities',
                                   'hospital_beds_per_thousand', 'life_expectancy', 'human_development_index']

        for col in numeric_cols_to_convert:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

        fillna_cols = ['total_cases', 'new_cases', 'total_deaths', 'new_deaths',
                       'new_cases_smoothed', 'new_deaths_smoothed']
        for col in fillna_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean.groupby('location')[col].ffill().fillna(0)

        if 'population' in df_clean.columns:
             df_clean['population'] = df_clean.groupby('location')['population'].ffill().bfill()

        for col in ['total_cases', 'total_deaths']:
            if col in df_clean.columns:
                df_clean[col] = df_clean.groupby('location')[col].cummax()

        if 'population' in df_clean.columns and 'total_cases' in df_clean.columns:
            df_clean['cases_per_million'] = (df_clean['total_cases'] / df_clean['population'] * 1000000)
            df_clean['cases_per_million'] = df_clean['cases_per_million'].replace([np.inf, -np.inf], np.nan).fillna(0)

        if 'population' in df_clean.columns and 'total_deaths' in df_clean.columns:
            df_clean['deaths_per_million'] = (df_clean['total_deaths'] / df_clean['population'] * 1000000)
            df_clean['deaths_per_million'] = df_clean['deaths_per_million'].replace([np.inf, -np.inf], np.nan).fillna(0)

        df_clean['year'] = df_clean['date'].dt.year
        df_clean['month'] = df_clean['date'].dt.month
        df_clean['week'] = pd.to_numeric(df_clean['date'].dt.isocalendar().week, errors='coerce').fillna(0).astype(int)

        df_clean['new_cases_7day_avg'] = df_clean.groupby('location')['new_cases'].rolling(7, min_periods=1).mean().reset_index(0, drop=True)
        df_clean['new_deaths_7day_avg'] = df_clean.groupby('location')['new_deaths'].rolling(7, min_periods=1).mean().reset_index(0, drop=True)

        st.success("✅ Data cleaning complete!")
        return df_clean

# Function to perform advanced analytics (copying from previous cell)
@st.cache_data # Cache the advanced analytics
def analyze_covid_trends_improved(df):
    """Perform advanced trend analysis, including additional metrics and time series features."""
    st.info("Performing advanced trend analysis...")
    df_sorted = df.sort_values(['location', 'date']).copy()

    df_sorted['case_fatality_rate'] = (df_sorted['total_deaths'] / df_sorted['total_cases'] * 100)
    df_sorted['case_fatality_rate'] = df_sorted['case_fatality_rate'].replace([np.inf, -np.inf, pd.NA], np.nan).fillna(0)

    if 'population' in df_sorted.columns:
        df_sorted['new_cases_per_million'] = (df_sorted['new_cases'] / df_sorted['population'] * 1000000)
        df_sorted['new_cases_per_million'] = df_sorted['new_cases_per_million'].replace([np.inf, -np.inf, pd.NA], np.nan).fillna(0)
    else:
        df_sorted['new_cases_per_million'] = 0

    if 'population' in df_sorted.columns:
        df_sorted['new_deaths_per_million'] = (df_sorted['new_deaths'] / df_sorted['population'] * 1000000)
        df_sorted['new_deaths_per_million'] = df_sorted['new_deaths_per_million'].replace([np.inf, -np.inf, pd.NA], np.nan).fillna(0)
    else:
        df_sorted['new_deaths_per_million'] = 0

    df_sorted['cases_growth_rate'] = df_sorted.groupby('location')['total_cases'].pct_change() * 100
    df_sorted['deaths_growth_rate'] = df_sorted.groupby('location')['total_deaths'].pct_change() * 100
    df_sorted['cases_growth_rate'] = df_sorted['cases_growth_rate'].replace([np.inf, -np.inf, pd.NA], np.nan).fillna(0)
    df_sorted['deaths_growth_rate'] = df_sorted['deaths_growth_rate'].replace([np.inf, -np.inf, pd.NA], np.nan).fillna(0)

    df_sorted['new_cases_per_million_30day_avg'] = df_sorted.groupby('location')['new_cases_per_million'].rolling(30, min_periods=1).mean().reset_index(0, drop=True)
    df_sorted['new_deaths_per_million_30day_avg'] = df_sorted.groupby('location')['new_deaths_per_million'].rolling(30, min_periods=1).mean().reset_index(0, drop=True)

    df_sorted['new_cases_per_million_90day_avg'] = df_sorted.groupby('location')['new_cases_per_million'].rolling(90, min_periods=1).mean().reset_index(0, drop=True)
    df_sorted['new_deaths_per_million_90day_avg'] = df_sorted.groupby('location')['new_deaths_per_million'].rolling(90, min_periods=1).mean().reset_index(0, drop=True)

    if 'new_cases_7day_avg' in df_sorted.columns:
         df_sorted['cases_avg_daily_change'] = df_sorted.groupby('location')['new_cases_7day_avg'].diff().fillna(0)
    else:
        df_sorted['cases_avg_daily_change'] = 0

    if 'new_deaths_7day_avg' in df_sorted.columns:
        df_sorted['deaths_avg_daily_change'] = df_sorted.groupby('location')['new_deaths_7day_avg'].diff().fillna(0)
    else:
        df_sorted['deaths_avg_daily_change'] = 0

    cols_to_check = ['cases_growth_rate', 'deaths_growth_rate', 'case_fatality_rate',
                     'new_cases_per_million', 'new_deaths_per_million', 'new_cases_per_million_30day_avg',
                     'new_deaths_per_million_30day_avg', 'new_cases_per_million_90day_avg',
                     'new_deaths_per_million_90day_avg', 'cases_avg_daily_change', 'deaths_avg_daily_change']

    for col in cols_to_check:
         if col in df_sorted.columns:
             df_sorted[col] = pd.to_numeric(df_sorted[col], errors='coerce').replace([np.inf, -np.inf, pd.NA], np.nan).fillna(0)
         else:
              df_sorted[col] = 0

    st.success("✅ Advanced trend analysis complete!")
    return df_sorted

# Function to create the global dashboard (parameterized version)
def create_covid_dashboard_parameterized(df, start_date=None, end_date=None):
    """Create a comprehensive COVID-19 dashboard using Plotly Express with improved visualizations and date filtering."""
    df_filtered = df.copy()
    if start_date:
        df_filtered = df_filtered[df_filtered['date'] >= pd.to_datetime(start_date)]
    if end_date:
        df_filtered = df_filtered[df_filtered['date'] <= pd.to_datetime(end_date)]

    if df_filtered.empty:
        st.warning(f"No data available for the selected date range: {start_date} to {end_date}")
        return go.Figure().update_layout(title_text=f"No data available between {start_date} and {end_date}")

    df_sorted = df_filtered.sort_values(['location', 'date']).copy()
    latest_data = df_sorted.loc[df_sorted.groupby('location')['date'].idxmax()].copy()
    latest_data = latest_data[(latest_data['iso_code'].str.len() == 3) & (latest_data['continent'].notna())].copy()

    if not latest_data['total_cases'].sum() > 0:
         st.warning("Total cases data is zero or missing for latest date in the selected range. Cannot generate Top 15 Countries chart.")
         top_15 = pd.DataFrame(columns=['location', 'total_cases'])
    else:
        top_15 = latest_data.nlargest(15, 'total_cases').reset_index()

    fig1_bar = px.bar(top_15, x='total_cases', y='location', orientation='h',
                      title=f'Top 15 Countries by Total Cases (Cumulative as of {df_filtered["date"].max().strftime("%Y-%m-%d")})',
                      color='total_cases', color_continuous_scale='Viridis',
                      labels={'total_cases': 'Total Confirmed Cases', 'location': 'Country'},
                      hover_data={'total_cases': ':, .0f', 'location': True})
    fig1_bar.update_layout(yaxis={'categoryorder':'total ascending'})

    global_daily_cases = df_sorted.groupby('date')['new_cases_smoothed'].sum().reset_index()
    fig1_line_cases = px.line(global_daily_cases, x='date', y='new_cases_smoothed',
                              title='Global Daily New Cases Trend (7-day smoothed)',
                              labels={'new_cases_smoothed': 'New Cases (7-day smoothed)', 'date': 'Date'},
                              color_discrete_sequence=px.colors.sequential.Plasma,
                              hover_data={'new_cases_smoothed': ':, .0f', 'date': True})

    global_daily_deaths = df_sorted.groupby('date')['new_deaths_smoothed'].sum().reset_index()
    fig1_line_deaths = px.line(global_daily_deaths, x='date', y='new_deaths_smoothed',
                               title='Global Daily New Deaths Trend (7-day smoothed)',
                               labels={'new_deaths_smoothed': 'New Deaths (7-day smoothed)', 'date': 'Date'},
                               color_discrete_sequence=px.colors.sequential.Plasma,
                               hover_data={'new_deaths_smoothed': ':, .0f', 'date': True})

    latest_data['cases_per_million'] = pd.to_numeric(latest_data['cases_per_million'], errors='coerce').fillna(0)
    fig1_map = px.choropleth(latest_data, locations='iso_code', color='cases_per_million',
                             hover_name='location',
                             hover_data={'cases_per_million': ':, .0f', 'total_cases': ':, .0f', 'location': False, 'iso_code': False},
                             color_continuous_scale='Plasma',
                             title=f'Global Cases per Million Population (as of {df_filtered["date"].max().strftime("%Y-%m-%d")})',
                             labels={'cases_per_million': 'Cases per Million'})
    fig1_map.update_layout(geo=dict(showframe=False, showcoastlines=False, projection_type='eckert4'))

    correlation_data = latest_data[(latest_data['total_cases'] > 0) & (latest_data['total_deaths'] > 0) & (latest_data['population'] > 0)].copy()
    fig1_scatter = px.scatter(correlation_data, x='total_cases', y='total_deaths', size='population',
                              color='location', hover_name='location',
                              hover_data={'total_cases': ':, .0f', 'total_deaths': ':, .0f', 'population': ':, .0f'},
                              title=f'Deaths vs Cases Correlation by Country (as of {df_filtered["date"].max().strftime("%Y-%m-%d")})',
                              labels={'total_cases': 'Total Cases', 'total_deaths': 'Total Deaths', 'population': 'Population'},
                              log_x=True, log_y=True, size_max=60)
    fig1_scatter.update_layout(showlegend=False)

    fig1 = make_subplots(rows=3, cols=2,
                         subplot_titles=(
                             f'Top 15 Countries by Total Cases (Cumulative as of {df_filtered["date"].max().strftime("%Y-%m-%d")})',
                             'Global Daily New Cases Trend',
                             'Global Daily New Deaths Trend',
                             f'Global Cases per Million Population (as of {df_filtered["date"].max().strftime("%Y-%m-%d")})',
                             f'Deaths vs Cases Correlation by Country (as of {df_filtered["date"].max().strftime("%Y-%m-%d")})'
                         ),
                         specs=[
                             [{"type": "bar", "colspan": 2}, None],
                             [{"type": "scatter"}, {"type": "scatter"}],
                             [{"type": "choropleth"}, {"type": "scatter"}]
                         ],
                         vertical_spacing=0.1, horizontal_spacing=0.1)

    for data in fig1_bar.data: fig1.add_trace(data, row=1, col=1)
    for data in fig1_line_cases.data: fig1.add_trace(data, row=2, col=1)
    for data in fig1_line_deaths.data: fig1.add_trace(data, row=2, col=2)
    for data in fig1_map.data: fig1.add_trace(data, row=3, col=1)
    for data in fig1_scatter.data: fig1.add_trace(data, row=3, col=2)

    fig1.update_layout(height=1200, title_text="<b>COVID-19 Global Overview Dashboard</b>", title_x=0.5)
    fig1.update_annotations(yshift=-20)
    return fig1

# Function to create country comparison charts (parameterized version)
def create_country_comparison_parameterized(df, countries=['United States', 'United Kingdom', 'Germany', 'France', 'Italy'], start_date=None, end_date=None):
    """Create country comparison charts using Plotly Express with improved visualizations."""
    df_filtered = df.copy()
    country_data = df_filtered[df_filtered['location'].isin(countries)].copy()

    if start_date:
        country_data = country_data[country_data['date'] >= pd.to_datetime(start_date)]
    if end_date:
        country_data = country_data[country_data['date'] <= pd.to_datetime(end_date)]

    if country_data.empty:
        st.warning(f"No data found for selected countries {countries} in the date range {start_date} to {end_date}")
        return go.Figure().update_layout(title_text=f"No data available for selected countries between {start_date} and {end_date}")

    country_data = country_data.sort_values(['location', 'date'])


    fig_cases_avg = px.line(country_data, x='date', y='new_cases_7day_avg', color='location',
                           title='Daily New Cases (7-day average)',
                           labels={'new_cases_7day_avg': 'New Cases (7-day average)', 'date': 'Date', 'location': 'Country'},
                           hover_data={'new_cases_7day_avg': ':, .0f', 'date': True, 'location': True},
                           color_discrete_sequence=px.colors.qualitative.Bold)

    fig_cases_cum = px.line(country_data, x='date', y='total_cases', color='location',
                           title='Cumulative Cases',
                           labels={'total_cases': 'Total Cases', 'date': 'Date', 'location': 'Country'},
                           hover_data={'total_cases': ':, .0f', 'date': True, 'location': True},
                           color_discrete_sequence=px.colors.qualitative.Bold)

    fig_deaths_avg = px.line(country_data, x='date', y='new_deaths_7day_avg', color='location',
                            title='Daily Deaths (7-day average)',
                            labels={'new_deaths_7day_avg': 'New Deaths (7-day average)', 'date': 'Date', 'location': 'Country'},
                            hover_data={'new_deaths_7day_avg': ':, .0f', 'date': True, 'location': True},
                            color_discrete_sequence=px.colors.qualitative.Bold)

    fig_cases_per_million = px.line(country_data, x='date', y='cases_per_million', color='location',
                                   title='Cases per Million Population',
                                   labels={'cases_per_million': 'Cases per Million', 'date': 'Date', 'location': 'Country'},
                                   hover_data={'cases_per_million': ':, .0f', 'total_cases': ':, .0f', 'population': ':, .0f', 'date': True, 'location': True},
                                   color_discrete_sequence=px.colors.qualitative.Bold)


    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=(
                            'Daily New Cases (7-day average)',
                            'Cumulative Cases',
                            'Daily Deaths (7-day average)',
                            'Cases per Million Population'
                        ),
                        specs=[[{"secondary_y": False}, {"secondary_y": False}],
                               [{"secondary_y": False}, {"secondary_y": False}]],
                        vertical_spacing=0.1, horizontal_spacing=0.1)

    for data in fig_cases_avg.data: fig.add_trace(data, row=1, col=1)
    for data in fig_cases_cum.data: fig.add_trace(data, row=1, col=2)
    for data in fig_deaths_avg.data: fig.add_trace(data, row=2, col=1)
    for data in fig_cases_per_million.data: fig.add_trace(data, row=2, col=2)

    fig.update_layout(height=800, title_text=f"<b>COVID-19 Country Comparison Dashboard ({', '.join(countries)})</b>", title_x=0.5, hovermode="x unified")
    fig.update_annotations(yshift=-10)
    return fig

# Function to create vaccination charts (parameterized version)
def create_vaccination_charts_parameterized(df, countries=['United States', 'United Kingdom', 'Germany', 'France', 'Italy'], start_date=None, end_date=None):
    """Create vaccination progress charts for selected countries and date range."""
    df_filtered = df.copy()

    # Filter for selected countries
    vaccination_data = df_filtered[df_filtered['location'].isin(countries)].copy()
    vaccination_data = vaccination_data.sort_values(['location', 'date'])

    # Apply date filtering if start_date and end_date are provided
    if start_date:
        vaccination_data = vaccination_data[vaccination_data['date'] >= pd.to_datetime(start_date)]
    if end_date:
        vaccination_data = vaccination_data[vaccination_data['date'] <= pd.to_datetime(end_date)]


    # Filter out rows with missing vaccination data for plotting
    vaccination_data_filtered = vaccination_data[
        (vaccination_data['total_vaccinations'].notna()) |
        (vaccination_data['people_vaccinated'].notna()) |
        (vaccination_data['people_fully_vaccinated'].notna())
    ].copy()

    if vaccination_data_filtered.empty:
         st.warning(f"No vaccination data available for selected countries {countries} in the date range {start_date} to {end_date}")
         fig = go.Figure()
         fig.update_layout(title_text=f"No vaccination data available for selected countries between {start_date} and {end_date}")
         return fig


    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Total Vaccinations',
            'People Vaccinated (at least 1 dose)',
            'People Fully Vaccinated',
            'People Fully Vaccinated (% of Population)'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]],
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )

    colors = px.colors.qualitative.Bold

    # Total Vaccinations over time
    for i, country in enumerate(countries):
        country_data = vaccination_data_filtered[vaccination_data_filtered['location'] == country].copy()
        if not country_data.empty and country_data['total_vaccinations'].max() > 0: # Only add if data exists and is > 0
             fig.add_trace(
                go.Scatter(
                    x=country_data['date'],
                    y=country_data['total_vaccinations'],
                    mode='lines',
                    name=country,
                    line=dict(color=colors[i % len(colors)]),
                    hovertemplate='<b>%{text}</b><br>Date: %{x|%Y-%m-%d}<br>Total Vaccinations: %{y:,.0f}<extra></extra>',
                    text = country_data['location'] # Use location for hover text
                ),
                row=1, col=1
            )

    # People Vaccinated (at least 1 dose) over time
    for i, country in enumerate(countries):
        country_data = vaccination_data_filtered[vaccination_data_filtered['location'] == country].copy()
        if not country_data.empty and country_data['people_vaccinated'].max() > 0:
            fig.add_trace(
                go.Scatter(
                    x=country_data['date'],
                    y=country_data['people_vaccinated'],
                    mode='lines',
                    name=country,
                    showlegend=(i == 0), # Show legend only once
                    line=dict(color=colors[i % len(colors)]),
                     hovertemplate='<b>%{text}</b><br>Date: %{x|%Y-%m-%d}<br>People Vaccinated: %{y:,.0f}<extra></extra>',
                     text = country_data['location']
                ),
                row=1, col=2
            )

    # People Fully Vaccinated over time
    for i, country in enumerate(countries):
        country_data = vaccination_data_filtered[vaccination_data_filtered['location'] == country].copy()
        if not country_data.empty and country_data['people_fully_vaccinated'].max() > 0:
            fig.add_trace(
                go.Scatter(
                    x=country_data['date'],
                    y=country_data['people_fully_vaccinated'],
                    mode='lines',
                    name=country,
                    showlegend=(i == 0),
                    line=dict(color=colors[i % len(colors)]),
                    hovertemplate='<b>%{text}</b><br>Date: %{x|%Y-%m-%d}<br>People Fully Vaccinated: %{y:,.0f}<extra></extra>',
                    text = country_data['location']
                ),
                row=2, col=1
            )

    # People Fully Vaccinated (% of Population) over time
    # Calculate percentage - ensure population is not zero
    vaccination_data_filtered['people_fully_vaccinated_pct'] = (
        vaccination_data_filtered['people_fully_vaccinated'] / vaccination_data_filtered['population'] * 100
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    for i, country in enumerate(countries):
        country_data = vaccination_data_filtered[vaccination_data_filtered['location'] == country].copy()
        if not country_data.empty and country_data['people_fully_vaccinated_pct'].max() > 0:
             fig.add_trace(
                go.Scatter(
                    x=country_data['date'],
                    y=country_data['people_fully_vaccinated_pct'],
                    mode='lines',
                    name=country,
                    showlegend=(i == 0),
                    line=dict(color=colors[i % len(colors)]),
                    hovertemplate='<b>%{text}</b><br>Date: %{x|%Y-%m-%d}<br>Fully Vaccinated: %{y:.2f} %<extra></extra>',
                    text = country_data['location']
                ),
                row=2, col=2
            )
    fig.update_yaxes(range=[0, 100], row=2, col=2) # Set y-axis to 0-100%


    # Update subplot titles and layout
    fig.update_layout(
        height=800, # Set height here
        title_text=f"<b>COVID-19 Vaccination Progress by Country ({', '.join(countries)})</b>",
        title_x=0.5,
        hovermode="x unified"
    )
    fig.update_annotations(yshift=-10)

    # Update axis labels
    fig.update_xaxes(title_text='Date', row=1, col=1)
    fig.update_yaxes(title_text='Count', row=1, col=1)
    fig.update_xaxes(title_text='Date', row=1, col=2)
    fig.update_yaxes(title_text='Count', row=1, col=2)
    fig.update_xaxes(title_text='Date', row=2, col=1)
    fig.update_yaxes(title_text='Count', row=2, col=1)
    fig.update_xaxes(title_text='Date', row=2, col=2)
    fig.update_yaxes(title_text='Percentage (%)', row=2, col=2)

    return fig

# Function to create demographic correlation charts (parameterized version)
def create_demographic_correlation_charts_parameterized(df, start_date=None, end_date=None):
    """Create scatter plots exploring correlations between COVID-19 metrics and demographic/health factors (latest data within date range)."""
    df_filtered = df.copy()
    if start_date:
        df_filtered = df_filtered[df_filtered['date'] >= pd.to_datetime(start_date)]
    if end_date:
        df_filtered = df_filtered[df_filtered['date'] <= pd.to_datetime(end_date)]

    if df_filtered.empty:
        st.warning(f"No data available for the selected date range: {start_date} to {end_date} for demographic correlation plots.")
        return go.Figure().update_layout(title_text=f"No data available between {start_date} and {end_date}")

    latest_data = df_filtered.loc[df_filtered.groupby('location')['date'].idxmax()].copy()
    latest_data = latest_data[(latest_data['iso_code'].str.len() == 3) & (latest_data['continent'].notna())].copy()

    if latest_data.empty:
         st.warning(f"No latest country data available for demographic correlation plots as of {df_filtered['date'].max().strftime('%Y-%m-%d')}")
         return go.Figure().update_layout(title_text=f"No data available for demographic correlation plots as of {df_filtered['date'].max().strftime('%Y-%m-%d')}")

    correlation_pairs = [
        ('median_age', 'cases_per_million', 'Cases per Million vs Median Age'),
        ('median_age', 'deaths_per_million', 'Deaths per Million vs Median Age'),
        ('gdp_per_capita', 'cases_per_million', 'Cases per Million vs GDP per Capita'),
        ('gdp_per_capita', 'deaths_per_million', 'Deaths per Million vs GDP per Capita'),
        ('hospital_beds_per_thousand', 'cases_per_million', 'Cases per Million vs Hospital Beds per Thousand'),
        ('hospital_beds_per_thousand', 'deaths_per_million', 'Deaths per Million vs Hospital Beds per Thousand'),
        ('life_expectancy', 'cases_per_million', 'Cases per Million vs Life Expectancy'),
        ('life_expectancy', 'deaths_per_million', 'Deaths per Million vs Life Expectancy'),
        ('cardiovasc_death_rate', 'deaths_per_million', 'Deaths per Million vs Cardiovascular Death Rate'),
        ('diabetes_prevalence', 'deaths_per_million', 'Deaths per Million vs Diabetes Prevalence'),
    ]

    available_pairs = [(x, y, title) for x, y, title in correlation_pairs if x in latest_data.columns and y in latest_data.columns]

    if not available_pairs:
        st.warning("None of the target correlation columns are available in the data.")
        return go.Figure().update_layout(title_text="No relevant columns for demographic correlation plots")

    n_cols = 2
    n_rows = (len(available_pairs) + n_cols - 1) // n_cols

    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=[pair[2] for pair in available_pairs],
                        specs=[[{"type": "scatter"}] * n_cols] * n_rows,
                        vertical_spacing=0.1, horizontal_spacing=0.1)

    continent_colors = px.colors.qualitative.Alphabet

    for i, (x_col, y_col, title) in enumerate(available_pairs):
        row = (i // n_cols) + 1
        col_idx = (i % n_cols) + 1

        # Filter out rows with NaN in the relevant columns for plotting
        plot_data = latest_data[latest_data[x_col].notna() & latest_data[y_col].notna()].copy()

        if not plot_data.empty:
             scatter_plot = go.Scatter(x=plot_data[x_col], y=plot_data[y_col], mode='markers',
                                     text=plot_data['location'], # Use country name as text label on hover
                                     marker=dict(size=8, color=[continent_colors[c_code % len(continent_colors)] for c_code in plot_data['continent'].astype('category').cat.codes]),
                                     name=title, # Name for potential legend
                                     showlegend=False, # Hide individual trace legends
                                     hovertemplate=f'<b>%{{text}}</b><br>{title.split(" vs ")[0]}: %{{x}}<br>{title.split(" vs ")[1]}: %{{y:,.0f}}<extra></extra>' # Improved hover
            )
             fig.add_trace(scatter_plot, row=row, col=col_idx)
             fig.update_xaxes(title_text=title.split(" vs ")[0], row=row, col=col_idx) # Set x-axis title
             fig.update_yaxes(title_text=title.split(" vs ")[1], row=row, col=col_idx) # Set y-axis title
        else:
            st.warning(f"No valid data for correlation plot: {title} in the selected date range.")

    fig.update_layout(height=400 * n_rows,
                      title_text=f"<b>COVID-19 Correlation Analysis with Demographic and Health Factors (as of {df_filtered['date'].max().strftime('%Y-%m-%d')})</b>",
                      title_x=0.5, showlegend=True, legend_title_text='Continent')
    return fig

# Function to create distribution charts (parameterized version)
def create_distribution_charts_parameterized(df, start_date=None, end_date=None):
    """Create distribution plots for key COVID-19 metrics (latest data within date range)."""
    df_filtered = df.copy()
    if start_date:
        df_filtered = df_filtered[df_filtered['date'] >= pd.to_datetime(start_date)]
    if end_date:
        df_filtered = df_filtered[df_filtered['date'] <= pd.to_datetime(end_date)]

    if df_filtered.empty:
        st.warning(f"No data available for the selected date range: {start_date} to {end_date} for distribution plots.")
        return go.Figure().update_layout(title_text=f"No data available between {start_date} and {end_date}")

    latest_data = df_filtered.loc[df_filtered.groupby('location')['date'].idxmax()].copy()
    latest_data = latest_data[(latest_data['iso_code'].str.len() == 3) & (latest_data['continent'].notna())].copy()

    if latest_data.empty:
         st.warning(f"No latest country data available for distribution plots as of {df_filtered['date'].max().strftime('%Y-%m-%d')}")
         return go.Figure().update_layout(title_text=f"No data available for distribution plots as of {df_filtered['date'].max().strftime('%Y-%m-%d')}")


    dist_cols = ['cases_per_million', 'deaths_per_million', 'stringency_index', 'reproduction_rate']
    dist_titles = ['Cases per Million Population', 'Deaths per Million Population', 'Stringency Index', 'Reproduction Rate']

    # Filter out columns that don't exist in the dataframe
    available_cols = [col for col in dist_cols if col in latest_data.columns]
    available_titles = [title for col, title in zip(dist_cols, dist_titles) if col in latest_data.columns]


    if not available_cols:
        st.warning("None of the target distribution columns are available.")
        fig = go.Figure()
        fig.update_layout(title_text="No relevant columns for distribution plots")
        return fig


    # Determine number of rows and columns for subplots
    n_cols = 2
    n_rows = (len(available_cols) + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=available_titles,
        # Use type 'xy' for box plots
         specs=[[{"type": "xy"}] * n_cols] * n_rows,
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )

    for i, col in enumerate(available_cols):
        row = (i // n_cols) + 1
        col_idx = (i % n_cols) + 1

        # Filter out NaN and negative values for meaningful distribution plots (allow zero)
        plot_data = latest_data[latest_data[col].notna() & (latest_data[col] >= 0)].copy()

        if not plot_data.empty:
            # Use Box plot to show distribution and outliers
            box_plot = go.Box(
                y=plot_data[col],
                name=available_titles[i], # Use title as name for legend/hover
                boxpoints='outliers', # Show outlier points
                jitter=0.3, # Add jitter to outlier points
                pointpos=-1.8, # Position outlier points
                 hovertemplate=f'{available_titles[i]}: %{{y:,.2f}}<extra></extra>' # Improved hover
            )
            fig.add_trace(box_plot, row=row, col=col_idx)
            # Fixed the positional argument error here by moving row and col_idx to be keyword arguments
            fig.update_yaxes(title_text=available_titles[i], row=row, col=col_idx)
        else:
            st.warning(f"No valid data for distribution plot of {available_titles[i]} in the selected date range.")


    fig.update_layout(
        height=400 * n_rows, # Set height here
        title_text=f"<b>Distribution of Key COVID-19 Metrics (as of {df_filtered['date'].max().strftime('%Y-%m-%d')})</b>",
        title_x=0.5,
        showlegend=False # Hide legend as name is in subplot title
    )


    return fig


# Function to create advanced analytical charts (parameterized version)
def create_advanced_charts_parameterized(df, countries=['United States', 'United Kingdom', 'Germany', 'France', 'Italy'], start_date=None, end_date=None):
    """Create advanced analytical charts using the enhanced advanced data with country and date filtering."""
    df_filtered = df.copy()
    country_data_filtered = df_filtered[df_filtered['location'].isin(countries)].copy()

    # Apply date filtering if start_date and end_date are provided
    if start_date:
        country_data_filtered = country_data_filtered[country_data_filtered['date'] >= pd.to_datetime(start_date)]
    if end_date:
        country_data_filtered = country_data_filtered[country_data_filtered['date'] <= pd.to_datetime(end_date)]


    if country_data_filtered.empty:
        st.warning(f"No data found for selected countries {countries} in the date range {start_date} to {end_date} for advanced charts.")
        fig = go.Figure()
        fig.update_layout(title_text=f"No data available for selected countries between {start_date} and {end_date}")
        return fig


    # Ensure data is sorted
    df_sorted = country_data_filtered.sort_values(['location', 'date']).copy()


    # Define the number of rows and columns for the subplot grid
    n_rows = 3
    n_cols = 2

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=(
            'Cases Growth Rate Trends (7-day rolling avg)',
            f'Case Fatality Rate (%) by Country (as of {df_sorted["date"].max().strftime("%Y-%m-%d")})',
            'Daily New Cases per Million (7-day average)',
            'Daily New Deaths per Million (7-day average)',
            f'Cases vs Population (as of {df_sorted["date"].max().strftime("%Y-%m-%d")})',
            f'Deaths vs Population (as of {df_sorted["date"].max().strftime("%Y-%m-%d")})'
        ),
         specs=[
             [{"secondary_y": False}, {"secondary_y": False}],
             [{"secondary_y": False}, {"secondary_y": False}],
             [{"secondary_y": False}, {"secondary_y": False}]
         ],
        vertical_spacing=0.08,
        horizontal_spacing=0.08
    )

    # Select countries for detailed line plots (use the filtered list)
    top_countries_analysis = countries # Use the provided list of countries

    colors = px.colors.qualitative.Set1 # Use a standard color set

    if not top_countries_analysis:
        st.warning("No countries selected for detailed advanced analysis charts.")
        fig.update_layout(title_text="No countries selected for advanced charts")
        return fig


    for i, country in enumerate(top_countries_analysis):
        country_data = df_sorted[df_sorted['location'] == country].copy()

        if country_data.empty:
            st.warning(f"No data found for {country} in the selected date range for advanced charts.")
            continue # Skip if no data for the country

        # Growth rate trends (using 7-day rolling mean of the growth rate)
        # Ensure 'cases_growth_rate' exists and is numeric before rolling mean
        if 'cases_growth_rate' in country_data.columns:
             # Ensure data is numeric and replace inf/NaN before rolling
             country_data['cases_growth_rate_smoothed'] = pd.to_numeric(country_data['cases_growth_rate'], errors='coerce').replace([np.inf, -np.inf], np.nan).rolling(7, min_periods=1).mean().fillna(0)
        else:
             country_data['cases_growth_rate_smoothed'] = 0 # Add column if missing


        fig.add_trace(
            go.Scatter(
                x=country_data['date'],
                y=country_data['cases_growth_rate_smoothed'],
                mode='lines',
                name=f'{country}', # Simplified name for legend consistency
                line=dict(color=colors[i % len(colors)]), # Cycle through colors
                hovertemplate=f'<b>{country}</b><br>Date: %{{x|%Y-%m-%d}}<br>Growth Rate (7-day avg): %{{y:.2f}}%<extra></extra>' # Improved hover
            ),
            row=1, col=1
        )

        # Case fatality rate (Latest value) - Use bar chart as before
        if not country_data.empty and 'case_fatality_rate' in country_data.columns:
            latest_cfr_data = country_data.iloc[-1] # Get the latest row within the date range
            latest_cfr = latest_cfr_data.get('case_fatality_rate', 0)

            fig.add_trace(
                go.Bar(
                    x=[country],
                    y=[latest_cfr],
                    name=country,
                    marker_color=colors[i % len(colors)],
                    showlegend=False, # Hide legend here, shown in row 1 col 1
                    hovertemplate=f'<b>{country}</b><br>Latest CFR: %{{y:.2f}}%<extra></extra>'
                ),
                row=1, col=2
            )
        elif not country_data.empty:
             st.warning(f"'case_fatality_rate' column not found for {country}. Cannot plot CFR.")


        # Daily New Cases per Million (7-day average)
        # Ensure 'new_cases_per_million' exists and is numeric before rolling
        if 'new_cases_per_million' in country_data.columns:
             country_data['new_cases_per_million_7day_avg'] = pd.to_numeric(country_data['new_cases_per_million'], errors='coerce').rolling(7, min_periods=1).mean().fillna(0)
        else:
            country_data['new_cases_per_million_7day_avg'] = 0


        fig.add_trace(
             go.Scatter(
                x=country_data['date'],
                y=country_data['new_cases_per_million_7day_avg'],
                mode='lines',
                name=f'{country}', # Use same name for legend consistency
                line=dict(color=colors[i % len(colors)]),
                showlegend=False, # Hide legend here, shown in row 1 col 1
                hovertemplate=f'<b>{country}</b><br>Date: %{{x|%Y-%m-%d}}<br>New Cases per Million (7-day avg): %{{y:.2f}}<extra></extra>' # Improved hover
            ),
            row=2, col=1
        )

        # Daily New Deaths per Million (7-day average)
        # Ensure 'new_deaths_per_million' exists and is numeric before rolling
        if 'new_deaths_per_million' in country_data.columns:
            country_data['new_deaths_per_million_7day_avg'] = pd.to_numeric(country_data['new_deaths_per_million'], errors='coerce').rolling(7, min_periods=1).mean().fillna(0)
        else:
            country_data['new_deaths_per_million_7day_avg'] = 0


        fig.add_trace(
             go.Scatter(
                x=country_data['date'],
                y=country_data['new_deaths_per_million_7day_avg'],
                mode='lines',
                name=f'{country}', # Use same name for legend consistency
                line=dict(color=colors[i % len(colors)]),
                showlegend=False, # Hide legend here, shown in row 1 col 1
                hovertemplate=f'<b>{country}</b><br>Date: %{{x|%Y-%m-%d}}<br>New Deaths per Million (7-day avg): %{{y:.2f}}<extra></extra>' # Improved hover
            ),
            row=2, col=2
        )


    # Population vs cases scatter (Latest Data within date range)
    latest_data_scatter = df_sorted[df_sorted['date'] == df_sorted['date'].max()].copy()
    # Filter out rows with missing or zero population or total cases for scatter plot
    scatter_data_cases = latest_data_scatter[(latest_data_scatter['population'].notna()) & (latest_data_scatter['population'] > 0) &
                                     (latest_data_scatter['total_cases'].notna()) & (latest_data_scatter['total_cases'] > 0) &
                                     (latest_data_scatter['iso_code'].str.len() == 3) & (latest_data_scatter['continent'].notna())].copy() # Ensure country data

    if not scatter_data_cases.empty:
         # Ensure 'life_expectancy' exists before using it for color
         marker_color = scatter_data_cases['life_expectancy'] if 'life_expectancy' in scatter_data_cases.columns else None
         colorbar_title = 'Life Expectancy' if 'life_expectancy' in scatter_data_cases.columns else ''

         fig.add_trace(
            go.Scatter(
                x=scatter_data_cases['population'],
                y=scatter_data_cases['total_cases'],
                mode='markers',
                text=scatter_data_cases['location'],
                marker=dict(
                    size=np.log10(scatter_data_cases['total_cases'] + 1) * 5, # Size markers by log of cases
                    color=marker_color, # Color by life expectancy or None
                    colorscale='Viridis' if marker_color is not None else 'Viridis', # Default colorscale if no color data
                    showscale=marker_color is not None, # Show scale only if color data is available
                    colorbar=dict(title=colorbar_title, x=1.08) if marker_color is not None else None # Add color bar title and position
                ),
                name='Cases vs Population', # Name for potential legend
                showlegend=False, # Hide individual trace legends
                # Adjusted hover template based on whether color data is available
                hovertemplate=f'<b>%{{text}}</b><br>Population: %{{x:,.0f}}<br>Total Cases: %{{y:,.0f}}' + (f'<br>{colorbar_title}: %{{marker.color:.1f}}' if marker_color is not None else '') + '<extra></extra>'
            ),
            row=3, col=1
        )
         fig.update_xaxes(title_text='Population (Log Scale)', type='log', row=3, col=1) # Use log scale for scatter
         fig.update_yaxes(title_text='Total Cases (Log Scale)', type='log', row=3, col=1) # Use log scale for scatter
    else:
        st.warning("No valid data for Population vs Cases scatter plot in the selected date range.")

    # Population vs deaths scatter (Latest Data within date range)
    scatter_data_deaths = latest_data_scatter[(latest_data_scatter['population'].notna()) & (latest_data_scatter['population'] > 0) &
                                      (latest_data_scatter['total_deaths'].notna()) & (latest_data_scatter['total_deaths'] > 0) & # Fixed typo here
                                      (latest_data_scatter['iso_code'].str.len() == 3) & (latest_data_scatter['continent'].notna())].copy() # Ensure country data


    if not scatter_data_deaths.empty:
         # Ensure 'cardiovasc_death_rate' exists before using it for color
         marker_color_deaths = scatter_data_deaths['cardiovasc_death_rate'] if 'cardiovasc_death_rate' in scatter_data_deaths.columns else None
         colorbar_title_deaths = 'Cardiovascular<br>Death Rate' if 'cardiovasc_death_rate' in scatter_data_deaths.columns else ''

         fig.add_trace(
            go.Scatter(
                x=scatter_data_deaths['population'],
                y=scatter_data_deaths['total_deaths'],
                mode='markers',
                text=scatter_data_deaths['location'],
                marker=dict(
                    size=np.log10(scatter_data_deaths['total_deaths'] + 1) * 5, # Size markers by log of deaths
                    color=marker_color_deaths, # Color by cardiovascular death rate or None
                    colorscale='Plasma' if marker_color_deaths is not None else 'Plasma', # Different colorscale
                    showscale=marker_color_deaths is not None, # Show scale only if color data is available
                    colorbar=dict(title=colorbar_title_deaths, x=1.08) if marker_color_deaths is not None else None # Add color bar title and position
                ),
                name='Deaths vs Population', # Name for potential legend
                showlegend=False, # Hide individual trace legends
                # Adjusted hover template based on whether color data is available
                hovertemplate=f'<b>%{{text}}</b><br>Population: %{{x:,.0f}}<br>Total Deaths: %{{y:,.0f}}' + (f'<br>{colorbar_title_deaths}: %{{marker.color:.1f}}' if marker_color_deaths is not None else '') + '<extra></extra>'
            ),
            row=3, col=2
        )
         fig.update_xaxes(title_text='Population (Log Scale)', type='log', row=3, col=2) # Use log scale for scatter
         fig.update_yaxes(title_text='Total Deaths (Log Scale)', type='log', row=3, col=2) # Use log scale for scatter
    else:
        st.warning("No valid data for Population vs Deaths scatter plot in the selected date range.")


    # Update layout for better titles and spacing
    fig.update_layout(
        height=1200, # Increased height to accommodate the new row
        title_text=f"<b>COVID-19 Advanced Analytics and Trends ({', '.join(countries)})</b>",
        title_x=0.5,
        hovermode="x unified", # Unified hover for line charts
        legend_title_text='Country', # Add legend title for country lines
        legend=dict(traceorder='reversed', yanchor='top', y=0.98, xanchor='left', x=0.01) # Position legend
    )
    fig.update_annotations(yshift=-15) # Adjust subplot title positioning

    # Update axis labels for consistency
    fig.update_xaxes(title_text='Date', row=1, col=1)
    fig.update_yaxes(title_text='Growth Rate (%)', row=1, col=1)

    fig.update_xaxes(title_text='Country', row=1, col=2)
    fig.update_yaxes(title_text='Case Fatality Rate (%)', row=1, col=2)

    fig.update_xaxes(title_text='Date', row=2, col=1)
    fig.update_yaxes(title_text='Cases per Million (7-day avg)', row=2, col=1)

    fig.update_xaxes(title_text='Date', row=2, col=2)
    fig.update_yaxes(title_text='Deaths per Million (7-day avg)', row=2, col=2)


    return fig


# Streamlit App Layout
st.set_page_config(layout="wide", page_title="COVID-19 Data Explorer")

st.title("COVID-19 Data Explorer")
st.markdown("Explore global and country-specific COVID-19 data and trends.")

# Load and prepare data
covid_df = download_covid_data()
covid_clean = clean_covid_data(covid_df)
covid_advanced = analyze_covid_trends_improved(covid_clean)

# Get date range from the data
min_date = covid_clean['date'].min().to_pydatetime()
max_date = covid_clean['date'].max().to_pydatetime()

# Sidebar for user inputs
st.sidebar.header("Filter Data and Select Charts")

# Date range slider
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if len(date_range) == 2:
    start_date = date_range[0]
    end_date = date_range[1]
elif len(date_range) == 1:
    start_date = date_range[0]
    end_date = max_date # Default end date if only one date is selected
else:
    start_date = min_date
    end_date = max_date


# Country selection for country-specific charts
all_countries = sorted(covid_clean['location'].unique().tolist())
default_countries = ['United States', 'United Kingdom', 'Germany', 'France', 'Italy']
selected_countries = st.sidebar.multiselect(
    "Select Countries for Comparison Charts",
    options=all_countries,
    default=default_countries
)

# Chart selection
chart_options = [
    "Global Overview Dashboard",
    "Country Comparison Charts",
    "Vaccination Progress Charts",
    "Advanced Analytics Charts",
    "Demographic Correlation Plots",
    "Distribution Plots"
]
selected_charts = st.sidebar.multiselect(
    "Select Charts to Display",
    options=chart_options,
    default=chart_options[:2] # Default to showing Global and Country Comparison
)

st.sidebar.markdown("---")
st.sidebar.info("Data Source: Our World in Data")

# Display selected charts based on user selection
if "Global Overview Dashboard" in selected_charts:
    st.header("Global Overview")
    global_fig = create_covid_dashboard_parameterized(covid_clean, start_date, end_date)
    st.plotly_chart(global_fig, use_container_width=True)

if "Country Comparison Charts" in selected_charts:
    if selected_countries:
        st.header(f"Country Comparison ({', '.join(selected_countries)})")
        country_comparison_fig = create_country_comparison_parameterized(covid_clean, selected_countries, start_date, end_date)
        st.plotly_chart(country_comparison_fig, use_container_width=True)
    else:
        st.warning("Please select at least one country for the Country Comparison Charts.")

if "Vaccination Progress Charts" in selected_charts:
    if selected_countries:
        st.header(f"Vaccination Progress ({', '.join(selected_countries)})")
        vaccination_fig = create_vaccination_charts_parameterized(covid_clean, selected_countries, start_date, end_date)
        st.plotly_chart(vaccination_fig, use_container_width=True)
    else:
        st.warning("Please select at least one country for the Vaccination Progress Charts.")

if "Advanced Analytics Charts" in selected_charts:
     if selected_countries:
        st.header(f"Advanced Analytics and Trends ({', '.join(selected_countries)})")
        # Use covid_advanced for advanced charts
        advanced_fig = create_advanced_charts_parameterized(covid_advanced, selected_countries, start_date, end_date)
        st.plotly_chart(advanced_fig, use_container_width=True)
     else:
        st.warning("Please select at least one country for the Advanced Analytics Charts.")


if "Demographic Correlation Plots" in selected_charts:
    st.header("Demographic Correlation Analysis")
    demographic_fig = create_demographic_correlation_charts_parameterized(covid_advanced, start_date, end_date) # Use covid_advanced
    st.plotly_chart(demographic_fig, use_container_width=True)

if "Distribution Plots" in selected_charts:
    st.header("Distribution of Key Metrics")
    distribution_fig = create_distribution_charts_parameterized(covid_advanced, start_date, end_date) # Use covid_advanced
    st.plotly_chart(distribution_fig, use_container_width=True)

st.markdown("---")
st.markdown("Created with Streamlit and Plotly")
