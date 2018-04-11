import pandas as pd
import logging
import numpy as np
import os

from bokeh.io import output_file, curdoc
from bokeh.layouts import widgetbox, row, column
from bokeh.models.widgets import Select
from bokeh.plotting import figure
from bokeh.models import LinearAxis, Range1d

def load_prediction_data(filename, sep=','):
    '''Load prediction data that was generated from the house_price_forecast.py file

    Args:
        filename (string):
        sep (string): Default ';'
    '''
    logging.info('Loading family data')
    family = pd.read_csv(filename, sep)
    return family

def format_for_visualization(df):
    '''Format the prediction data so that it can be easily used in bokeh visualization.
    '''
    # get family size
    #family_avg_size = pd.pivot_table(df, index=['year'], values=['avg_size_of_family'],
    #                    columns=['area'], aggfunc=np.mean)
    #family_avg_size = family_avg_size.T.reset_index().rename(columns={'level_0':
    #                    'information'}).groupby(['area','information']).mean().T

    # get family nubmers
    #family_number = pd.pivot_table(df, index=['year'], values=['num_of_families'],
    #                    columns=['area'], aggfunc=np.mean)
    #family_number = family_number.T.reset_index().rename(columns={'level_0':
    #                    'information' }).groupby(['area','information']).mean().T

    # get price
    price = pd.pivot_table(df, index=['year'], values=['price_per_square_meter'],
                        columns=['area', 'information'], aggfunc=np.sum)
    price = price.T.reset_index().drop(columns=['level_0']).groupby(['area',
                        'information']).mean().T
    price.columns.set_levels(price.columns.levels[1] + '_price', level=1, inplace=True)

    # get sale observations
    observations = pd.pivot_table(df, index=['year'], values=['observations'],
                        columns=['area', 'information'], aggfunc=np.sum)
    observations = observations.T.reset_index().drop(columns=['level_0']).groupby(
                        ['area','information']).mean().T
    #TODO somehow NaN values become 0 after this step, which leads to Visualization issue.
    # Need to be investigate further
    observations.loc[observations.index == 2017,:] = np.nan
    observations.columns.set_levels(observations.columns.levels[1] + '_observations', level=1, inplace=True)

    # join the dataframes
    #df = pd.concat([family_number, family_avg_size, price, observations],
    df = pd.concat([price, observations],
            axis=1).T.reset_index().sort_values('area').set_index(['area', 'information']).T
    return df

def update(attr, old, new):
    '''Update function that is used by bokeh to update the web interface when there is
    change on value.
    '''
    layout.children[0].children[0] = create_figure_house_price()
    layout.children[1].children[0] = create_figure_observations()
    #layout.children[1].children[1] = create_figure_family()

def get_house_data(t_area, t_type):
    '''Used to fetch house price information for a given area (city or Finland) and type.
    '''
    logging.info('Getting house data')
    columns = []
    if t_type == 'both':
        t_type = ['block_of_flats','single_family_houses']
    else:
        t_type = [t_type]
    for ttype in t_type:
        for tinfo in ['price','observations']:
            columns.append(ttype + '_' + tinfo)
    rtn = dd[t_area][columns]
    return rtn

def get_family_data(t_area):
    '''Used to fetch family size and number of family for the given area (city or Finland).
    '''
    logging.info('Getting family data')
    rtn = dd[t_area][['avg_size_of_family', 'num_of_families']]
    return rtn

def create_figure_house_price():
    '''Create chart figure for house price.
    '''
    logging.info('Creating bokeh house price figure')
    t_area, t_type = house_area_ticker.value, house_type_ticker.value
    logging.info('Area: {} type: {}'.format(t_area, t_type))
    data = get_house_data(t_area, t_type)
    tools = 'pan,wheel_zoom,box_select,reset'
    # columns of observations and price of each type of houses
    column_price = [column for column in data.columns if column in
                        ['block_of_flats_price', 'single_family_houses_price']]
    # ylimit ranges
    ymin = np.min(data[column_price].min(axis=0))
    ymax = np.max(data[column_price].max())
    ystd = np.max(data[column_price].std())
    # create a figure
    fig = figure(plot_width=400, plot_height=400, y_range=(ymin - 2*ystd, ymax + 2*ystd))
    # set figure parameters
    fig.xaxis.axis_label = 'Year'
    fig.xaxis.axis_label_text_color = 'firebrick'
    fig.xaxis.axis_label_standoff = 30
    fig.yaxis.axis_label = 'Price per square meter (EUR)'
    fig.yaxis.axis_label_text_color = 'firebrick'
    fig.title.text = 'Hosue price of %s' % (t_area)

    colors = ['hotpink', 'darkorange', 'yellowgreen', 'lightseagreen',
                    'deepskyblue', 'slateblue', 'darkmagenta']
    # plot price of the selected city or Finland
    for i in xrange(len(column_price)):
        column = column_price[i]
        logging.debug('columns_price ({}): {}'.format(i, column))
        fig.circle(data.index.values[:-1], data[column].iloc[:-1], legend=column,
                        color=colors[i], size=10, selection_color='firebrick')
        fig.line(data.index.values[:-1], data[column].iloc[:-1], legend=column,
                        color=colors[i],line_width=2)
        fig.square(data.index.values[-1], data[column].iloc[-1], legend='pred_'+column,
                        color=colors[i], size=10, selection_color='firebrick')
        fig.line(data.index.values[-2:], data[column].iloc[-2:], legend='pred_'+column,
                        color=colors[i],line_width=2, line_dash=[4, 4])
    fig.legend.label_text_font_size = "8pt"
    fig.legend.border_line_width = 1.5
    fig.legend.border_line_color = 'navy'
    fig.legend.border_line_alpha = 0.3
    fig.legend.padding = 2
    fig.legend.spacing = 1
    return fig

def create_figure_observations():
    '''Create chart figure for house sale observations.
    '''
    logging.info('Creating bokeh house sale observation figure')
    t_area, t_type = house_area_ticker.value, house_type_ticker.value
    logging.info('Area: {} type: {}'.format(t_area, t_type))
    data = get_house_data(t_area, t_type)
    tools = 'pan,wheel_zoom,box_select,reset'
    # columns of observations and price of each type of houses
    column_obs = [column for column in data.columns if column in
                        ['block_of_flats_observations', 'single_family_houses_observations']]
    # ylimit ranges
    ymin = np.min(data[column_obs].min(axis=0))
    ymax = np.max(data[column_obs].max())
    ystd = np.max(data[column_obs].std())
    # create a figure
    fig = figure(plot_width=400, plot_height=400, y_range=(ymin- 2*ystd, ymax + 2*ystd))
    # set figure parameters
    fig.xaxis.axis_label = 'Year'
    fig.xaxis.axis_label_text_color = 'firebrick'
    fig.xaxis.axis_label_standoff = 30
    fig.yaxis.axis_label = 'Number of sale observations'
    fig.yaxis.axis_label_text_color = 'firebrick'

    fig.title.text = 'Number of sale observations in %s' % (t_area)
    colors = ['hotpink', 'darkorange', 'yellowgreen', 'lightseagreen',
                    'deepskyblue', 'slateblue', 'darkmagenta']
    # plot price of the selected city or Finland
    for i in xrange(len(column_obs)):
        column = column_obs[i]
        logging.info('columns_obs ({}): {}'.format(i, column))
        fig.circle(data.index.values, data[column], legend=column, color=colors[i], size=10,
                        selection_color='firebrick')
        fig.line(data.index.values, data[column], legend=column, color=colors[i],line_width=2)
    fig.legend.label_text_font_size = "8pt"
    fig.legend.border_line_width = 1.5
    fig.legend.border_line_color = 'navy'
    fig.legend.border_line_alpha = 0.3
    fig.legend.padding = 2
    fig.legend.spacing = 1
    return fig

def create_figure_family():
    '''Create chart figure for family information.
    '''
    logging.info('Creating bokeh family figure')
    t_area, t_type = house_area_ticker.value, house_type_ticker.value
    logging.info('Area: {} type: {}'.format(t_area, t_type))
    data = get_family_data(t_area)
    tools = 'pan,wheel_zoom,box_select,reset'
    # columns of family info
    column_obs = [column for column in data.columns if column in
                        ['avg_size_of_family', 'num_of_families']]
    # ylimit ranges
    ymin = np.min(data['num_of_families'].min(axis=0))
    ymax = np.max(data['num_of_families'].max())
    ystd = np.max(data['num_of_families'].std())
    # create a figure
    fig = figure(plot_width=500, plot_height=400, y_range=(ymin- 2*ystd, ymax + 2*ystd))
    # set figure parameters
    fig.xaxis.axis_label = 'Year'
    fig.xaxis.axis_label_text_color = 'firebrick'
    fig.xaxis.axis_label_standoff = 30
    fig.yaxis.axis_label = 'Number of families'
    fig.yaxis.axis_label_text_color = 'firebrick'
    # ylimit ranges
    ymin = np.min(data['avg_size_of_family'].min(axis=0))
    ymax = np.max(data['avg_size_of_family'].max())
    ystd = np.max(data['avg_size_of_family'].std())
    fig.extra_y_ranges = {"familysize": Range1d(start=ymin- 2*ystd, end=ymax + 2*ystd)}

    fig.add_layout(LinearAxis(y_range_name="familysize"), 'right')
    fig.yaxis[1].axis_label = 'Average size ofe family'
    fig.yaxis[1].axis_label_text_color = 'firebrick'

    fig.title.text = 'Avg size and number of families in %s' % (t_area)
    colors = ['hotpink', 'darkorange', 'yellowgreen', 'lightseagreen']
    # plot price of the selected city or Finland
    for i in xrange(len(column_obs)):
        column = column_obs[i]
        logging.debug('columns_family ({}): {}'.format(i, column))
        if column == 'num_of_families':
            fig.circle(data.index.values, data[column], legend=column, color=colors[i], size=10,
                            selection_color='firebrick')
            fig.line(data.index.values, data[column], legend=column, color=colors[i],line_width=2)
        else:
            fig.circle(data.index.values, data[column], legend=column, color=colors[i], size=10,
                            selection_color='firebrick', y_range_name="familysize")
            fig.line(data.index.values, data[column], legend=column, color=colors[i],line_width=2,
                            y_range_name="familysize")
    fig.legend.label_text_font_size = "8pt"
    fig.legend.border_line_width = 1.5
    fig.legend.border_line_color = 'navy'
    fig.legend.border_line_alpha = 0.3
    fig.legend.padding = 2
    fig.legend.spacing = 1
    return fig

dir_path = os.path.dirname(os.path.realpath(__file__))
# load prediction data that was generated from house_price_forecast.py file
filename = os.path.join(dir_path, '../data/prediction.csv')

dd = load_prediction_data(filename)
dd = format_for_visualization(dd)

## Visualization
# output to static HTML file
output_file('housing_price.html', title='Housing price of Finland')
# House ticker options
HOUSE_TYPE_TICKERS = ['block_of_flats','single_family_houses', 'both']
AREA_TICKERS = [i for i in dd.columns.levels[0].tolist()]
# Select object for areas (city or Finland)
house_area_ticker = Select(title='City', value= 'Helsinki', options=AREA_TICKERS)
house_area_ticker.on_change('value', update)
# Select object for types of houses
house_type_ticker = Select(title='House type', value= 'both', options=HOUSE_TYPE_TICKERS)
house_type_ticker.on_change('value', update)
# create widget for the tickers
widgets = widgetbox([house_area_ticker, house_type_ticker], width=300)
# create a layout of two rows two columns.
layout = column(row(create_figure_house_price(), widgets),
                row(create_figure_observations()))
                #row(create_figure_observations(), create_figure_family()))

curdoc().add_root(layout)
curdoc().title = 'Forecast House Price'
