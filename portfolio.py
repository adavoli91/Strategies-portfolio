import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import os
import datetime
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from plotly.subplots import make_subplots

dict_margin = {'AD': 3000, 'BP': 3000, 'C': 2000, 'CD': 3000, 'CL': 18000, 'EC': 3800, 'ES': 13000, 'FDAX': 40000, 'GC': 15000, 'HG': 10000, 'NG': 5000, 'NQ': 23000,
               'RTY': 8000, 'S': 4000, 'VX': 26000, 'VXX': 900, 'YM': 15000}

class Portfolio:
    def __init__(self):
        # self.list_strategies = [i for i in os.listdir('./reports/') if '.txt' in i]
        self.list_strategies = st.file_uploader('Upload strategies', accept_multiple_files = True, type = ['csv', 'txt'])
        self.list_strategies = [strat.name for strat in self.list_strategies]
        self.dict_margin = {}

    def _compute_np_dd(self, daily_profit: np.array) -> tuple[np.array, np.array, np.array]:
        '''
        Function to compute NP, equity peak and DD as functions of time.

        Args:
            daily_profit: Array containing the daily profit.
        
        Returns:
            cum_profit: Array containing cumulated profit over time.
            max_equity: Array containing equity peak over time.
            dd: Array containing drawdown over time.
        '''
        cum_profit = daily_profit.cumsum()
        max_equity = pd.Series(cum_profit).cummax().values
        dd = cum_profit - max_equity
        #
        return cum_profit, max_equity, dd
    
    

    def _correlation_classic(self, data_1: np.array, data_2: np.array) -> float:
        '''
        Function to compute the correlation between two strategies as lag-0 cross-correlation.

        Args:
            data_1: Array containing two columns: the first one are the dates, and the second one the P/L.
            data_2: Array containing two columns: the first one are the dates, and the second one the P/L.

        Returns:
            corr: Correlation coefficient.
        '''
        # get common dates and slice data
        _, idx_1, idx_2 = np.intersect1d(data_1['date'].values, data_2['date'].values, return_indices = True)
        data_1 = data_1.loc[idx_1, 'daily_profit'].astype(float)
        data_2 = data_2.loc[idx_2, 'daily_profit'].astype(float)
        n = data_1.shape[0]
        # mean values and standard deviations
        mu_1 = data_1.mean()
        mu_2 = data_2.mean()
        sigma_1 = data_1.std()
        sigma_2 = data_2.std()
        # compute correlation
        corr = 1/(n*sigma_1*sigma_2)*np.sum((data_1 - mu_1)*(data_2 - mu_2))
        #
        return corr

    def _choose_strategies(self):
        '''
        Function to select the strategies of the portfolio.
        '''
        list_strategies = self.list_strategies
        list_strategies = ['_'.join(i.split('.txt')[0].split('_')[1:]) for i in list_strategies if '.txt' in i]
        list_strategies = np.concatenate((
            np.sort([i for i in list_strategies if '_Sign' in i]),
            np.sort([i for i in list_strategies if '_Sign' not in i])
        ))
        # Multi-select box for strategies
        self.portfolio = st.sidebar.multiselect(label='Strategies to include in the portfolio:', options=list_strategies)

    def _choose_volumes(self, run):
        '''
        Function to select the number of contracts for each strategy of the portfolio.
        '''
        if (run == True) and (st.session_state.dict_strat is not None):
            return

        dict_strat = {}        
        for strat in self.portfolio:
            dict_strat[strat] = st.number_input(f'Number of contracts for {strat}:', min_value = 1, value = 1)
        
        if run == True:
            # restore original names, including instrument and '.txt'
            dict_strat = {[strat for strat in self.list_strategies if key in strat][0]: value for key, value in dict_strat.items()}
            st.session_state.dict_strat = dict_strat

    def _filter_dates(self):
        '''
        Function to create the filter for dates.

        Args: None.

        Returns: None.
        '''
        # sidebar - choose the way to filter dates
        filt_date = st.sidebar.selectbox(label = 'Filter date by: ', options = ['Slider', 'Calendar'])
        filter_date_start = datetime.datetime.strptime('2010-01-01', '%Y-%m-%d')
        filter_date_end = datetime.datetime.strptime(f'{datetime.datetime.now().year}-12-31', '%Y-%m-%d')
        # sidebar - filter date with slider
        if filt_date == 'Slider':
            filter_date = st.sidebar.slider(label = 'Date range', min_value = filter_date_start, max_value = filter_date_end,
                                            value = [filter_date_start, filter_date_end])
        # sidebar - filter date with calendar
        elif filt_date == 'Calendar':
            filter_date = st.sidebar.date_input(label = 'Date range', min_value = filter_date_start, max_value = filter_date_end,
                                                value = [filter_date_start, filter_date_end])
        # no date filter
        else:
            filter_date = [filter_date_start, filter_date_end]
        #
        self.date_start = filter_date[0].strftime('%Y-%m-%d')
        self.date_end = filter_date[1].strftime('%Y-%m-%d')

    def _plot_preferences(self):
        self.agg_month = st.sidebar.radio(label = 'Aggregate plot by month:', options = ['No', 'Yes'], horizontal = True)

    def _read_strats(self):
        '''
        Function to import strategies results.
        '''
        dict_results = {}
        for strat in self.dict_strat.keys():
            instrument = strat.split('_')[0]
            # read data
            with open(f'./reports/{strat}') as f:
                data = f.readlines()
            data = np.array([i.split(' ') for i in data])
            # get strategy parameters
            dates = pd.to_datetime(data[:, 0], format = '%d/%m/%Y')
            daily_profit = data[:, 1].astype(float)
            curr_contract = data[:, 2].astype(float)*self.dict_strat[strat]
            n_trades = data[:, 5].astype(int)
            # build dataframe
            df = pd.DataFrame({'date': dates, 'daily_profit': daily_profit, 'curr_contract': curr_contract, 'n_trades': n_trades})
            df['margin'] = dict_margin[instrument]*df['curr_contract'].abs()
            dict_results[strat] = df
        #
        self.dict_results = dict_results

    def _portfolio_performance(self) -> pd.DataFrame:
        '''
        Function to compute the performance of a portfolio.
        
        Args: None.

        Returns:
            df_portfolio: Dataframe containing the portfolio performance.
        '''
        #
        dict_strat_vol = self.dict_strat
        date_start = self.date_start
        date_end = self.date_end
        #
        df_portfolio = {instr: [] for instr in np.unique([strat.split('_')[0] for strat in dict_strat_vol.keys()])}
        for i in range(len(dict_strat_vol)):
            # get strategy features
            strategy = list(dict_strat_vol.keys())[i]
            instrument = strategy.split('_')[0]
            n_contracts = dict_strat_vol[strategy]
            #
            df_temp = self.dict_results[strategy]
            # keep relevant dates
            df_temp = df_temp[(df_temp['date'] >= date_start) & (df_temp['date'] < date_end)].reset_index(drop = True)
            # adjust profit and margins by number of contracts
            df_temp['daily_profit'] *= n_contracts
            df_temp['margin'] *= n_contracts*np.sign(df_temp['curr_contract'])
            # combine portfolio strategies
            df_temp.index = df_temp['date']
            df_portfolio[instrument].append(df_temp)
        #
        list_first_dates = np.unique([df_temp['date'].min() for list_df in df_portfolio.values() for df_temp in list_df])
        if list_first_dates.shape[0] > 1:
            st.write(f'The backtest starts on {np.max(list_first_dates).strftime("%Y-%m-%d")} because that is the first available date of at least one strategy.')
            for instr in df_portfolio.keys():
                list_df = df_portfolio[instr]
                for i in range(len(list_df)):
                    df_temp = list_df[i]
                    df_temp = df_temp[df_temp['date'] >= np.max(list_first_dates).strftime("%Y-%m-%d")].reset_index(drop = True)
                    df_temp['n_trades'] -= df_temp['n_trades'].min()
                    list_df[i] = df_temp
        # get the first date for each instrument
        # list_first_date = [df_temp['date'].min() for df_temp]
        df_portfolio = {instr: performance for instr, performance in df_portfolio.items() if len(performance) > 0}
        # combine performances for each instrument
        for instr in df_portfolio.keys():
            df_portfolio[instr] = pd.concat(df_portfolio[instr], axis = 1)
            if type(df_portfolio[instr]['daily_profit']) == pd.DataFrame:
                df_portfolio[instr]['daily_profit'] = df_portfolio[instr]['daily_profit'].sum(axis = 1)
            if type(df_portfolio[instr]['margin']) == pd.DataFrame:
                df_portfolio[instr]['margin'] = df_portfolio[instr]['margin'].sum(axis = 1)
            df_portfolio[instr] = df_portfolio[instr].loc[:, ~df_portfolio[instr].columns.duplicated()][['date', 'daily_profit', 'margin']]
            df_portfolio[instr]['margin'] = abs(df_portfolio[instr]['margin'])
        # compute aggregated performance
        df_portfolio = [value for value in df_portfolio.values()]
        df_portfolio = pd.concat(df_portfolio, axis = 1)
        if type(df_portfolio['daily_profit']) == pd.DataFrame:
            df_portfolio['daily_profit'] = df_portfolio['daily_profit'].sum(axis = 1)
        if type(df_portfolio['margin']) == pd.DataFrame:
            df_portfolio['margin'] = df_portfolio['margin'].sum(axis = 1)
        df_portfolio = df_portfolio.loc[:, ~df_portfolio.columns.duplicated()][['date', 'daily_profit', 'margin']]
        # add portfolio cumulative statistics
        df_portfolio['cum_profit'], df_portfolio['max_equity'], df_portfolio['dd'] = self._compute_np_dd(df_portfolio['daily_profit'].values)
        #
        self.df_portfolio = df_portfolio.reset_index(drop = True).sort_values(by = 'date')

    def _plot_profit(self):
        '''
        Function to plot the results of a portfolio dataframe.

        Args: None.
            
        Returns: None.
        '''
        df = self.df_portfolio.copy()
        agg_month = self.agg_month
        #
        if agg_month == 'Yes':
            df_month = df.copy()
            df_month['year'] = df_month['date'].dt.year
            df_month['month'] = df_month['date'].dt.month
            df_month = df_month.groupby(['year', 'month']).agg({'daily_profit': 'sum', 'date': 'max'}).reset_index()
            df_month = df_month.rename(columns = {'daily_profit': 'monthly_profit'})
            df_month['cum_profit'], _, df_month['dd'] = self._compute_np_dd(df_month['monthly_profit'])
        #
        figure = make_subplots(rows = 2, cols = 1, shared_xaxes = True, row_heights = [0.67, 0.33], vertical_spacing = 0)
        figure.update_layout(go.Layout(margin = dict(l = 20, r = 20, t = 20, b = 20), template = 'simple_white', showlegend = False,
                                    xaxis1 = {'showgrid': True, 'showline': True, 'mirror': True, 'titlefont': {'size': 20}, 'tickfont': {'size': 16},
                                                'tickangle': -90},
                                    yaxis1 = {'showgrid': True, 'showline': True, 'mirror': True, 'titlefont': {'size': 20}, 'tickfont': {'size': 16},
                                                'tickformat': f'.{2}f', 'title': 'NP [$]'},
                                    xaxis2 = {'showgrid': True, 'showline': True, 'mirror': True, 'titlefont': {'size': 20}, 'tickfont': {'size': 16},
                                                'tickangle': -90},
                                    yaxis2 = {'showgrid': True, 'showline': True, 'mirror': True, 'titlefont': {'size': 20}, 'tickfont': {'size': 16},
                                                'tickformat': f'.{2}f', 'title': 'DD [$]'},
                                    font = {'size': 28}, autosize = False, width = 900, height = 500, hovermode = 'closest'))
        #
        figure.add_trace(go.Scatter(x = df['date'].values, y = df['cum_profit'].values, mode = 'lines', line_color = 'blue'), row = 1, col = 1)
        figure.add_trace(go.Scatter(x = df['date'].values, y = df['dd'].values, mode = 'lines', line_color = 'blue'), row = 2, col = 1)
        if agg_month == True:
            figure.add_trace(go.Scatter(x = df_month['date'].values, y = df_month['cum_profit'].values, line = {'dash': 'dot', 'color': 'red'}), row = 1, col = 1)
            figure.add_trace(go.Scatter(x = df_month['date'].values, y = df_month['dd'].values, line = {'dash': 'dot', 'color': 'red'}), row = 2, col = 1)
        st.plotly_chart(figure)

    def _portfolio_metrics(self):
        '''
        Function to compute portfolio performances.

        Args: None.
            
        Returns: None.
        '''
        df = self.df_portfolio
        # check if current drawdown is the maximum of all history
        date_peak_equity = df.loc[df['cum_profit'] == df['cum_profit'].max(), 'date'].values[0]
        df_filt = df[df['date'] >= date_peak_equity].reset_index(drop = True).copy()
        df_filt['cum_profit'], df_filt['max_equity'], df_filt['dd'] = self._compute_np_dd(df_filt['daily_profit'])
        max_dd_recent = False
        if df_filt['dd'].min() <= df['dd'].min():
            max_dd_recent = True
        # NP
        net_profit = df['cum_profit'].values[-1]
        # NP/DD
        np_dd = -df['cum_profit'].values[-1]/df['dd'].min()
        # max DD
        max_dd = -df['dd'].min()
        # average DD duration 
        idx_betwee_peaks = df['max_equity'].expanding().apply(lambda x: x.argmax())
        mean_duration_dd = np.mean(np.diff(idx_betwee_peaks.drop_duplicates()))
        # average DD
        df_temp = pd.DataFrame(tuple(zip(idx_betwee_peaks, idx_betwee_peaks.shift(-1))), columns = ['min', 'max']).dropna().astype(int)
        avg_dd = -np.mean(df_temp.apply(lambda row: df['dd'][row['min']: row['max']].min(), axis = 1).drop_duplicates())
        # R^2
        lr = LinearRegression().fit(np.arange(1, df.shape[0] + 1).reshape(-1, 1), df['cum_profit'])
        r_2 = r2_score(y_true = df['cum_profit'], y_pred = lr.coef_*np.arange(1, df.shape[0] + 1) + lr.intercept_)
        #
        df_metrics = pd.DataFrame([[net_profit, max_dd, np_dd, avg_dd, mean_duration_dd, r_2, -df_filt['dd'].min(), str(max_dd_recent)]])
        df_metrics.columns = ['NP', 'Max DD', 'NP/DD', 'Avg. DD', 'Avg. DD duration [days]', 'R^2', 'Current DD', 'Currently max DD']
        #
        st.table(df_metrics.style.format({'NP': '{:.2f}', 'Max DD': '{:.2f}', 'NP/DD': '{:.2f}', 'Avg. DD': '{:.2f}', 'Avg. DD duration [days]': '{:.0f}',
                                          'R^2': '{:.3f}', 'Current DD': '{:.2f}'}))

    def _plot_monte_carlo(self):
        '''
        Function to plot the results of a monte carlo analysis on the portfolio dataframe.

        Args: None.
            
        Returns: None.
        '''
        df = self.df_portfolio.copy()
        #
        figure = make_subplots(rows = 2, cols = 1, shared_xaxes = True, row_heights = [0.67, 0.33], vertical_spacing = 0)
        figure.update_layout(go.Layout(margin = dict(l = 20, r = 20, t = 20, b = 20), template = 'simple_white', showlegend = False,
                                    xaxis1 = {'showgrid': True, 'showline': True, 'mirror': True, 'titlefont': {'size': 20}, 'tickfont': {'size': 16},
                                                'tickangle': -90},
                                    yaxis1 = {'showgrid': True, 'showline': True, 'mirror': True, 'titlefont': {'size': 20}, 'tickfont': {'size': 16},
                                                'tickformat': f'.{2}f', 'title': 'NP [$]'},
                                    xaxis2 = {'showgrid': True, 'showline': True, 'mirror': True, 'titlefont': {'size': 20}, 'tickfont': {'size': 16},
                                                'tickangle': -90},
                                    yaxis2 = {'showgrid': True, 'showline': True, 'mirror': True, 'titlefont': {'size': 20}, 'tickfont': {'size': 16},
                                                'tickformat': f'.{2}f', 'title': 'DD [$]'},
                                    font = {'size': 28}, autosize = False, width = 900, height = 500, hovermode = 'closest'))
        #
        for i in range(100):
            df_temp = df.copy()
            shuffle_idx = np.random.choice(range(df_temp.shape[0]), df_temp.shape[0], replace = False)
            df_temp['daily_profit'] = df_temp.loc[shuffle_idx, 'daily_profit'].values
            df_temp['cum_profit'], _, df_temp['dd'] = self._compute_np_dd(df_temp['daily_profit'])
            figure.add_trace(go.Scatter(x = df_temp['date'].values, y = df_temp['cum_profit'].values, mode = 'lines', line_color = 'gray', opacity = 0.5), row = 1, col = 1)
            figure.add_trace(go.Scatter(x = df_temp['date'].values, y = df_temp['dd'].values, mode = 'lines', line_color = 'gray', opacity = 0.5), row = 2, col = 1)
        #
        figure.add_trace(go.Scatter(x = df['date'].values, y = df['cum_profit'].values, mode = 'lines', line_color = 'blue'), row = 1, col = 1)
        figure.add_trace(go.Scatter(x = df['date'].values, y = df['dd'].values, mode = 'lines', line_color = 'blue'), row = 2, col = 1)
        st.plotly_chart(figure)
        
    def _plot_dd_hist(self):
        '''
        Function to plot the histogram of daily drawdowns.

        Args: None.
            
        Returns: None.
        '''
        df = self.df_portfolio
        #
        vals = df.loc[df['dd'] != 0, 'dd'].abs().values
        q_1 = np.quantile(vals, 0.25)
        q_2 = np.quantile(vals, 0.50)
        q_3 = np.quantile(vals, 0.75)
        q_3_iqr = np.quantile(vals, 0.75) + (q_3 - q_1)
        #
        figure = go.Figure()
        figure.update_layout(go.Layout(margin = dict(l = 20, r = 20, t = 20, b = 20), template = 'simple_white', showlegend = False,
                                    xaxis1 = {'showgrid': True, 'showline': True, 'mirror': True, 'titlefont': {'size': 20}, 'tickfont': {'size': 16},
                                                'tickangle': 0, 'title': 'Daily drawdown [$]'},
                                    yaxis1 = {'showgrid': True, 'showline': True, 'mirror': True, 'titlefont': {'size': 20}, 'tickfont': {'size': 16},
                                                'tickformat': f'.{0}f'},
                                    font = {'size': 28}, autosize = False, width = 900, height = 500, hovermode = 'closest'))
        figure.add_trace(go.Histogram(x = vals, marker_color = 'lime', opacity = 0.5))
        figure.add_vline(x = q_1, line_width = 2, line_dash = 'dash', line_color = 'blue', annotation_text = 'Q1',
                        annotation_position = 'top right', annotation = {'font_color': 'blue', 'font_size': 16, 'borderwidth': 10})
        figure.add_vline(x = q_2, line_width = 2, line_dash = 'dash', line_color = 'yellow', annotation_text = 'Q2',
                        annotation_position = 'top right', annotation = {'font_color': 'yellow', 'font_size': 16, 'borderwidth': 10})
        figure.add_vline(x = q_3, line_width = 2, line_dash = 'dash', line_color = 'red', annotation_text = 'Q3',
                        annotation_position = 'top right', annotation = {'font_color': 'red', 'font_size': 16, 'borderwidth': 10})
        figure.add_vline(x = q_3_iqr, line_width = 2, line_dash = 'dash', line_color = 'cyan', annotation_text = 'Q3 + IQR',
                        annotation_position = 'top right', annotation = {'font_color': 'cyan', 'font_size': 16, 'borderwidth': 10})
        st.plotly_chart(figure)

    def _plot_correlation(self):
        '''
        Function to plot the correlation matrix between strategies.

        Args: None.
            
        Returns: None.
        '''
        dict_results = self.dict_results
        #
        dict_corr = {}
        for strat_1 in dict_results.keys():
            for strat_2 in dict_results.keys():
                data_1, data_2 = dict_results[strat_1], dict_results[strat_2]
                dict_corr[('_'.join(strat_1.split('_')[1:]),
                           '_'.join(strat_2.split('_')[1:]))] = round(self._correlation_classic(data_1[['date', 'daily_profit']],
                                                                                                data_2[['date', 'daily_profit']]), 3)
        #
        df_corr = pd.DataFrame(np.array(list(dict_corr.values())).reshape(int(np.sqrt(len(dict_corr))), -1))
        df_corr.columns = ['_'.join(i.split('_')[1:]) for i in dict_results.keys()]
        df_corr.index = ['_'.join(i.split('_')[1:]) for i in dict_results.keys()]
        #
        figure = go.Figure()
        figure.update_layout(go.Layout(margin = dict(l = 20, r = 20, t = 20, b = 20), template = 'simple_white', showlegend = False,
                                       xaxis = {'showgrid': True, 'showline': True, 'mirror': True, 'titlefont': {'size': 20}, 'tickfont': {'size': 12},
                                                'tickangle': -40}))
        figure.add_trace(go.Heatmap(z = df_corr.values[::-1, :], x = df_corr.columns, y = df_corr.index[::-1],
                                    colorscale = 'Spectral_r', text = df_corr.values[::-1, :], texttemplate="%{text}", zmin = -1, zmax = 1))
        st.plotly_chart(figure)
        
    def _plot_margin_hist(self):
        '''
        Function to plot the histogram of daily margins.

        Args: None.
            
        Returns: None.
        '''
        df = self.df_portfolio
        #
        vals = df.loc[df['margin'] != 0, 'margin'].values
        q_1 = np.quantile(vals, 0.25)
        q_2 = np.quantile(vals, 0.50)
        q_3 = np.quantile(vals, 0.75)
        q_3_iqr = np.quantile(vals, 0.75) + (q_3 - q_1)
        #
        figure = go.Figure()
        figure.update_layout(go.Layout(margin = dict(l = 20, r = 20, t = 20, b = 20), template = 'simple_white', showlegend = False,
                                    xaxis1 = {'showgrid': True, 'showline': True, 'mirror': True, 'titlefont': {'size': 20}, 'tickfont': {'size': 16},
                                                'tickangle': 0, 'title': 'Daily margin [$]'},
                                    yaxis1 = {'showgrid': True, 'showline': True, 'mirror': True, 'titlefont': {'size': 20}, 'tickfont': {'size': 16},
                                                'tickformat': f'.{0}f'},
                                    font = {'size': 28}, autosize = False, width = 900, height = 500, hovermode = 'closest'))
        figure.add_trace(go.Histogram(x = vals, marker_color = 'orange', opacity = 0.5))
        figure.add_vline(x = q_1, line_width = 2, line_dash = 'dash', line_color = 'lime', annotation_text = 'Q1',
                        annotation_position = 'top right', annotation = {'font_color': 'lime', 'font_size': 16, 'borderwidth': 10})
        figure.add_vline(x = q_2, line_width = 2, line_dash = 'dash', line_color = 'yellow', annotation_text = 'Q2',
                        annotation_position = 'top right', annotation = {'font_color': 'yellow', 'font_size': 16, 'borderwidth': 10})
        figure.add_vline(x = q_3, line_width = 2, line_dash = 'dash', line_color = 'blue', annotation_text = 'Q3',
                        annotation_position = 'top right', annotation = {'font_color': 'blue', 'font_size': 16, 'borderwidth': 10})
        figure.add_vline(x = q_3_iqr, line_width = 2, line_dash = 'dash', line_color = 'cyan', annotation_text = 'Q3 + IQR',
                        annotation_position = 'top right', annotation = {'font_color': 'cyan', 'font_size': 16, 'borderwidth': 10})
        st.plotly_chart(figure)

    def _plot_prob_ruin(self):
        '''
        Function to plot the probability of ruin.

        Args: None.
            
        Returns: None.
        '''
        df = self.df_portfolio
        #
        capital = self.capital
        prob_ruin = {}
        for max_dd_accepted_perc in [10, 20, 30, 50, 100]:
            counter_tot, counter_ruin = {0: 0, 20: 0, 40: 0, 100: 0, 200: 0, 400: 0}, {0: 0, 20: 0, 40: 0, 100: 0, 200: 0, 400: 0}
            profit_mean = np.mean(df[df['daily_profit'] != 0]['daily_profit'].values)
            profit_std = np.std(df[df['daily_profit'] != 0]['daily_profit'].values)
            for i in range(2000):
                for n_days in counter_tot.keys():
                    history_profit = capital + np.cumsum(np.random.normal(loc = profit_mean, scale = profit_std, size = n_days))
                    if np.sum(history_profit <= capital*(1 - max_dd_accepted_perc/100)) > 0:
                        counter_ruin[n_days] += 1
                    counter_tot[n_days] += 1
            prob_ruin[max_dd_accepted_perc] = {n_days: round(counter_ruin[n_days]/counter_tot[n_days], 3) for n_days in counter_tot.keys()}
        #
        figure = go.Figure()
        figure.update_layout(go.Layout(margin = dict(l = 20, r = 20, t = 20, b = 20), template = 'simple_white', showlegend = True,
                                    legend = {'font': {'size': 13}, 'x': 0.},
                                    xaxis = {'showgrid': True, 'showline': True, 'mirror': True, 'titlefont': {'size': 20}, 'tickfont': {'size': 16},
                                                'tickformat': f'.{0}f', 'title': 'Number of days'},
                                    yaxis = {'showgrid': True, 'showline': True, 'mirror': True, 'titlefont': {'size': 20}, 'tickfont': {'size': 16},
                                                'tickformat': f'.{0}%', 'title': 'Probability of ruin'},
                                    font = {'size': 28}, autosize = False, width = 900, height = 500, hovermode = 'closest'))
        for max_dd_accepted_perc in prob_ruin.keys():
            figure.add_trace(go.Scatter(x = list(prob_ruin[max_dd_accepted_perc].keys()),
                                        y = list(prob_ruin[max_dd_accepted_perc].values()), name = f'Max DD = {max_dd_accepted_perc}%'))
        st.plotly_chart(figure)

if __name__ == '__main__':
    # Page width
    st.set_page_config(layout='wide')

    # Initialize session state for tracking user input
    if 'dict_strat' not in st.session_state:
        st.session_state.load = None
        st.session_state.dict_strat = None
    with st.form(key='Main run'):
        if st.session_state.load != True:
            portfolio = Portfolio()
            load = st.form_submit_button(label = 'Load strategies')
            st.session_state.load = load
            st.session_state.portfolio = portfolio
        if st.session_state.load == True:
            portfolio = st.session_state.portfolio
            # Create the Run button
            run = st.form_submit_button(label = 'Run')
            #
            st.session_state.run = run
            portfolio._choose_strategies()
            portfolio._choose_volumes(run)
            portfolio._filter_dates()
            # choose what to show
            plot_equity = st.sidebar.radio(label = 'Portfolio equity:', options = ['Yes', 'No'], horizontal = True)
            if plot_equity == 'Yes':
                portfolio._plot_preferences()
            portfolio_metrics = st.sidebar.radio(label = 'Portfolio metrics:', options = ['Yes', 'No'], horizontal = True)
            mc_analysis = st.sidebar.radio(label = 'Monte Carlo analysis:', options = ['Yes', 'No'], horizontal = True)
            drawdown_analysis = st.sidebar.radio(label = 'Drawdown analysis:', options = ['Yes', 'No'], horizontal = True)
            correlation_analysis = st.sidebar.radio(label = 'Correlation analysis:', options = ['Yes', 'No'], horizontal = True)
            prob_ruin = st.sidebar.radio(label = 'Probability of ruin:', options = ['Yes', 'No'], horizontal = True)
            if prob_ruin == 'Yes':
                portfolio.capital = st.sidebar.number_input(f'Initial capital in $:', value = 100000)
            margin_analysis = st.sidebar.radio(label = 'Margin analysis:', options = ['Yes', 'No'], horizontal = True)
            if st.session_state.dict_strat is not None:
                #
                portfolio.dict_strat = st.session_state.dict_strat
                portfolio._read_strats()
                portfolio._portfolio_performance()
                if plot_equity == 'Yes':
                    st.header('Portfolio equity')
                    portfolio._plot_profit()
                if portfolio_metrics == 'Yes':
                    st.header('Portfolio metrics')
                    portfolio._portfolio_metrics()
                if mc_analysis == 'Yes':
                    st.header('Monte Carlo analysis')
                    portfolio._plot_monte_carlo()
                if drawdown_analysis == 'Yes':
                    st.header('Drawdown analysis')
                    st.markdown('Values above Q2 have 50% chance; values above Q3 + IQR are rare.')
                    portfolio._plot_dd_hist()
                if len(portfolio.dict_strat) > 1:
                    if correlation_analysis == 'Yes':
                        st.header('Correlation between strategies')
                        portfolio._plot_correlation()
                if prob_ruin == 'Yes':
                    st.header('Probability of ruin')
                    st.markdown(f'Probability of having a given drawdown over time, starting from {portfolio.capital/1000} k$.')
                    portfolio._plot_prob_ruin()
                if margin_analysis == 'Yes':
                    st.header('Margin analysis')
                    st.markdown('Values above Q2 have 50% chance; values above Q3 + IQR are rare.')
                    portfolio._plot_margin_hist()