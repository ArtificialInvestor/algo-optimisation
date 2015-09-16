
#Parameter Optimisation
##### Marcus Williamson - 01/09/15

1. Setup Environment
1. Import Algorithm
1. Setup Optimisation Tests
1. Review Performance

##All Weather V1 - Reversion Optimisation

###1. Setup Environment
####Import Libraries


    import zipline
    import pytz
    from datetime import datetime
    import matplotlib.pyplot as pyplot
    from collections import defaultdict
    
    from zipline import TradingAlgorithm
    from zipline.api import order_target, record, symbol, history, add_history, order_target_percent
    from zipline.api import schedule_function, date_rules, time_rules, order, get_open_orders, get_datetime
    from zipline.api import set_slippage, set_commission
    from zipline.api import slippage
    from zipline.api import commission
    
    from zipline.utils import tradingcalendar
    
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    import time

####Define functions for evaluating performance


    # define a bunch of performance statistics for analysis of our backtests
    def normalize(returns, starting_value=1):
        return starting_value * (returns / returns.iloc[0])
    
    
    def cum_returns(returns, starting_value=None):
    
        # df_price.pct_change() adds a nan in first position, we can use
        # that to have cum_returns start at the origin so that
        # df_cum.iloc[0] == starting_value
        # Note that we can't add that ourselves as we don't know which dt
        # to use.
        if pd.isnull(returns.iloc[0]):
            returns.iloc[0] = 0.
    
        df_cum = np.exp(np.log(1 + returns).cumsum())
    
        if starting_value is None:
            return df_cum - 1
        else:
            return df_cum * starting_value
    
    
    def aggregate_returns(df_daily_rets, convert_to):
    
        def cumulate_returns(x):
            return cum_returns(x)[-1]
    
        if convert_to == 'weekly':
            return df_daily_rets.groupby(
                [lambda x: x.year,
                 lambda x: x.month,
                 lambda x: x.isocalendar()[1]]).apply(cumulate_returns)
        elif convert_to == 'monthly':
            return df_daily_rets.groupby(
                [lambda x: x.year, lambda x: x.month]).apply(cumulate_returns)
        elif convert_to == 'yearly':
            return df_daily_rets.groupby(
                [lambda x: x.year]).apply(cumulate_returns)
        else:
            ValueError('convert_to must be weekly, monthly or yearly')
    
    
    def max_drawdown(returns):
    
        if returns.size < 1:
            return np.nan
    
        df_cum_rets = cum_returns(returns, starting_value=100)
    
        MDD = 0
        DD = 0
        peak = -99999
        for value in df_cum_rets:
            if (value > peak):
                peak = value
            else:
                DD = (peak - value) / peak
            if (DD > MDD):
                MDD = DD
        return -1 * MDD
    
    
    def annual_return(returns, style='compound'):
    
        if returns.size < 1:
            return np.nan
    
        if style == 'calendar':
            num_years = len(returns) / 252.0
            df_cum_rets = cum_returns(returns, starting_value=100)
            start_value = df_cum_rets[0]
            end_value = df_cum_rets[-1]
            return ((end_value - start_value) / start_value) / num_years
        if style == 'compound':
            return pow((1 + returns.mean()), 252) - 1
        else:
            return returns.mean() * 252
    
    
    def annual_volatility(returns):
    
        if returns.size < 2:
            return np.nan
    
        return returns.std() * np.sqrt(252)
    
    
    def calmar_ratio(returns, returns_style='calendar'):
    
        temp_max_dd = max_drawdown(returns=returns)
        if temp_max_dd < 0:
            temp = annual_return(
                returns=returns,
                style=returns_style) / abs(max_drawdown(returns=returns))
        else:
            return np.nan
    
        if np.isinf(temp):
            return np.nan
    
        return temp
    
    
    def omega_ratio(returns, annual_return_threshhold=0.0):
    
        daily_return_thresh = pow(1 + annual_return_threshhold, 1 / 252) - 1
    
        returns_less_thresh = returns - daily_return_thresh
    
        numer = sum(returns_less_thresh[returns_less_thresh > 0.0])
        denom = -1.0 * sum(returns_less_thresh[returns_less_thresh < 0.0])
    
        if denom > 0.0:
            return numer / denom
        else:
            return np.nan
    
    
    def sortino_ratio(returns, returns_style='compound'):
    
        numer = annual_return(returns, style=returns_style)
        denom = annual_volatility(returns[returns < 0.0])
    
        if denom > 0.0:
            return numer / denom
        else:
            return np.nan
    
    
    def sharpe_ratio(returns, returns_style='compound'):
    
        numer = annual_return(returns, style=returns_style)
        denom = annual_volatility(returns)
    
        if denom > 0.0:
            return numer / denom
        else:
            return np.nan
    
    
    def stability_of_timeseries(returns, logValue=True):
    
        if returns.size < 2:
            return np.nan
    
        df_cum_rets = cum_returns(returns, starting_value=100)
        temp_values = np.log10(
            df_cum_rets.values) if logValue else df_cum_rets.values
        len_returns = df_cum_rets.size
    
        X = list(range(0, len_returns))
        X = sm.add_constant(X)
    
        model = sm.OLS(temp_values, X).fit()
    
        return model.rsquared
    
    def perf_stats(returns, returns_style='compound', return_as_dict=False):
    
        all_stats = {}
        all_stats['annual_return'] = annual_return(
            returns,
            style=returns_style)
        all_stats['annual_volatility'] = annual_volatility(returns)
        all_stats['sharpe_ratio'] = sharpe_ratio(
            returns,
            returns_style=returns_style)
        all_stats['calmar_ratio'] = calmar_ratio(
            returns,
            returns_style=returns_style)
        all_stats['stability'] = stability_of_timeseries(returns)
        all_stats['max_drawdown'] = max_drawdown(returns)
        all_stats['omega_ratio'] = omega_ratio(returns)
        all_stats['sortino_ratio'] = sortino_ratio(returns)
    
        if return_as_dict:
            return all_stats
        else:
            all_stats_df = pd.DataFrame(
                index=list(all_stats.keys()),
                data=list(all_stats.values()))
            all_stats_df.columns = ['perf_stats']
            return all_stats_df
        

###2. Import Algorithm
####Load data for backtests


    data = get_pricing(['SPY'],start_date='2007-01-01',end_date = '2010-01-01',frequency='minute')

####Define Algorithm


    lookback = 20
    sdev = 2.5
    
    def initialize(context): 
        
        context.input1 = symbol('SPY')
        context.lookback1 = lookback
        context.stdevparam = sdev
        context.average = 0
        context.stdev = 0
           
        
    def handle_data(context,data):
        
        if get_datetime().hour == 14 and get_datetime().minute == 30: #do this once per day
            
                context.average = history(context.lookback1,'1d', 'price')[context.input1].mean()
                context.stdev = history(context.lookback1,'1d', 'price')[context.input1].std()
            
    
        if (data[context.input1].price > context.average and context.portfolio.positions[context.input1].amount<0) or(data[context.input1].price < context.average and context.portfolio.positions[context.input1].amount>0):
            order_target_percent(context.input1,0) #close position
    
        elif data[context.input1].price > context.average + (context.stdevparam*context.stdev) and not context.portfolio.positions[context.input1].amount>0: #above top band
            order_target_percent(context.input1,1.0) #buying
    
        elif data[context.input1].price < context.average - (context.stdevparam*context.stdev) and not context.portfolio.positions[context.input1].amount<0: #below bottom band
            order_target_percent(context.input1,-1.0) #selling 
              

####Run test to ensure algorithm is functioning


    # RUN this cell to run a single backtest
    algo_obj = TradingAlgorithm(initialize=initialize, handle_data=handle_data, 
                                data_frequency='minute')
    perf_manual = algo_obj.run(data.transpose(2,1,0))
    perf_returns = perf_manual.returns     # grab the daily returns from the algo backtest
    (np.cumprod(1+perf_returns)).plot()    # plots the performance of your algo




    <matplotlib.axes._subplots.AxesSubplot at 0x7f02aea20f90>




![png](output_11_1.png)


###3. Setup Optimisation Tests
####Setup Parameters


    param_range_1 = map(int, np.linspace(1, 10, 10))   # lookback
    param_range_2 = map(float, np.around(np.linspace(0.89, 1.00, 10),decimals = 6))    # standard deviation
    print(param_range_1, param_range_2)

    ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [0.89, 0.902222, 0.914444, 0.926667, 0.938889, 0.951111, 0.963333, 0.975556, 0.987778, 1.0])


####Creating Tests


    # Show time when all the backtests started
    print time.ctime()
    
    count = 0
    results_df = pd.DataFrame()
    
    for param_1 in param_range_1:
        for param_2 in param_range_2:
            print "Run #" + str(count) + ", Parameters: " + str(param_1) + ", " + str(param_2)
            
            def initialize(context): 
        
                context.input1 = symbol('SPY')
                context.lookback1 = param_1
                context.stdevparam = param_2
                context.average = 0
                context.stdev = 0
            
            
            # this runs a backtest unique to the values in 'param_1' and 'param_2'
            algo_obj = TradingAlgorithm(initialize=initialize, handle_data=handle_data, 
                                data_frequency='minute')
            
            # compute the performance stats for this backtest run and then append to a dataframe
            # that is accumulating all of the backtest performance stats
            perf_algo = algo_obj.run(data.transpose(2,1,0))
            perf_returns = perf_algo.returns
            perf_stats_df = perf_stats( perf_returns ).T
    
            perf_stats_df['param_1'] = param_1
            perf_stats_df['param_2'] = param_2
            perf_stats_df.index = [count]
            
            if count < 1:
                results_df = perf_stats_df
            else:
                results_df = results_df.append(perf_stats_df)
            
            count += 1
    
    # Show time when all the backtests completed
    print time.ctime()
    
    results_df.sort_index(axis=1)

    Sun Sep 13 22:58:26 2015
    Run #0, Parameters: 1, 0.89
    Run #1, Parameters: 1, 0.902222
    Run #2, Parameters: 1, 0.914444
    Run #3, Parameters: 1, 0.926667
    Run #4, Parameters: 1, 0.938889
    Run #5, Parameters: 1, 0.951111
    Run #6, Parameters: 1, 0.963333
    Run #7, Parameters: 1, 0.975556
    Run #8, Parameters: 1, 0.987778
    Run #9, Parameters: 1, 1.0
    Run #10, Parameters: 2, 0.89
    Run #11, Parameters: 2, 0.902222
    Run #12, Parameters: 2, 0.914444
    Run #13, Parameters: 2, 0.926667
    Run #14, Parameters: 2, 0.938889
    Run #15, Parameters: 2, 0.951111
    Run #16, Parameters: 2, 0.963333
    Run #17, Parameters: 2, 0.975556
    Run #18, Parameters: 2, 0.987778
    Run #19, Parameters: 2, 1.0
    Run #20, Parameters: 3, 0.89
    Run #21, Parameters: 3, 0.902222
    Run #22, Parameters: 3, 0.914444
    Run #23, Parameters: 3, 0.926667
    Run #24, Parameters: 3, 0.938889
    Run #25, Parameters: 3, 0.951111
    Run #26, Parameters: 3, 0.963333
    Run #27, Parameters: 3, 0.975556
    Run #28, Parameters: 3, 0.987778
    Run #29, Parameters: 3, 1.0
    Run #30, Parameters: 4, 0.89
    Run #31, Parameters: 4, 0.902222
    Run #32, Parameters: 4, 0.914444
    Run #33, Parameters: 4, 0.926667

###4. Review Performance
####Tabulated Results


    # you should modify these 2 string labels to match the variables which you ran the above _for_ loops over
    # it's just to label the axes properly in the heatmaps
    
    param_name_1 = 'lookback'
    param_name_2 = 'stddev'
    
    results_df[param_name_1] = results_df.param_1
    results_df[param_name_2] = results_df.param_2
    
    results_df_sharpe = results_df.pivot(index=param_name_1, columns=param_name_2, values='sharpe_ratio') 
    results_df_max_drawdown = results_df.pivot(index=param_name_1, columns=param_name_2, values='max_drawdown') 
    results_df_annual_return = results_df.pivot(index=param_name_1, columns=param_name_2, values='annual_return') 
    results_df_volatility = results_df.pivot(index=param_name_1, columns=param_name_2, values='annual_volatility') 
    results_df_stability = results_df.pivot(index=param_name_1, columns=param_name_2, values='stability') 
    results_df_sortino = results_df.pivot(index=param_name_1, columns=param_name_2, values='sortino_ratio') 
    results_df_omega = results_df.pivot(index=param_name_1, columns=param_name_2, values='omega_ratio') 
    results_df_calmar = results_df.pivot(index=param_name_1, columns=param_name_2, values='calmar_ratio') 
    
    results_df




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sortino_ratio</th>
      <th>omega_ratio</th>
      <th>max_drawdown</th>
      <th>calmar_ratio</th>
      <th>annual_return</th>
      <th>stability</th>
      <th>sharpe_ratio</th>
      <th>annual_volatility</th>
      <th>param_1</th>
      <th>param_2</th>
      <th>lookback</th>
      <th>stddev</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.030399</td>
      <td>0.907434</td>
      <td>-0.057620</td>
      <td>-0.062011</td>
      <td>-0.003573</td>
      <td>0.093347</td>
      <td>-0.121806</td>
      <td>0.029334</td>
      <td>1</td>
      <td>0.890000</td>
      <td>1</td>
      <td>0.890000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.030399</td>
      <td>0.907434</td>
      <td>-0.057620</td>
      <td>-0.062011</td>
      <td>-0.003573</td>
      <td>0.093347</td>
      <td>-0.121806</td>
      <td>0.029334</td>
      <td>1</td>
      <td>0.902222</td>
      <td>1</td>
      <td>0.902222</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.030399</td>
      <td>0.907434</td>
      <td>-0.057620</td>
      <td>-0.062011</td>
      <td>-0.003573</td>
      <td>0.093347</td>
      <td>-0.121806</td>
      <td>0.029334</td>
      <td>1</td>
      <td>0.914444</td>
      <td>1</td>
      <td>0.914444</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.030399</td>
      <td>0.907434</td>
      <td>-0.057620</td>
      <td>-0.062011</td>
      <td>-0.003573</td>
      <td>0.093347</td>
      <td>-0.121806</td>
      <td>0.029334</td>
      <td>1</td>
      <td>0.926667</td>
      <td>1</td>
      <td>0.926667</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.030399</td>
      <td>0.907434</td>
      <td>-0.057620</td>
      <td>-0.062011</td>
      <td>-0.003573</td>
      <td>0.093347</td>
      <td>-0.121806</td>
      <td>0.029334</td>
      <td>1</td>
      <td>0.938889</td>
      <td>1</td>
      <td>0.938889</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.030399</td>
      <td>0.907434</td>
      <td>-0.057620</td>
      <td>-0.062011</td>
      <td>-0.003573</td>
      <td>0.093347</td>
      <td>-0.121806</td>
      <td>0.029334</td>
      <td>1</td>
      <td>0.951111</td>
      <td>1</td>
      <td>0.951111</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.030399</td>
      <td>0.907434</td>
      <td>-0.057620</td>
      <td>-0.062011</td>
      <td>-0.003573</td>
      <td>0.093347</td>
      <td>-0.121806</td>
      <td>0.029334</td>
      <td>1</td>
      <td>0.963333</td>
      <td>1</td>
      <td>0.963333</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-0.030399</td>
      <td>0.907434</td>
      <td>-0.057620</td>
      <td>-0.062011</td>
      <td>-0.003573</td>
      <td>0.093347</td>
      <td>-0.121806</td>
      <td>0.029334</td>
      <td>1</td>
      <td>0.975556</td>
      <td>1</td>
      <td>0.975556</td>
    </tr>
    <tr>
      <th>8</th>
      <td>-0.030399</td>
      <td>0.907434</td>
      <td>-0.057620</td>
      <td>-0.062011</td>
      <td>-0.003573</td>
      <td>0.093347</td>
      <td>-0.121806</td>
      <td>0.029334</td>
      <td>1</td>
      <td>0.987778</td>
      <td>1</td>
      <td>0.987778</td>
    </tr>
    <tr>
      <th>9</th>
      <td>-0.030399</td>
      <td>0.907434</td>
      <td>-0.057620</td>
      <td>-0.062011</td>
      <td>-0.003573</td>
      <td>0.093347</td>
      <td>-0.121806</td>
      <td>0.029334</td>
      <td>1</td>
      <td>1.000000</td>
      <td>1</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>-0.394701</td>
      <td>0.949264</td>
      <td>-0.423320</td>
      <td>-0.159773</td>
      <td>-0.067635</td>
      <td>0.379125</td>
      <td>-0.272108</td>
      <td>0.248559</td>
      <td>2</td>
      <td>0.890000</td>
      <td>2</td>
      <td>0.890000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>-0.394922</td>
      <td>0.949284</td>
      <td>-0.418203</td>
      <td>-0.161414</td>
      <td>-0.067504</td>
      <td>0.361087</td>
      <td>-0.271930</td>
      <td>0.248240</td>
      <td>2</td>
      <td>0.902222</td>
      <td>2</td>
      <td>0.902222</td>
    </tr>
    <tr>
      <th>12</th>
      <td>-0.384546</td>
      <td>0.950656</td>
      <td>-0.415238</td>
      <td>-0.158284</td>
      <td>-0.065726</td>
      <td>0.346221</td>
      <td>-0.264790</td>
      <td>0.248218</td>
      <td>2</td>
      <td>0.914444</td>
      <td>2</td>
      <td>0.914444</td>
    </tr>
    <tr>
      <th>13</th>
      <td>-0.338752</td>
      <td>0.956652</td>
      <td>-0.413859</td>
      <td>-0.139984</td>
      <td>-0.057934</td>
      <td>0.279771</td>
      <td>-0.232769</td>
      <td>0.248889</td>
      <td>2</td>
      <td>0.926667</td>
      <td>2</td>
      <td>0.926667</td>
    </tr>
    <tr>
      <th>14</th>
      <td>-0.290003</td>
      <td>0.962762</td>
      <td>-0.399370</td>
      <td>-0.124611</td>
      <td>-0.049766</td>
      <td>0.237835</td>
      <td>-0.199759</td>
      <td>0.249129</td>
      <td>2</td>
      <td>0.938889</td>
      <td>2</td>
      <td>0.938889</td>
    </tr>
    <tr>
      <th>15</th>
      <td>-0.274877</td>
      <td>0.964804</td>
      <td>-0.397054</td>
      <td>-0.118310</td>
      <td>-0.046976</td>
      <td>0.226874</td>
      <td>-0.189034</td>
      <td>0.248504</td>
      <td>2</td>
      <td>0.951111</td>
      <td>2</td>
      <td>0.951111</td>
    </tr>
    <tr>
      <th>16</th>
      <td>-0.275487</td>
      <td>0.964600</td>
      <td>-0.392754</td>
      <td>-0.120321</td>
      <td>-0.047257</td>
      <td>0.256128</td>
      <td>-0.190011</td>
      <td>0.248705</td>
      <td>2</td>
      <td>0.963333</td>
      <td>2</td>
      <td>0.963333</td>
    </tr>
    <tr>
      <th>17</th>
      <td>-0.259095</td>
      <td>0.966907</td>
      <td>-0.382147</td>
      <td>-0.116257</td>
      <td>-0.044427</td>
      <td>0.219501</td>
      <td>-0.177407</td>
      <td>0.250425</td>
      <td>2</td>
      <td>0.975556</td>
      <td>2</td>
      <td>0.975556</td>
    </tr>
    <tr>
      <th>18</th>
      <td>-0.231428</td>
      <td>0.970450</td>
      <td>-0.374383</td>
      <td>-0.105917</td>
      <td>-0.039654</td>
      <td>0.193159</td>
      <td>-0.158521</td>
      <td>0.250146</td>
      <td>2</td>
      <td>0.987778</td>
      <td>2</td>
      <td>0.987778</td>
    </tr>
    <tr>
      <th>19</th>
      <td>-0.211237</td>
      <td>0.972956</td>
      <td>-0.363663</td>
      <td>-0.099795</td>
      <td>-0.036292</td>
      <td>0.187257</td>
      <td>-0.144940</td>
      <td>0.250391</td>
      <td>2</td>
      <td>1.000000</td>
      <td>2</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.865758</td>
      <td>1.095205</td>
      <td>-0.187895</td>
      <td>0.629149</td>
      <td>0.118214</td>
      <td>0.571595</td>
      <td>0.528441</td>
      <td>0.223704</td>
      <td>3</td>
      <td>0.890000</td>
      <td>3</td>
      <td>0.890000</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.803312</td>
      <td>1.088802</td>
      <td>-0.195956</td>
      <td>0.563314</td>
      <td>0.110385</td>
      <td>0.545631</td>
      <td>0.492495</td>
      <td>0.224134</td>
      <td>3</td>
      <td>0.902222</td>
      <td>3</td>
      <td>0.902222</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.784442</td>
      <td>1.086636</td>
      <td>-0.196747</td>
      <td>0.548159</td>
      <td>0.107849</td>
      <td>0.526620</td>
      <td>0.480857</td>
      <td>0.224284</td>
      <td>3</td>
      <td>0.914444</td>
      <td>3</td>
      <td>0.914444</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.814461</td>
      <td>1.089579</td>
      <td>-0.201706</td>
      <td>0.552762</td>
      <td>0.111495</td>
      <td>0.541877</td>
      <td>0.495670</td>
      <td>0.224938</td>
      <td>3</td>
      <td>0.926667</td>
      <td>3</td>
      <td>0.926667</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.777083</td>
      <td>1.085722</td>
      <td>-0.200101</td>
      <td>0.532955</td>
      <td>0.106645</td>
      <td>0.526376</td>
      <td>0.473973</td>
      <td>0.225002</td>
      <td>3</td>
      <td>0.938889</td>
      <td>3</td>
      <td>0.938889</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.739478</td>
      <td>1.081780</td>
      <td>-0.199158</td>
      <td>0.510195</td>
      <td>0.101609</td>
      <td>0.497388</td>
      <td>0.452876</td>
      <td>0.224364</td>
      <td>3</td>
      <td>0.951111</td>
      <td>3</td>
      <td>0.951111</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.682262</td>
      <td>1.075630</td>
      <td>-0.207271</td>
      <td>0.452377</td>
      <td>0.093764</td>
      <td>0.450778</td>
      <td>0.418554</td>
      <td>0.224020</td>
      <td>3</td>
      <td>0.963333</td>
      <td>3</td>
      <td>0.963333</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.797452</td>
      <td>1.088368</td>
      <td>-0.206488</td>
      <td>0.531295</td>
      <td>0.109706</td>
      <td>0.517328</td>
      <td>0.489121</td>
      <td>0.224292</td>
      <td>3</td>
      <td>0.975556</td>
      <td>3</td>
      <td>0.975556</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.642075</td>
      <td>1.071416</td>
      <td>-0.216759</td>
      <td>0.407114</td>
      <td>0.088246</td>
      <td>0.443749</td>
      <td>0.394706</td>
      <td>0.223573</td>
      <td>3</td>
      <td>0.987778</td>
      <td>3</td>
      <td>0.987778</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0.609242</td>
      <td>1.067906</td>
      <td>-0.216307</td>
      <td>0.387451</td>
      <td>0.083808</td>
      <td>0.430977</td>
      <td>0.376000</td>
      <td>0.222894</td>
      <td>3</td>
      <td>1.000000</td>
      <td>3</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>70</th>
      <td>-0.261760</td>
      <td>0.962538</td>
      <td>-0.387650</td>
      <td>-0.119895</td>
      <td>-0.046477</td>
      <td>0.651981</td>
      <td>-0.194416</td>
      <td>0.239062</td>
      <td>8</td>
      <td>0.890000</td>
      <td>8</td>
      <td>0.890000</td>
    </tr>
    <tr>
      <th>71</th>
      <td>-0.278870</td>
      <td>0.960079</td>
      <td>-0.388816</td>
      <td>-0.127349</td>
      <td>-0.049515</td>
      <td>0.660904</td>
      <td>-0.207064</td>
      <td>0.239131</td>
      <td>8</td>
      <td>0.902222</td>
      <td>8</td>
      <td>0.902222</td>
    </tr>
    <tr>
      <th>72</th>
      <td>-0.315064</td>
      <td>0.954979</td>
      <td>-0.393129</td>
      <td>-0.142366</td>
      <td>-0.055968</td>
      <td>0.683543</td>
      <td>-0.233772</td>
      <td>0.239414</td>
      <td>8</td>
      <td>0.914444</td>
      <td>8</td>
      <td>0.914444</td>
    </tr>
    <tr>
      <th>73</th>
      <td>-0.346856</td>
      <td>0.950127</td>
      <td>-0.393620</td>
      <td>-0.156941</td>
      <td>-0.061775</td>
      <td>0.700848</td>
      <td>-0.258312</td>
      <td>0.239149</td>
      <td>8</td>
      <td>0.926667</td>
      <td>8</td>
      <td>0.926667</td>
    </tr>
    <tr>
      <th>74</th>
      <td>-0.375088</td>
      <td>0.945772</td>
      <td>-0.397163</td>
      <td>-0.168844</td>
      <td>-0.067059</td>
      <td>0.706888</td>
      <td>-0.280490</td>
      <td>0.239077</td>
      <td>8</td>
      <td>0.938889</td>
      <td>8</td>
      <td>0.938889</td>
    </tr>
    <tr>
      <th>75</th>
      <td>-0.389589</td>
      <td>0.943351</td>
      <td>-0.398268</td>
      <td>-0.175752</td>
      <td>-0.069996</td>
      <td>0.707430</td>
      <td>-0.292512</td>
      <td>0.239294</td>
      <td>8</td>
      <td>0.951111</td>
      <td>8</td>
      <td>0.951111</td>
    </tr>
    <tr>
      <th>76</th>
      <td>-0.424658</td>
      <td>0.938334</td>
      <td>-0.402080</td>
      <td>-0.189525</td>
      <td>-0.076204</td>
      <td>0.726481</td>
      <td>-0.318382</td>
      <td>0.239348</td>
      <td>8</td>
      <td>0.963333</td>
      <td>8</td>
      <td>0.963333</td>
    </tr>
    <tr>
      <th>77</th>
      <td>-0.404131</td>
      <td>0.941287</td>
      <td>-0.389190</td>
      <td>-0.185889</td>
      <td>-0.072346</td>
      <td>0.730670</td>
      <td>-0.302868</td>
      <td>0.238870</td>
      <td>8</td>
      <td>0.975556</td>
      <td>8</td>
      <td>0.975556</td>
    </tr>
    <tr>
      <th>78</th>
      <td>-0.454829</td>
      <td>0.933659</td>
      <td>-0.404392</td>
      <td>-0.201799</td>
      <td>-0.081606</td>
      <td>0.746684</td>
      <td>-0.341394</td>
      <td>0.239037</td>
      <td>8</td>
      <td>0.987778</td>
      <td>8</td>
      <td>0.987778</td>
    </tr>
    <tr>
      <th>79</th>
      <td>-0.464434</td>
      <td>0.931948</td>
      <td>-0.401173</td>
      <td>-0.207959</td>
      <td>-0.083427</td>
      <td>0.747746</td>
      <td>-0.348744</td>
      <td>0.239222</td>
      <td>8</td>
      <td>1.000000</td>
      <td>8</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>80</th>
      <td>0.020197</td>
      <td>1.002850</td>
      <td>-0.301309</td>
      <td>0.011872</td>
      <td>0.003577</td>
      <td>0.404650</td>
      <td>0.014804</td>
      <td>0.241646</td>
      <td>9</td>
      <td>0.890000</td>
      <td>9</td>
      <td>0.890000</td>
    </tr>
    <tr>
      <th>81</th>
      <td>-0.008266</td>
      <td>0.998833</td>
      <td>-0.300478</td>
      <td>-0.004873</td>
      <td>-0.001464</td>
      <td>0.443263</td>
      <td>-0.006057</td>
      <td>0.241749</td>
      <td>9</td>
      <td>0.902222</td>
      <td>9</td>
      <td>0.902222</td>
    </tr>
    <tr>
      <th>82</th>
      <td>0.016808</td>
      <td>1.002379</td>
      <td>-0.295340</td>
      <td>0.010108</td>
      <td>0.002985</td>
      <td>0.387145</td>
      <td>0.012341</td>
      <td>0.241894</td>
      <td>9</td>
      <td>0.914444</td>
      <td>9</td>
      <td>0.914444</td>
    </tr>
    <tr>
      <th>83</th>
      <td>-0.006491</td>
      <td>0.999072</td>
      <td>-0.302812</td>
      <td>-0.003836</td>
      <td>-0.001162</td>
      <td>0.443375</td>
      <td>-0.004797</td>
      <td>0.242166</td>
      <td>9</td>
      <td>0.926667</td>
      <td>9</td>
      <td>0.926667</td>
    </tr>
    <tr>
      <th>84</th>
      <td>-0.002304</td>
      <td>0.999670</td>
      <td>-0.301475</td>
      <td>-0.001369</td>
      <td>-0.000413</td>
      <td>0.426622</td>
      <td>-0.001705</td>
      <td>0.242171</td>
      <td>9</td>
      <td>0.938889</td>
      <td>9</td>
      <td>0.938889</td>
    </tr>
    <tr>
      <th>85</th>
      <td>0.024993</td>
      <td>1.003590</td>
      <td>-0.299394</td>
      <td>0.014990</td>
      <td>0.004488</td>
      <td>0.365802</td>
      <td>0.018534</td>
      <td>0.242152</td>
      <td>9</td>
      <td>0.951111</td>
      <td>9</td>
      <td>0.951111</td>
    </tr>
    <tr>
      <th>86</th>
      <td>0.015272</td>
      <td>1.002194</td>
      <td>-0.300933</td>
      <td>0.009109</td>
      <td>0.002741</td>
      <td>0.385760</td>
      <td>0.011321</td>
      <td>0.242130</td>
      <td>9</td>
      <td>0.963333</td>
      <td>9</td>
      <td>0.963333</td>
    </tr>
    <tr>
      <th>87</th>
      <td>0.058981</td>
      <td>1.008517</td>
      <td>-0.274385</td>
      <td>0.038394</td>
      <td>0.010535</td>
      <td>0.324892</td>
      <td>0.043780</td>
      <td>0.240629</td>
      <td>9</td>
      <td>0.975556</td>
      <td>9</td>
      <td>0.975556</td>
    </tr>
    <tr>
      <th>88</th>
      <td>0.049183</td>
      <td>1.007144</td>
      <td>-0.276244</td>
      <td>0.031788</td>
      <td>0.008781</td>
      <td>0.303927</td>
      <td>0.036597</td>
      <td>0.239940</td>
      <td>9</td>
      <td>0.987778</td>
      <td>9</td>
      <td>0.987778</td>
    </tr>
    <tr>
      <th>89</th>
      <td>0.013903</td>
      <td>1.002030</td>
      <td>-0.283303</td>
      <td>0.008775</td>
      <td>0.002486</td>
      <td>0.373988</td>
      <td>0.010368</td>
      <td>0.239792</td>
      <td>9</td>
      <td>1.000000</td>
      <td>9</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>90</th>
      <td>0.212142</td>
      <td>1.030036</td>
      <td>-0.212291</td>
      <td>0.178327</td>
      <td>0.037857</td>
      <td>0.000436</td>
      <td>0.156023</td>
      <td>0.242639</td>
      <td>10</td>
      <td>0.890000</td>
      <td>10</td>
      <td>0.890000</td>
    </tr>
    <tr>
      <th>91</th>
      <td>0.189027</td>
      <td>1.027111</td>
      <td>-0.218735</td>
      <td>0.155086</td>
      <td>0.033923</td>
      <td>0.006090</td>
      <td>0.140398</td>
      <td>0.241618</td>
      <td>10</td>
      <td>0.902222</td>
      <td>10</td>
      <td>0.902222</td>
    </tr>
    <tr>
      <th>92</th>
      <td>0.140469</td>
      <td>1.020201</td>
      <td>-0.231355</td>
      <td>0.109182</td>
      <td>0.025260</td>
      <td>0.055424</td>
      <td>0.104509</td>
      <td>0.241699</td>
      <td>10</td>
      <td>0.914444</td>
      <td>10</td>
      <td>0.914444</td>
    </tr>
    <tr>
      <th>93</th>
      <td>0.142193</td>
      <td>1.020456</td>
      <td>-0.237222</td>
      <td>0.107611</td>
      <td>0.025528</td>
      <td>0.073825</td>
      <td>0.105803</td>
      <td>0.241276</td>
      <td>10</td>
      <td>0.926667</td>
      <td>10</td>
      <td>0.926667</td>
    </tr>
    <tr>
      <th>94</th>
      <td>0.169947</td>
      <td>1.024277</td>
      <td>-0.239115</td>
      <td>0.126515</td>
      <td>0.030252</td>
      <td>0.045638</td>
      <td>0.125813</td>
      <td>0.240449</td>
      <td>10</td>
      <td>0.938889</td>
      <td>10</td>
      <td>0.938889</td>
    </tr>
    <tr>
      <th>95</th>
      <td>0.146121</td>
      <td>1.020981</td>
      <td>-0.243269</td>
      <td>0.107102</td>
      <td>0.026055</td>
      <td>0.076404</td>
      <td>0.108441</td>
      <td>0.240265</td>
      <td>10</td>
      <td>0.951111</td>
      <td>10</td>
      <td>0.951111</td>
    </tr>
    <tr>
      <th>96</th>
      <td>0.127446</td>
      <td>1.018338</td>
      <td>-0.248438</td>
      <td>0.091641</td>
      <td>0.022767</td>
      <td>0.111541</td>
      <td>0.094687</td>
      <td>0.240447</td>
      <td>10</td>
      <td>0.963333</td>
      <td>10</td>
      <td>0.963333</td>
    </tr>
    <tr>
      <th>97</th>
      <td>0.116868</td>
      <td>1.016833</td>
      <td>-0.249378</td>
      <td>0.083708</td>
      <td>0.020875</td>
      <td>0.125095</td>
      <td>0.086766</td>
      <td>0.240589</td>
      <td>10</td>
      <td>0.975556</td>
      <td>10</td>
      <td>0.975556</td>
    </tr>
    <tr>
      <th>98</th>
      <td>0.081472</td>
      <td>1.011726</td>
      <td>-0.255163</td>
      <td>0.056961</td>
      <td>0.014534</td>
      <td>0.193610</td>
      <td>0.060424</td>
      <td>0.240539</td>
      <td>10</td>
      <td>0.987778</td>
      <td>10</td>
      <td>0.987778</td>
    </tr>
    <tr>
      <th>99</th>
      <td>0.067908</td>
      <td>1.009764</td>
      <td>-0.257274</td>
      <td>0.047106</td>
      <td>0.012119</td>
      <td>0.225512</td>
      <td>0.050339</td>
      <td>0.240748</td>
      <td>10</td>
      <td>1.000000</td>
      <td>10</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 12 columns</p>
</div>



####Heatmaps - Small


    fig = plt.figure(figsize=(15,15))
    
    ax1 = fig.add_subplot(3,3,1)
    ax1.set_title("Sharpe Ratio", fontsize=16)
    ax1 = sns.heatmap(results_df_sharpe, annot=True, cbar=False, annot_kws={"size": 8}, cmap='Greens')
    
    ax2 = fig.add_subplot(3,3,2)
    ax2.set_title("Annual Return", fontsize=16)
    ax2 = sns.heatmap(results_df_annual_return, annot=True, cbar=False, annot_kws={"size": 8}, cmap='Greens')
    
    ax3 = fig.add_subplot(3,3,3)
    ax3.set_title("Max Drawdown", fontsize=16)
    ax3 = sns.heatmap(results_df_max_drawdown, annot=True, cbar=False, annot_kws={"size": 8}, cmap='Reds_r')
    
    ax4 = fig.add_subplot(3,3,4)
    ax4.set_title("Volatility", fontsize=16)
    ax4 = sns.heatmap(results_df_volatility, annot=True, cbar=False, annot_kws={"size": 8}, cmap='Reds')
    
    ax5 = fig.add_subplot(3,3,5)
    ax5.set_title("Stability", fontsize=16)
    ax5 = sns.heatmap(results_df_stability, annot=True, cbar=False, annot_kws={"size": 8}, cmap='Blues')
    
    ax6 = fig.add_subplot(3,3,6)
    ax6.set_title("Calmar Ratio", fontsize=16)
    ax6 = sns.heatmap(results_df_calmar, annot=True, cbar=False, annot_kws={"size": 8}, cmap='Blues')
    
    ax7 = fig.add_subplot(3,3,7)
    ax7.set_title("Sortino", fontsize=16)
    ax7 = sns.heatmap(results_df_sortino, annot=True, cbar=False, annot_kws={"size": 8}, cmap='Greens')
    
    ax8 = fig.add_subplot(3,3,8)
    ax8.set_title("Omega Ratio", fontsize=16)
    ax8 = sns.heatmap(results_df_omega, annot=True, cbar=False, annot_kws={"size": 8}, cmap='Blues')
    
    for ax in fig.get_axes():
        ax.tick_params(axis='x', labelsize=3)
        ax.tick_params(axis='y', labelsize=3)
        ax.set_xlabel(ax.get_xlabel(), fontdict={'fontsize' : 6})
        ax.set_ylabel(ax.get_ylabel(), fontdict={'fontsize' : 6})


![png](output_19_0.png)


####Heatmaps - Large


    fig = plt.figure(figsize=(15,80))
    
    ax1 = fig.add_subplot(8,1,1)
    ax1.set_title("Sharpe Ratio", fontsize=16)
    ax1 = sns.heatmap(results_df_sharpe, annot=True, cbar=False, annot_kws={"size": 14}, cmap='Greens')
    
    ax2 = fig.add_subplot(8,1,2)
    ax2.set_title("Annual Return", fontsize=16)
    ax2 = sns.heatmap(results_df_annual_return, annot=True, cbar=False, annot_kws={"size": 14}, cmap='Greens')
    
    ax3 = fig.add_subplot(8,1,3)
    ax3.set_title("Max Drawdown", fontsize=16)
    ax3 = sns.heatmap(results_df_max_drawdown, annot=True, cbar=False, annot_kws={"size": 14}, cmap='Reds_r')
    
    ax4 = fig.add_subplot(8,1,4)
    ax4.set_title("Volatility", fontsize=16)
    ax4 = sns.heatmap(results_df_volatility, annot=True, cbar=False, annot_kws={"size": 14}, cmap='Reds')
    
    ax5 = fig.add_subplot(8,1,5)
    ax5.set_title("Stability", fontsize=16)
    ax5 = sns.heatmap(results_df_stability, annot=True, cbar=False, annot_kws={"size": 14}, cmap='Blues')
    
    ax6 = fig.add_subplot(8,1,6)
    ax6.set_title("Calmar Ratio", fontsize=16)
    ax6 = sns.heatmap(results_df_calmar, annot=True, cbar=False, annot_kws={"size": 14}, cmap='Blues')
    
    ax7 = fig.add_subplot(8,1,7)
    ax7.set_title("Sortino", fontsize=16)
    ax7 = sns.heatmap(results_df_sortino, annot=True, cbar=False, annot_kws={"size": 14}, cmap='Greens')
    
    ax8 = fig.add_subplot(8,1,8)
    ax8.set_title("Omega Ratio", fontsize=16)
    ax8 = sns.heatmap(results_df_omega, annot=True, cbar=False, annot_kws={"size": 14}, cmap='Blues')
    
    for ax in fig.get_axes():
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.set_xlabel(ax.get_xlabel(), fontdict={'fontsize' : 15})
        ax.set_ylabel(ax.get_ylabel(), fontdict={'fontsize' : 15})


![png](output_21_0.png)

