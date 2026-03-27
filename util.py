import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt

def get_stock_pool(df: pd.DataFrame, date_col: str = 'date', ticker_col: str = 'ticker',
				   list_date_col: str = 'list_date', st_col: str = 'is_st', suspend_col: str = 'is_suspended',
				   exclude_new_days: int = 60) -> pd.DataFrame:
    """
	从全市场日历数据中剔除ST股、停牌股、以及上市未满 N 天的次新股。
    
        df: 包含日度状态的数据框 
        date_col: 交易日期字段名
        ticker_col: 股票代码字段名
        list_date_col: 股票上市日期字段名 (datetime 格式)
        st_col: 是否 ST 的布尔/整数字段 (1/True代表ST)
        suspend_col: 是否停牌的布尔/整数字段 (1/True代表停牌)
        exclude_new_days: 剔除上市天数少于该值的股票 (默认 60 天)

    """
    df_clean = df.copy()
    
    #剔除 ST/PT 股票
    if st_col in df_clean.columns:
        df_clean = df_clean[~df_clean[st_col].astype(bool)]
        
    #剔除停牌股票
    if suspend_col in df_clean.columns:
        df_clean = df_clean[~df_clean[suspend_col].astype(bool)]
        
    # 除上市未满 exclude_new_days 天的新股
    if list_date_col in df_clean.columns:
        df_clean[date_col] = pd.to_datetime(df_clean[date_col])
        df_clean[list_date_col] = pd.to_datetime(df_clean[list_date_col])
        
        # 计算上市天数
        listed_days = (df_clean[date_col] - df_clean[list_date_col]).dt.days
        df_clean = df_clean[listed_days >= exclude_new_days]
        
    return df_clean


def get_valid_days(df: pd.DataFrame, ticker_col: str = 'ticker', limit_up_col: str = 'is_limit_up',
                   limit_down_col: str = 'is_limit_down', suspend_col: str = 'is_suspended',
                   lookback_window: int = 20, min_valid_days: int = 10) -> pd.Series:
    """
    获取有效交易日掩码：非停牌且非一字涨跌停，且过去N天满足最低有效天数要求。
    """
    df_tmp = df[[ticker_col]].copy()
    
    # 标记当天是否有效 (1为有效，0为无效)
    is_invalid_today = (df[limit_up_col].astype(bool) | df[limit_down_col].astype(bool) | df[suspend_col].astype(bool))
    df_tmp['is_valid_today'] = (~is_invalid_today).astype(int)
    
    # 滚动计算有效天数
    valid_days_count = df_tmp.groupby(ticker_col)['is_valid_today'].rolling(
        window=lookback_window, min_periods=1).sum().reset_index(level=0, drop=True)
    
    valid_days_count = valid_days_count.reindex(df.index)
    return valid_days_count >= min_valid_days


def calc_fwd_ret(price_df: pd.DataFrame, date_col: str = 'date', ticker_col: str = 'ticker', 
                 price_col: str = 'close', periods: list = [1, 5, 20]) -> pd.DataFrame:
    """
    计算并对齐未来收益率 (Forward Returns)。
    """
    df = price_df.sort_values(by=[ticker_col, date_col]).copy()
    
    for period in periods:
        future_price = df.groupby(ticker_col)[price_col].shift(-period)
        df[f'next_ret_{period}d'] = future_price / df[price_col] - 1 
    return df


def fillna_with_industry(df: pd.DataFrame, factor_col: str, ind_col: str = 'industry') -> pd.Series:
    """
    用该股票所属行业的因子中位数填充 NaN。
    """
    s = df[factor_col].copy()
    # 按行业分组，计算中位数并填充
    s = s.groupby(df[ind_col]).transform(lambda x: x.fillna(x.median()))
    # 如果整个行业都是 NaN，再用全市场的全局中位数兜底
    return s.fillna(s.median())

def rank_factor(s: pd.Series) -> pd.Series:
    """
    将因子值转化为横截面上的百分比排名 (0 到 1 之间)。应对长尾分布和异常值。
    """
    return s.rank(pct=True)

def handle_outliers(s: pd.Series) -> pd.Series:
    """中位数去极值 (MAD法)"""
    median = s.median()
    mad = (s - median).abs().median()
    threshold = 3 * 1.4826 * mad
    return s.clip(median - threshold, median + threshold)


def standardize(s: pd.Series) -> pd.Series:
    """Z-Score 标准化"""
    return (s - s.mean()) / (s.std() + 1e-12)

	
def neutralize(df: pd.DataFrame, factor_col: str, mcap_col: str = 'market_cap', ind_col: str = 'industry') -> pd.Series:
    """通用市值行业中性化函数"""
    # 剔除NaN
    df_clean = df.dropna(subset=[factor_col, mcap_col, ind_col]).copy()
    
    # 如果当天有效股票太少，不够跑线性回归，直接返回 NaN
    if len(df_clean) < 10:
        return pd.Series(index=df.index, dtype=float)
        
    y = df_clean[factor_col]
    size = np.log(df_clean[mcap_col])
    # 哑变量处理，drop_first=True 防止完美共线性
    industry = pd.get_dummies(df_clean[ind_col], drop_first=True)
    
    X = pd.concat([size, industry], axis=1)
    X = sm.add_constant(X)
    
    # 拟合线性回归并获取残差
    model = sm.OLS(y, X).fit()
    
    # 将算出的残差贴回原来的 index 上，保证长度不变
    return model.resid.reindex(df.index)


def calc_ic(df: pd.DataFrame, factor_col: str, ret_col: str = 'next_ret') -> float:
    """通用的每日 Rank IC 计算函数"""
    df_clean = df.dropna(subset=[factor_col, ret_col])
    if len(df_clean) < 5: 
        return np.nan 
    return df_clean[factor_col].corr(df_clean[ret_col], method='spearman')

	
def calc_ic_metrics(panel_df: pd.DataFrame, factor_col: str, date_col: str = 'date', 
                    ret_cols: dict = None) -> pd.Series:
    """计算多个持有期的 IC 均值、ICIR 和 IC 胜率。"""
    if ret_cols is None:
        ret_cols = {'1D': 'ret_1d', '5D': 'ret_5d'}
        
    metrics = {}
    for period_name, ret_col in ret_cols.items():
        daily_ic = panel_df.groupby(date_col).apply(
            lambda x: x[factor_col].corr(x[ret_col], method='spearman')).dropna()
        
        if len(daily_ic) == 0:
            continue
            
        ic_mean = daily_ic.mean()
        ic_std = daily_ic.std()
        
        # 提取数字部分进行年化处理
        period_days = int(''.join(filter(str.isdigit, period_name))) 
        annualizer = np.sqrt(252 / period_days) if period_days > 0 else np.sqrt(252)
        
        icir = (ic_mean / (ic_std + 1e-12)) * annualizer
        ic_win_rate = (daily_ic > 0).mean()
        
        metrics[f'IC_Mean_{period_name}'] = ic_mean
        metrics[f'ICIR_{period_name}'] = icir
        metrics[f'IC_WinRate_{period_name}'] = f"{ic_win_rate*100:.2f}%"
        
    return pd.Series(metrics, dtype='object')

def calc_factor_autocorr(panel_df: pd.DataFrame, factor_col: str, date_col: str = 'date', ticker_col: str = 'ticker') -> float:
    """计算因子的日度秩自相关性 (衡量衰减速度)"""
    pivot_factor = panel_df.pivot(index=date_col, columns=ticker_col, values=factor_col)
    rank_df = pivot_factor.rank(axis=1, pct=True)
    autocorr = rank_df.corrwith(rank_df.shift(1), axis=1, method='spearman')
    return autocorr.mean()


def plot_rolling_ic(panel_df: pd.DataFrame, factor_col: str, ret_col: str = 'next_ret', 
                    date_col: str = 'date', window: int = 20):
    """计算并绘制因子的滚动 IC 曲线"""
    # 避免硬编码 'date'，使用传入的 date_col
    daily_ic = panel_df.dropna(subset=[factor_col, ret_col]).groupby(date_col).apply(
        lambda x: x[factor_col].corr(x[ret_col], method='spearman')).dropna()
    
    rolling_ic = daily_ic.rolling(window=window).mean()
    
    # 绘图
    plt.figure(figsize=(12, 5))
    plt.plot(rolling_ic.index, rolling_ic, label=f'{window}D Rolling IC', color='blue', alpha=0.8)
    plt.axhline(0, color='red', linestyle='--', linewidth=1.5)
    plt.axhline(daily_ic.mean(), color='green', linestyle='-', label=f'IC Mean ({daily_ic.mean():.4f})')
    
    plt.title(f'Rolling Rank IC - {factor_col}', fontsize=14)
    plt.ylabel('Rank IC', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return daily_ic


def safe_qcut(s: pd.Series, q: int = 5) -> pd.Series:
    """安全的因子横截面分组函数，跳过 NaN"""
    s_valid = s.dropna()
    if len(s_valid) < q: 
        return pd.Series(index=s.index, dtype=float)
        
    try:
        res = pd.qcut(s_valid, q, labels=False, duplicates='drop')
        return res.reindex(s.index)
    except Exception:
        return pd.Series(index=s.index, dtype=float)

		
def calc_long_group_turnover(panel_df: pd.DataFrame, factor_col: str, date_col: str = 'date', 
                             ticker_col: str = 'ticker', q: int = 5) -> float:
    """计算做多组 (Top Group) 的日均换手率预估"""
    df = panel_df.copy()
    df['group'] = df.groupby(date_col)[factor_col].transform(lambda x: safe_qcut(x, q=q))
    df = df.dropna(subset=['group'])
    
    top_group_idx = q - 1
    long_holdings = df[df['group'] == top_group_idx].groupby(date_col)[ticker_col].apply(set)
    
    turnovers = []
    dates = long_holdings.index
    
    for i in range(1, len(dates)):
        today_set = long_holdings.iloc[i]
        yesterday_set = long_holdings.iloc[i-1]
        
        new_stocks = today_set - yesterday_set
        if len(today_set) > 0:
            turnover = len(new_stocks) / len(today_set)
            turnovers.append(turnover)
            
    avg_daily_turnover = sum(turnovers) / len(turnovers) if turnovers else 0
    print(f"[{factor_col}] 预估做多组日均换手率: {avg_daily_turnover * 100:.2f}%")
    print(f"[{factor_col}] 预估做多组年化换手率: {avg_daily_turnover * 252 * 100:.2f}% (单边)")
    
    return avg_daily_turnover

def calc_long_short_returns(panel_df: pd.DataFrame, factor_col: str, ret_col: str = 'next_ret', 
                            date_col: str = 'date', q: int = 5, fee_rate: float = 0.0015) -> pd.DataFrame:
    """计算因子的 Q1 到 Q5 分组收益，以及 Top - Bottom 多空收益。"""
    df = panel_df.copy()
    
    df['group'] = df.groupby(date_col)[factor_col].transform(lambda x: safe_qcut(x, q=q))
    df = df.dropna(subset=['group'])
    
    # 计算各组每日等权收益率
    group_ret = df.groupby([date_col, 'group'])[ret_col].mean().unstack()
    group_ret.columns = [f'Group_{int(c)}' for c in group_ret.columns]
    
    top_group = f'Group_{q-1}'
    bottom_group = 'Group_0'
    
    if top_group in group_ret.columns and bottom_group in group_ret.columns:
        group_ret['Long_Short'] = group_ret[top_group] - group_ret[bottom_group] - fee_rate
    else:
        group_ret['Long_Short'] = np.nan
        
    return group_ret


def get_metrics(series: pd.Series) -> pd.Series:
    """输出年化收益率、年化波动率、最大回撤和夏普比率"""
    series = series.dropna()
    if len(series) == 0: 
        return pd.Series(dtype='float64') 
    
    n = len(series)
    nav = (1 + series).prod()
    ann_ret = nav ** (252 / n) - 1
    ann_vol = series.std() * np.sqrt(252)
    
    rf_daily = 0.03 / 252
    excess = series - rf_daily 
    sharpe = (excess.mean() * 252) / (excess.std() * np.sqrt(252) + 1e-12)
    
    cum = (1 + series).cumprod() 
    running_max = cum.cummax()  
    drawdown = (cum - running_max) / running_max 
    mdd = drawdown.min() 

    return pd.Series({
        '年化收益率': f"{ann_ret*100:.2f}%",
        '年化波动率': f"{ann_vol*100:.2f}%",
        '最大回撤': f"{mdd*100:.2f}%",  
        '夏普比率': f"{sharpe:.2f}"})