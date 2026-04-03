import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt


import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt


# ------------------- 数据处理 ----------------------------------------

def calc_fwd_ret(price_df: pd.DataFrame, date_col: str = 'date', ticker_col: str = 'ticker',
                 price_col: str = 'close', periods: list = [1, 5, 20]) -> pd.DataFrame:
    """
    计算并对齐未来收益率。日频
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
    s = s.groupby(df[ind_col]).transform(lambda x: x.fillna(x.median()))
    return s.fillna(s.median())


def rank_factor(s: pd.Series) -> pd.Series:
    """
    将因子值转化为横截面上的百分比排名 (0 到 1)。应对长尾分布和异常值。
    """
    return s.rank(pct=True)


def handle_outliers(s: pd.Series) -> pd.Series:
    """
    中位数去极值 MAD法
    """
    median = s.median()
    mad = (s - median).abs().median()
    threshold = 3 * 1.4826 * mad
    return s.clip(median - threshold, median + threshold)

def build_price_volume_features(df: pd.DataFrame,ticker_col: str = 'ticker',date_col: str = 'date',close_col: str = 'close',
                                amount_col: str = 'amount',windows: dict = None,beta_window: int = 60, min_beta_obs: int = 20,
								market_ret: pd.Series = None) -> pd.DataFrame:
    """
    从价格量数据构造常见风格特征。
    
    生成列包括：
    - ret
    - momentum
    - volatility
    - liquidity
    - beta

    参数:
    - windows: {'momentum': 20, 'volatility': 20, 'liquidity': 20}
    - market_ret: 可选，外部传入市场收益率；若不传则默认按全市场等权收益率构造
    """
    if windows is None:
        windows = {'momentum': 20, 'volatility': 20, 'liquidity': 20}

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out = out.sort_values([ticker_col, date_col]).copy()

    out['ret'] = out.groupby(ticker_col)[close_col].pct_change()

    if market_ret is None:
        market_ret = out.groupby(date_col)['ret'].mean().rename('market_ret')

    out = out.merge(market_ret.rename('market_ret'), on=date_col, how='left')

    mom_w = windows.get('momentum', 20)
    vol_w = windows.get('volatility', 20)
    liq_w = windows.get('liquidity', 20)

    out['momentum'] = (
        out.groupby(ticker_col)['ret']
        .transform(lambda x: (1 + x).rolling(mom_w).apply(np.prod, raw=True) - 1))

    out['volatility'] = (
        out.groupby(ticker_col)['ret']
        .transform(lambda x: x.rolling(vol_w).std()))

    out['liquidity'] = (
        out.groupby(ticker_col)[amount_col]
        .transform(lambda x: np.log1p(x).rolling(liq_w).mean()))

    def _rolling_beta(g):
        x = g['market_ret']
        y = g['ret']
        cov_xy = y.rolling(beta_window, min_periods=min_beta_obs).cov(x)
        var_x = x.rolling(beta_window, min_periods=min_beta_obs).var()
        return cov_xy / (var_x + 1e-12)

    out['beta'] = (
        out.groupby(ticker_col, group_keys=False)
        .apply(_rolling_beta)
        .reset_index(level=0, drop=True))

    return out


def standardize(s: pd.Series) -> pd.Series:
    """
    Z-Score 标准化
    """
    return (s - s.mean()) / (s.std() + 1e-12)

def neutralize_factors(df: pd.DataFrame,factor_col: str,numeric_exposure_cols: list = None,categorical_exposure_cols: list = None,
					   log_transform_cols: list = None,min_obs: int = 20, add_constant: bool = True) -> pd.Series:
    """
    通用横截面中性化函数。
    
    用数值型暴露 + 类别型暴露对 factor_col 做横截面 OLS 回归，
    返回残差序列。
    
    示例:
    numeric_exposure_cols = ['market_cap', 'momentum', 'volatility', 'liquidity', 'beta']
    categorical_exposure_cols = ['industry']
    log_transform_cols = ['market_cap']
    """
    if numeric_exposure_cols is None:
        numeric_exposure_cols = []
    if categorical_exposure_cols is None:
        categorical_exposure_cols = []
    if log_transform_cols is None:
        log_transform_cols = []

    use_cols = [factor_col] + numeric_exposure_cols + categorical_exposure_cols
    df_clean = df.dropna(subset=use_cols).copy()

    if len(df_clean) < min_obs:
        return pd.Series(index=df.index, dtype=float)

    y = df_clean[factor_col].astype(float)

    X_parts = []

    if len(numeric_exposure_cols) > 0:
        X_num = df_clean[numeric_exposure_cols].astype(float).copy()

        for col in log_transform_cols:
            if col in X_num.columns:
                X_num[col] = np.log(X_num[col].replace(0, np.nan))

        X_num = X_num.replace([np.inf, -np.inf], np.nan)
        valid_mask = X_num.notna().all(axis=1)

        df_clean = df_clean.loc[valid_mask].copy()
        y = y.loc[valid_mask]
        X_num = X_num.loc[valid_mask]

        if len(df_clean) < min_obs:
            return pd.Series(index=df.index, dtype=float)

        X_parts.append(X_num)

    for col in categorical_exposure_cols:
        dummies = pd.get_dummies(df_clean[col], prefix=col, drop_first=True, dtype=float)
        X_parts.append(dummies)

    if len(X_parts) == 0:
        return pd.Series(index=df.index, dtype=float)

    X = pd.concat(X_parts, axis=1)

    if add_constant:
        X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()
    resid = pd.Series(model.resid, index=df_clean.index)

    return resid.reindex(df.index)
	

def preprocess_factor_general(df: pd.DataFrame,factor_col: str, date_col: str = 'date', winsorize_func=None, standardize_func=None,
							  fillna_func=None,neutralize: bool = False, numeric_exposure_cols: list = None, categorical_exposure_cols: list = None,
							  log_transform_cols: list = None, min_obs: int = 20, new_col: str = None) -> pd.DataFrame:
    """
    通用因子预处理流水线：
    - 可选填充缺失
    - 可选去极值
    - 可选标准化
    - 可选中性化
    - 可选再次标准化
    """
    out = df.copy()
    out_col = new_col if new_col is not None else factor_col
    out[out_col] = out[factor_col].copy()

    if fillna_func is not None:
        out[out_col] = out.groupby(date_col, group_keys=False).apply(
            lambda x: fillna_func(x, out_col)).reset_index(level=0, drop=True)

    if winsorize_func is not None:
        out[out_col] = out.groupby(date_col)[out_col].transform(winsorize_func)

    if standardize_func is not None:
        out[out_col] = out.groupby(date_col)[out_col].transform(standardize_func)

    if neutralize:
        out[out_col] = out.groupby(date_col, group_keys=False).apply(
            lambda x: neutralize_factors(
                x, factor_col=out_col,
                numeric_exposure_cols=numeric_exposure_cols,
                categorical_exposure_cols=categorical_exposure_cols,
                log_transform_cols=log_transform_cols,
                min_obs=min_obs)).reset_index(level=0, drop=True)

        if standardize_func is not None:
            out[out_col] = out.groupby(date_col)[out_col].transform(standardize_func)

    return out
	
def safe_qcut(s: pd.Series, q: int = 5) -> pd.Series:
    """
    安全的因子横截面分组函数，跳过 NaN
    """
    s_valid = s.dropna()
    if len(s_valid) < q:
        return pd.Series(index=s.index, dtype=float)

    try:
        res = pd.qcut(s_valid, q, labels=False, duplicates='drop')
        return res.reindex(s.index)
    except Exception:
        return pd.Series(index=s.index, dtype=float)


def _get_periods_per_year(freq: str) -> int:
    """
    根据频率返回年化周期数
    freq:
    - 'D' / 'daily'      : 日频，252
    - 'W' / 'weekly'     : 周频，52
    - 'M' / 'monthly'    : 月频，12
    - 'Q' / 'quarterly'  : 季频，4
    - 'Y' / 'yearly'     : 年频，1
    """
    freq_map = {
        'D': 252, 'DAILY': 252,
        'W': 52, 'WEEKLY': 52,
        'M': 12, 'MONTHLY': 12,
        'Q': 4, 'QUARTERLY': 4,
        'Y': 1, 'YEARLY': 1}

    freq_key = str(freq).upper()
    if freq_key not in freq_map:
        raise ValueError("freq 只能是 D/W/M/Q/Y 或 daily/weekly/monthly/quarterly/yearly")
    return freq_map[freq_key]


# ------------------- 1. 频率无关的 IC 工具 ----------------------------------------

def calc_ic(df: pd.DataFrame, factor_col: str, ret_col: str = 'next_ret') -> float:
    """
    单期横截面 Rank IC 计算函数
    """
    df_clean = df.dropna(subset=[factor_col, ret_col])
    if len(df_clean) < 5:
        return np.nan
    return df_clean[factor_col].corr(df_clean[ret_col], method='spearman')


def calc_cross_sectional_ic_series(panel_df: pd.DataFrame, factor_col: str, ret_col: str,
                                   date_col: str = 'date', method: str = 'spearman',
                                   min_obs: int = 5) -> pd.Series:
    """
    计算按 date_col 分组的横截面 IC 序列
    """
    def _single_ic(x):
        x = x.dropna(subset=[factor_col, ret_col])
        if len(x) < min_obs:
            return np.nan
        return x[factor_col].corr(x[ret_col], method=method)

    ic_series = panel_df.groupby(date_col).apply(_single_ic).dropna()
    ic_series.name = f'IC_{factor_col}'
    return ic_series


def summarize_ic_series(ic_series: pd.Series, freq: str = 'D', direction: int = 1) -> pd.Series:
    """
    direction: 1 表示正向因子，-1 表示负向因子
    """
    ic_series = ic_series.dropna()
    if len(ic_series) == 0:
        return pd.Series(dtype='object')

    # 如果是负相关因子，先整体翻转 IC 值，这样均值、ICIR 和胜率都会自动修正
    if direction == -1:
        ic_series = -ic_series

    periods_per_year = _get_periods_per_year(freq)

    ic_mean = ic_series.mean()
    ic_std = ic_series.std(ddof=1)
    icir = (ic_mean / (ic_std + 1e-12)) * np.sqrt(periods_per_year)
    win_rate = (ic_series > 0).mean() 

    return pd.Series({
        'IC均值': ic_mean,
        'IC标准差': ic_std,
        'ICIR': icir,
        'IC胜率': f"{win_rate * 100:.2f}%")


def calc_ic_metrics(panel_df: pd.DataFrame, factor_col: str, date_col: str = 'date',
                    ret_cols: dict = None, freq_map: dict = None,
                    method: str = 'spearman', min_obs: int = 5) -> pd.Series:
    """
    计算多个收益列对应的 IC 均值、ICIR 和 IC 胜率。支持日频/月频/季频。
    """
    if ret_cols is None:
        ret_cols = {'1D': 'next_ret_1d'}

    if freq_map is None:
        freq_map = {k: 'D' for k in ret_cols.keys()}

    metrics = {}

    for period_name, ret_col in ret_cols.items():
        ic_series = calc_cross_sectional_ic_series(
            panel_df=panel_df,
            factor_col=factor_col,
            ret_col=ret_col,
            date_col=date_col,
            method=method,
            min_obs=min_obs)

        if len(ic_series) == 0:
            continue

        summary = summarize_ic_series(ic_series, freq=freq_map.get(period_name, 'D'))

        metrics[f'IC_Mean_{period_name}'] = summary['IC均值']
        metrics[f'IC_Std_{period_name}'] = summary['IC标准差']
        metrics[f'ICIR_{period_name}'] = summary['ICIR']
        metrics[f'IC_WinRate_{period_name}'] = summary['IC胜率']

    return pd.Series(metrics, dtype='object')


def calc_factor_autocorr(panel_df: pd.DataFrame, factor_col: str,
                         date_col: str = 'date', ticker_col: str = 'ticker') -> float:
    """
    计算因子的秩自相关性 (衡量衰减速度)
    """
    pivot_factor = panel_df.pivot(index=date_col, columns=ticker_col, values=factor_col)
    rank_df = pivot_factor.rank(axis=1, pct=True)
    autocorr = rank_df.corrwith(rank_df.shift(1), axis=1, method='spearman')
    return autocorr.mean()


def plot_rolling_ic(panel_df: pd.DataFrame, factor_col: str, ret_col: str = 'next_ret',
                    date_col: str = 'date', window: int = 20, method: str = 'spearman'):
    """
    计算并绘制因子的滚动 IC 曲线
    """
    ic_series = calc_cross_sectional_ic_series(
        panel_df=panel_df,
        factor_col=factor_col,
        ret_col=ret_col,
        date_col=date_col,
        method=method)

    rolling_ic = ic_series.rolling(window=window).mean()

    plt.figure(figsize=(12, 5))
    plt.plot(rolling_ic.index, rolling_ic, label=f'{window}期 Rolling IC', alpha=0.8)
    plt.axhline(0, color='red', linestyle='--', linewidth=1.5)
    plt.axhline(ic_series.mean(), color='green', linestyle='-', label=f'IC Mean ({ic_series.mean():.4f})')

    plt.title(f'Rolling Rank IC - {factor_col}', fontsize=14)
    plt.ylabel('Rank IC', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    return ic_series


# ------------------- 2. 多空净值 / 分组净值函数 ----------------------------------------

def calc_long_short_returns(panel_df: pd.DataFrame, factor_col: str, ret_col: str,
                            date_col: str, q: int = 5, fee_rate: float = 0.0,
                            factor_order: str = 'ascending') -> pd.DataFrame:
    """
    通用分组收益函数：支持日频 / 周频 / 月频 / 季频，只要传入对应收益列和日期列即可。

    factor_order:
    - 'ascending': 因子越小越好，做多 Group_0，做空 Group_{q-1}
    - 'descending': 因子越大越好，做多 Group_{q-1}，做空 Group_0
    """
    df = panel_df.copy()

    required_cols = [date_col, factor_col, ret_col]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少必要列: {missing_cols}")

    df['group'] = df.groupby(date_col)[factor_col].transform(lambda x: safe_qcut(x, q=q))
    df = df.dropna(subset=['group', ret_col])

    group_ret = df.groupby([date_col, 'group'])[ret_col].mean().unstack()

    for g in range(q):
        if g not in group_ret.columns:
            group_ret[g] = np.nan

    group_ret = group_ret[[g for g in range(q)]]
    group_ret.columns = [f'Group_{int(c)}' for c in group_ret.columns]

    low_group = 'Group_0'
    high_group = f'Group_{q-1}'

    if factor_order.lower() == 'ascending':
        group_ret['Long_Short'] = group_ret[low_group] - group_ret[high_group] - fee_rate
    elif factor_order.lower() == 'descending':
        group_ret['Long_Short'] = group_ret[high_group] - group_ret[low_group] - fee_rate
    else:
        raise ValueError("factor_order 只能是 'ascending' 或 'descending'")

    return group_ret


def calc_nav(series: pd.Series, fillna_value: float = 0.0, start_value: float = 1.0) -> pd.Series:
    """
    由收益率序列计算净值序列
    """
    s = series.fillna(fillna_value).copy()
    nav = (1 + s).cumprod() * start_value
    return nav


def calc_group_nav(group_ret_df: pd.DataFrame, fillna_value: float = 0.0,
                   start_value: float = 1.0) -> pd.DataFrame:
    """
    将分组收益 DataFrame 转成分组净值 DataFrame
    """
    nav_df = group_ret_df.copy()
    for col in nav_df.columns:
        nav_df[col] = calc_nav(nav_df[col], fillna_value=fillna_value, start_value=start_value)
    return nav_df


def get_metrics(series: pd.Series, freq: str = 'M', rf_annual: float = 0.03) -> pd.Series:
    """
    通用绩效指标函数，支持日频 / 周频 / 月频 / 季频 / 年频
    """
    series = series.dropna()
    if len(series) == 0:
        return pd.Series(dtype='object')

    periods_per_year = _get_periods_per_year(freq)

    n = len(series)
    nav = (1 + series).prod()
    ann_ret = nav ** (periods_per_year / n) - 1

    vol = series.std(ddof=1)
    ann_vol = vol * np.sqrt(periods_per_year)

    rf_per_period = rf_annual / periods_per_year
    excess = series - rf_per_period
    sharpe = (excess.mean() / (vol + 1e-12)) * np.sqrt(periods_per_year)

    cum = (1 + series).cumprod()
    running_max = cum.cummax()
    drawdown = cum / running_max - 1
    mdd = drawdown.min()

    return pd.Series({
        '年化收益率': f"{ann_ret * 100:.2f}%",
        '年化波动率': f"{ann_vol * 100:.2f}%",
        '最大回撤': f"{mdd * 100:.2f}%",
        '夏普比率': f"{sharpe:.2f}"})


# ------------------- 3. 通用回测主函数 ----------------------------------------

def run_factor_backtest(panel_df: pd.DataFrame, factor_col: str, ret_col: str, date_col: str,
                        freq: str = 'D', q: int = 5, fee_rate: float = 0.0,
                        factor_order: str = 'ascending', method: str = 'spearman',
                        min_obs: int = 5) -> dict:
    """
    通用单因子回测主函数

    返回：
    - ic_series
    - ic_summary
    - group_ret
    - nav_df
    - long_short_nav
    - performance
    - monotonicity
    - factor_direction
    - coverage
    """
    ic_series = calc_cross_sectional_ic_series(
        panel_df=panel_df,
        factor_col=factor_col,
        ret_col=ret_col,
        date_col=date_col,
        method=method,
        min_obs=min_obs)

    ic_summary = summarize_ic_series(ic_series, freq=freq)

    group_ret = calc_long_short_returns(
        panel_df=panel_df,
        factor_col=factor_col,
        ret_col=ret_col,
        date_col=date_col,
        q=q,
        fee_rate=fee_rate,
        factor_order=factor_order)

    nav_df = calc_group_nav(group_ret)
    long_short_nav = nav_df['Long_Short'] if 'Long_Short' in nav_df.columns else pd.Series(dtype=float)

    performance = get_metrics(group_ret['Long_Short'], freq=freq) if 'Long_Short' in group_ret.columns else pd.Series(dtype='object')

    monotonicity = check_group_monotonicity(group_ret, q=q)
    factor_direction = summarize_factor_direction(ic_series)
    coverage = calc_factor_coverage(panel_df, factor_col=factor_col, date_col=date_col)

    return {
        'ic_series': ic_series,
        'ic_summary': ic_summary,
        'group_ret': group_ret,
        'nav_df': nav_df,
        'long_short_nav': long_short_nav,
        'performance': performance,
        'monotonicity': monotonicity,
        'factor_direction': factor_direction,
        'coverage': coverage}


# ------------------- 5. 方向判断和单调性检查 ----------------------------------------

def summarize_factor_direction(ic_series: pd.Series) -> pd.Series:
    """
    根据 IC 序列判断因子方向
    """
    ic_series = ic_series.dropna()
    if len(ic_series) == 0:
        return pd.Series(dtype='object')

    ic_mean = ic_series.mean()

    if ic_mean > 0:
        direction = 'descending'
        interpretation = '因子越大越好'
    elif ic_mean < 0:
        direction = 'ascending'
        interpretation = '因子越小越好'
    else:
        direction = 'neutral'
        interpretation = '方向不明显'

    return pd.Series({
        'IC均值': ic_mean,
        '方向判断': direction,
        '解释': interpretation})


def check_group_monotonicity(group_ret_df: pd.DataFrame, q: int = 5, freq: str = 'M') -> pd.Series:
    """
    检查分组年化收益是否单调
    """
    group_cols = [f'Group_{i}' for i in range(q) if f'Group_{i}' in group_ret_df.columns]

    if len(group_cols) < 2:
        return pd.Series(dtype='object')

    ann_returns = []
    for col in group_cols:
        metric = get_metrics(group_ret_df[col], freq=freq)
        ann_ret = metric.get('年化收益率', np.nan)
        if isinstance(ann_ret, str):
            ann_ret = float(ann_ret.replace('%', ''))
        ann_returns.append(ann_ret)

    increasing = all(x <= y for x, y in zip(ann_returns[:-1], ann_returns[1:]))
    decreasing = all(x >= y for x, y in zip(ann_returns[:-1], ann_returns[1:]))

    if increasing:
        mono = '单调递增'
    elif decreasing:
        mono = '单调递减'
    else:
        mono = '非单调'

    return pd.Series({
        '各组年化收益率(%)': ann_returns,
        '单调性': mono})


# ------------------- 6. 更通用的换手率函数 ----------------------------------------

def calc_group_turnover(panel_df: pd.DataFrame, factor_col: str, date_col: str = 'date',
                        ticker_col: str = 'ticker', q: int = 5, freq: str = 'D',
                        factor_order: str = 'ascending', side: str = 'long') -> pd.Series:
    """
    计算分组换手率。支持 long / short / long_short
    """
    df = panel_df.copy()
    df['group'] = df.groupby(date_col)[factor_col].transform(lambda x: safe_qcut(x, q=q))
    df = df.dropna(subset=['group'])

    low_group_idx = 0
    high_group_idx = q - 1

    if factor_order.lower() == 'ascending':
        long_idx = low_group_idx
        short_idx = high_group_idx
    elif factor_order.lower() == 'descending':
        long_idx = high_group_idx
        short_idx = low_group_idx
    else:
        raise ValueError("factor_order 只能是 'ascending' 或 'descending'")

    periods_per_year = _get_periods_per_year(freq)

    result = {}

    if side in ['long', 'long_short']:
        long_holdings = df[df['group'] == long_idx].groupby(date_col)[ticker_col].apply(set)
        long_turnovers = []

        for i in range(1, len(long_holdings)):
            today_set = long_holdings.iloc[i]
            prev_set = long_holdings.iloc[i - 1]
            if len(today_set) > 0:
                turnover = len(today_set - prev_set) / len(today_set)
                long_turnovers.append(turnover)

        avg_long_turnover = np.mean(long_turnovers) if len(long_turnovers) > 0 else np.nan
        result['做多组平均单期换手率'] = avg_long_turnover
        result['做多组年化换手率'] = avg_long_turnover * periods_per_year if pd.notna(avg_long_turnover) else np.nan

    if side in ['short', 'long_short']:
        short_holdings = df[df['group'] == short_idx].groupby(date_col)[ticker_col].apply(set)
        short_turnovers = []

        for i in range(1, len(short_holdings)):
            today_set = short_holdings.iloc[i]
            prev_set = short_holdings.iloc[i - 1]
            if len(today_set) > 0:
                turnover = len(today_set - prev_set) / len(today_set)
                short_turnovers.append(turnover)

        avg_short_turnover = np.mean(short_turnovers) if len(short_turnovers) > 0 else np.nan
        result['做空组平均单期换手率'] = avg_short_turnover
        result['做空组年化换手率'] = avg_short_turnover * periods_per_year if pd.notna(avg_short_turnover) else np.nan

    if side == 'long_short':
        vals = [v for k, v in result.items() if '平均单期换手率' in k and pd.notna(v)]
        result['多空平均单期换手率'] = np.mean(vals) if len(vals) > 0 else np.nan
        result['多空年化换手率'] = result['多空平均单期换手率'] * periods_per_year if pd.notna(result['多空平均单期换手率']) else np.nan

    return pd.Series(result)


def calc_long_group_turnover(panel_df: pd.DataFrame, factor_col: str, date_col: str = 'date',
                             ticker_col: str = 'ticker', q: int = 5, freq: str = 'D',
                             factor_order: str = 'ascending') -> float:
    """
    兼容旧接口：只返回做多组平均单期换手率
    """
    res = calc_group_turnover(
        panel_df=panel_df,
        factor_col=factor_col,
        date_col=date_col,
        ticker_col=ticker_col,
        q=q,
        freq=freq,
        factor_order=factor_order,
        side='long')
    return res.get('做多组平均单期换手率', np.nan)


# ------------------- 7. 因子覆盖率 / 有效样本统计 ----------------------------------------

def calc_factor_coverage(panel_df: pd.DataFrame, factor_col: str, date_col: str = 'date') -> pd.Series:
    """
    统计因子覆盖情况
    """
    grouped = panel_df.groupby(date_col)

    coverage_series = grouped[factor_col].apply(lambda x: x.notna().mean())
    valid_count_series = grouped[factor_col].apply(lambda x: x.notna().sum())
    total_count_series = grouped[factor_col].size()

    return pd.Series({
        '平均覆盖率': coverage_series.mean(),
        '最小覆盖率': coverage_series.min(),
        '最大覆盖率': coverage_series.max(),
        '平均有效样本数': valid_count_series.mean(),
        '最小有效样本数': valid_count_series.min(),
        '最大有效样本数': valid_count_series.max(),
        '平均总样本数': total_count_series.mean()})


def calc_cross_section_count(panel_df: pd.DataFrame, factor_col: str, ret_col: str = None,
                             date_col: str = 'date') -> pd.DataFrame:
    """
    统计每期横截面的有效样本数
    """
    df = panel_df.copy()

    if ret_col is None:
        count_df = df.groupby(date_col).agg(
            total_count=(factor_col, 'size'),
            factor_valid_count=(factor_col, lambda x: x.notna().sum()))
    else:
        count_df = df.groupby(date_col).apply(
            lambda x: pd.Series({
                'total_count': len(x),
                'factor_valid_count': x[factor_col].notna().sum(),
                'ret_valid_count': x[ret_col].notna().sum(),
                'joint_valid_count': x[[factor_col, ret_col]].dropna().shape[0]}))

    return count_df


# ------------------- 额外：预处理流水线（ ----------------------------------------

def preprocess_factor(panel_df: pd.DataFrame, factor_col: str, date_col: str = 'date',
                      ind_col: str = 'industry', mcap_col: str = 'market_cap',
                      fillna_industry: bool = False, winsorize: bool = False,
                      zscore: bool = False, rank: bool = False,
                      neutralize_flag: bool = False, new_col: str = None) -> pd.DataFrame:
    """
    通用因子预处理流水线
    """
    df = panel_df.copy()
    out_col = new_col if new_col is not None else factor_col

    df[out_col] = df[factor_col]

    if fillna_industry:
        df[out_col] = fillna_with_industry(df, out_col, ind_col=ind_col)

    if winsorize:
        df[out_col] = df.groupby(date_col)[out_col].transform(handle_outliers)

    if zscore:
        df[out_col] = df.groupby(date_col)[out_col].transform(standardize)

    if neutralize_flag:
        df[out_col] = df.groupby(date_col, group_keys=False).apply(
            lambda x: neutralize(x, out_col, mcap_col=mcap_col, ind_col=ind_col)).reset_index(level=0, drop=True)

    if rank:
        df[out_col] = df.groupby(date_col)[out_col].transform(rank_factor)

    return df