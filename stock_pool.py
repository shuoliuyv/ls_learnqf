import pandas as pd
import numpy as np
import time
from datetime import timedelta

def is_bj_stock(ts_code):
    """判断是否为北交所股票"""
    return isinstance(ts_code, str) and ts_code.endswith('.BJ')


def get_market_type(ts_code):
    """获取市场类型"""
    if ts_code.endswith('.BJ'):
        return 'BJ'
    elif ts_code.startswith('30') and ts_code.endswith('.SZ'):
        return 'CY'
    elif ts_code.startswith('68') and ts_code.endswith('.SH'):
        return 'KC'
    else:
        return 'MAIN'

class TushareDataCache:
    """
    用于缓存回测所需的各类 Tushare 数据
    """

    def __init__(self, pro, sleep_interval=0.0):
        self.pro = pro
        self.sleep_interval = sleep_interval

        # 静态/半静态缓存
        self.trade_cal_cache = {}
        self.stock_basic_cache = None

        # 按日期缓存
        self.daily_cache = {}    
        self.daily_basic_cache = {}  
        self.bak_basic_cache = {}   

        # 指数成分缓存
        self.index_weight_cache = {}  # {(index_code, signal_date): DataFrame}

    def _sleep(self):
        if self.sleep_interval and self.sleep_interval > 0:
            time.sleep(self.sleep_interval)

    # ---------- 交易日历 ----------

    def get_trade_calendar(self, start_date, end_date, exchange='SSE'):
        key = (exchange, start_date, end_date)
        if key not in self.trade_cal_cache:
            cal = self.pro.trade_cal(exchange=exchange, start_date=start_date, end_date=end_date)
            cal = cal.copy()
            cal['cal_date'] = pd.to_datetime(cal['cal_date'])
            self.trade_cal_cache[key] = cal
            self._sleep()
        return self.trade_cal_cache[key].copy()

    def get_open_days(self, start_date, end_date, exchange='SSE'):
        cal = self.get_trade_calendar(start_date, end_date, exchange=exchange)
        open_days = cal[cal['is_open'] == 1].copy().sort_values('cal_date')
        return open_days

    def get_prev_open_date(self, date_str, exchange='SSE', lookback_days=60):
        """
        获取某一日期之前最近一个交易日
        """
        end_dt = pd.to_datetime(date_str)
        start_dt = end_dt - pd.Timedelta(days=lookback_days)

        open_days = self.get_open_days(start_dt.strftime('%Y%m%d'), end_dt.strftime('%Y%m%d'), exchange=exchange)
        prev_days = open_days[open_days['cal_date'] < end_dt]
        if prev_days.empty:
            return None
        return prev_days.iloc[-1]['cal_date'].strftime('%Y%m%d')

    def preload_stock_basic(self):
        """
        一次性拉取 stock_basic
        """
        if self.stock_basic_cache is not None:
            return

        # A股正常上市
        main_info = self.pro.stock_basic(exchange='', list_status='L', fields='ts_code,list_date')
        self._sleep()

        dfs = [main_info] if main_info is not None and not main_info.empty else []

        # 北交所
        try:
            bj_info = self.pro.stock_basic(exchange='北交所', list_status='L', fields='ts_code,list_date')
            self._sleep()
            if bj_info is not None and not bj_info.empty:
                dfs.append(bj_info)
        except Exception:
            pass

        if dfs:
            stock_info = pd.concat(dfs, axis=0).drop_duplicates(subset=['ts_code']).reset_index(drop=True)
            stock_info['list_date'] = pd.to_datetime(stock_info['list_date'], errors='coerce')
        else:
            stock_info = pd.DataFrame(columns=['ts_code', 'list_date'])

        self.stock_basic_cache = stock_info

    def get_stock_basic(self):
        if self.stock_basic_cache is None:
            self.preload_stock_basic()
        return self.stock_basic_cache.copy()

    # ---------- daily ----------

    def preload_daily(self, trade_dates):
        """
        一次性把所有需要的 trade_date 的 daily 拉到缓存
        """
        unique_dates = sorted(set([d for d in trade_dates if d is not None]))
        for d in unique_dates:
            if d not in self.daily_cache:
                df = self.pro.daily(trade_date=d, fields='ts_code,amount')
                if df is None or df.empty:
                    df = pd.DataFrame(columns=['ts_code', 'amount'])
                self.daily_cache[d] = df.copy()
                self._sleep()

    def get_daily(self, trade_date):
        if trade_date not in self.daily_cache:
            self.preload_daily([trade_date])
        return self.daily_cache[trade_date].copy()

    # ---------- daily_basic ----------

    def preload_daily_basic(self, trade_dates):
        """
        一次性把所有需要的 trade_date 的 daily_basic 拉到缓存
        """
        unique_dates = sorted(set([d for d in trade_dates if d is not None]))
        for d in unique_dates:
            if d not in self.daily_basic_cache:
                df = self.pro.daily_basic(trade_date=d, fields='ts_code,total_mv')
                if df is None or df.empty:
                    df = pd.DataFrame(columns=['ts_code', 'total_mv'])
                self.daily_basic_cache[d] = df.copy()
                self._sleep()

    def get_daily_basic(self, trade_date):
        if trade_date not in self.daily_basic_cache:
            self.preload_daily_basic([trade_date])
        return self.daily_basic_cache[trade_date].copy()

    # ---------- bak_basic ----------

    def preload_bak_basic(self, trade_dates):
        """
        一次性把所有需要的 trade_date 的 bak_basic 拉到缓存
        """
        unique_dates = sorted(set([d for d in trade_dates if d is not None]))
        for d in unique_dates:
            if d not in self.bak_basic_cache:
                try:
                    df = self.pro.bak_basic(trade_date=d, fields='ts_code,name,status')
                    if df is None or df.empty:
                        df = pd.DataFrame(columns=['ts_code', 'name', 'status'])
                except Exception:
                    df = pd.DataFrame(columns=['ts_code', 'name', 'status'])
                self.bak_basic_cache[d] = df.copy()
                self._sleep()

    def get_bak_basic(self, trade_date):
        if trade_date not in self.bak_basic_cache:
            self.preload_bak_basic([trade_date])
        return self.bak_basic_cache[trade_date].copy()

    # ---------- index_weight ----------

    def get_index_pool(self, index_code, signal_date, lookback_days=180):
        """
        获取单个指数成分股。
        未来函数规避：使用 signal_date（调仓前一个交易日）获取成分。
        若当日无数据，则向前回退最近一期 <= signal_date 的成分。
        """
        key = (index_code, signal_date)

        if key in self.index_weight_cache:
            return self.index_weight_cache[key].copy()

        weight_df = self.pro.index_weight(index_code=index_code, trade_date=signal_date)
        self._sleep()

        if weight_df is None or weight_df.empty:
            prev_date = (pd.to_datetime(signal_date) - timedelta(days=lookback_days)).strftime('%Y%m%d')
            weight_df = self.pro.index_weight(
                index_code=index_code,
                start_date=prev_date,
                end_date=signal_date)
            self._sleep()

            if weight_df is not None and not weight_df.empty:
                latest_date = weight_df['trade_date'].max()
                weight_df = weight_df[weight_df['trade_date'] == latest_date]

        if weight_df is None or weight_df.empty:
            pool = pd.DataFrame(columns=['ts_code'])
        else:
            pool = weight_df[['con_code']].rename(columns={'con_code': 'ts_code'})
            pool = pool.drop_duplicates().reset_index(drop=True)

        self.index_weight_cache[key] = pool.copy()
        return pool

    def preload_index_weights(self, pool_type, signal_dates, lookback_days=30):
        """
        一次性把所有会用到的指数成分预拉到缓存
        """
        if pool_type == 'ALL':
            return

        if isinstance(pool_type, str):
            index_codes = [pool_type]
        elif isinstance(pool_type, list):
            index_codes = list(pool_type)
        else:
            raise ValueError(f"不支持的 pool_type: {pool_type}")

        for index_code in index_codes:
            for signal_date in sorted(set(signal_dates)):
                self.get_index_pool(index_code, signal_date, lookback_days=lookback_days)


# ==================== 调仓日====================

def get_rebalance_dates(cache, start_date, end_date, freq='M', exchange='SSE'):
    """
    获取调仓日（执行日。默认取每个周期的第一个交易日
    """
    open_days = cache.get_open_days(start_date, end_date, exchange=exchange)

    if open_days.empty:
        return []

    if freq == 'M':
        rebalance_dates = open_days.groupby(open_days['cal_date'].dt.to_period('M')).first()
    elif freq == 'W':
        rebalance_dates = open_days.groupby(open_days['cal_date'].dt.to_period('W')).first()
    elif freq == 'Q':
        rebalance_dates = open_days.groupby(open_days['cal_date'].dt.to_period('Q')).first()
    else:
        raise ValueError(f"不支持的频率: {freq}")

    return rebalance_dates['cal_date'].dt.strftime('%Y%m%d').tolist()


def get_rebalance_schedule(cache, start_date, end_date, freq='M', exchange='SSE'):
    """
    返回调仓计划：
    - signal_date: 调仓日前一个交易日，用于选股和过滤（避免未来函数）
    - trade_date: 实际调仓日
    """
    trade_dates = get_rebalance_dates(cache, start_date, end_date, freq=freq, exchange=exchange)

    schedule = []
    for trade_date in trade_dates:
        signal_date = cache.get_prev_open_date(trade_date, exchange=exchange, lookback_days=60)
        if signal_date is None:
            continue
        schedule.append({
            'signal_date': signal_date,
            'trade_date': trade_date})

    return schedule


# ==================== 过滤函数 ====================

def get_base_and_traded_pool(cache, signal_date, pool_type='ALL', include_bj=False):
    """
    获取基础池并剔除停牌。注意：这里用 signal_date，而不是 trade_date
    """
    daily_data = cache.get_daily(signal_date)

    if daily_data.empty:
        return pd.DataFrame(columns=['ts_code'])

    traded_stocks = daily_data[daily_data['amount'] > 0][['ts_code']].copy()

    # 获取基础股票池
    if pool_type == 'ALL':
        pool = traded_stocks.copy()

    elif isinstance(pool_type, str):
        pool = cache.get_index_pool(pool_type, signal_date)

    elif isinstance(pool_type, list):
        all_pools = []
        for idx in pool_type:
            idx_pool = cache.get_index_pool(idx, signal_date)
            if not idx_pool.empty:
                all_pools.append(idx_pool)
        if all_pools:
            pool = pd.concat(all_pools).drop_duplicates().reset_index(drop=True)
        else:
            return pd.DataFrame(columns=['ts_code'])
    else:
        raise ValueError(f"不支持的 pool_type: {pool_type}")

    if pool.empty:
        return pool

    # 剔除北交所
    if not include_bj:
        pool = pool[~pool['ts_code'].str.endswith('.BJ')]

    # 剔除停牌（确认有成交额）
    pool = pool[pool['ts_code'].isin(traded_stocks['ts_code'])]

    return pool.drop_duplicates().reset_index(drop=True)


def filter_st_stocks(cache, signal_date, pool_df):
    """剔除 ST、*ST、SST、退市股票"""
    if pool_df.empty:
        return pool_df

    day_basic = cache.get_bak_basic(signal_date)
    if day_basic.empty:
        return pool_df

    valid_stocks = day_basic[
        ~day_basic['name'].str.contains('ST|\\*ST|SST|退', regex=True, na=False)]
    valid_stocks = valid_stocks[valid_stocks['status'] == 'L']

    filtered_pool = pool_df[pool_df['ts_code'].isin(valid_stocks['ts_code'])]
    return filtered_pool.drop_duplicates().reset_index(drop=True)


def filter_new_stocks(cache, signal_date, pool_df, min_list_days=60):
    """剔除上市天数不足的股票"""
    if min_list_days <= 0 or pool_df.empty:
        return pool_df

    stock_info = cache.get_stock_basic()
    if stock_info.empty:
        return pool_df

    merged = pd.merge(pool_df, stock_info, on='ts_code', how='left')
    merged = merged.dropna(subset=['list_date']).copy()

    current_dt = pd.to_datetime(signal_date)
    merged['listed_days'] = (current_dt - merged['list_date']).dt.days

    filtered_pool = merged[merged['listed_days'] >= min_list_days]
    return filtered_pool[['ts_code']].drop_duplicates().reset_index(drop=True)


def filter_by_liquidity(cache, signal_date, pool_df, min_amount=10000000, bj_min_amount=3000000):
    """
    剔除流动性不足的股票
    注意：amount 字段口径与 Tushare daily 保持一致
    """
    if pool_df.empty:
        return pool_df

    daily_data = cache.get_daily(signal_date)
    if daily_data.empty:
        return pool_df

    daily_data = daily_data.copy()
    daily_data['min_amount'] = daily_data['ts_code'].apply(
        lambda x: bj_min_amount if x.endswith('.BJ') else min_amount)

    liquid_stocks = daily_data[daily_data['amount'] >= daily_data['min_amount']]
    filtered_pool = pool_df[pool_df['ts_code'].isin(liquid_stocks['ts_code'])]

    return filtered_pool.drop_duplicates().reset_index(drop=True)


def filter_by_market_cap(cache, signal_date, pool_df, min_mcap=None, max_mcap=None):
    """按市值过滤股票"""
    if pool_df.empty:
        return pool_df

    if min_mcap is None and max_mcap is None:
        return pool_df

    daily_basic = cache.get_daily_basic(signal_date)
    if daily_basic.empty:
        return pool_df

    filtered_basic = daily_basic.copy()

    if min_mcap is not None:
        filtered_basic = filtered_basic[filtered_basic['total_mv'] >= min_mcap]
    if max_mcap is not None:
        filtered_basic = filtered_basic[filtered_basic['total_mv'] <= max_mcap]

    filtered_pool = pool_df[pool_df['ts_code'].isin(filtered_basic['ts_code'])]
    return filtered_pool.drop_duplicates().reset_index(drop=True)

def get_clean_stock_pool(cache, signal_date, pool_type='ALL',include_bj=False,min_list_days=60,
                         min_amount=None,bj_min_amount=None,min_mcap=None,max_mcap=None):
    """选取样本空间"""
    pool = get_base_and_traded_pool(cache, signal_date, pool_type, include_bj)
    if pool.empty:
        return []

    pool = filter_st_stocks(cache, signal_date, pool)
    if pool.empty:
        return []

    pool = filter_new_stocks(cache, signal_date, pool, min_list_days)
    if pool.empty:
        return []

    if min_amount is not None or bj_min_amount is not None:
        pool = filter_by_liquidity(
            cache,
            signal_date,
            pool,
            0 if min_amount is None else min_amount,
            0 if bj_min_amount is None else bj_min_amount)
        if pool.empty:
            return []

    if min_mcap is not None or max_mcap is not None:
        pool = filter_by_market_cap(cache, signal_date, pool, min_mcap, max_mcap)
        if pool.empty:
            return []

    return pool['ts_code'].drop_duplicates().tolist()
	
# ==================== 预加载入口 ====================

def preload_backtest_data(cache, start_date, end_date,pool_type='ALL',freq='M',
                          exchange='SSE',min_mcap=None,max_mcap=None,index_lookback_days=180):
    """
    一次性预拉回测所需数据：
    - trade_cal
    - stock_basic
    - 所有 signal_date 对应的 daily
    - 所有 signal_date 对应的 bak_basic
    - 若需要市值过滤，则拉所有 signal_date 对应的 daily_basic
    - 若 pool_type 为指数/指数列表，则预拉所有相关指数成分
    """
    schedule = get_rebalance_schedule(
        cache,
        start_date,
        end_date,
        freq=freq,
        exchange=exchange)
    signal_dates = [x['signal_date'] for x in schedule]

    cache.preload_stock_basic()

    cache.preload_daily(signal_dates)
    cache.preload_bak_basic(signal_dates)

    if min_mcap is not None or max_mcap is not None:
        cache.preload_daily_basic(signal_dates)

    cache.preload_index_weights(
        pool_type,
        signal_dates,
        lookback_days=index_lookback_days)

    return schedule


# ==================== 主函数：生成历史样本空间 ====================

def get_history_sample_space_df(pro, start_date, end_date,pool_type='ALL',include_bj=False, min_list_days=60,min_amount=None,
                                bj_min_amount=None,min_mcap=None,max_mcap=None,freq='M', sleep_interval=0.0, exchange='SSE',
                                index_lookback_days=30,return_signal_date=True):
    """
    遍历调仓日，返回包含所有有效样本的 DataFrame

    参数说明：
    - min_amount / bj_min_amount:
        None 表示不做成交额门槛过滤
        但默认仍会剔除当日无成交股票（停牌）
    - min_mcap / max_mcap:
        None 表示不做市值过滤
    """
    cache = TushareDataCache(pro, sleep_interval=sleep_interval)

    schedule = preload_backtest_data(
        cache=cache,
        start_date=start_date,
        end_date=end_date,
        pool_type=pool_type,
        freq=freq,
        exchange=exchange,
        min_mcap=min_mcap,
        max_mcap=max_mcap,
        index_lookback_days=index_lookback_days)

    records = []
    total_dates = len(schedule)

    for i, item in enumerate(schedule):
        signal_date = item['signal_date']
        trade_date = item['trade_date']


        valid_pool = get_clean_stock_pool(
            cache=cache,
            signal_date=signal_date,
            pool_type=pool_type,
            include_bj=include_bj,
            min_list_days=min_list_days,
            min_amount=min_amount,
            bj_min_amount=bj_min_amount,
            min_mcap=min_mcap,
            max_mcap=max_mcap)

        if return_signal_date:
            for ticker in valid_pool:
                records.append({
                    'signal_date': signal_date,
                    'trade_date': trade_date,
                    'ts_code': ticker})
        else:
            for ticker in valid_pool:
                records.append({
                    'trade_date': trade_date,
                    'ts_code': ticker})

    df_sample_space = pd.DataFrame(records)

    return df_sample_space


# ==================== 单次调仓获取股票池 ====================

def get_single_rebalance_stock_pool(pro, trade_date,pool_type='ALL',include_bj=False, min_list_days=60,min_amount=None, bj_min_amount=None,min_mcap=None,
                                    max_mcap=None,sleep_interval=0.0, exchange='SSE', index_lookback_days=30):
    """
    单次调仓获取股票池：
    - trade_date: 实际调仓日
    - signal_date: 前一个交易日

    参数说明：
    - min_amount / bj_min_amount:
        None 表示不做成交额门槛过滤
        但默认仍会剔除当日无成交股票（停牌）
    - min_mcap / max_mcap:
        None 表示不做市值过滤
    """
    cache = TushareDataCache(pro, sleep_interval=sleep_interval)
    signal_date = cache.get_prev_open_date(
        trade_date,
        exchange=exchange,
        lookback_days=60)

    if signal_date is None:
        return []

    cache.preload_stock_basic()
    cache.preload_daily([signal_date])
    cache.preload_bak_basic([signal_date])

    if min_mcap is not None or max_mcap is not None:
        cache.preload_daily_basic([signal_date])

    cache.preload_index_weights(
        pool_type,
        [signal_date],
        lookback_days=index_lookback_days)

    return get_clean_stock_pool(
        cache=cache,
        signal_date=signal_date,
        pool_type=pool_type,
        include_bj=include_bj,
        min_list_days=min_list_days,
        min_amount=min_amount,
        bj_min_amount=bj_min_amount,
        min_mcap=min_mcap,
        max_mcap=max_mcap)


def get_daily_price_for_universe(pro, universe_df,
                                 start_date=None,
                                 end_date=None,
                                 fields='ts_code,trade_date,open,high,low,close,vol,amount'):
    """
    只获取股票池内股票的日行情数据

    参数:
        pro: tushare pro 接口
        universe_df: 股票池 DataFrame（至少包含 ts_code）
        start_date: 可选，YYYYMMDD
        end_date: 可选，YYYYMMDD
        fields: 返回字段

    返回:
        DataFrame: 股票池对应的行情数据
    """

    # ========= 提取股票列表 =========
    ts_codes = universe_df['ts_code'].drop_duplicates().tolist()

    if len(ts_codes) == 0:
        return pd.DataFrame(columns=fields.split(','))

    # ========= 分批请求（Tushare限制）=========
    all_data = []

    batch_size = 50  # tushare 一般限制
    for i in range(0, len(ts_codes), batch_size):
        batch_codes = ts_codes[i:i + batch_size]

        df = pro.daily(
            ts_code=','.join(batch_codes),
            start_date=start_date,
            end_date=end_date,
            fields=fields)

        if df is not None and not df.empty:
            all_data.append(df)

    if not all_data:
        return pd.DataFrame(columns=fields.split(','))

    price_df = pd.concat(all_data, axis=0)

    price_df = price_df.copy()
    price_df['trade_date'] = pd.to_datetime(price_df['trade_date'])
    price_df = price_df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)

    return price_df