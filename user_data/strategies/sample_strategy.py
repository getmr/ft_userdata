# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Dict, Optional, Union, Tuple

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,  # @informative decorator
    # Hyperopt Parameters
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    # timeframe helpers
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    # Strategy helper functions
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib


class MACDStrategy(IStrategy):
    """
    基于MACD的多时间框架策略，适用于ETH5L交易
    同时使用日线和1小时线的MACD指标进行交易决策
    """
    INTERFACE_VERSION = 3

    # 主要交易时间框架
    timeframe = "1h"

    # 是否允许做空
    can_short: bool = False

    # 最小ROI设置
    minimal_roi = {
        "480": 0.02,    # 8小时后，如果利润达到2%就卖出
        "240": 0.03,    # 4小时后，如果利润达到3%就卖出
        "120": 0.04,    # 2小时后，如果利润达到4%就卖出
        "0": 0.05       # 从开始就检查，如果利润达到5%就卖出
    }

    # 止损设置
    stoploss = -0.08 # 止损比例

    # 追踪止损设置
    trailing_stop = True                      # 启用追踪止损功能
    trailing_stop_positive = 0.01             # 1%的追踪止损距离
    trailing_stop_positive_offset = 0.02      # 2%的激活阈值
    trailing_only_offset_is_reached = True    # 只有达到激活阈值后才开始追踪止损

    # 其他设置
    process_only_new_candles = True    # 只在新K线出现时执行策略
    use_exit_signal = True             # 使用策略中的出场信号
    exit_profit_only = False           # 是否只在盈利时允许出场信号
    ignore_roi_if_entry_signal = False # 是否在有入场信号时忽略ROI

    # 启动所需蜡烛数
    startup_candle_count: int = 50 # 启动所需蜡烛数

    # MACD参数 # 
    # 快速线周期 EMA移动平均线快线
    macd_fast = IntParameter(10, 20, default=12, space="buy")  
    # 慢速线周期 EMA移动平均线慢线
    macd_slow = IntParameter(20, 30, default=26, space="buy")  
    # 信号线周期 EMA移动平均线信号线   DEA （Diff of Exponential Moving Average）MACD 慢线
    macd_signal = IntParameter(5, 15, default=9, space="buy")  
    @property
    def plot_config(self):
        """
        用于定义FreqTrade回测或实时交易时的图表可视化配置。
        它控制了策略中各个指标在图表上的显示方式。
        """
        return {
            # 主图指标配置
            "main_plot": {
                "tema": {},                # 在主图显示TEMA指标，使用默认颜色
                "sar": {"color": "white"}, # 在主图显示SAR指标，颜色设为白色
            },
            "subplots": {
                # 子图配置 - 每个字典定义一个额外的图表
                "MACD": {
                    "macd": {"color": "blue"},      # MACD线，蓝色
                    "macdsignal": {"color": "orange"}, # 信号线，橙色
                },
                "RSI": {
                    "rsi": {"color": "red"},        # RSI指标，红色
                }
            }
        }

    def informative_pairs(self):
        """
        定义额外的时间框架
        """
        return [
            ("ETH5L/USDT", "1h"),  # 1小时线
            ("ETH5L/USDT", "1d")  # 日线
        ]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        计算技术指标
        """
        # 计算1小时线MACD
        macd = ta.MACD(
            dataframe,
            fastperiod=self.macd_fast.value,
            slowperiod=self.macd_slow.value,
            signalperiod=self.macd_signal.value
        )
        dataframe["macd"] = macd["macd"]  # 这是DIF值  # 快线
        dataframe["macdsignal"] = macd["macdsignal"]  # 这是DEA值  # 慢线
        dataframe["macdhist"] = macd["macdhist"]  # 这是柱状图值  # 柱状图

        # 获取日线数据
        informative_1d = self.dp.get_pair_dataframe("ETH5L/USDT", "1d") # 获取日线数据

        # 计算日线MACD
        macd_1d = ta.MACD(
            informative_1d,
            fastperiod=self.macd_fast.value,
            slowperiod=self.macd_slow.value,
            signalperiod=self.macd_signal.value
        )
        
        # 合并日线数据
        dataframe = merge_informative_pair(dataframe, informative_1d, self.timeframe, "1d", ffill=True) # 合并日线数据

        # 计算日线MACD指标
        dataframe["macd_1d"] = macd_1d["macd"] # 计算日线MACD指标  # 快线
        dataframe["macdsignal_1d"] = macd_1d["macdsignal"] # 计算日线MACD信号线  # 慢线
        dataframe["macdhist_1d"] = macd_1d["macdhist"] # 计算日线MACD柱状图  # 柱状图

        # 计算RSI作为辅助指标
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)


        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        入场信号
        """
        dataframe.loc[
            (
                # 日线MACD金叉
                (qtpylib.crossed_above(dataframe["macd_1d"], dataframe["macdsignal_1d"])) &
                # 1小时线MACD金叉
                (qtpylib.crossed_above(dataframe["macd"], dataframe["macdsignal"])) &
                # RSI超卖
                (dataframe["rsi"] < 30) &
                # 确保有成交量
                (dataframe["volume"] > 0)
            ),
            "enter_long"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        出场信号
        """
        dataframe.loc[
            (
                # 日线MACD死叉
                (qtpylib.crossed_below(dataframe["macd_1d"], dataframe["macdsignal_1d"])) &
                # 1小时线MACD死叉
                (qtpylib.crossed_below(dataframe["macd"], dataframe["macdsignal"])) &
                # RSI超买
                (dataframe["rsi"] > 70) &
                # 确保有成交量
                (dataframe["volume"] > 0)
            ),
            "exit_long"] = 1

        return dataframe