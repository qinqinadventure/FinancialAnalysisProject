import directParams as dp

class StockDataExtractor:
    """股票数据提取器"""

    def __init__(self, df):
        self.df = df
        self.column_mapping = {
            'date': '日期',
            'code': '股票代码',
            'open': '开盘',
            'close': '收盘',
            'high': '最高',
            'low': '最低',
            'volume': '成交量',
            'amount': '成交额',
            'amplitude': '振幅',
            'change_rate': '涨跌幅',
            'change_amount': '涨跌额',
            'turnover_rate': '换手率'
        }

    def get_column(self, column_alias):
        """通过别名获取列"""
        if column_alias in self.column_mapping:
            actual_name = self.column_mapping[column_alias]
            return dp.get_column_by_name(self.df, actual_name)
        else:
            return dp.get_column_by_name(self.df, column_alias)

    def get_price_data(self, include_derived=True):
        """获取价格相关数据"""
        base_columns = ['开盘', '收盘', '最高', '最低']
        if include_derived:
            base_columns.extend(['涨跌幅', '涨跌额'])
        return dp.get_multiple_columns(self.df, base_columns)

    def get_trading_data(self):
        """获取交易量相关数据"""
        return dp.get_multiple_columns(self.df, ['成交量', '成交额', '换手率'])

    def get_daily_summary(self, date=None):
        """
        获取某日的完整数据摘要

        参数:
        date: 指定日期，如果为None则返回所有数据
        """
        if date:
            daily_data = self.df[self.df['日期'] == date]
            if len(daily_data) == 0:
                print(f"未找到日期为 {date} 的数据")
                return None
            return daily_data.iloc[0]  # 返回Series
        else:
            return self.df
        