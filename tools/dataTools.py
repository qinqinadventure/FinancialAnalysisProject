import akshare as ak
import datetime
import pandas as pd


def getData(stock_code="000001", days=1000):
    """
    获取指定股票代码最近days天的历史数据

    参数:
    stock_code: 股票代码，默认为"000001"
    days: 获取最近多少天的数据，默认为1000天

    返回:
    DataFrame: 包含股票历史数据的DataFrame
    """
    # 获取当前日期
    end_date = datetime.date.today()

    # 计算开始日期（当前日期往前推days天）
    start_date = end_date - datetime.timedelta(days=days)

    # 格式化日期为字符串（YYYYMMDD格式）
    end_date_str = end_date.strftime("%Y%m%d")
    start_date_str = start_date.strftime("%Y%m%d")

    try:
        # 获取股票历史数据
        df = ak.stock_zh_a_hist(
            symbol=stock_code,
            period="daily",
            start_date=start_date_str,
            end_date=end_date_str,
            adjust="qfq"
        )

        # 按日期排序（确保数据按时间顺序排列）
        if not df.empty and '日期' in df.columns:
            df = df.sort_values('日期')
            print(f"成功获取股票 {stock_code} 从 {start_date_str} 到 {end_date_str} 的数据，共 {len(df)} 条记录")
        else:
            print("未获取到数据，请检查股票代码和日期范围")

        return df

    except Exception as e:
        print(f"获取数据时出错: {e}")
        return pd.DataFrame()