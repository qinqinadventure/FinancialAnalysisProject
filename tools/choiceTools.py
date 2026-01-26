import json
import analysis.getPCA as getPCA
import tools.dataTools as DT

def choice_analysis(choice):
    if choice == '1':
        # 加载PCA分析参数
        cfg = json.loads("D:\project\pycharm\FinancialAnalysisProject\cfg\PCA_config.json")
        # 获取关键参数
        features = cfg["features"]["value"]
        # 获取数据
        stock_data = DT.getData(cfg["stock_code"]["value"],cfg["days"]["value"])
        # 进行PCA分析
        getPCA.PCAResult(features, stock_data)
    else:
        print("choice error")
        exit(0)