import json
import analysis.getPCA as getPCA
import tools.dataTools as DT

def choice_analysis(choice):
    # 选择进行PCA分析
    if choice == '1':
        # 使用原始字符串避免转义问题
        with open(r"D:\project\pycharm\FinancialAnalysisProject\cfg\PCA_config.json", 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        # 获取关键参数
        features = cfg["features"]["value"]
        # 获取数据
        stock_data = DT.getData(cfg["stock_code"]["value"],cfg["days"]["value"])
        # 进行PCA分析
        getPCA.PCAResult(features, stock_data)
    # 选择判断当前位置
    elif choice == '2':
        # 使用原始字符串避免转义问题
        with open(r"D:\project\pycharm\FinancialAnalysisProject\cfg\PressLevel_config.json", 'r', encoding='utf-8') as f:
            cfg = json.load(f)
    else:
        print("choice error")
        exit(0)