import json
import analysis.getPCA as getPCA
import tools.dataTools as DT
import analysis.getLevel as GT
import analysis.getHisAnalysis as GH

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
        # 获取关键参数
        code = cfg["stock_code"]["value"]
        days = cfg["days"]["value"]
        # 获取数据
        stock_data  = DT.getData(code, days)
        # 进行金融分析并判断位置
        GT.outputLevelInfo(stock_data)
    # 判断当前点位量比
    elif choice == '3':
        # 使用原始字符串避免转义问题
        with open(r"D:\project\pycharm\FinancialAnalysisProject\cfg\nowday_config.json", 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        # 获取关键参数
        code = cfg["stock_code"]["value"]
        history_days = cfg["history_days"]["value"]
        quantity_days = cfg["quantity_days"]["value"]
        analysis_target = cfg["analysis_target"]["value"]
        # 获取数据
        history_data  = DT.getData(code, history_days)
        quantity_data = DT.getData(code, quantity_days)
        # 进行历史分位分析
        GH.outputHisAnalysis(history_data, quantity_data, analysis_target)
    else:
        print("choice error")
        exit(0)