import json

def choice_analysis(choice):
    if choice == '1':
        # 加载PCA分析参数
        cfg = json.loads("D:\project\pycharm\FinancialAnalysisProject\cfg\PCA_config.json")
        # 获取关键参数
