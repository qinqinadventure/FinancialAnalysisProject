import analysis.HISAnalysis.getHis as getHis

def outputHisAnalysis(history_data, quantity_data, analysis_target):
    output = getHis.getHisAnalysis(history_data, quantity_data, analysis_target)
    print(output)