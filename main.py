import menu
import tools.choiceTools as CT

if __name__ == '__main__':
    # 获取选项
    menu.OperationMenu()
    operate_choice = input("请输出你的选择:")
    # 根据选项进行操作
    CT.choice_analysis(operate_choice)

