import os
import pandas as pd

# 修改为你 chronics 文件夹路径
chronics_dir = r"C:\Users\18410\Desktop\AssignmentP1\MachineLearning\ProjectE\l2rpn_case14_storage_train\chronics"

# 保存有问题的 scenario
problem_scenarios = []

for scenario in os.listdir(chronics_dir):
    scenario_path = os.path.join(chronics_dir, scenario)
    load_q_path = os.path.join(scenario_path, "load_q.csv")
    
    if os.path.isfile(load_q_path):
        try:
            df_q = pd.read_csv(load_q_path)
            if (df_q == 0).any().any():
                problem_scenarios.append(scenario)
        except Exception as e:
            print(f"读取出错：{scenario} -> {e}")

print(f"\n⚠️ 以下场景 load_q.csv 中含有 Q=0 的风险行：")
for p in problem_scenarios:
    print(f"  - {p}")

print(f"\n共发现 {len(problem_scenarios)} 个问题场景。")