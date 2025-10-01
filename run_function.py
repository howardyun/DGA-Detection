import subprocess

# 定义命令和参数
cmd = [
    'python',
    'main_ysx_iterator_trans_improve.py',
    '1', '0', '0', '0',
    'Transformer-improve-2m_1:7-0.00001',
    '0.1428',
    '5',
    '0.00001'
]

# 执行命令
result = subprocess.run(cmd, capture_output=True, text=True)

# 打印输出结果
print("标准输出：")
print(result.stdout)
print("标准错误：")
print(result.stderr)
