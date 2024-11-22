import xlrd
import xlwt
from xlutils.copy import copy
import pandas as pd


# 翻译分类结果
def translate_classification(row):
    if row['是否颜色偏差']:
        return 'Color Bias'
    elif row['是否低光']:
        return 'Low Light'
    elif row['是否模糊']:
        return 'Blurry'
    else:
        return 'Normal'


# 读取 CSV 文件
csv_file_path = '../../results/metrics/plots/T1.csv'  # 替换为你的 CSV 文件路径
csv_data = pd.read_csv(csv_file_path)

# 添加翻译列并生成目标格式
csv_data['Degraded Image Classification'] = csv_data.apply(translate_classification, axis=1)

# 选择并重命名列
final_data = csv_data.rename(columns={
    '文件名': 'image file name'
})[['image file name', 'Degraded Image Classification']]

# 添加额外列并保持空值
final_data['PSNR'] = ''
final_data['UCIQE'] = ''
final_data['UIQM'] = ''

# 打开现有的 .xls 文件
xls_file_path = '../../results/metrics/Answer.xls'  # 替换为目标 .xls 文件路径
book = xlrd.open_workbook(xls_file_path, formatting_info=True)
sheet_names = book.sheet_names()

# 检查目标工作表是否存在
if 'attachment1 results' not in sheet_names:
    raise ValueError("工作表 'attachment1 results' 不存在，请检查文件结构！")

# 复制现有文件以进行写操作
wb = copy(book)
sheet = wb.get_sheet(sheet_names.index('attachment1 results'))

# 写入数据到 'attachment1 results' 工作表
# 表头
headers = ['image file name', 'Degraded Image Classification', 'PSNR', 'UCIQE', 'UIQM']
for col, header in enumerate(headers):
    sheet.write(0, col, header)

# 填入数据
for row_idx, row_data in enumerate(final_data.itertuples(index=False), start=1):
    sheet.write(row_idx, 0, row_data[0])  # image file name
    sheet.write(row_idx, 1, row_data[1])  # Degraded Image Classification
    sheet.write(row_idx, 2, row_data[2])  # PSNR (empty)
    sheet.write(row_idx, 3, row_data[3])  # UCIQE (empty)
    sheet.write(row_idx, 4, row_data[4])  # UIQM (empty)

# 保存文件
wb.save(xls_file_path)
print(f"数据已成功写入工作表 'attachment1 results'，文件路径: {xls_file_path}")
