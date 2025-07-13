import os
import glob
import csv

# 获取当前目录下所有csv文件
csv_files = glob.glob('*.csv')

# 检查是否存在csv文件
if not csv_files:
    print("当前文件夹中没有找到CSV文件")
    exit()

# 创建合并后的文件名
output_file = 'all.csv'

# 确保不合并自己（如果已存在all.csv）
if output_file in csv_files:
    csv_files.remove(output_file)

# 再次检查剩余文件数量
if not csv_files:
    print("没有可合并的CSV文件（仅存在all.csv）")
    exit()

# 开始合并
with open(output_file, 'w', newline='', encoding='utf-8') as out_f:
    writer = csv.writer(out_f)
    header_written = False

    for file in csv_files:
        with open(file, 'r', newline='', encoding='utf-8') as in_f:
            reader = csv.reader(in_f)
            header = next(reader)

            # 如果是第一个文件，写入标题
            if not header_written:
                writer.writerow(header)
                header_written = True

            # 写入当前文件的所有行
            for row in reader:
                writer.writerow(row)

        print(f"已合并: {os.path.basename(file)}")

print(f"合并完成！共合并 {len(csv_files)} 个文件 -> {output_file}")