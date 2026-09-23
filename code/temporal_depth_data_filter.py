import pandas as pd
import numpy as np
import os
import json
import glob
from datetime import datetime
from collections import Counter

def extract_date_from_parts(parts):
    """从文件名各部分中提取8位数字日期字符串"""
    for p in parts:
        if len(p) == 8 and p.isdigit():
            return p
    return None

def filter_temporal_depth_data(input_dir, output_dir, depth_ranges=None, time_range=None):
    """
    按深度区间分组全球时空数据（不筛选海域和季节）
    input_dir: 包含按时间点组织的Excel文件目录
    output_dir: 输出根目录，将在其下创建 depth_0_200, depth_200_400 等子目录
    depth_ranges: 深度区间列表，格式如 [(0,200), (200,400), (400,600), (600,800)]，默认上述值
    time_range: 时间范围筛选 (start_date, end_date)
    """
    # 默认深度区间（左闭右开，最后一个区间包含右端点）
    if depth_ranges is None:
        depth_ranges = [
            (0, 200, 'depth_0_200'),
            (200, 400, 'depth_200_400'),
            (400, 600, 'depth_400_600'),
            (600, 800, 'depth_600_800')
        ]
    else:
        # 如果传入自定义区间，需要包含文件夹名，此处简化处理，假设传入三元组 (low, high, folder_name)
        pass

    # 创建输出目录及深度子目录
    os.makedirs(output_dir, exist_ok=True)
    for (low, high, folder) in depth_ranges:
        os.makedirs(os.path.join(output_dir, folder), exist_ok=True)

    # 统计信息
    filtered_files = []          # 所有成功输出的文件（相对路径）
    buoy_time_stats = {}          # 浮标全局统计
    depth_stats = {folder: {'files': [], 'buoys': set()} for (_, _, folder) in depth_ranges}

    invalid_counter = Counter()
    invalid_details = []

    excel_files = glob.glob(os.path.join(input_dir, "*.xlsx"))
    print(f"找到 {len(excel_files)} 个Excel文件")

    for idx, file_path in enumerate(excel_files):
        filename = os.path.basename(file_path)
        base_name = os.path.splitext(filename)[0]
        parts = base_name.split('_')
        reason = None

        try:
            # 1. 提取日期（用于时间筛选和浮标统计）
            date_str = extract_date_from_parts(parts)
            if date_str is None:
                reason = "未找到有效的8位日期"
                invalid_counter[reason] += 1
                invalid_details.append((filename, reason))
                continue

            # 2. 解析日期
            try:
                year = int(date_str[0:4])
                month = int(date_str[4:6])
                day = int(date_str[6:8])
                current_date = datetime(year, month, day)
            except ValueError as e:
                reason = f"日期解析错误: {e}"
                invalid_counter[reason] += 1
                invalid_details.append((filename, reason))
                continue

            # 3. 时间范围筛选
            if time_range:
                start_date, end_date = time_range
                if current_date < start_date or current_date > end_date:
                    reason = "时间范围外"
                    invalid_counter[reason] += 1
                    invalid_details.append((filename, reason))
                    continue

            # 4. 读取Excel
            df = pd.read_excel(file_path, header=None)
            if len(df) == 0:
                reason = "Excel文件为空"
                invalid_counter[reason] += 1
                invalid_details.append((filename, reason))
                continue

            # 浮标ID（取第一个部分）
            buoy_id = parts[0]
            # 提取经纬度（仅用于统计）
            lon = df.iloc[0, 1] if df.shape[1] > 1 else None
            lat = df.iloc[0, 2] if df.shape[1] > 2 else None

            # 5. 深度列检查（假设深度在第6列，索引5）
            depth_column = 5
            if df.shape[1] <= depth_column:
                reason = f"深度列索引{depth_column}超出范围（总列数{df.shape[1]}）"
                invalid_counter[reason] += 1
                invalid_details.append((filename, reason))
                continue

            # 获取深度数据（跳过第一行元数据）
            depth_data = df.iloc[1:, depth_column]
            if len(depth_data) == 0:
                reason = "无深度观测数据"
                invalid_counter[reason] += 1
                invalid_details.append((filename, reason))
                continue

            # 6. 按深度区间分组
            # 为每个区间收集行索引（相对于原始df，保留元数据行0）
            rows_by_range = {folder: [0] for (_, _, folder) in depth_ranges}  # 每个区间至少包含元数据行
            for i, depth in enumerate(depth_data):
                if pd.isna(depth):
                    continue
                depth_val = float(depth)
                # 找到所属区间
                for low, high, folder in depth_ranges:
                    # 最后一个区间包含high，其余左闭右开
                    if folder == depth_ranges[-1][2]:
                        if low <= depth_val <= high:
                            rows_by_range[folder].append(i+1)  # +1因为跳过元数据行
                            break
                    else:
                        if low <= depth_val < high:
                            rows_by_range[folder].append(i+1)
                            break

            # 7. 保存各区间文件（如果该区间有数据行）
            file_saved_for_this_buoy = False
            for (low, high, folder) in depth_ranges:
                rows = rows_by_range[folder]
                if len(rows) > 1:  # 至少有一个数据行+元数据行
                    filtered_df = df.iloc[rows]
                    depth_dir = os.path.join(output_dir, folder)
                    output_path = os.path.join(depth_dir, filename)
                    filtered_df.to_excel(output_path, header=False, index=False)

                    # 记录统计
                    filtered_files.append(os.path.join(folder, filename))
                    depth_stats[folder]['files'].append(os.path.join(folder, filename))
                    depth_stats[folder]['buoys'].add(buoy_id)
                    file_saved_for_this_buoy = True

            # 8. 如果至少有一个区间保存了文件，则更新浮标全局统计
            if file_saved_for_this_buoy:
                if buoy_id not in buoy_time_stats:
                    buoy_time_stats[buoy_id] = {
                        'files': [],
                        'timestamps': [],
                        'longitude': lon,
                        'latitude': lat
                    }
                buoy_time_stats[buoy_id]['files'].append(filename)  # 记录原始文件名（非区间路径）
                buoy_time_stats[buoy_id]['timestamps'].append(date_str)
            else:
                # 所有深度数据都不在定义的区间内（如深度>800或<0）
                reason = "无有效深度区间内的数据"
                invalid_counter[reason] += 1
                invalid_details.append((filename, reason))

            if (idx+1) % 1000 == 0:
                print(f"已处理 {idx+1} 个文件...")

        except Exception as e:
            reason = f"其他异常: {str(e)}"
            invalid_counter[reason] += 1
            invalid_details.append((filename, reason))

    # 输出统计
    print(f"\n筛选完成! 总文件数: {len(excel_files)}")
    print(f"成功输出的文件片段总数: {len(filtered_files)}")
    print(f"无效文件数: {sum(invalid_counter.values())}")
    print(f"有效浮标数: {len(buoy_time_stats)}")
    print("\n=== 各深度区间统计 ===")
    for folder, stats in depth_stats.items():
        print(f"{folder}: {len(stats['files'])} 个文件片段, {len(stats['buoys'])} 个浮标")

    print("\n=== 无效原因统计 ===")
    for reason, count in invalid_counter.most_common():
        print(f"{reason}: {count}")

    print("\n=== 前10个无效文件示例 ===")
    for fname, reason in invalid_details[:10]:
        print(f"{fname} -> {reason}")

    # 保存筛选信息
    filter_info = {
        'time_range': str(time_range) if time_range else None,
        'depth_ranges': [(low, high) for (low, high, folder) in depth_ranges],
        'depth_folders': [folder for (_, _, folder) in depth_ranges],
        'filtered_file_fragments': filtered_files,
        'buoy_time_stats': buoy_time_stats,
        'depth_stats': {
            folder: {
                'file_count': len(stats['files']),
                'buoy_count': len(stats['buoys']),
                'buoys': list(stats['buoys'])
            } for folder, stats in depth_stats.items()
        },
        'invalid_stats': dict(invalid_counter),
        'total_file_fragments': len(filtered_files)
    }

    info_path = os.path.join(output_dir, 'depth_filter_info.json')
    with open(info_path, 'w') as f:
        json.dump(filter_info, f, indent=2, default=str)

    print(f"\n筛选信息已保存到: {info_path}")
    return filtered_files, buoy_time_stats, depth_stats

if __name__ == "__main__":
    input_directory = r"D:\cdomproject\temporal_data"
    output_directory = r"D:\cdomproject\temporal_depth_data"
    time_range = None  # 可选： (datetime(2010,1,1), datetime(2023,12,31))
    filtered_files, buoy_stats, depth_stats = filter_temporal_depth_data(
        input_directory,
        output_directory,
        depth_ranges=None,  # 使用默认区间
        time_range=time_range
    )