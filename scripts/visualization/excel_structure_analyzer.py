import pandas as pd

def analyze_excel_structure(file_path, max_show=10):
    """
    读取Excel/CSV文件，打印表结构，行列过多时只显示前10
    """
    try:
        # ----------------------
        # 处理 CSV 文件
        # ----------------------
        if file_path.lower().endswith(".csv"):
            print("=" * 60)
            print(f"📊 分析 CSV 文件: {file_path}")
            df = pd.read_csv(file_path)

            print(f"✅ 数据总行数: {df.shape[0]}")
            print(f"✅ 数据总列数: {df.shape[1]}")
            print(f"📌 仅展示前 {max_show} 列结构")
            print("-" * 80)

            # 只取前10列
            cols_to_show = df.columns[:max_show]

            for col in cols_to_show:
                dtype = str(df[col].dtype)
                non_null = df[col].notna().sum()
                null_count = df[col].isna().sum()
                sample = df[col].dropna().iloc[0] if non_null > 0 else "无数据"
                print(f"列名: {col:<20} 类型: {dtype:<10} 非空: {non_null:<6} 空值: {null_count:<6} 示例: {sample}")

            print("\n✅ CSV 分析完成！")
            print("=" * 60)
            return

        # ----------------------
        # 处理 Excel 文件
        # ----------------------
        excel_file = pd.ExcelFile(file_path)
        sheet_names = excel_file.sheet_names

        print("=" * 60)
        print(f"📊 Excel 文件路径: {file_path}")
        print(f"📋 工作表数量: {len(sheet_names)}")
        print(f"📑 表名: {sheet_names}")
        print("=" * 60)

        for sheet in sheet_names:
            print(f"\n🔍 工作表: [{sheet}]")
            print("-" * 50)
            df = pd.read_excel(file_path, sheet_name=sheet)

            print(f"✅ 总行数: {df.shape[0]}")
            print(f"✅ 总列数: {df.shape[1]}")
            print(f"📌 仅展示前 {max_show} 列结构")
            print("-" * 80)

            # 只取前10列
            cols_to_show = df.columns[:max_show]

            for col in cols_to_show:
                dtype = str(df[col].dtype)
                non_null = df[col].notna().sum()
                null_count = df[col].isna().sum()
                sample = df[col].dropna().iloc[0] if non_null > 0 else "无数据"
                print(f"列名: {col:<20} 类型: {dtype:<10} 非空: {non_null:<6} 空值: {null_count:<6} 示例: {sample}")

        print("\n✅ Excel 分析完成！")
        print("=" * 60)

    except FileNotFoundError:
        print("❌ 错误：文件不存在，请检查路径！")
    except Exception as e:
        print(f"❌ 读取失败: {str(e)}")

if __name__ == "__main__":
    # 你的文件路径（用 / 不要用 \）
    YOUR_FILE = "data/datasets/ST_EVCDP_v2/inf.csv"
    
    # 执行分析
    analyze_excel_structure(YOUR_FILE, max_show=10)