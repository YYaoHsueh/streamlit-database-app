import sqlite3
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def initialize_database(csv_file, db_name="data.db", table_name="data_table"):
    """
    初始化 SQLite 資料庫，根據 CSV 檔案建立資料表。
    """
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        
        # 讀取 CSV 檔案
        data = pd.read_csv(csv_file)
        column_names = data.columns.tolist()
        
        # 推測資料型態
        def infer_sqlite_type(dtype):
            if pd.api.types.is_integer_dtype(dtype):
                return "INTEGER"
            elif pd.api.types.is_float_dtype(dtype) or pd.api.types.is_numeric_dtype(dtype):
                return "REAL"
            else:
                return "TEXT"
        
        column_definitions = ", ".join([f'"{col}" {infer_sqlite_type(data[col].dtype)}' for col in column_names])
        create_table_sql = f'CREATE TABLE IF NOT EXISTS "{table_name}" (id INTEGER PRIMARY KEY AUTOINCREMENT, {column_definitions})'
        
        cursor.execute(create_table_sql)
        conn.commit()
    except Exception as e:
        print(f"資料庫初始化錯誤: {e}")
    finally:
        conn.close()

def import_csv_to_db(csv_file, db_name="data.db", table_name="data_table", chunksize=500):
    """
    匯入 CSV 資料至 SQLite 資料庫，支援分批寫入以提高效能。
    """
    try:
        conn = sqlite3.connect(db_name)
        initialize_database(csv_file, db_name, table_name)
        
        for chunk in pd.read_csv(csv_file, chunksize=chunksize):
            chunk.to_sql(table_name, conn, if_exists='append', index=False)
        
        conn.commit()
    except Exception as e:
        print(f"資料匯入錯誤: {e}")
    finally:
        conn.close()

def query_data(db_name="data.db", table_name="data_table", **conditions):
    """
    查詢 SQLite 資料表，支援條件篩選。
    """
    try:
        conn = sqlite3.connect(db_name)
        query = f'SELECT * FROM "{table_name}"'
        data = pd.read_sql_query(query, conn)
        
        if conditions:
            query_conditions = " & ".join([f'{key} == {repr(value)}' for key, value in conditions.items()])
            data = data.query(query_conditions)
        
        return data
    except Exception as e:
        print(f"查詢錯誤: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

def export_to_excel(output_file, db_name="data.db", table_name="data_table"):
    """
    匯出資料庫內容至 Excel。
    """
    try:
        data = query_data(db_name, table_name)
        data.to_excel(output_file, index=False)
    except Exception as e:
        print(f"匯出錯誤: {e}")

def find_best_efficiency(df):
    """
    找出效率最高的資料。
    """
    if df.empty:
        print("資料表為空！")
        return None
    return df.loc[df["Efficiency"].idxmax()]

def plot_scatter_trend(df, x_col, y_col, title, output_file="scatter_trend.png"):
    """
    繪製散點圖與趨勢線，並儲存圖片而不是直接顯示
    """
    if df.empty:
        print("無法繪圖，數據表為空！")
        return
    
    df_sorted = df.sort_values(by=x_col)
    coeffs = np.polyfit(df_sorted[x_col], df_sorted[y_col], deg=3)
    trend_line = np.poly1d(coeffs)
    
    plt.figure(figsize=(8, 5))
    plt.scatter(df[x_col], df[y_col], color='blue', alpha=0.6, label="Data")
    plt.plot(df_sorted[x_col], trend_line(df_sorted[x_col]), linestyle='-', color='red', label="Trend")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    plt.savefig(output_file)  # 儲存圖片
    print(f"Scatter trend saved to {output_file}")  # 確保檔案成功儲存
    plt.close()  # 釋放記憶體


def load_data(db_path, table_name="data_table"):
    """自動讀取 SQLite 資料庫中的所有欄位"""
    conn = sqlite3.connect(db_path)
    query = f"PRAGMA table_info('{table_name}');"
    columns_info = pd.read_sql_query(query, conn)
    column_names = columns_info['name'].tolist()
    
    query = f"SELECT {', '.join(column_names)} FROM '{table_name}';"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def analyze_sensitivity(df, input_vars=None, output_vars=None):
    """計算相關係數矩陣，允許手動選擇 input 和 output 變數"""
    if input_vars is None or output_vars is None:
        print("請輸入有效的 input_vars 和 output_vars 清單。")
        return None
    
    missing_vars = [var for var in input_vars + output_vars if var not in df.columns]
    if missing_vars:
        print(f"以下變數不存在於資料集中: {missing_vars}")
        return None
    
    correlation_matrix = df[input_vars + output_vars].corr()
    return correlation_matrix.loc[input_vars, output_vars]


def plot_heatmap(correlation_matrix, output_file="temp_heatmap.png"):
    """繪製相關性熱圖並儲存為圖片"""
    if correlation_matrix is None or correlation_matrix.empty:
        print("無法繪製熱圖，請確認輸入的變數是否正確。")
        return
    
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title("Sensitivity Analysis (Pearson Correlation)")
        
        plt.savefig(output_file)  # 儲存圖片
        print(f"Heatmap saved to {output_file}")  # 確保檔案成功儲存

        plt.close()  # 確保圖表關閉，釋放資源
    except Exception as e:
        print(f"Error saving heatmap: {e}")  # 紀錄錯誤日誌

def plot_trend(df, x_col, y_col, title, output_file="two_parameters_trend.png"):
    """
    繪製散點圖與趨勢線，並儲存圖片而不是直接顯示
    """
    if df.empty:
        print("無法繪圖，數據表為空！")
        return
    
    plt.figure(figsize=(8, 5))
    plt.plot(df[x_col], df[y_col], color='blue', alpha=0.6, label="Data")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    plt.savefig(output_file)  # 儲存圖片
    print(f"Two parameters trend saved to {output_file}")  # 確保檔案成功儲存
    plt.close()  # 釋放記憶體

if __name__ == "__main__":
    db_name = "Design6056_2025.db"
    table_name = "Parameters_Results"
    csv_file = "D:\jason_hsueh\Code\Python\Database_SQL\Design60-new\Total_Parameter.csv"
    output_excel = "output.xlsx"
    
    import_csv_to_db(csv_file, db_name, table_name)
    data = query_data(db_name, table_name)
    
    if not data.empty:
        best_eff = find_best_efficiency(data)
        print("效率最高的組合:", best_eff)
        plot_scatter_trend(data, "Straight_rib", "Efficiency", "Straight Rib vs Efficiency")
        sensitivity_analysis = analyze_sensitivity(data)
        plot_heatmap(sensitivity_analysis)
