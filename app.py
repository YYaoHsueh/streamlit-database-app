import streamlit as st
import pandas as pd
import os
import DataBase_function_v2 as db2

st.set_page_config(page_title="Database Explorer", layout="wide")
st.title("📊 Database Explorer Web App")

# Upload CSV
st.header("1. Upload CSV File")
uploaded_file = st.file_uploader("Choose a CSV file to upload", type="csv")
if uploaded_file:
    temp_path = "temp_uploaded.csv"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    db_path = "database.db"
    table_name = "data_table"

    db2.import_csv_to_db(temp_path, db_path, table_name)
    st.success(f"CSV imported into {db_path}, table: {table_name}")

    df = pd.read_csv(temp_path)
    st.dataframe(df.head())

    # Select columns for filtering
    st.header("2. Filter and Query Data")
    st.write("Enter pandas-compatible query (e.g., column1 > 50 and column2 == 'A')")
    query_str = st.text_input("Query String")

    if st.button("Run Query"):
        try:
            df_all = db2.load_data(db_path, table_name)
            df_filtered = df_all.query(query_str)
            st.success("Query successful")
            st.dataframe(df_filtered)
        except Exception as e:
            st.error(f"Query error: {e}")

    # Export filtered data to Excel
    st.header("3. Export Data")
    if st.button("Export Table to Excel"):
        output_file = "exported_data.xlsx"
        db2.export_to_excel(output_file, db_path, table_name)
        with open(output_file, "rb") as f:
            st.download_button("Download Excel", f, file_name=output_file)

    # Variable selection for analysis
    st.header("4. Analyze Variable Relationships")
    columns = df.columns.tolist()
    input_vars = st.multiselect("Select Input Variables", columns)
    output_vars = st.multiselect("Select Output Variables", columns)

    # Heatmap
    if st.button("Generate Heatmap"):
        if input_vars and output_vars:
            corr = db2.analyze_sensitivity(df, input_vars, output_vars)
            heatmap_path = "heatmap.png"
            db2.plot_heatmap(corr, heatmap_path)
            st.image(heatmap_path, caption="Correlation Heatmap", use_column_width=True)
        else:
            st.warning("Please select at least one input and one output variable")

    # Scatter plot
    if st.button("Generate Scatter Trend"):
        if len(input_vars) == 1 and len(output_vars) == 1:
            scatter_path = f"{input_vars[0]}_vs_{output_vars[0]}.png"
            db2.plot_scatter_trend(df, input_vars[0], output_vars[0], "Scatter Plot", scatter_path)
            st.image(scatter_path, caption="Scatter Trend", use_column_width=True)
        else:
            st.warning("Please select exactly one input and one output variable")
        
    # 新增：折線圖即時顯示
    st.header("5. Line Chart for Two Columns")
    st.write("Select two columns to plot a line chart (X vs Y)")

    if df is not None:
        col_options = df.columns.tolist()
        x_col = st.selectbox("X-axis", col_options, key="line_x")
        y_col = st.selectbox("Y-axis", col_options, key="line_y")

        if st.button("Plot Line Chart"):
            if x_col and y_col:
                chart_df = df[[x_col, y_col]].dropna().sort_values(by=x_col)
                chart_df.set_index(x_col, inplace=True)
                st.line_chart(chart_df)
            else:
                st.warning("Please select both X and Y columns to plot.")
