import sqlite3
import pandas as pd

def main():
    conn = sqlite3.connect("backtests.sqlite")
    df = pd.read_sql("""
        SELECT *
        FROM runs 
        WHERE logical_run_id = 'b3f3a6c7bd9c679f6a98dfdb3f40fe6823e659dbeca024bb49d9158d1f6921e2'
    """, conn)
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    for col in df.columns:
        print(f"[{col}]: {df[col].iloc[0]}")
        print(f"[{col} Vec]: {df[col].iloc[1]}")

if __name__ == "__main__":
    main()
