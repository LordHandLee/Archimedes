import sqlite3
import pandas as pd

def main():
    conn = sqlite3.connect("backtests.sqlite")
    
    run_vec = "941ed23b43c6b7bf74e0fafea050bb1249b75db6afb2f120eec2ad3ffacdc5ed"
    run_ref = "b3f3a6c7bd9c679f6a98dfdb3f40fe6823e659dbeca024bb49d9158d1f6921e2"
    
    df_vec = pd.read_sql(f"SELECT * FROM trades WHERE run_id = '{run_vec}' ORDER BY timestamp", conn)
    df_ref = pd.read_sql(f"SELECT * FROM trades WHERE run_id = '{run_ref}' ORDER BY timestamp", conn)
    
    print("Vectorized Trades:")
    print(df_vec.head(10)[['timestamp', 'side', 'qty', 'price', 'fee', 'equity_after']])
    
    print("\nReference Trades:")
    print(df_ref.head(10)[['timestamp', 'side', 'qty', 'price', 'fee', 'equity_after']])

if __name__ == "__main__":
    main()
