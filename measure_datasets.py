import time
from ui_qt_dashboard import CatalogReader
import warnings
warnings.filterwarnings('ignore')

def main():
    start = time.perf_counter()
    reader = CatalogReader("backtests.sqlite")
    df = reader.load_acquisition_datasets(include_untracked_local=True)
    end = time.perf_counter()
    print(f"Loaded {len(df)} datasets in {end - start:.2f} seconds.")

if __name__ == "__main__":
    main()
