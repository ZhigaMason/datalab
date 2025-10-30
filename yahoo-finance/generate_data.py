import yfinance as yf

START_DATE = '2022-01-01'
END_DATE = '2025-01-01'
TICKERS = [
    'KO',
    'NVDA',
    'JNJ',
    '^GSPC'
]

if __name__ == '__main__':
    data = yf.download(TICKERS, start=START_DATE, end=END_DATE)
    if data is None:
        raise RuntimeError("Could not download data.")
    data.to_csv('data/KO_NVDA_JNJ_GSPC-2022-2025.csv')
