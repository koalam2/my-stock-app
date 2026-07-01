import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json

# ---------------------------------------------------------
# 1. 초기 설정 및 데이터 관리 함수 (구글 시트 연동)
# ---------------------------------------------------------

st.set_page_config(page_title="미국 주식 관리 - StockWise", layout="wide")

# 구글 시트 연결 설정 (캐싱)
@st.cache_resource
def init_connection():
    scope = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    
    if "sheet_url" not in st.secrets:
        st.error("🚨 `sheet_url` 설정이 없습니다.")
        st.stop()
        
    try:
        if "gcp_json" in st.secrets:
            creds_dict = json.loads(st.secrets["gcp_json"])
        else:
            creds_dict = dict(st.secrets["gcp_service_account"])
            
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        return client
    except Exception as e:
        st.error(f"🚨 구글 시트 연결 실패: {e}")
        st.stop()

# 시트 데이터 로드 함수
def load_data_from_sheet(sheet_name):
    try:
        client = init_connection()
        sheet = client.open_by_url(st.secrets["sheet_url"]).worksheet(sheet_name)
        data = sheet.get_all_records()
        
        if not data:
            if sheet_name == 'transactions':
                return pd.DataFrame(columns=['Date', 'Type', 'Ticker', 'Sector', 'Amount_USD', 'Quantity', 'Exchange_Rate', 'Total_KRW'])
            elif sheet_name == 'favorites':
                return pd.DataFrame(columns=['Ticker', 'Sector'])
            elif sheet_name == 'memos':  
                return pd.DataFrame(columns=['Date', 'Title', 'Content', 'Color'])
            elif sheet_name == 'targets':  
                return pd.DataFrame(columns=['Ticker', 'Target_Ratio'])
            elif sheet_name == 'config':
                return {} 
        
        if sheet_name == 'config':
            return {row['Key']: row['Value'] for row in data}
            
        df = pd.DataFrame(data)
        
        if sheet_name == 'transactions':
            df['Date'] = pd.to_datetime(df['Date']).dt.date
            num_cols = ['Amount_USD', 'Quantity', 'Exchange_Rate', 'Total_KRW']
            for col in num_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                
        return df
    except Exception as e:
        if sheet_name == 'transactions':
            return pd.DataFrame(columns=['Date', 'Type', 'Ticker', 'Sector', 'Amount_USD', 'Quantity', 'Exchange_Rate', 'Total_KRW'])
        elif sheet_name == 'favorites':
            return pd.DataFrame(columns=['Ticker', 'Sector'])
        elif sheet_name == 'memos':
            return pd.DataFrame(columns=['Date', 'Title', 'Content', 'Color'])
        elif sheet_name == 'targets':
            return pd.DataFrame(columns=['Ticker', 'Target_Ratio'])
        elif sheet_name == 'config':
            return {}

# 시트 데이터 저장 함수
def save_data_to_sheet(data, sheet_name):
    try:
        client = init_connection()
        spreadsheet = client.open_by_url(st.secrets["sheet_url"])
        
        try:
            sheet = spreadsheet.worksheet(sheet_name)
        except:
            sheet = spreadsheet.add_worksheet(title=sheet_name, rows=1000, cols=20)
        
        sheet.clear() 
        
        if sheet_name == 'config':
            rows = [['Key', 'Value']]
            for k, v in data.items():
                rows.append([k, v])
            sheet.update(rows)
        else:
            df_save = data.copy()
            if 'Date' in df_save.columns:
                df_save['Date'] = df_save['Date'].astype(str)
            sheet.update([df_save.columns.values.tolist()] + df_save.values.tolist())
            
    except Exception as e:
        st.error(f"구글 시트 저장 중 오류 발생: {e}")

# 설정 로드 함수
def load_config():
    default_config = {'goal1': 100000000, 'goal2': 1000000000}
    sheet_config = load_data_from_sheet('config')
    if sheet_config:
        for k, v in sheet_config.items():
            try:
                sheet_config[k] = int(str(v).replace(',', '').replace('.', '').split('.')[0])
            except:
                pass
        default_config.update(sheet_config)
    return default_config

# 설정 저장 함수
def save_config(goal1, goal2):
    config_data = {'goal1': goal1, 'goal2': goal2}
    save_data_to_sheet(config_data, 'config')

# 섹터 및 그룹 정의
SECTOR_OPTIONS = [
    'IT/반도체', '커뮤니케이션', '경기소비재', 
    '필수소비재', '헬스케어', '유틸리티',   
    '금융', '에너지/소재', '산업재',        
    '채권', '기타'
]

GROUP_ORDER_LIST = ['성장주', '방어주', '가치주/기반주', '채권', '기타']

SECTOR_COLOR_MAP = {
    'IT/반도체': '#E05D5D', '커뮤니케이션': '#FF8B8B', '경기소비재': '#FFB4B4',
    '헬스케어': '#2B9348', '필수소비재': '#55A630', '유틸리티': '#80B918',
    '금융': '#0077B6', '에너지/소재': '#0096C7', '산업재': '#48CAE4',
    '채권': '#FFD166', '기타': '#ADB5BD'
}

GROUP_COLOR_MAP = {
    '성장주': '#D00000', '방어주': '#2B9348', '가치주/기반주': '#023E8A',
    '채권': '#FFC300', '기타': '#6C757D'
}

def get_group_by_sector(sector):
    growth = ['IT/반도체', '커뮤니케이션', '경기소비재']
    defense = ['필수소비재', '헬스케어', '유틸리티']
    value = ['금융', '에너지/소재', '산업재']
    bond = ['채권']
    
    if sector in growth: return "성장주"
    elif sector in defense: return "방어주"
    elif sector in value: return "가치주/기반주"
    elif sector in bond: return "채권"
    else: return "기타"

# API 데이터 가져오기
@st.cache_data(ttl=3600)
def get_exchange_rate():
    try:
        df = fdr.DataReader('USD/KRW', start=datetime.now() - timedelta(days=7))
        return df['Close'].iloc[-1]
    except:
        return 1300.0

@st.cache_data(ttl=600)
def get_current_price(ticker):
    try:
        df = fdr.DataReader(ticker, start=datetime.now() - timedelta(days=7))
        return df['Close'].iloc[-1]
    except:
        return 0.0

@st.cache_data(ttl=3600*24)
def get_sp500_data():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    df = fdr.DataReader('US500', start_date, end_date)
    if df.empty:
        df = fdr.DataReader('SPY', start_date, end_date)
    return df

@st.cache_data(ttl=3600)
def calculate_historical_assets(transactions_df):
    if transactions_df.empty:
        return pd.DataFrame()

    transactions_df['Date'] = pd.to_datetime(transactions_df['Date'])
    start_date = transactions_df['Date'].min()
    end_date = datetime.now()
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    daily_df = pd.DataFrame(index=date_range)
    daily_df.index.name = 'Date'
    
    try:
        usdkrw = fdr.DataReader('USD/KRW', start_date, end_date)['Close']
        usdkrw = usdkrw[~usdkrw.index.duplicated(keep='last')]
        
        spy_data = fdr.DataReader('SPY', start_date - timedelta(days=7), end_date)['Close']
        spy_data = spy_data[~spy_data.index.duplicated(keep='last')]
    except:
        return pd.DataFrame()

    daily_df['Exchange_Rate'] = usdkrw
    daily_df['SPY_Price'] = spy_data
    
    daily_df['Exchange_Rate'] = daily_df['Exchange_Rate'].ffill().bfill()
    daily_df['SPY_Price'] = daily_df['SPY_Price'].ffill().bfill()

    tickers = transactions_df[transactions_df['Ticker'].notna() & (transactions_df['Ticker'] != 'CASH')]['Ticker'].unique()
    price_data = {}
    for t in tickers:
        try:
            df = fdr.DataReader(t, start_date - timedelta(days=7), end_date)
            df = df[~df.index.duplicated(keep='last')]
            price_data[t] = df['Close']
        except:
            price_data[t] = pd.Series(0, index=date_range) 
    
    prices_df = pd.DataFrame(price_data).reindex(date_range).ffill().bfill()
    
    daily_df['Cash_Change'] = 0.0
    daily_df['Principal_Change'] = 0.0
    daily_df['SPY_Qty_Change'] = 0.0
    
    for t in tickers:
        daily_df[f'Qty_Change_{t}'] = 0.0
    
    for _, row in transactions_df.iterrows():
        d = row['Date']
        if d not in daily_df.index: continue 
        
        amt_krw = row['Total_KRW']
        rate_then = daily_df.at[d, 'Exchange_Rate']
        spy_price_then = daily_df.at[d, 'SPY_Price']
        
        if pd.isna(rate_then) or rate_then == 0: rate_then = 1300.0
        if pd.isna(spy_price_then) or spy_price_then == 0: spy_price_then = 400.0

        if row['Type'] == '입금':
            daily_df.at[d, 'Cash_Change'] += amt_krw
            daily_df.at[d, 'Principal_Change'] += amt_krw
            usd_amt = amt_krw / rate_then
            spy_qty = usd_amt / spy_price_then
            daily_df.at[d, 'SPY_
