import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import os
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
    
    # Secrets 설정 확인
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
            # 시트별 빈 데이터프레임 구조 정의
            if sheet_name == 'transactions':
                return pd.DataFrame(columns=['Date', 'Type', 'Ticker', 'Sector', 'Amount_USD', 'Quantity', 'Exchange_Rate', 'Total_KRW'])
            elif sheet_name == 'favorites':
                return pd.DataFrame(columns=['Ticker', 'Sector'])
            elif sheet_name == 'memos':  
                return pd.DataFrame(columns=['Date', 'Title', 'Content', 'Color'])
            elif sheet_name == 'targets':  # [NEW] 목표 비중 시트
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
        # 에러 발생 시 빈 객체 반환
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
        
        # 탭이 없으면 자동으로 생성하는 로직 추가
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
        spy_data = fdr.DataReader('SPY', start_date - timedelta(days=7), end_date)['Close']
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
            daily_df.at[d, 'SPY_Qty_Change'] += spy_qty

        elif row['Type'] == '출금':
            daily_df.at[d, 'Cash_Change'] -= amt_krw
            daily_df.at[d, 'Principal_Change'] -= amt_krw
            
            usd_amt = amt_krw / rate_then
            spy_qty = usd_amt / spy_price_then
            daily_df.at[d, 'SPY_Qty_Change'] -= spy_qty

        elif row['Type'] == '매수':
            daily_df.at[d, 'Cash_Change'] -= amt_krw
            if row['Ticker'] in tickers:
                daily_df.at[d, f"Qty_Change_{row['Ticker']}"] += row['Quantity']

        elif row['Type'] == '매도':
            daily_df.at[d, 'Cash_Change'] += amt_krw
            if row['Ticker'] in tickers:
                daily_df.at[d, f"Qty_Change_{row['Ticker']}"] -= row['Quantity']
        
        elif row['Type'] == '배당':
            daily_df.at[d, 'Cash_Change'] += amt_krw

        elif row['Type'] == '수수료':
            daily_df.at[d, 'Cash_Change'] -= amt_krw
            
        elif row['Type'] == '양도세매매':
            pass

    daily_df['Cash_Balance'] = daily_df['Cash_Change'].cumsum()
    daily_df['Invested_Principal'] = daily_df['Principal_Change'].cumsum()
    daily_df['SPY_Sim_Qty'] = daily_df['SPY_Qty_Change'].cumsum()
    
    for t in tickers:
        daily_df[f'Qty_{t}'] = daily_df[f'Qty_Change_{t}'].cumsum()

    daily_df['Stock_Eval_KRW'] = 0.0
    for t in tickers:
        qty_col = f'Qty_{t}'
        daily_val = daily_df[qty_col] * prices_df[t] * daily_df['Exchange_Rate']
        daily_df['Stock_Eval_KRW'] += daily_val.fillna(0)
        
    daily_df['Total_Asset_KRW'] = daily_df['Stock_Eval_KRW'] + daily_df['Cash_Balance']
    daily_df['Profit_KRW'] = daily_df['Total_Asset_KRW'] - daily_df['Invested_Principal']
    
    daily_df['SP500_Sim_Asset_KRW'] = daily_df['SPY_Sim_Qty'] * daily_df['SPY_Price'] * daily_df['Exchange_Rate']

    return daily_df

def calculate_tax_fifo(df, target_year):
    df = df.sort_values(by='Date')
    portfolio_queue = {} 
    realized_gains = [] 
    
    for _, row in df.iterrows():
        t_type = row['Type']
        ticker = row['Ticker']
        qty = row['Quantity']
        price = row['Amount_USD']
        rate = row['Exchange_Rate']
        date = pd.to_datetime(row['Date']).date()
        
        if t_type == '매수':
            if ticker not in portfolio_queue:
                portfolio_queue[ticker] = []
            portfolio_queue[ticker].append({
                'qty': qty,
                'price_usd': price,
                'rate': rate,
                'date': date
            })
            
        elif t_type == '매도':
            if ticker not in portfolio_queue:
                continue 
            
            remaining_sell_qty = qty
            total_buy_cost_krw = 0
            
            while remaining_sell_qty > 0 and portfolio_queue[ticker]:
                batch = portfolio_queue[ticker][0] 
                
                if batch['qty'] <= remaining_sell_qty:
                    cost = batch['qty'] * batch['price_usd'] * batch['rate']
                    total_buy_cost_krw += cost
                    remaining_sell_qty -= batch['qty']
                    portfolio_queue[ticker].pop(0) 
                else:
                    cost = remaining_sell_qty * batch['price_usd'] * batch['rate']
                    total_buy_cost_krw += cost
                    batch['qty'] -= remaining_sell_qty 
                    remaining_sell_qty = 0
            
            sell_revenue_krw = qty * price * rate
            gain_krw = sell_revenue_krw - total_buy_cost_krw
            
            if date.year == target_year:
                realized_gains.append({
                    '날짜': date,
                    '티커': ticker,
                    '수량': qty,
                    '매도금액(KRW)': sell_revenue_krw,
                    '매수금액(KRW, FIFO)': total_buy_cost_krw, 
                    '실현손익(KRW)': gain_krw
                })
        
        elif t_type == '양도세매매':
            if ticker in portfolio_queue:
                temp_sell_qty = qty
                temp_buy_cost_krw = 0
                
                while temp_sell_qty > 0 and portfolio_queue[ticker]:
                    batch = portfolio_queue[ticker][0]
                    if batch['qty'] <= temp_sell_qty:
                        cost = batch['qty'] * batch['price_usd'] * batch['rate']
                        temp_buy_cost_krw += cost
                        temp_sell_qty -= batch['qty']
                        portfolio_queue[ticker].pop(0)
                    else:
                        cost = temp_sell_qty * batch['price_usd'] * batch['rate']
                        temp_buy_cost_krw += cost
                        batch['qty'] -= temp_sell_qty
                        temp_sell_qty = 0
                
                sell_rev = qty * price * rate
                gain = sell_rev - temp_buy_cost_krw
                
                if date.year == target_year:
                    realized_gains.append({
                        '날짜': date,
                        '티커': ticker + " (양도세)",
                        '수량': qty,
                        '매도금액(KRW)': sell_rev,
                        '매수금액(KRW, FIFO)': temp_buy_cost_krw,
                        '실현손익(KRW)': gain
                    })
            
            if ticker not in portfolio_queue:
                portfolio_queue[ticker] = []
            portfolio_queue[ticker].append({
                'qty': qty,
                'price_usd': price,
                'rate': rate,
                'date': date
            })

    df['Date_dt'] = pd.to_datetime(df['Date'])
    fees_df = df[(df['Type'] == '수수료') & (df['Date_dt'].dt.year == target_year)]
    total_fees = fees_df['Total_KRW'].sum()
    
    return realized_gains, total_fees

# ---------------------------------------------------------
# 2. 전역 변수 계산 및 사이드바 설정
# ---------------------------------------------------------

st.sidebar.title("📈 StockWise")

menu = st.sidebar.radio("메뉴 이동", [
    "1. 총 자산 확인", 
    "2. 포트폴리오 분석", 
    "3. 수익 분석", 
    "4. 거래 기록 (입출금/매매)", 
    "5. 세금 관리 (양도세)",
    "6. 투자 메모 (Post-it)"
])

if 'last_menu' not in st.session_state:
    st.session_state['last_menu'] = menu

if st.session_state['last_menu'] != menu:
    st.session_state['last_menu'] = menu
    if menu == "4. 거래 기록 (입출금/매매)":
        st.session_state['tx_type_radio'] = "매수"
        if 'fav_selector' in st.session_state:
            del st.session_state['fav_selector']

# [데이터 로드] 구글 시트 사용
df = load_data_from_sheet('transactions')
current_rate = get_exchange_rate()

# [GLOBAL] 포트폴리오 및 현재 자산 계산
portfolio = {}
total_deposit_krw = 0
total_withdraw_krw = 0
current_cash_krw = 0 

if not df.empty:
    df = df.sort_values(by='Date')

for index, row in df.iterrows():
    if row['Type'] == '입금':
        total_deposit_krw += row['Total_KRW']
        current_cash_krw += row['Total_KRW']
    elif row['Type'] == '출금':
        total_withdraw_krw += row['Total_KRW']
        current_cash_krw -= row['Total_KRW']
    elif row['Type'] == '매수':
        current_cash_krw -= row['Total_KRW']
        if row['Ticker'] not in portfolio:
            portfolio[row['Ticker']] = {'qty': 0, 'invested_usd': 0, 'invested_krw': 0, 'sector': row['Sector']}
        portfolio[row['Ticker']]['qty'] += row['Quantity']
        portfolio[row['Ticker']]['invested_usd'] += (row['Amount_USD'] * row['Quantity'])
        portfolio[row['Ticker']]['invested_krw'] += row['Total_KRW']
    elif row['Type'] == '매도':
        current_cash_krw += row['Total_KRW']
        if row['Ticker'] in portfolio:
            if portfolio[row['Ticker']]['qty'] > 0:
                avg_price_usd = portfolio[row['Ticker']]['invested_usd'] / portfolio[row['Ticker']]['qty']
                avg_price_krw = portfolio[row['Ticker']]['invested_krw'] / portfolio[row['Ticker']]['qty']
                portfolio[row['Ticker']]['qty'] -= row['Quantity']
                portfolio[row['Ticker']]['invested_usd'] -= (avg_price_usd * row['Quantity'])
                portfolio[row['Ticker']]['invested_krw'] -= (avg_price_krw * row['Quantity'])
    elif row['Type'] == '배당':
        current_cash_krw += row['Total_KRW']
    elif row['Type'] == '수수료':
        current_cash_krw -= row['Total_KRW']
    elif row['Type'] == '양도세매매':
        pass

portfolio = {k: v for k, v in portfolio.items() if v['qty'] > 0.000001}

current_total_stock_val_krw = 0
for ticker, data in portfolio.items():
    curr_price = get_current_price(ticker)
    current_total_stock_val_krw += (curr_price * data['qty'] * current_rate)

current_total_asset_krw = current_total_stock_val_krw + current_cash_krw

st.sidebar.markdown("---")
st.sidebar.markdown("### 🚀 자산 목표 달성률")

# [설정 로드] 구글 시트 사용
app_config = load_config()
saved_goal1 = int(app_config.get('goal1', 100000000))
saved_goal2 = int(app_config.get('goal2', 1000000000))

with st.sidebar.expander("🎯 목표 금액 설정", expanded=False):
    goal1_target = st.number_input("1차 목표 (원)", value=saved_goal1, step=10_000_000, format="%d")
    goal2_target = st.number_input("2차 목표 (원)", value=saved_goal2, step=100_000_000, format="%d")
    
    if st.button("목표 저장"):
        save_config(goal1_target, goal2_target)
        st.success("목표 금액이 저장되었습니다!")
        st.rerun()

st.sidebar.caption(f"🥇 1차: {goal1_target:,.0f}원")
prog1 = min(current_total_asset_krw / goal1_target, 1.0) if goal1_target > 0 else 0
st.sidebar.progress(prog1)
st.sidebar.caption(f"{prog1*100:.1f}% ({current_total_asset_krw:,.0f}원)")

st.sidebar.caption(f"🥈 2차: {goal2_target:,.0f}원")
prog2 = min(current_total_asset_krw / goal2_target, 1.0) if goal2_target > 0 else 0
st.sidebar.progress(prog2)
st.sidebar.caption(f"{prog2*100:.1f}% ({current_total_asset_krw:,.0f}원)")


# ---------------------------------------------------------
# 3. 화면별 로직 구현
# ---------------------------------------------------------

def color_negative_red(val):
    if val > 0:
        return 'color: blue' 
    elif val < 0:
        return 'color: red' 
    else:
        return 'color: black'

if menu == "1. 총 자산 확인":
    st.title("💰 총 자산 현황")
    
    daily_df = calculate_historical_assets(df)
    
    diff_val = 0
    yesterday_asset = 0
    
    if not daily_df.empty:
        yesterday = datetime.now().date() - timedelta(days=1)
        yesterday_ts = pd.Timestamp(yesterday)
        
        if yesterday_ts in daily_df.index:
            yesterday_asset = daily_df.loc[yesterday_ts]['Total_Asset_KRW']
        else:
            past_data = daily_df[daily_df.index < pd.Timestamp(datetime.now().date())]
            if not past_data.empty:
                yesterday_asset = past_data.iloc[-1]['Total_Asset_KRW']
            else:
                yesterday_asset = current_total_asset_krw 
        
        diff_val = current_total_asset_krw - yesterday_asset
    
    total_stock_eval_usd = 0
    stock_details = []
    
    total_tickers = len(portfolio)
    if total_tickers > 0:
        progress_bar = st.progress(0)
    
    for i, (ticker, data) in enumerate(portfolio.items()):
        curr_price_usd = get_current_price(ticker)
        qty = data['qty']
        
        eval_value_usd = curr_price_usd * qty
        eval_value_krw = eval_value_usd * current_rate 
        
        total_stock_eval_usd += eval_value_usd
        
        invested_krw = data['invested_krw']
        invested_usd = data['invested_usd']
        
        stock_gain_usd = eval_value_usd - invested_usd
        stock_gain_krw = stock_gain_usd * current_rate
        
        total_gain_krw = eval_value_krw - invested_krw
        roi_percent = (total_gain_krw / invested_krw * 100) if invested_krw > 0 else 0
        avg_price_usd = invested_usd / qty if qty > 0 else 0

        stock_details.append({
            "티커": ticker,
            "보유수량": qty,
            "평단가($)": avg_price_usd,
            "현재가($)": curr_price_usd,
            "매수금액(₩)": invested_krw,      
            "평가금액(₩)": eval_value_krw,    
            "주가수익(₩)": stock_gain_krw,    
            "총손익(₩)": total_gain_krw,      
            "수익률(%)": roi_percent
        })
        if total_tickers > 0:
            progress_bar.progress((i + 1) / total_tickers)
    
    if total_tickers > 0:
        progress_bar.empty()
    
    if stock_details:
        stock_details.sort(key=lambda x: x["평가금액(₩)"], reverse=True)

    total_stock_eval_krw = total_stock_eval_usd * current_rate
    
    net_invest_krw = total_deposit_krw - total_withdraw_krw
    total_roi_krw = current_total_asset_krw - net_invest_krw
    total_roi_percent = (total_roi_krw / net_invest_krw * 100) if net_invest_krw != 0 else 0

    st.markdown(f"### 🏦 총 자산: {current_total_asset_krw:,.0f} 원")
    st.caption(f"전일 대비: {diff_val:+,.0f} 원 ({ (diff_val/yesterday_asset*100) if yesterday_asset>0 else 0 :+.2f}%)")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("투자 원금 (순입금)", f"{net_invest_krw:,.0f} 원")
    col2.metric("주식 평가금액", f"{total_stock_eval_krw:,.0f} 원")
    col3.metric("보유 현금 (예수금)", f"{current_cash_krw:,.0f} 원")
    col4.metric("총 수익률 (현금포함)", f"{total_roi_percent:.2f} %", f"{total_roi_krw:,.0f} 원")

    st.markdown("---")
    with st.expander("💵 예수금 잔고 보정 (배당/수수료 오차 수정)"):
        st.info("실제 잔고와 차이가 나는 경우, 그 원인이 **배당금 수령**이나 **수수료 차감** 등이라면 여기서 수정하세요. \n\n**주의:** 단순 입금/출금(원금 추가)은 '거래 기록' 탭을 이용해야 정확한 수익률이 계산됩니다.")
        
        adj_col1, adj_col2 = st.columns(2)
        with adj_col1:
            adj_currency = st.radio("통화 선택", ["KRW (원)", "USD (달러)"], horizontal=True)
        
        with adj_col2:
            target_balance = 0.0
            diff_krw = 0.0
            if adj_currency == "KRW (원)":
                target_balance = st.number_input("실제 잔고 (KRW)", value=float(current_cash_krw), step=1000.0, format="%.0f")
                diff_krw = target_balance - current_cash_krw
            else:
                est_usd = current_cash_krw / current_rate if current_rate > 0 else 0
                target_balance = st.number_input("실제 잔고 (USD)", value=float(est_usd), step=1.0, format="%.2f")
                target_krw_from_usd = target_balance * current_rate
                diff_krw = target_krw_from_usd - current_cash_krw
        
        if st.button("잔고 수정 적용 (수익/비용 반영)"):
            if abs(diff_krw) < 1:
                st.warning("변경 사항이 없습니다.")
            else:
                adj_type = '배당' if diff_krw > 0 else '수수료'
                adj_amount = abs(diff_krw)
                
                new_adj_data = {
                    'Date': datetime.now().date(),
                    'Type': adj_type,
                    'Ticker': 'CASH', 
                    'Sector': '-',
                    'Amount_USD': 0.0,
                    'Quantity': 1,
                    'Exchange_Rate': current_rate,
                    'Total_KRW': adj_amount
                }
                
                df = pd.concat([df, pd.DataFrame([new_adj_data])], ignore_index=True)
                save_data_to_sheet(df, 'transactions')
                st.success(f"잔고 보정이 완료되었습니다! ({adj_type} {adj_amount:,.0f}원 처리)")
                st.rerun()

    st.markdown("---")
    st.markdown("### 📋 보유 주식 상세 (수익 분석)")
    if stock_details:
        details_df = pd.DataFrame(stock_details)
        
        st.dataframe(
            details_df.style
            .format({
                "평단가($)": "{:.2f}", 
                "현재가($)": "{:.2f}", 
                "보유수량": "{:,.4f}",
                "매수금액(₩)": "{:,.0f}", 
                "평가금액(₩)": "{:,.0f}", 
                "주가수익(₩)": "{:,.0f}", 
                "총손익(₩)": "{:,.0f}", 
                "수익률(%)": "{:.2f}%"
            })
            .map(color_negative_red, subset=["주가수익(₩)", "총손익(₩)", "수익률(%)"]),
            use_container_width=True
        )
    else:
        st.write("보유 중인 주식이 없습니다.")

elif menu == "2. 포트폴리오 분석":
    st.title("📊 포트폴리오 분석")
    
    if not portfolio:
        st.warning("분석할 보유 주식이 없습니다.")
    else:
        data_list = []
        for ticker, data in portfolio.items():
            curr_price = get_current_price(ticker)
            val_usd = curr_price * data['qty']
            group = get_group_by_sector(data['sector'])
            invested_krw = data['invested_krw']

            data_list.append({
                'Ticker': ticker, 'Sector': data['sector'], 
                'Group': group, 'Value_USD': val_usd, 
                'Value_KRW': val_usd * current_rate,
                'Invested_KRW': invested_krw
            })
        
        pf_df = pd.DataFrame(data_list)
        
        group_order_map = {g: i for i, g in enumerate(GROUP_ORDER_LIST)}
        sector_order_map = {s: i for i, s in enumerate(SECTOR_OPTIONS)}
        
        pf_df['Group_Order'] = pf_df['Group'].map(group_order_map).fillna(99)
        pf_df['Sector_Order'] = pf_df['Sector'].map(sector_order_map).fillna(99)

        pf_df.sort_values(by=['Group_Order', 'Sector_Order', 'Value_USD'], ascending=[True, True, False], inplace=True)
        
        # [모바일 최적화] 좌우 컬럼(Columns) 대신 탭(Tabs)을 사용하여 모바일 화면 겹침 방지
        tab_pie1, tab_pie2 = st.tabs(["📊 주식 및 그룹별 비중", "📊 섹터별 비중"])
        
        def prepare_pie_data(df, group_col, value_col, threshold=0.01):
            total = df[value_col].sum()
            df_ratio = df.copy()
            df_ratio['ratio'] = df_ratio[value_col] / total
            
            main_df = df_ratio[df_ratio['ratio'] >= threshold].copy()
            small_df = df_ratio[df_ratio['ratio'] < threshold].copy()
            
            main_df['extra_hover'] = ""

            if not small_df.empty:
                other_data = {col: '기타' for col in df.columns}
                other_data[value_col] = small_df[value_col].sum()
                for col in ['Value_KRW', 'Invested_KRW']: 
                    if col in df.columns: other_data[col] = small_df[col].sum()
                
                if 'Group_Order' in df.columns: other_data['Group_Order'] = 999
                if 'Sector_Order' in df.columns: other_data['Sector_Order'] = 999

                if group_col == 'Ticker': other_data['Sector'] = '기타'
                if group_col == 'Sector': other_data['Group'] = '기타'
                
                details = []
                small_df_sorted = small_df.sort_values(by=value_col, ascending=False)
                for _, row in small_df_sorted.iterrows():
                    pct = (row[value_col] / total) * 100
                    details.append(f"{row[group_col]} ({pct:.2f}%)")
                
                other_desc = "<br>".join(details)
                other_row = pd.DataFrame([other_data])
                other_row['extra_hover'] = f"<br><br><b>[포함된 항목]</b><br>{other_desc}"
                
                main_df = pd.concat([main_df, other_row], ignore_index=True)
            
            return main_df

        with tab_pie1:
            st.subheader("1. 주식별 비중")
            stock_pie_df = prepare_pie_data(pf_df, 'Ticker', 'Value_USD', threshold=0.01)
            
            fig1 = px.pie(stock_pie_df, values='Value_USD', names='Ticker', color='Sector', 
                          color_discrete_map=SECTOR_COLOR_MAP, hole=0.4,
                          custom_data=['extra_hover'],
                          labels={'Ticker': '종목', 'Sector': '섹터', 'Group': '그룹', 'Value_USD': '평가액($)'})
            
            fig1.update_traces(
                sort=False, 
                rotation=180,
                textposition='inside',
                textinfo='percent+label', 
                texttemplate='%{label}<br>%{percent:.0%}',
                hovertemplate='<b>%{label}</b><br>비중: %{percent}<br>평가금: $%{value:,.2f}%{customdata[0]}<extra></extra>'
            )
            # [모바일 최적화] 범례(Legend)를 가로(아래쪽)로 배치하여 차트 크기 극대화
            fig1.update_layout(uniformtext_minsize=12, uniformtext_mode='hide', legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5))
            st.plotly_chart(fig1, use_container_width=True)
            
            st.subheader("2. 그룹별 비중")
            group_agg = pf_df.groupby(['Group', 'Group_Order'], as_index=False)['Value_USD'].sum()
            group_agg.sort_values(by='Group_Order', inplace=True)
            
            group_pie_df = prepare_pie_data(group_agg, 'Group', 'Value_USD', threshold=0)
            
            fig3 = px.pie(group_pie_df, values='Value_USD', names='Group', hole=0.4, 
                          color='Group', 
                          color_discrete_map=GROUP_COLOR_MAP,
                          custom_data=['extra_hover'],
                          labels={'Group': '그룹', 'Value_USD': '평가액($)'})
            
            fig3.update_traces(
                sort=False, 
                textposition='inside',
                textinfo='percent+label', 
                texttemplate='%{label}<br>%{percent:.0%}',
                hovertemplate='<b>%{label}</b><br>비중: %{percent}<br>평가금: $%{value:,.2f}%{customdata[0]}<extra></extra>'
            )
            fig3.update_layout(uniformtext_minsize=12, uniformtext_mode='hide', legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5))
            st.plotly_chart(fig3, use_container_width=True)

        with tab_pie2:
            st.subheader("3. 섹터별 비중")
            sector_agg = pf_df.groupby(['Group', 'Group_Order', 'Sector', 'Sector_Order'], as_index=False)['Value_USD'].sum()
            sector_agg.sort_values(by=['Group_Order', 'Sector_Order'], inplace=True)
            
            sector_pie_df = prepare_pie_data(sector_agg, 'Sector', 'Value_USD', threshold=0)
            
            fig2 = px.pie(sector_pie_df, values='Value_USD', names='Sector', hole=0.4,
                          color='Sector',
                          color_discrete_map=SECTOR_COLOR_MAP,
                          custom_data=['extra_hover'],
                          labels={'Sector': '섹터', 'Value_USD': '평가액($)'})
            
            fig2.update_traces(
                sort=False, 
                rotation=180,
                textposition='inside',
                textinfo='percent+label', 
                texttemplate='%{label}<br>%{percent:.0%}',
                hovertemplate='<b>%{label}</b><br>비중: %{percent}<br>평가금: $%{value:,.2f}%{customdata[0]}<extra></extra>'
            )
            fig2.update_layout(uniformtext_minsize=12, uniformtext_mode='hide', legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5))
            st.plotly_chart(fig2, use_container_width=True)
            
        st.markdown("---")
        st.subheader("4. 섹터별 수익 현황")
        
        sector_stats = pf_df.groupby('Sector')[['Invested_KRW', 'Value_KRW']].sum().reset_index()
        sector_stats['Profit_KRW'] = sector_stats['Value_KRW'] - sector_stats['Invested_KRW']
        sector_stats['수익금(만원)'] = sector_stats['Profit_KRW'] / 10000
        
        sector_stats['ROI'] = (sector_stats['Profit_KRW'] / sector_stats['Invested_KRW'] * 100).fillna(0)
        sector_stats = sector_stats.sort_values(by='ROI', ascending=False)
        
        # [모바일 최적화] 중첩된 표와 차트를 1단 탭으로 분리하여 쾌적하게 표시
        tab_sec_tbl, tab_sec_roi, tab_sec_profit = st.tabs(["📋 표로 보기", "📊 수익률 차트", "💰 수익금 차트"])
        
        with tab_sec_tbl:
            st.dataframe(
                sector_stats.style.format({
                    "Invested_KRW": "{:,.0f}",
                    "Value_KRW": "{:,.0f}",
                    "Profit_KRW": "{:,.0f}",
                    "ROI": "{:.2f}%"
                })
                .map(color_negative_red, subset=["Profit_KRW", "ROI"]),
                use_container_width=True,
                column_config={
                    "Sector": "섹터",
                    "Invested_KRW": "총 투자금(₩)",
                    "Value_KRW": "평가금액(₩)",
                    "Profit_KRW": "손익(₩)",
                    "ROI": "수익률(%)"
                }
            )

        with tab_sec_roi:
            fig_roi = px.bar(sector_stats, x='Sector', y='ROI', color='Sector', 
                             text_auto='.2f',
                             title="섹터별 수익률 (%)",
                             color_discrete_map=SECTOR_COLOR_MAP,
                             labels={'Sector': '섹터', 'ROI': '수익률(%)'}) 
            fig_roi.update_layout(showlegend=False, yaxis_title="수익률 (%)")
            st.plotly_chart(fig_roi, use_container_width=True)
        
        with tab_sec_profit:
            fig_profit = px.bar(sector_stats, x='Sector', y='수익금(만원)', color='Sector',
                                text_auto=',.0f',
                                title="섹터별 수익금 (단위: 만원)",
                                color_discrete_map=SECTOR_COLOR_MAP,
                                labels={'Sector': '섹터', '수익금(만원)': '수익금(만원)'}) 
            fig_profit.update_layout(showlegend=False, yaxis_title="수익금 (만원)")
            st.plotly_chart(fig_profit, use_container_width=True)

        st.markdown("---")
        st.subheader("5. 그룹별 수익 현황")
        
        group_stats = pf_df.groupby('Group')[['Invested_KRW', 'Value_KRW']].sum().reset_index()
        group_stats['Profit_KRW'] = group_stats['Value_KRW'] - group_stats['Invested_KRW']
        group_stats['수익금(만원)'] = group_stats['Profit_KRW'] / 10000
        
        group_stats['ROI'] = (group_stats['Profit_KRW'] / group_stats['Invested_KRW'] * 100).fillna(0)
        group_stats = group_stats.sort_values(by='ROI', ascending=False)
        
        # [모바일 최적화] 중첩된 표와 차트를 1단 탭으로 분리하여 쾌적하게 표시
        tab_grp_tbl, tab_grp_roi, tab_grp_profit = st.tabs(["📋 표로 보기", "📊 수익률 차트", "💰 수익금 차트"])
        
        with tab_grp_tbl:
            st.dataframe(
                group_stats.style.format({
                    "Invested_KRW": "{:,.0f}",
                    "Value_KRW": "{:,.0f}",
                    "Profit_KRW": "{:,.0f}",
                    "ROI": "{:.2f}%"
                })
                .map(color_negative_red, subset=["Profit_KRW", "ROI"]),
                use_container_width=True,
                column_config={
                    "Group": "그룹",
                    "Invested_KRW": "총 투자금(₩)",
                    "Value_KRW": "평가금액(₩)",
                    "Profit_KRW": "손익(₩)",
                    "ROI": "수익률(%)"
                }
            )
            
        with tab_grp_roi:
            fig_roi_g = px.bar(group_stats, x='Group', y='ROI', color='Group', 
                             text_auto='.2f',
                             title="그룹별 수익률 (%)",
                             color_discrete_map=GROUP_COLOR_MAP,
                             labels={'Group': '그룹', 'ROI': '수익률(%)'}) 
            fig_roi_g.update_layout(showlegend=False, yaxis_title="수익률 (%)")
            st.plotly_chart(fig_roi_g, use_container_width=True)
            
        with tab_grp_profit:
            fig_profit_g = px.bar(group_stats, x='Group', y='수익금(만원)', color='Group',
                                text_auto=',.0f',
                                title="그룹별 수익금 (단위: 만원)",
                                color_discrete_map=GROUP_COLOR_MAP,
                                labels={'Group': '그룹', '수익금(만원)': '수익금(만원)'}) 
            fig_profit_g.update_layout(showlegend=False, yaxis_title="수익금 (만원)")
            st.plotly_chart(fig_profit_g, use_container_width=True)

        # -------------------------------------------------------------
        # [NEW] 🎯 6. 목표 비중 설정 및 리밸런싱
        # -------------------------------------------------------------
        st.markdown("---")
        st.subheader("🎯 6. 목표 비중 설정 및 리밸런싱")
        st.caption("각 종목의 **목표 비중(%)** 열을 수정하세요. 현재 자산과 비교하여 필요한 매수/매도 수량을 실시간으로 알려줍니다.")

        # 1. Session State에 타겟 비율 초기화 (DB -> Session)
        if 'target_ratios' not in st.session_state:
            targets_df = load_data_from_sheet('targets')
            if not targets_df.empty:
                st.session_state.target_ratios = dict(zip(targets_df['Ticker'], targets_df['Target_Ratio']))
            else:
                st.session_state.target_ratios = {}

        # 2. 에디터 렌더링 순서를 기억하기 위한 리스트
        st.session_state.rebal_tickers = list(portfolio.keys())

        # 3. 데이터 변경 시 작동할 콜백 함수
        def update_target_ratio_callback():
            changes = st.session_state.rebalance_editor['edited_rows']
            for row_idx_str, col_changes in changes.items():
                if 'Target_Ratio' in col_changes:
                    row_idx = int(row_idx_str)
                    ticker = st.session_state.rebal_tickers[row_idx]
                    new_ratio = col_changes['Target_Ratio']
                    st.session_state.target_ratios[ticker] = new_ratio

        # 4. 리밸런싱 데이터프레임 계산
        total_pf_value_usd = sum([get_current_price(t) * d['qty'] for t, d in portfolio.items()])
        
        rebal_list = []
        for ticker in st.session_state.rebal_tickers:
            data = portfolio[ticker]
            curr_price = get_current_price(ticker)
            qty = data['qty']
            val_usd = curr_price * qty
            curr_ratio = (val_usd / total_pf_value_usd * 100) if total_pf_value_usd > 0 else 0
            
            # session_state에 값이 있으면 가져오고, 없으면 현재 비율로 초기 세팅
            target_ratio = st.session_state.target_ratios.get(ticker, round(curr_ratio, 1))
            
            target_val_usd = total_pf_value_usd * (target_ratio / 100)
            diff_usd = target_val_usd - val_usd
            
            action = "-"
            if curr_price > 0:
                shares = diff_usd / curr_price
                ratio_diff = target_ratio - curr_ratio
                
                # 1주 이상 차이 나거나, 비중이 5% 이상 차이날 때 액션 표시
                if shares >= 1.0 or ratio_diff >= 5.0:
                    shares_disp = f"{int(shares):,}" if shares >= 1.0 else f"{shares:.2f}"
                    action = f"🛒 +{shares_disp}주 매수"
                elif shares <= -1.0 or ratio_diff <= -5.0:
                    shares_disp = f"{-int(shares):,}" if shares <= -1.0 else f"{-shares:.2f}"
                    action = f"💰 -{shares_disp}주 매도"

            rebal_list.append({
                'Ticker': ticker,
                'Price_USD': curr_price,
                'Quantity': qty,
                'Value_USD': val_usd,
                'Current_Ratio': curr_ratio,
                'Target_Ratio': target_ratio,
                'Diff_USD': diff_usd,
                'Action': action
            })
            
        rebal_df = pd.DataFrame(rebal_list)
        
        # 5. UI 출력: 메시지 및 저장 버튼
        total_target_ratio = sum([r['Target_Ratio'] for r in rebal_list])
        
        col_msg, col_btn = st.columns([3, 1])
        with col_msg:
            if abs(total_target_ratio - 100.0) > 0.1:
                st.warning(f"⚠️ 설정된 목표 비중 합계: **{total_target_ratio:.1f}%** (100%에 맞춰주세요)")
            else:
                st.success(f"✅ 목표 비중 합계: **{total_target_ratio:.1f}%**")
                
        with col_btn:
            # 실시간 뷰어 역할은 에디터가 하고, 실제 DB 저장은 이 버튼으로 수행 (최적화)
            if st.button("💾 목표 비중 저장하기", use_container_width=True, type="primary"):
                save_df = pd.DataFrame({
                    'Ticker': list(st.session_state.target_ratios.keys()),
                    'Target_Ratio': list(st.session_state.target_ratios.values())
                })
                save_data_to_sheet(save_df, 'targets')
                st.toast("목표 비중이 구글 시트에 안전하게 저장되었습니다!", icon="✅")

        # 6. 실시간 데이터 에디터 렌더링
        st.data_editor(
            rebal_df,
            disabled=['Ticker', 'Price_USD', 'Quantity', 'Value_USD', 'Current_Ratio', 'Diff_USD', 'Action'],
            hide_index=True,
            column_config={
                "Ticker": st.column_config.TextColumn("종목", width="medium"),
                "Price_USD": st.column_config.NumberColumn("현재가($)", format="$%.2f"),
                "Quantity": st.column_config.NumberColumn("보유 수량", format="%.4f"),
                "Value_USD": st.column_config.NumberColumn("평가액($)", format="$%.2f"),
                "Current_Ratio": st.column_config.NumberColumn("현재 비중(%)", format="%.1f%%"),
                "Target_Ratio": st.column_config.NumberColumn("🎯 목표 비중(%) ✏️", format="%.1f", min_value=0.0, step=0.1),
                "Diff_USD": st.column_config.NumberColumn("조정 필요 금액($)", format="$%.2f"),
                "Action": st.column_config.TextColumn("리밸런싱 액션 (1주 또는 5% 이상)", width="medium")
            },
            use_container_width=True,
            key="rebalance_editor",
            on_change=update_target_ratio_callback
        )


elif menu == "3. 수익 분석":
    st.title("📈 수익 분석")
    
    if df.empty:
        st.warning("거래 내역이 없어 수익을 분석할 수 없습니다.")
    else:
        with st.spinner('과거 자산 데이터를 계산 중입니다... (종목 수에 따라 시간이 걸릴 수 있습니다)'):
            daily_df = calculate_historical_assets(df)

        if not daily_df.empty:
            daily_df['Invested_Principal_10k'] = daily_df['Invested_Principal'] / 10000
            daily_df['Total_Asset_KRW_10k'] = daily_df['Total_Asset_KRW'] / 10000
            daily_df['SP500_Sim_Asset_KRW_10k'] = daily_df['SP500_Sim_Asset_KRW'] / 10000
            daily_df['Profit_KRW_10k'] = daily_df['Profit_KRW'] / 10000

            st.subheader("1. 벤치마크 비교 (시장 vs 내 자산)")
            st.caption("선택한 기간의 **시작일 자산**을 기준으로, S&P 500 투자 가정 시 성과와 실제 내 자산 성과를 비교합니다.")
            
            period_option = st.radio("기간 선택", ["올해 (YTD)", "최근 1년", "전체 기간", "직접 입력"], horizontal=True, key="benchmark_period_select")
            
            plot_df = daily_df.copy()

            if period_option == "올해 (YTD)":
                start_of_year = datetime(datetime.now().year, 1, 1)
                plot_df = daily_df[daily_df.index >= pd.Timestamp(start_of_year)].copy()

            elif period_option == "최근 1년":
                one_year_ago = datetime.now() - timedelta(days=365)
                if daily_df.index.min() < one_year_ago:
                    plot_df = daily_df[daily_df.index >= one_year_ago].copy()
            
            elif period_option == "직접 입력":
                min_date = daily_df.index.min().date()
                max_date = daily_df.index.max().date()
                custom_start = st.date_input("비교 시작일 선택", value=min_date, min_value=min_date, max_value=max_date)
                custom_start_ts = pd.Timestamp(custom_start)
                plot_df = daily_df[daily_df.index >= custom_start_ts].copy()

            if not plot_df.empty:
                # [개선된 로직] 기준점(시작일)에 맞춰 그래프 Rebase
                start_my_asset = plot_df['Total_Asset_KRW_10k'].iloc[0]
                start_sp500_sim = plot_df['SP500_Sim_Asset_KRW_10k'].iloc[0]
                start_principal = plot_df['Invested_Principal_10k'].iloc[0]

                # 1. 내 자산 (실제 데이터)
                plot_df['Rebased_My_Asset'] = plot_df['Total_Asset_KRW_10k']

                # 2. S&P 500 (비율 조정: 시작점에서 내 자산과 동일하게 출발)
                if start_sp500_sim != 0:
                    sp500_ratio = start_my_asset / start_sp500_sim
                    plot_df['Rebased_SP500'] = plot_df['SP500_Sim_Asset_KRW_10k'] * sp500_ratio
                else:
                    plot_df['Rebased_SP500'] = plot_df['SP500_Sim_Asset_KRW_10k']

                # 3. 투자 원금 (차액 보정: 시작점에서 내 자산과 동일하게 출발하고 입출금만 반영)
                plot_df['Rebased_Principal'] = (plot_df['Invested_Principal_10k'] - start_principal) + start_my_asset

            else:
                plot_df['Rebased_My_Asset'] = 0
                plot_df['Rebased_SP500'] = 0
                plot_df['Rebased_Principal'] = 0

            fig_bm = go.Figure()
            fig_bm.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Rebased_My_Asset'], mode='lines', name='내 총 자산 (실제)', line=dict(color='#d62728', width=2)))
            fig_bm.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Rebased_SP500'], mode='lines', name='S&P 500 투자 가정', line=dict(color='#1f77b4', width=2)))
            fig_bm.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Rebased_Principal'], mode='lines', name='현금 보유 가정 (입출금 반영)', line=dict(color='gray', dash='dash', width=1)))

            fig_bm.update_layout(
                xaxis_title="날짜", yaxis_title="평가금액 (단위: 만원)",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_bm, use_container_width=True)
            
            st.subheader("2. 누적 수익금 추이 (전체 기간)")
            fig_profit = px.line(daily_df, x=daily_df.index, y='Profit_KRW_10k', title="일별 누적 수익금 변화")
            fig_profit.update_traces(line_color='#2ca02c') 
            fig_profit.add_hline(y=0, line_dash="dot", line_color="black")
            fig_profit.update_layout(xaxis_title="날짜", yaxis_title="수익금 (단위: 만원)", hovermode="x unified")
            st.plotly_chart(fig_profit, use_container_width=True)

            st.subheader("3. 연도별 수익 현황")
            daily_df['Prev_Profit'] = daily_df['Profit_KRW'].shift(1).fillna(0)
            daily_df['Daily_Profit_Change'] = daily_df['Profit_KRW'] - daily_df['Prev_Profit']
            daily_df['Year'] = daily_df.index.year
            yearly_stats = daily_df.groupby('Year')['Daily_Profit_Change'].sum().reset_index()
            yearly_stats.rename(columns={'Daily_Profit_Change': 'Yearly_Profit_KRW'}, inplace=True)
            yearly_stats['Yearly_Profit_KRW_10k'] = yearly_stats['Yearly_Profit_KRW'] / 10000
            yearly_stats['Color'] = yearly_stats['Yearly_Profit_KRW'].apply(lambda x: '#e53935' if x >= 0 else '#1e88e5')

            fig_year = go.Figure()
            fig_year.add_trace(go.Bar(
                x=yearly_stats['Year'], y=yearly_stats['Yearly_Profit_KRW_10k'],
                marker_color=yearly_stats['Color'], name='수익금(만원)',
                text=yearly_stats['Yearly_Profit_KRW_10k'].apply(lambda x: f"{x:,.0f}"), textposition='auto'
            ))
            fig_year.update_layout(title="연도별 발생 수익금", xaxis_title="연도", xaxis=dict(tickmode='linear'), yaxis_title="수익금 (단위: 만원)")
            st.plotly_chart(fig_year, use_container_width=True)
        else:
            st.info("데이터를 계산할 수 없습니다. 거래 내역을 확인해주세요.")

elif menu == "4. 거래 기록 (입출금/매매)":
    st.title("📝 입금/출금/매수/매도 기록")
    
    with st.expander("⭐ 즐겨찾기(단축) 종목 관리 (클릭해서 열기/닫기)"):
        st.caption("자주 거래하는 종목을 등록하면 아래에서 쉽게 선택할 수 있습니다. 행을 선택하고 Delete 키를 누르면 삭제됩니다.")
        
        fav_df = load_data_from_sheet('favorites')
        
        edited_fav_df = st.data_editor(
            fav_df,
            num_rows="dynamic",
            column_config={
                "Ticker": st.column_config.TextColumn("티커 (예: AAPL)", required=True),
                "Sector": st.column_config.SelectboxColumn(
                    "섹터",
                    options=SECTOR_OPTIONS,
                    required=True
                )
            },
            key="fav_editor",
            use_container_width=True
        )
        
        if st.button("즐겨찾기 변경사항 저장"):
            save_data_to_sheet(edited_fav_df, 'favorites')
            st.success("즐겨찾기 목록이 업데이트되었습니다!")
            st.rerun()

    st.divider()
    
    tx_type = st.radio("거래 종류 선택", ["매수", "매도", "입금", "출금", "배당", "수수료", "양도세매매"], horizontal=True, key="tx_type_radio")

    if tx_type in ["매수", "매도", "양도세매매"]:
        current_favs = load_data_from_sheet('favorites')
        def update_form_from_fav():
            selection = st.session_state.fav_selector
            loaded_favs = load_data_from_sheet('favorites')
            if selection != "직접 입력" and not loaded_favs.empty:
                row = loaded_favs[loaded_favs['Ticker'] == selection].iloc[0]
                st.session_state.form_ticker = row['Ticker']
                if row['Sector'] in SECTOR_OPTIONS:
                    st.session_state.form_sector = row['Sector']
        
        if not current_favs.empty:
            fav_options = ["직접 입력"] + current_favs['Ticker'].tolist()
            st.selectbox(
                "⚡ 빠른 입력 (즐겨찾기 선택)", 
                fav_options, 
                key="fav_selector", 
                on_change=update_form_from_fav
            )

    with st.form("transaction_form"):
        col1, col2 = st.columns(2)
        date = col1.date_input("날짜", datetime.now())
        
        if tx_type in ["매수", "매도", "양도세매매"]:
            col3, col4 = st.columns(2)
            
            if 'form_ticker' not in st.session_state: st.session_state['form_ticker'] = ""
            if 'form_sector' not in st.session_state: st.session_state['form_sector'] = SECTOR_OPTIONS[0]
            
            ticker = col3.text_input("티커 (예: AAPL, TSLA)", key="form_ticker").upper()
            
            if st.session_state.form_sector not in SECTOR_OPTIONS:
                st.session_state.form_sector = SECTOR_OPTIONS[0]
                
            sector = col4.selectbox("섹터", SECTOR_OPTIONS, key="form_sector")
            
            col5, col6, col7 = st.columns(3)
            amount_usd = col5.number_input("달러 단가 ($)", min_value=0.0, format="%.2f")
            quantity = col6.number_input("수량", min_value=0.0001, step=0.0001, format="%.4f")
            ex_rate_input = col7.number_input("적용 환율 (₩/$)", value=float(round(current_rate, 2)), format="%.2f")
            estimated_krw = amount_usd * quantity * ex_rate_input
            
            if tx_type == "양도세매매":
                st.caption(f"💡 양도세 절세용 매매: 현재가로 매도 후 즉시 재매수한 것으로 처리합니다. (평단가는 유지되나 실현 손익은 발생)")
            else:
                st.caption(f"💡 예상 원화 금액: {estimated_krw:,.0f} 원")
            
            input_krw_amount = 0 

        else:
            msg = "금액은 원화(KRW) 기준으로 기록됩니다."
            if tx_type == "배당": msg += " (수익 처리, 원금 증가 안 함)"
            elif tx_type == "수수료": msg += " (비용 처리, 원금 감소 안 함)"
            st.info(msg)
            
            input_krw_amount = st.number_input("금액 (KRW)", min_value=0, step=1000)
            ticker = "CASH"
            sector = "-"
            amount_usd = 0.0
            quantity = 1
            ex_rate_input = 1.0
            estimated_krw = input_krw_amount 

        submitted = st.form_submit_button("기록 저장")
        
        if submitted:
            valid = True
            if tx_type in ['매수', '매도', '양도세매매']:
                if not ticker:
                    st.error("티커를 입력해주세요.")
                    valid = False
                elif quantity <= 0:
                    st.error("수량은 0보다 커야 합니다.")
                    valid = False
                final_total_krw = estimated_krw
            else:
                if input_krw_amount <= 0:
                    st.error("금액은 0보다 커야 합니다.")
                    valid = False
                final_total_krw = input_krw_amount

            if valid:
                new_data = {
                    'Date': date,
                    'Type': tx_type,
                    'Ticker': ticker,
                    'Sector': sector,
                    'Amount_USD': amount_usd,
                    'Quantity': quantity,
                    'Exchange_Rate': ex_rate_input,
                    'Total_KRW': final_total_krw
                }
                
                df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
                save_data_to_sheet(df, 'transactions')

                if tx_type in ['매수', '매도', '양도세매매']:
                    fav_df = load_data_from_sheet('favorites')
                    
                    if ticker in fav_df['Ticker'].values:
                        fav_df.loc[fav_df['Ticker'] == ticker, 'Sector'] = sector
                    else:
                        new_fav = pd.DataFrame([{'Ticker': ticker, 'Sector': sector}])
                        fav_df = pd.concat([fav_df, new_fav], ignore_index=True)
                    
                    save_data_to_sheet(fav_df, 'favorites')
                    st.toast(f"⭐ '{ticker}' 종목이 즐겨찾기에 반영되었습니다!", icon="✅")

                st.success("거래 내역이 저장되었습니다!")
                st.rerun()

    st.markdown("### 📜 최근 거래 내역 (수정/삭제)")
    st.caption("표의 내용을 더블 클릭해 수정하거나, 행을 선택(왼쪽 체크박스) 후 **Delete 키**를 눌러 삭제할 수 있습니다.")

    if not df.empty:
        sorted_df = df.sort_values(by='Date', ascending=False).reset_index(drop=True)
        
        edited_df = st.data_editor(
            sorted_df,
            num_rows="dynamic",
            column_config={
                "Date": st.column_config.DateColumn("날짜", format="YYYY-MM-DD"),
                "Type": st.column_config.SelectboxColumn("종류", options=["매수", "매도", "입금", "출금", "배당", "수수료", "양도세매매"], required=True),
                "Ticker": st.column_config.TextColumn("티커"),
                "Sector": st.column_config.SelectboxColumn("섹터", options=SECTOR_OPTIONS),
                "Amount_USD": st.column_config.NumberColumn("달러 단가($)", format="%.2f"),
                "Quantity": st.column_config.NumberColumn("수량", format="%.4f"),
                "Exchange_Rate": st.column_config.NumberColumn("환율(₩/$)", format="%.2f"),
                "Total_KRW": st.column_config.NumberColumn("원화 합계(₩)", format="%d")
            },
            use_container_width=True,
            key="history_editor"
        )
        
        if st.button("거래 내역 변경사항 저장", type="primary"):
            save_data_to_sheet(edited_df, 'transactions')
            st.success("거래 내역이 업데이트되었습니다!")
            st.rerun()
            
    else:
        st.write("아직 기록된 거래가 없습니다.")

elif menu == "5. 세금 관리 (양도세)":
    st.title("💸 세금 관리 (양도소득세)")
    st.caption("미국 주식 양도소득세는 **선입선출법(FIFO)**을 기준으로 계산되며, 연간 **250만 원**까지 기본 공제됩니다.")
    
    current_year = datetime.now().year
    target_year = st.selectbox("조회 연도 선택", range(current_year, current_year - 5, -1))
    
    if df.empty:
        st.warning("거래 내역이 없습니다.")
    else:
        realized_gains, total_fees = calculate_tax_fifo(df, target_year)
        
        total_revenue = sum(item['매도금액(KRW)'] for item in realized_gains)
        total_cost = sum(item['매수금액(KRW, FIFO)'] for item in realized_gains)
        gross_profit = total_revenue - total_cost
        
        net_profit = gross_profit - total_fees
        taxable_income = max(0, net_profit - 2_500_000)
        estimated_tax = taxable_income * 0.22 
        
        col1, col2, col3 = st.columns(3)
        col1.metric("총 실현 손익 (수수료 차감후)", f"{net_profit:,.0f} 원")
        col2.metric("양도세 과세 표준 (공제후)", f"{taxable_income:,.0f} 원", f"공제 250만원")
        col3.metric("예상 납부 세액 (22%)", f"{estimated_tax:,.0f} 원")
        
        st.subheader("📊 기본 공제(250만원) 사용 현황")
        
        deduction_used = min(net_profit, 2_500_000) if net_profit > 0 else 0
        deduction_percent = (deduction_used / 2_500_000) * 100
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = deduction_used,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"공제 사용액: {deduction_used:,.0f}원 ({deduction_percent:.1f}%)"},
            number = {'valueformat': ",.0f"}, 
            gauge = {
                'axis': {'range': [None, 2500000], 'tickformat': ",.0f"}, 
                'bar': {'color': "#2ca02c" if deduction_used < 2500000 else "#d62728"},
                'steps': [
                    {'range': [0, 2000000], 'color': "lightgray"},
                    {'range': [2000000, 2500000], 'color': "gray"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 2500000}
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        if net_profit < 2500000:
            remaining = 2500000 - max(0, net_profit)
            st.success(f"💡 아직 **{remaining:,.0f}원**의 비과세 수익 한도가 남아있습니다! 수익 중인 종목을 매도하여 절세할 수 있습니다.")
        else:
            st.warning(f"⚠️ 기본 공제 한도를 초과했습니다. 초과분에 대해 **22%**의 세금이 발생합니다.")

        st.markdown("---")
        st.subheader("📜 상세 매도 내역 (선입선출 기준)")
        st.caption("각 매도 건별로 어떤 매수 물량(FIFO)과 매칭되었는지 계산된 결과입니다.")
        
        if realized_gains:
            gain_df = pd.DataFrame(realized_gains)
            
            st.dataframe(
                gain_df.style.format({
                    "수량": "{:,.4f}",
                    "매도환율": "{:,.2f}",
                    "매도금액(KRW)": "{:,.0f}",
                    "매수금액(KRW, FIFO)": "{:,.0f}",
                    "실현손익(KRW)": "{:,.0f}"
                })
                .map(color_negative_red, subset=["실현손익(KRW)"]),
                use_container_width=True
            )
            
            st.info(f"➕ 이 해에 납부한 총 수수료: **{total_fees:,.0f}원** (실현 손익에서 일괄 차감됨)")
        else:
            st.write("해당 연도의 매도 내역이 없습니다.")

elif menu == "6. 투자 메모 (Post-it)":
    st.title("🗒️ 투자 메모 (Idea Note)")
    st.caption("투자 원칙, 매수 아이디어, 반성할 점 등을 포스트잇처럼 기록하세요.")

    # [데이터 로드]
    memos_df = load_data_from_sheet('memos')

    # --- 메모 작성 폼 ---
    with st.expander("✍️ 새 메모 작성하기", expanded=False):
        with st.form("memo_form"):
            col1, col2 = st.columns([3, 1])
            input_title = col1.text_input("제목", placeholder="예: 테슬라 매수 이유")
            input_color = col2.selectbox("색상 선택", ["노랑 (Yellow)", "분홍 (Pink)", "파랑 (Blue)", "초록 (Green)"])
            
            input_content = st.text_area("내용", height=150, placeholder="자유롭게 내용을 작성하세요...")
            
            submitted = st.form_submit_button("메모 붙이기")
            
            if submitted:
                if not input_title or not input_content:
                    st.error("제목과 내용을 모두 입력해주세요.")
                else:
                    # 색상 코드 매핑
                    color_map = {
                        "노랑 (Yellow)": "#FFF475",
                        "분홍 (Pink)": "#F28B82",
                        "파랑 (Blue)": "#A7FFEB",
                        "초록 (Green)": "#CCFF90"
                    }
                    
                    new_memo = {
                        'Date': datetime.now().strftime("%Y-%m-%d %H:%M"),
                        'Title': input_title,
                        'Content': input_content,
                        'Color': color_map[input_color]
                    }
                    
                    memos_df = pd.concat([memos_df, pd.DataFrame([new_memo])], ignore_index=True)
                    save_data_to_sheet(memos_df, 'memos')
                    st.success("메모가 저장되었습니다!")
                    st.rerun()

    st.divider()

    # --- 메모 보여주기 (Post-it 스타일) ---
    if not memos_df.empty:
        # 최신순 정렬 (선택 사항)
        memos_df = memos_df.sort_values(by='Date', ascending=False).reset_index(drop=True)
        
        # 3열 그리드로 배치
        cols = st.columns(3)
        
        for idx, row in memos_df.iterrows():
            with cols[idx % 3]:
                # 포스트잇 스타일 CSS 적용
                st.markdown(f"""
                <div style="
                    background-color: {row['Color']};
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 10px;
                    box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
                    color: black;
                ">
                    <small style="color: #555;">📅 {row['Date']}</small>
                    <h4 style="margin-top: 5px; margin-bottom: 10px; border-bottom: 1px solid rgba(0,0,0,0.1); padding-bottom:5px;">{row['Title']}</h4>
                    <div style="white-space: pre-wrap; font-family: sans-serif; font-size: 14px; line-height: 1.5;">{row['Content']}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # 삭제 버튼 (각 포스트잇 아래 배치)
                if st.button("🗑️ 삭제", key=f"del_memo_{idx}"):
                    memos_df = memos_df.drop(index=idx)
                    save_data_to_sheet(memos_df, 'memos')
                    st.rerun()
    else:
        st.info("작성된 메모가 없습니다. 위의 '새 메모 작성하기'를 눌러 첫 메모를 남겨보세요!")
