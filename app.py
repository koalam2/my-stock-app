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

def color_negative_red(val):
    if val > 0:
        return 'color: blue' 
    elif val < 0:
        return 'color: red' 
    else:
        return 'color: black'

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

# [데이터 및 설정 로드] 구글 시트 사용
df = load_data_from_sheet('transactions')
app_config = load_config()
saved_goal1 = int(app_config.get('goal1', 100000000))
saved_goal2 = int(app_config.get('goal2', 1000000000))
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


# ---------------------------------------------------------
# 3. 화면별 로직 구현
# ---------------------------------------------------------

if menu == "1. 총 자산 확인":
    st.title("💰 총 자산 현황")
    
    # --- 자산 목표 설정 및 현황 (대시보드 상단 배치) ---
    st.markdown("### 🚀 자산 목표 달성률")
    col_g1, col_g2 = st.columns(2)
    
    prog1 = min(current_total_asset_krw / saved_goal1, 1.0) if saved_goal1 > 0 else 0
    with col_g1:
        st.caption(f"🥇 1차 목표: {saved_goal1:,.0f}원 (달성률: {prog1*100:.1f}%)")
        st.progress(prog1)

    prog2 = min(current_total_asset_krw / saved_goal2, 1.0) if saved_goal2 > 0 else 0
    with col_g2:
        st.caption(f"🥈 2차 목표: {saved_goal2:,.0f}원 (달성률: {prog2*100:.1f}%)")
        st.progress(prog2)

    with st.expander("🎯 목표 금액 수정", expanded=False):
        c1, c2 = st.columns(2)
        goal1_target = c1.number_input("1차 목표 (원)", value=saved_goal1, step=10_000_000, format="%d", key="g1_input")
        goal2_target = c2.number_input("2차 목표 (원)", value=saved_goal2, step=100_000_000, format="%d", key="g2_input")
        
        if st.button("목표 저장"):
            save_config(goal1_target, goal2_target)
            st.success("목표 금액이 성공적으로 저장되었습니다!")
            st.rerun()
            
    st.markdown("---")
    
    # --- 기존 총 자산 확인 로직 ---
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
        
        def prepare_pie_data(df, group_col, value_col, threshold=0.01):
