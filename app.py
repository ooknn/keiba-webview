import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import requests
from bs4 import BeautifulSoup
import re
import os
import io
from supabase import create_client, Client

# スマホ閲覧に最適化したページ設定
st.set_page_config(page_title="地方競馬 AI予想アプリ", page_icon="🏇", layout="centered")

st.title("🏇 地方競馬 AI予想アプリ")
st.markdown("**(V5 Supabase直結・マスタークラス版)**")
st.markdown("---")

# ==========================================
# 1. 初期設定 (AIモデルとSupabase接続)
# ==========================================
MODEL_FILE_NAME = 'model_win_v5.pkl'

# ★ここにSupabase Storageで「Get URL」した公開URLを貼り付け直してください！
SUPABASE_MODEL_URL = "https://fxvkgfrebghqkwlqshfa.supabase.co/storage/v1/object/public/models/model_win_v5.pkl" 

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILE_NAME):
        if SUPABASE_MODEL_URL == "あなたのSupabase_Storageの公開URL": return None, "URL_NOT_SET"
        try:
            st.info("☁️ SupabaseからAIモデルをダウンロード中...")
            response = requests.get(SUPABASE_MODEL_URL)
            response.raise_for_status()
            with open(MODEL_FILE_NAME, 'wb') as f: f.write(response.content)
            st.success("✅ ダウンロード完了！")
        except Exception as e: return None, f"ダウンロードエラー: {e}"
    try: return joblib.load(MODEL_FILE_NAME), "SUCCESS"
    except Exception as e: return None, f"読み込みエラー: {e}"

model, load_status = load_model()
if model is None:
    st.error(f"⚠️ {load_status}")
    st.stop()
else:
    st.success("✅ AIモデル(V5) 準備完了！")

# 🔒 Secrets（秘密の金庫）からAPIキーを読み込んでSupabaseに接続
@st.cache_resource
def init_supabase():
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        return create_client(url, key)
    except Exception as e:
        st.error("⚠️ Supabaseの接続設定(Secrets)が見つかりません。")
        return None

supabase_client = init_supabase()

# ==========================================
# 2. 当日の出馬表取得関数
# ==========================================
@st.cache_data(ttl=60)
def scrape_shutuba(race_id):
    url = f"https://nar.netkeiba.com/race/shutuba.html?race_id={race_id}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser', from_encoding='euc-jp')
        html_text = str(soup)
        
        if "該当するデータがありません" in html_text: return None, "レースが見つかりません。"
            
        dfs = pd.read_html(io.StringIO(html_text))
        if not dfs: return None, "出馬表が見つかりません。"
        
        df = dfs[0]
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(-1)
            
        df.columns = df.columns.str.replace(r'\s+', '', regex=True)
        rename_map = {col: 'オッズ' for col in df.columns if 'オッズ' in col}
        rename_map.update({col: '人気' for col in df.columns if '人気' in col})
        df = df.rename(columns=rename_map)
        
        if '馬番' not in df.columns: return None, "出馬表の形式が想定と異なります。"
            
        # 必要な列だけ抽出 (厩舎=調教師、馬体重なども取得を試みる)
        target_cols = [c for c in df.columns if c in ['枠', '馬番', '馬名', '騎手', '厩舎', '斤量', '馬体重(増減)', 'オッズ', '人気']]
        df = df[target_cols].copy()
        df = df.dropna(subset=['馬番'])
        
        info_div = soup.find('div', class_='RaceData01')
        distance = 1200
        if info_div:
            dist_match = re.search(r'(\d+)m', info_div.text)
            if dist_match: distance = int(dist_match.group(1))
            
        return df, distance
    except Exception as e:
        return None, f"取得エラー: {e}"

# ==========================================
# 3. 🧠 超重要: Supabaseから過去データを引き出し特徴量を作る関数
# ==========================================
def enrich_features_from_supabase(df_shutuba, distance, supabase):
    horse_names = df_shutuba['馬名'].tolist()
    jockeys = df_shutuba['騎手'].tolist()
    
    prev_ranks, prev_speeds, is_first_runs = [], [], []
    jockey_win_rates, jockey_top3_rates = [], []
    
    if supabase is None:
        st.error("データベースに接続できないため、仮の数字で計算します。")
        return df_shutuba # 失敗時はそのまま返す

    with st.spinner("📡 Supabaseから各馬と騎手の過去データをリアルタイム解析中..."):
        # ① 馬の過去データ（前走着順・スピード）を取得
        try:
            res_h = supabase.table('race_results').select('horse_name, rank, distance, time_seconds').in_('horse_name', horse_names).order('race_id', desc=True).limit(2000).execute()
            df_hist = pd.DataFrame(res_h.data)
            
            for horse in horse_names:
                if not df_hist.empty and horse in df_hist['horse_name'].values:
                    h_df = df_hist[df_hist['horse_name'] == horse]
                    last_race = h_df.iloc[0] # 最新のレース（前走）
                    
                    r = pd.to_numeric(last_race['rank'], errors='coerce')
                    prev_ranks.append(r if pd.notna(r) else 99)
                    is_first_runs.append(0)
                    
                    # スピード指数の計算 (距離 ÷ 秒数)
                    dist = pd.to_numeric(last_race['distance'], errors='coerce')
                    t_sec = pd.to_numeric(last_race['time_seconds'], errors='coerce')
                    if pd.notna(dist) and pd.notna(t_sec) and t_sec > 0:
                        prev_speeds.append(dist / t_sec)
                    else:
                        prev_speeds.append(15.5) # 計算不能時の平均スピード
                else:
                    # 過去データが見つからない（初出走や中央からの移籍など）
                    prev_ranks.append(99)
                    is_first_runs.append(1)
                    prev_speeds.append(15.5)
        except:
            prev_ranks = [99] * len(horse_names)
            is_first_runs = [1] * len(horse_names)
            prev_speeds = [15.5] * len(horse_names)

        # ② 騎手の過去データ（勝率・複勝率）を取得
        try:
            res_j = supabase.table('race_results').select('jockey, rank').in_('jockey', jockeys).order('race_id', desc=True).limit(3000).execute()
            df_j_hist = pd.DataFrame(res_j.data)
            
            for j in jockeys:
                if not df_j_hist.empty and j in df_j_hist['jockey'].values:
                    j_df = df_j_hist[df_j_hist['jockey'] == j]
                    j_df['rank'] = pd.to_numeric(j_df['rank'], errors='coerce')
                    total = len(j_df)
                    wins = len(j_df[j_df['rank'] == 1])
                    top3s = len(j_df[j_df['rank'] <= 3])
                    
                    # ベイズ平滑化（少ない出走回数の極端な数値を均す）
                    weight = 20
                    overall_win, overall_top3 = 0.08, 0.23 # 地方競馬の平均的な勝率
                    smoothed_win = (wins + weight * overall_win) / (total + weight)
                    smoothed_top3 = (top3s + weight * overall_top3) / (total + weight)
                    
                    jockey_win_rates.append(smoothed_win)
                    jockey_top3_rates.append(smoothed_top3)
                else:
                    jockey_win_rates.append(0.08)
                    jockey_top3_rates.append(0.23)
        except:
            jockey_win_rates = [0.08] * len(jockeys)
            jockey_top3_rates = [0.23] * len(jockeys)
            
    # データフレームに結合
    df_shutuba['prev_rank'] = prev_ranks
    df_shutuba['is_first_run'] = is_first_runs
    df_shutuba['prev_speed'] = prev_speeds
    df_shutuba['jockey_win_rate'] = jockey_win_rates
    df_shutuba['jockey_top3_rate'] = jockey_top3_rates
    
    # ※調教師はAPI負荷軽減のため一旦平均値で埋める
    df_shutuba['trainer_win_rate'] = 0.08
    df_shutuba['trainer_top3_rate'] = 0.23
    
    return df_shutuba

# ==========================================
# 4. 画面UIと予測処理
# ==========================================
st.write("予想したいレースID（12桁）を入力してください。")
st.caption("例: 202444012211 (2024年 大井競馬 1回22日目 11R)")

race_id_input = st.text_input("レースIDを入力", "202444012211")

if st.button("AI予想を開始する", type="primary", use_container_width=True):
    with st.spinner("出馬表と最新オッズを取得中..."):
        df_shutuba, distance_or_err = scrape_shutuba(race_id_input)
        
    if df_shutuba is None:
        st.error(distance_or_err)
    elif 'オッズ' not in df_shutuba.columns:
        st.error("⚠️ オッズ情報が見つかりません。")
    else:
        # ★大進化：Supabaseから本物の過去データを取得して特徴量に追加！
        df_shutuba = enrich_features_from_supabase(df_shutuba, distance_or_err, supabase_client)
        
        # === 特徴量の生成（AIへ渡す準備） ===
        feature_names = model.feature_name()
        X_pred = pd.DataFrame(index=df_shutuba.index, columns=feature_names)
        
        # 取得できたデータをAIの入力形式にマッピング
        col_map = {
            'waku': '枠', 'umaban': '馬番', 'weight': '斤量', 
            'distance': distance_or_err, 
            'prev_rank': 'prev_rank', 'prev_speed': 'prev_speed', 'is_first_run': 'is_first_run',
            'jockey_win_rate': 'jockey_win_rate', 'jockey_top3_rate': 'jockey_top3_rate',
            'trainer_win_rate': 'trainer_win_rate', 'trainer_top3_rate': 'trainer_top3_rate'
        }
        
        for ai_col, df_col in col_map.items():
            if ai_col in feature_names:
                if isinstance(df_col, str) and df_col in df_shutuba.columns:
                    X_pred[ai_col] = pd.to_numeric(df_shutuba[df_col], errors='coerce')
                elif isinstance(df_col, int): # 距離など直接数値を入れる場合
                    X_pred[ai_col] = df_col
                    
        # 欠損値は0埋めし、全列をfloat型に統一（LightGBMエラー対策）
        X_pred = X_pred.fillna(0).astype(float)
        
        # === AIによる予測 ===
        with st.spinner("🧠 V5 AIが本物のデータで勝率を計算中..."):
            pred_probs = model.predict(X_pred)
            
        df_shutuba['AI勝率'] = pred_probs
        df_shutuba['オッズ'] = pd.to_numeric(df_shutuba['オッズ'], errors='coerce')
        df_shutuba['期待値'] = df_shutuba['AI勝率'] * df_shutuba['オッズ']
        
        df_shutuba = df_shutuba.dropna(subset=['オッズ'])
        df_shutuba = df_shutuba.sort_values('期待値', ascending=False).reset_index(drop=True)
        
        df_display = df_shutuba[['枠', '馬番', '馬名', '騎手', 'オッズ', 'AI勝率', '期待値']].copy()
        df_display['AI勝率'] = (df_display['AI勝率'] * 100).round(1).astype(str) + '%'
        df_display['オッズ'] = df_display['オッズ'].map('{:.1f}'.format)
        df_display['期待値'] = df_display['期待値'].map('{:.2f}'.format)
        
        # === 画面に表示 ===
        st.subheader("📊 AI予想・期待値ランキング")
        
        if distance_or_err in [850, 1600]:
             st.success(f"🔥 【激アツ】この距離({distance_or_err}m)は、AIの回収率が100%を超える大得意条件です！")
        else:
             st.warning(f"⚠️ この距離({distance_or_err}m)は、AIの得意条件(850m, 1600m)ではありません。見送りを推奨します。")
        
        def highlight_expected(val):
            try:
                if float(val) >= 1.0:
                    return 'background-color: #ffcccc; color: red; font-weight: bold'
            except: pass
            return ''
            
        st.dataframe(
            df_display.style.map(highlight_expected, subset=['期待値']), 
            use_container_width=True, 
            hide_index=True
        )
        
        st.info("💡 ピンク色に光っている馬が、AIが算出した「期待値1.0超えの買うべき馬」です！")
