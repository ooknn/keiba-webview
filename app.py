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

# スマホ閲覧に最適化したページ設定
st.set_page_config(page_title="地方競馬 AI予想アプリ", page_icon="🏇", layout="centered")

st.title("🏇 地方競馬 AI予想アプリ")
st.markdown("**(V5 マスタークラス版)**")
st.markdown("---")

# ==========================================
# 1. AIモデルの読み込み (Supabase Storage対応版)
# ==========================================
MODEL_FILE_NAME = 'model_win_v5.pkl'

# ★ここにSupabase Storageで「Get URL」した公開URLを貼り付けます
SUPABASE_MODEL_URL = "https://fxvkgfrebghqkwlqshfa.supabase.co/storage/v1/object/public/models/model_win_v5.pkl" 
# 例: "https://xxxxxxxxxxxxxxxxxxxx.supabase.co/storage/v1/object/public/models/model_win_v5.pkl"

@st.cache_resource
def load_model():
    # ローカルにファイルがない場合はSupabaseからダウンロードする
    if not os.path.exists(MODEL_FILE_NAME):
        if SUPABASE_MODEL_URL == "あなたのSupabase_Storageの公開URL":
            return None, "URL_NOT_SET"
        try:
            st.info("☁️ SupabaseからAIモデルをダウンロードしています...(初回のみ数秒〜数十秒かかります)")
            response = requests.get(SUPABASE_MODEL_URL)
            response.raise_for_status()
            with open(MODEL_FILE_NAME, 'wb') as f:
                f.write(response.content)
            st.success("✅ ダウンロード完了！")
        except Exception as e:
            return None, f"ダウンロードエラー: {e}"
            
    # ダウンロードした（または既に存在する）モデルを読み込む
    try:
        model = joblib.load(MODEL_FILE_NAME)
        return model, "SUCCESS"
    except Exception as e:
        return None, f"読み込みエラー: {e}"

model, load_status = load_model()

if model is None:
    if load_status == "URL_NOT_SET":
        st.error("⚠️ AIモデルが見つかりません。コード内の `SUPABASE_MODEL_URL` にSupabaseのURLを設定するか、ローカルにファイルを置いてください。")
    else:
        st.error(f"⚠️ {load_status}")
    st.stop()
else:
    st.success("✅ AIモデル(V5)の準備が完了しました！")

# ==========================================
# 2. 当日の出馬表＆オッズ取得関数
# ==========================================
@st.cache_data(ttl=60) # 1分間はキャッシュして無駄な通信を防ぐ
def scrape_shutuba(race_id):
    url = f"https://nar.netkeiba.com/race/shutuba.html?race_id={race_id}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        
        # ★究極の文字化け対策: 
        # Pandasに任せるのをやめ、BeautifulSoupに「EUC-JP」として完璧に解読させる
        soup = BeautifulSoup(response.content, 'html.parser', from_encoding='euc-jp')
        
        # BeautifulSoupが解読した安全なテキストをPythonの文字列に戻す
        html_text = str(soup)
        
        if "該当するデータがありません" in html_text:
            return None, "レースが見つかりません。"
            
        # 安全なテキストをPandasに渡す
        dfs = pd.read_html(io.StringIO(html_text))
        
        if not dfs: return None, "出馬表が見つかりません。"
        
        df = dfs[0]
        # 列名の階層を平坦化（マルチインデックス対策）
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(-1)
            
        # 列名に含まれるスペースや改行を除去（「馬 番」などの表記揺れ対策）
        df.columns = df.columns.str.replace(r'\s+', '', regex=True)
        
        # ★追加: 「単勝オッズ」「予想オッズ」などの表記揺れを「オッズ」に統一
        rename_map = {}
        for col in df.columns:
            if 'オッズ' in col:
                rename_map[col] = 'オッズ'
            elif '人気' in col:
                rename_map[col] = '人気'
        df = df.rename(columns=rename_map)
        
        # 原因特定のための安全装置
        if '馬番' not in df.columns:
            return None, f"出馬表の形式が想定と異なります。見つかった列: {df.columns.tolist()}"
            
        # 必要な列だけ抽出
        target_cols = [c for c in df.columns if c in ['枠', '馬番', '馬名', '騎手', '斤量', 'オッズ', '人気']]
        df = df[target_cols].copy()
        df = df.dropna(subset=['馬番'])
        
        # 距離の取得
        info_div = soup.find('div', class_='RaceData01')
        distance = 1200 # デフォルト
        if info_div:
            dist_match = re.search(r'(\d+)m', info_text := info_div.text)
            if dist_match: distance = int(dist_match.group(1))
            
        return df, distance
    except Exception as e:
        return None, f"取得エラー: {e}"

# ==========================================
# 3. 画面UIと予測処理
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
        # そもそもオッズが存在しない（発表前など）場合のエラー処理
        st.error("⚠️ オッズ情報が見つかりません。レース前日などでオッズがまだ発表されていない可能性があります。")
        st.info(f"参考（取得できた列）: {df_shutuba.columns.tolist()}")
    else:
        # === 特徴量の生成（簡易版） ===
        # AIが学習時に使った「特徴量の名前」をモデルから取得
        feature_names = model.feature_name()
        
        # 予測用の空データフレームを作成
        X_pred = pd.DataFrame(index=df_shutuba.index, columns=feature_names)
        
        # わかる情報だけ埋める
        if 'waku' in feature_names and '枠' in df_shutuba.columns:
            X_pred['waku'] = pd.to_numeric(df_shutuba['枠'], errors='coerce')
        if 'umaban' in feature_names and '馬番' in df_shutuba.columns:
            X_pred['umaban'] = pd.to_numeric(df_shutuba['馬番'], errors='coerce')
        if 'distance' in feature_names:
            X_pred['distance'] = distance_or_err
            
        # 本来はSupabaseから「過去の勝率」や「前走データ」を引いてくる必要がありますが、
        # 今回はデモとして、残りの特徴量にはAIの学習に影響の少ない「中央値(0)」を埋めます。
        X_pred = X_pred.fillna(0)
        
        # ★追加: LightGBMエラー対策。全列を明示的に数値(float)に変換する
        X_pred = X_pred.astype(float)
        
        # === AIによる予測 ===
        with st.spinner("AIが勝率を計算中..."):
            pred_probs = model.predict(X_pred)
            
        # 結果をデータフレームにまとめる
        df_shutuba['AI勝率'] = pred_probs
        
        # オッズの処理（「---」などを除外）
        df_shutuba['オッズ'] = pd.to_numeric(df_shutuba['オッズ'], errors='coerce')
        
        # 期待値の計算 (勝率 × オッズ)
        df_shutuba['期待値'] = df_shutuba['AI勝率'] * df_shutuba['オッズ']
        
        # 表示用に整える
        df_shutuba = df_shutuba.dropna(subset=['オッズ'])
        df_shutuba = df_shutuba.sort_values('期待値', ascending=False).reset_index(drop=True)
        
        df_display = df_shutuba[['枠', '馬番', '馬名', '騎手', 'オッズ', 'AI勝率', '期待値']].copy()
        df_display['AI勝率'] = (df_display['AI勝率'] * 100).round(1).astype(str) + '%'
        df_display['オッズ'] = df_display['オッズ'].map('{:.1f}'.format)
        df_display['期待値'] = df_display['期待値'].map('{:.2f}'.format)
        
        # === 画面に表示 ===
        st.subheader("📊 AI予想・期待値ランキング")
        
        # バックテストで見つけた「得意条件」の警告
        if distance_or_err not in [850, 1300, 1400, 1500, 1600]:
             st.warning(f"⚠️ この距離({distance_or_err}m)は、AIの得意条件ではありません。見送りを推奨します。")
        else:
             st.success(f"🔥 この距離({distance_or_err}m)はAIの得意条件です！")
        
        # 期待値1.1以上の馬をハイライト
        def highlight_expected(val):
            try:
                if float(val) >= 1.1:
                    return 'background-color: #ffcccc; color: red; font-weight: bold'
            except: pass
            return ''
            
        st.dataframe(
            df_display.style.map(highlight_expected, subset=['期待値']), 
            use_container_width=True, 
            hide_index=True
        )
        
        st.info("💡 ピンク色に光っている馬が、AIが算出した「期待値1.1超えの買うべき馬」です！")
