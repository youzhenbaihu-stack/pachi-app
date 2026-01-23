import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract
import re
import os

# ==========================================
# ★設定エリア：ここだけ変えればOK★
# ==========================================
# アプリのロック解除パスワード（これをNoteの有料エリアに書く）
APP_PASSWORD = "777" 

# ==========================================
# ページ設定 (Wideモード)
# ==========================================
st.set_page_config(page_title="サイトセブン専用 回転率アナライザー", page_icon="🎰", layout="wide")

# ==========================================
# 🔐 ログイン認証システム
# ==========================================
def check_password():
    """パスワード認証のロジック"""
    def password_entered():
        if st.session_state["password"] == APP_PASSWORD:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # パスワードをセッションから消す（安全のため）
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # まだパスワードが入力されていない場合
        st.markdown("""
        <style>
        .stApp { background-color: #1a1a2e; color: white; }
        .stTextInput > div > div > input { color: black; }
        </style>
        <h1 style='text-align: center; color: #FFD700;'>🔒 PRO ANALYZER LOGIN</h1>
        <p style='text-align: center;'>このツールを利用するにはパスワードが必要です。<br>
        パスワードはNote記事の有料エリアに記載されています。</p>
        """, unsafe_allow_html=True)
        
        st.text_input("パスワードを入力", type="password", on_change=password_entered, key="password")
        return False
    
    elif not st.session_state["password_correct"]:
        # パスワードが間違っている場合
        st.markdown("""
        <style>.stApp { background-color: #1a1a2e; color: white; }</style>
        <h1 style='text-align: center; color: #FFD700;'>🔒 PRO ANALYZER LOGIN</h1>
        """, unsafe_allow_html=True)
        st.text_input("パスワードを入力", type="password", on_change=password_entered, key="password")
        st.error("パスワードが違います")
        return False
    
    else:
        # パスワード正解
        return True

# 認証チェック実行
if not check_password():
    st.stop()  # 認証されていない場合はここで処理を止める（下のアプリ画面を見せない）

# ==========================================
# 👇 ここから下がいつものアプリ本体 👇
# ==========================================

# ... (ここから下のコードは、前回の「決定版コード」と同じ中身が続きます)
# デザイン設定、関数定義、画面レイアウトなどをそのまま続けてください。
# 長くなるので、以下に「認証通過後の中身」として貼り付けるべきコードを記載します。

# ==========================================
# ★★★ デザイン設定 (Dark & Gold) ★★★
# ==========================================
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #ffffff;
    }
    h1 {
        color: #FFD700 !important;
        text-shadow: 0 0 10px #FFD700, 0 0 20px #ff00de;
        font-family: 'Helvetica Neue', sans-serif;
        text-align: center;
        padding-bottom: 20px;
        border-bottom: 2px solid #FFD700;
    }
    section[data-testid="stSidebar"] {
        background-color: #1a1a2e;
        border-right: 1px solid #FFD700;
    }
    .stNumberInput, .stFileUploader, .stSelectbox {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 10px;
        border: 1px solid rgba(255, 215, 0, 0.3);
    }
    .stButton > button {
        background: linear-gradient(90deg, #FFD700, #FDB931);
        color: black;
        font-weight: bold;
        border: none;
        border-radius: 30px;
        padding: 15px 30px;
        font-size: 20px;
        box-shadow: 0 0 15px rgba(255, 215, 0, 0.6);
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 25px rgba(255, 215, 0, 1);
        color: #fff;
        background: linear-gradient(90deg, #ff0000, #ff5e00);
    }
    .stMarkdown, p, label, .stInfo {
        color: #e0e0e0 !important;
    }
    .stSuccess {
        background-color: rgba(0, 255, 0, 0.1);
        border: 1px solid #00ff00;
        color: #00ff00;
    }
    .stAlert {
        background-color: rgba(255, 215, 0, 0.1);
        border: 1px solid #FFD700;
        color: #FFD700;
    }
    </style>
    """, unsafe_allow_html=True)

# サイドバー
st.sidebar.title("MENU")
mode = st.sidebar.radio("機種タイプを選択", ["① 時短なし (スマパチ・ST機)", "② 時短あり (エヴァ・海など)"])

if mode == "① 時短なし (スマパチ・ST機)":
    st.title("🎰 PRO ANALYZER (ST)")
else:
    st.title("🎰 PRO ANALYZER (JITAN)")

st.markdown("<p style='text-align: center;'>グラフと履歴をアップロードして、真の回転率を暴く。</p>", unsafe_allow_html=True)

# ---------------------------------------------------------
# 関数定義
# ---------------------------------------------------------
def extract_graph_area(img):
    """ベージュ領域の自動切り抜き"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    height, width = img.shape[:2]
    lower_bg = np.array([0, 5, 200])
    upper_bg = np.array([40, 60, 255])
    mask_bg = cv2.inRange(hsv, lower_bg, upper_bg)
    kernel = np.ones((5,5), np.uint8)
    mask_bg = cv2.morphologyEx(mask_bg, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask_bg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_cnt)
        image_area = width * height
        rect_area = w * h
        if rect_area > (image_area * 0.8):
            return img, (0, 0, width, height)
        else:
            return img[y:y+h, x:x+w], (x, y, w, h)
    return img, (0, 0, width, height)

def analyze_graph_final(img):
    """グラフ解析（スケール70000発・5色対応）"""
    cropped_img, rect = extract_graph_area(img)
    hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
    height, width = cropped_img.shape[:2]
    balls_per_pixel = 70000 / height 
    gx, gy, gw, gh = 0, 0, width, height 

    # 0ライン検出
    mid_start = int(height * 0.3)
    mid_end = int(height * 0.7)
    roi_mid = cropped_img[mid_start:mid_end, :]
    gray_mid = cv2.cvtColor(roi_mid, cv2.COLOR_BGR2GRAY)
    sobel_y = cv2.Sobel(gray_mid, cv2.CV_8U, 0, 1, ksize=3)
    _, binary_line = cv2.threshold(sobel_y, 50, 255, cv2.THRESH_BINARY)
    line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width // 5, 1))
    detected_lines = cv2.morphologyEx(binary_line, cv2.MORPH_OPEN, line_kernel)
    contours_line, _ = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    zero_line_y = 0
    if contours_line:
        c = max(contours_line, key=lambda c: cv2.boundingRect(c)[2])
        lx, ly, lw, lh = cv2.boundingRect(c)
        zero_line_y = mid_start + ly + (lh // 2)
    else:
        zero_line_y = height // 2
    
    # グラフ線検出
    hsv_roi = hsv 
    mask_green = cv2.inRange(hsv_roi, np.array([30, 40, 40]), np.array([90, 255, 255]))
    mask_purple = cv2.inRange(hsv_roi, np.array([120, 40, 40]), np.array([165, 255, 255]))
    mask_orange1 = cv2.inRange(hsv_roi, np.array([0, 100, 100]), np.array([25, 255, 255]))
    mask_orange2 = cv2.inRange(hsv_roi, np.array([150, 100, 100]), np.array([180, 255, 255]))
    mask_cyan = cv2.inRange(hsv_roi, np.array([80, 40, 40]), np.array([100, 255, 255]))
    mask_line = cv2.bitwise_or(mask_green, mask_purple)
    mask_line = cv2.bitwise_or(mask_line, mask_orange1)
    mask_line = cv2.bitwise_or(mask_line, mask_orange2)
    mask_line = cv2.bitwise_or(mask_line, mask_cyan)
    
    contours_line_graph, _ = cv2.findContours(mask_line, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours_line_graph: return None, "グラフ線が見つかりませんでした"

    all_points = []
    for cnt in contours_line_graph:
        for p in cnt: all_points.append(p[0])
    if not all_points: return None, "線データなし"

    all_points.sort(key=lambda p: p[0])
    end_point_local = all_points[-1]
    end_point_y = end_point_local[1]
    diff_pixels = zero_line_y - end_point_y
    est_diff_balls = diff_pixels * balls_per_pixel
    return int(est_diff_balls), cropped_img

def sum_red_start_counts(img):
    """OCR集計"""
    height, width = img.shape[:2]
    roi_width = int(width * 0.35) 
    roi = img[:, width - roi_width : width]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    kernel = np.ones((2,2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
    mask_inverted = cv2.bitwise_not(mask)
    config = r'--oem 3 --psm 6 outputbase digits'
    text = pytesseract.image_to_string(mask_inverted, config=config)
    numbers = re.findall(r'\d+', text)
    numbers = [int(n) for n in numbers]
    return sum(numbers), numbers

# ---------------------------------------------------------
# メイン画面レイアウト
# ---------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📸 画像解析エリア")
    st.markdown("---")
    st.info("💡 **Hint**: 余白が多い画像は、自動でグラフ部分だけ切り抜いて解析します。")

    uploaded_graph = st.file_uploader("① グラフ画像をアップロード", type=['jpg', 'png', 'jpeg'], key="graph")
    diff_balls = 0

    if uploaded_graph is not None:
        file_bytes = np.asarray(bytearray(uploaded_graph.read()), dtype=np.uint8)
        img_graph = cv2.imdecode(file_bytes, 1)
        result, msg_or_img = analyze_graph_final(img_graph)
        
        if result is not None:
            diff_balls = result
            st.image(cv2.cvtColor(msg_or_img, cv2.COLOR_BGR2RGB), caption=f"解析範囲", use_column_width=True)
            st.success(f"推定差玉: {diff_balls} 発")
        else:
            st.error(f"エラー: {msg_or_img}")

    st.markdown("<br>", unsafe_allow_html=True)

    uploaded_histories = st.file_uploader(
        "② 履歴画像（赤数字）をアップロード (複数枚可)", 
        type=['jpg', 'png', 'jpeg'], 
        accept_multiple_files=True,
        key="history"
    )
    
    st_spins_auto = 0
    all_st_details = []

    if uploaded_histories:
        for uploaded_file in uploaded_histories:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img_hist = cv2.imdecode(file_bytes, 1)
            st_sum, num_list = sum_red_start_counts(img_hist)
            st_spins_auto += st_sum
            all_st_details.extend(num_list)
        
        st.info(f"検出: {all_st_details}")
        st.success(f"★ 合計ST回転数: {st_spins_auto} 回転")

with col2:
    st.markdown("### 🔢 データ入力エリア")
    st.markdown("---")
    
    # 1. 基本データ入力
    total_spins = st.number_input("現在の総回転数", min_value=0, value=3000, step=1)
    st_spins_final = st.number_input("ラッシュ(ST)の回転数", min_value=0, value=st_spins_auto, step=1)
    
    jitan_spins = 0
    if mode == "② 時短あり (エヴァ・海など)":
        st.warning("⚠️ 時短モードON")
        jitan_spins = st.number_input("時短中に回した回転数", min_value=0, value=0, step=1)

    # 2. 当たりデータ入力（新ロジック）
    st.markdown("#### ▼ 当たりデータ (データ機通りに入力)")
    
    c_data1, c_data2 = st.columns(2)
    with c_data1:
        total_hits = st.number_input("総当たり回数", min_value=0, value=0)
    with c_data2:
        first_hits = st.number_input("初当たり回数", min_value=0, value=0)
        
    # 自動計算：ST中当たり回数
    st_hits = total_hits - first_hits
    if st_hits < 0: st_hits = 0
    st.info(f"📊 計算上のST中当たり回数: **{st_hits} 回**")

    st.markdown("#### ▼ 出玉詳細設定")
    
    # ST中の出玉設定
    st_payout = st.number_input("ST中の平均出玉 (基本1500)", value=1500, step=10)

    # 初当たりの内訳設定
    c_fail1, c_fail2 = st.columns(2)
    with c_fail1:
        # 負けた時の出玉（選択式）
        fail_payout = st.selectbox("通常(ST落ち)の出玉", [1500, 1200, 1050, 450, 300], index=4)
    with c_fail2:
        # 負けた回数
        fail_count = st.number_input("通常(ST落ち)の回数", min_value=0, max_value=first_hits, value=0)
    
    # RUSH突入回数（自動）
    rush_entry_count = first_hits - fail_count
    # RUSH突入時の出玉（基本1500だが、機種によっては300などあるので変更可能に）
    rush_entry_payout = st.number_input("RUSH突入時の出玉 (基本1500)", value=1500, step=10)

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🔥 解析開始 (ANALYZE) 🔥", type="primary"):
        real_spins = total_spins - st_spins_final - jitan_spins
        
        # ★計算ロジック
        # 1. ST中出玉 = (総当たり - 初当たり) * ST平均出玉
        income_st = st_hits * st_payout
        
        # 2. 初当たり出玉
        # A. 通常(ST落ち) = 回数 * 選択した出玉
        income_fail = fail_count * fail_payout
        # B. RUSH突入 = (初当たり - 落ちた回数) * 突入出玉
        income_entry = rush_entry_count * rush_entry_payout
        
        total_payout = income_st + income_fail + income_entry
        used_balls = total_payout - diff_balls
        
        st.markdown(f"""
        <div style="background-color: rgba(0,0,0,0.5); padding: 20px; border-radius: 10px; border: 2px solid #FFD700; text-align: center;">
            <h3 style="color: #FFD700; margin-bottom: 0;">RESULT</h3>
            <p style="color: #ccc;">実質通常回転数: {real_spins} 回転</p>
            <p style="color: #ccc;">推定投資: {int(used_balls):,}発 ({int(used_balls)*4:,}円)</p>
        </div>
        """, unsafe_allow_html=True)
        
        if used_balls > 0:
            rate = (real_spins / used_balls) * 250
            st.markdown(f"""
            <div style="text-align: center; margin-top: 20px;">
                <p style="font-size: 1.5em; color: white;">1000円あたりの回転数</p>
                <h1 style="font-size: 5em; color: #00ff00; text-shadow: 0 0 20px #00ff00; margin: 0;">{rate:.2f}</h1>
                <p style="font-size: 1.5em; color: white;">回転</p>
            </div>
            """, unsafe_allow_html=True)
            if rate >= 20:
                st.balloons()
                st.markdown("<h2 style='color: gold; text-align: center;'>🏆 優秀台 (Excellent) 🏆</h2>", unsafe_allow_html=True)
            elif rate <= 15:
                st.markdown("<h2 style='color: red; text-align: center;'>💀 回収台 (Danger) 💀</h2>", unsafe_allow_html=True)
            else:
                st.markdown("<h2 style='color: orange; text-align: center;'>⚠️ ボーダー付近 (Average) ⚠️</h2>", unsafe_allow_html=True)
        else:
            st.error("計算エラー：投資がマイナスです。")
