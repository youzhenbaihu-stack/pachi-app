import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract
import re

# ページ設定 (Wideモードで広々と使う)
st.set_page_config(page_title="パチンコ回転率アナライザー", page_icon="🎰", layout="wide")

# ==========================================
# ★★★ デザイン変更エリア（ここが魔法） ★★★
# ==========================================
st.markdown("""
    <style>
    /* 全体の背景をダークモード風グラデーションに */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #ffffff;
    }
    
    /* タイトルを金色に光らせる */
    h1 {
        color: #FFD700 !important;
        text-shadow: 0 0 10px #FFD700, 0 0 20px #ff00de;
        font-family: 'Helvetica Neue', sans-serif;
        text-align: center;
        padding-bottom: 20px;
        border-bottom: 2px solid #FFD700;
    }
    
    /* サイドバーのデザイン */
    section[data-testid="stSidebar"] {
        background-color: #1a1a2e;
        border-right: 1px solid #FFD700;
    }
    
    /* 入力フォームの背景を半透明に */
    .stNumberInput, .stFileUploader {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 10px;
        border: 1px solid rgba(255, 215, 0, 0.3);
    }
    
    /* ボタンを「激アツ」風にする */
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
        background: linear-gradient(90deg, #ff0000, #ff5e00); /* ホバー時は赤く */
    }
    
    /* 文字の色を読みやすく白系に調整 */
    .stMarkdown, p, label {
        color: #e0e0e0 !important;
    }
    
    /* 成功メッセージ（緑）をネオン風に */
    .stSuccess {
        background-color: rgba(0, 255, 0, 0.1);
        border: 1px solid #00ff00;
        color: #00ff00;
    }
    </style>
    """, unsafe_allow_html=True)

# サイドバーでモード選択
st.sidebar.title("MENU")
mode = st.sidebar.radio("機種タイプを選択", ["① 時短なし (スマパチ・ST機)", "② 時短あり (エヴァ・海など)"])

# メインタイトル
if mode == "① 時短なし (スマパチ・ST機)":
    st.title("🎰 PRO ANALYZER (ST)")
else:
    st.title("🎰 PRO ANALYZER (JITAN)")

st.markdown("<p style='text-align: center;'>グラフと履歴をアップロードして、真の回転率を暴く。</p>", unsafe_allow_html=True)

# ---------------------------------------------------------
# 関数定義
# ---------------------------------------------------------
def analyze_graph_final(img):
    """グラフの画像を解析して差玉を算出する（0.027固定・線描画なし）"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    height, width = img.shape[:2]

    # 1. 枠検出
    lower_bg = np.array([0, 5, 200])
    upper_bg = np.array([40, 60, 255])
    mask_bg = cv2.inRange(hsv, lower_bg, upper_bg)
    kernel = np.ones((5,5), np.uint8)
    mask_bg = cv2.morphologyEx(mask_bg, cv2.MORPH_CLOSE, kernel)
    
    contours_bg, _ = cv2.findContours(mask_bg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    graph_rect = None
    if contours_bg:
        sorted_cnts = sorted(contours_bg, key=cv2.contourArea, reverse=True)
        for cnt in sorted_cnts:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > width * 0.5 and h > height * 0.2:
                graph_rect = (x, y, w, h)
                break
    
    if graph_rect is None: return None, "グラフ枠が見つかりませんでした"

    gx, gy, gw, gh = graph_rect
    balls_per_pixel = 66000 / gh 

    # 2. 0ライン検出
    mid_start = gy + int(gh * 0.3)
    mid_end = gy + int(gh * 0.7)
    roi_mid = img[mid_start:mid_end, gx:gx+gw]
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
        zero_line_y = gy + (gh // 2)

    # ★0ライン補正：0.027固定
    correction_y = int(gh * 0.027) 
    zero_line_y -= correction_y

    # 3. グラフ線検出
    roi = img[gy:gy+gh, gx:gx+gw]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    mask_green = cv2.inRange(hsv_roi, np.array([30, 40, 40]), np.array([90, 255, 255]))
    mask_purple = cv2.inRange(hsv_roi, np.array([120, 40, 40]), np.array([165, 255, 255]))
    mask_orange1 = cv2.inRange(hsv_roi, np.array([0, 100, 100]), np.array([25, 255, 255]))
    mask_orange2 = cv2.inRange(hsv_roi, np.array([150, 100, 100]), np.array([180, 255, 255]))

    mask_line = cv2.bitwise_or(mask_green, mask_purple)
    mask_line = cv2.bitwise_or(mask_line, mask_orange1)
    mask_line = cv2.bitwise_or(mask_line, mask_orange2)
    
    contours_line_graph, _ = cv2.findContours(mask_line, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours_line_graph: return None, "グラフ線が見つかりませんでした"

    all_points = []
    for cnt in contours_line_graph:
        for p in cnt: all_points.append(p[0])
    if not all_points: return None, "線データなし"

    all_points.sort(key=lambda p: p[0])
    end_point_local = all_points[-1]
    end_point_y = gy + end_point_local[1]

    diff_pixels = zero_line_y - end_point_y
    est_diff_balls = diff_pixels * balls_per_pixel

    return int(est_diff_balls), img

def sum_red_start_counts(img):
    """履歴画像の赤文字をOCRで集計する"""
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
    
    # OCR実行
    config = r'--oem 3 --psm 6 outputbase digits'
    text = pytesseract.image_to_string(mask_inverted, config=config)
    numbers = re.findall(r'\d+', text)
    numbers = [int(n) for n in numbers]
    
    return sum(numbers), numbers

# ---------------------------------------------------------
# メイン画面処理
# ---------------------------------------------------------

# カラムを使ってレイアウトを整える
col1, col2 = st.columns(2)

# 左側：画像アップロードエリア
with col1:
    st.markdown("### 📸 画像解析エリア")
    st.markdown("---")
    
    uploaded_graph = st.file_uploader("① グラフ画像をアップロード", type=['jpg', 'png', 'jpeg'], key="graph")
    diff_balls = 0

    if uploaded_graph is not None:
        file_bytes = np.asarray(bytearray(uploaded_graph.read()), dtype=np.uint8)
        img_graph = cv2.imdecode(file_bytes, 1)
        result, msg_or_img = analyze_graph_final(img_graph)
        
        if result is not None:
            diff_balls = result
            st.image(cv2.cvtColor(msg_or_img, cv2.COLOR_BGR2RGB), caption=f"解析完了", use_column_width=True)
            st.success(f"推定差玉: {diff_balls} 発")
        else:
            st.error(f"エラー: {msg_or_img}")

    st.markdown("<br>", unsafe_allow_html=True) # 余白

    uploaded_history = st.file_uploader("② 履歴画像（赤数字）をアップロード", type=['jpg', 'png', 'jpeg'], key="history")
    st_spins_auto = 0
    st_details = []

    if uploaded_history is not None:
        file_bytes = np.asarray(bytearray(uploaded_history.read()), dtype=np.uint8)
        img_hist = cv2.imdecode(file_bytes, 1)
        st_sum, num_list = sum_red_start_counts(img_hist)
        st_spins_auto = st_sum
        st_details = num_list
        st.info(f"検出: {st_details}")
        st.success(f"ST回転数: {st_spins_auto} 回転")

# 右側：計算入力エリア
with col2:
    st.markdown("### 🔢 データ入力エリア")
    st.markdown("---")

    total_spins = st.number_input("現在の総回転数", min_value=0, value=3000, step=1)
    st_spins_final = st.number_input("ラッシュ(ST)の回転数", min_value=0, value=st_spins_auto, step=1)

    # 時短モードのみ表示
    jitan_spins = 0
    if mode == "② 時短あり (エヴァ・海など)":
        st.warning("⚠️ 時短モードON")
        jitan_spins = st.number_input("時短中に回した回転数", min_value=0, value=0, step=1)

    st.markdown("#### ▼ 当たり内訳")
    c_sub1, c_sub2 = st.columns(2)
    with c_sub1:
        count_3000 = st.number_input("上位(3000発) 回数", min_value=0, value=0)
        payout_3000 = st.number_input("上位 出玉/回", value=2800)
    with c_sub2:
        count_1500 = st.number_input("通常(1500発) 回数", min_value=0, value=0)
        payout_1500 = st.number_input("通常 出玉/回", value=1400)

    c_sub3, c_sub4 = st.columns(2)
    with c_sub3:
        count_300 = st.number_input("チャージ(300発) 回数", min_value=0, value=0)
        payout_300 = st.number_input("チャージ 出玉/回", value=280)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 計算ボタン
    if st.button("🔥 解析開始 (ANALYZE) 🔥", type="primary"):
        real_spins = total_spins - st_spins_final - jitan_spins
        total_payout = (count_3000 * payout_3000) + (count_1500 * payout_1500) + (count_300 * payout_300)
        used_balls = total_payout - diff_balls
        
        # 結果表示エリア（カード風に）
        st.markdown(f"""
        <div style="background-color: rgba(0,0,0,0.5); padding: 20px; border-radius: 10px; border: 2px solid #FFD700; text-align: center;">
            <h3 style="color: #FFD700; margin-bottom: 0;">RESULT</h3>
            <p style="color: #ccc;">実質通常回転数: {real_spins} 回転</p>
            <p style="color: #ccc;">推定投資: {int(used_balls):,}発 ({int(used_balls)*4:,}円)</p>
        </div>
        """, unsafe_allow_html=True)
        
        if used_balls > 0:
            rate = (real_spins / used_balls) * 250
            
            # 回転率をデカデカと表示
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
