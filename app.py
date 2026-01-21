import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract
import re

# ページ設定 (Wideモード)
st.set_page_config(page_title="パチンコ回転率アナライザー", page_icon="🎰", layout="wide")

# ==========================================
# ★★★ デザイン設定 (Dark & Gold) ★★★
# ==========================================
st.markdown("""
    <style>
    /* 全体の背景をダークモード風グラデーションに */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #ffffff;
    }
    
    /* タイトルデザイン */
    h1 {
        color: #FFD700 !important;
        text-shadow: 0 0 10px #FFD700, 0 0 20px #ff00de;
        font-family: 'Helvetica Neue', sans-serif;
        text-align: center;
        padding-bottom: 20px;
        border-bottom: 2px solid #FFD700;
    }
    
    /* サイドバー */
    section[data-testid="stSidebar"] {
        background-color: #1a1a2e;
        border-right: 1px solid #FFD700;
    }
    
    /* 入力フォーム */
    .stNumberInput, .stFileUploader {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 10px;
        border: 1px solid rgba(255, 215, 0, 0.3);
    }
    
    /* ボタンデザイン */
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
    
    /* 文字色調整 */
    .stMarkdown, p, label, .stInfo {
        color: #e0e0e0 !important;
    }
    
    /* 成功メッセージ */
    .stSuccess {
        background-color: rgba(0, 255, 0, 0.1);
        border: 1px solid #00ff00;
        color: #00ff00;
    }
    
    /* 注意書き（info）のデザイン */
    .stAlert {
        background-color: rgba(255, 215, 0, 0.1);
        border: 1px solid #FFD700;
        color: #FFD700;
    }
    </style>
    """, unsafe_allow_html=True)

# サイドバーでモード選択
st.sidebar.title("MENU")
mode = st.sidebar.radio("機種タイプを選択", ["① 時短なし (スマパチ・ST機)", "② 時短あり (エヴァ・海など)"])

# タイトル表示
if mode == "① 時短なし (スマパチ・ST機)":
    st.title("🎰 PRO ANALYZER (ST)")
else:
    st.title("🎰 PRO ANALYZER (JITAN)")

st.markdown("<p style='text-align: center;'>グラフと履歴をアップロードして、真の回転率を暴く。</p>", unsafe_allow_html=True)

# ---------------------------------------------------------
# 関数定義
# ---------------------------------------------------------
def extract_graph_area(img):
    """
    画像の中からベージュ色のグラフ領域だけを特定して切り抜く関数。
    すでに切り抜かれている場合は、そのままの画像を返す。
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    height, width = img.shape[:2]
    
    # ベージュ色（グラフ背景）の定義
    lower_bg = np.array([0, 5, 200])
    upper_bg = np.array([40, 60, 255])
    mask_bg = cv2.inRange(hsv, lower_bg, upper_bg)
    
    # ノイズ除去
    kernel = np.ones((5,5), np.uint8)
    mask_bg = cv2.morphologyEx(mask_bg, cv2.MORPH_CLOSE, kernel)
    
    # 輪郭検出
    contours, _ = cv2.findContours(mask_bg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # 一番大きなベージュ領域を探す
        largest_cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_cnt)
        
        # 判定ロジック:
        # もし検出されたベージュ領域が、画像の面積の80%以上を占めているなら
        # 「すでにトリミング済みの画像」と判断して、元の画像をそのまま使う。
        # 逆に、もっと小さければ「スクショ全体画像」と判断して、その部分だけ切り抜く。
        
        image_area = width * height
        rect_area = w * h
        
        if rect_area > (image_area * 0.8):
            # すでにほぼ全体がグラフなので、そのまま返す
            return img, (0, 0, width, height)
        else:
            # 周りに黒い余白があるので、切り抜く
            cropped_img = img[y:y+h, x:x+w]
            return cropped_img, (x, y, w, h)
            
    # ベージュが見つからない場合は、とりあえずそのまま返す
    return img, (0, 0, width, height)

def analyze_graph_final(img):
    """グラフ解析（自動トリミング・0.027固定・5色対応）"""
    
    # ★ステップ1：まずはグラフ領域だけを綺麗に抽出する
    cropped_img, rect = extract_graph_area(img)
    
    # ここから先は「cropped_img（グラフ部分だけ）」に対して処理を行う
    hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
    height, width = cropped_img.shape[:2]

    # --- 基準設定 ---
    # グラフ領域の高さそのものを使って計算する
    # ※66000という定数は、グラフ画像の縦幅が6万発分のスケールであるという前提
    balls_per_pixel = 66000 / height 
    
    gx, gy, gw, gh = 0, 0, width, height # 切り抜き済みなので全体を使う

    # --- 0ライン検出 ---
    # 中央付近を探す
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
        # 見つからない場合は画像のちょうど真ん中とする
        zero_line_y = height // 2

    # 0ライン補正（0.027固定）
    correction_y = int(height * 0.027) 
    zero_line_y -= correction_y

    # --- グラフ線検出 ---
    # 画像全体から線を探す
    hsv_roi = hsv # すでに切り抜き済みなので全体
    
    # 色の定義
    mask_green = cv2.inRange(hsv_roi, np.array([30, 40, 40]), np.array([90, 255, 255]))
    mask_purple = cv2.inRange(hsv_roi, np.array([120, 40, 40]), np.array([165, 255, 255]))
    mask_orange1 = cv2.inRange(hsv_roi, np.array([0, 100, 100]), np.array([25, 255, 255]))
    mask_orange2 = cv2.inRange(hsv_roi, np.array([150, 100, 100]), np.array([180, 255, 255]))
    mask_cyan = cv2.inRange(hsv_roi, np.array([80, 40, 40]), np.array([100, 255, 255])) # 水色

    # 全ての色を合体
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

    # 一番右の点（最新の差玉）を探す
    all_points.sort(key=lambda p: p[0])
    end_point_local = all_points[-1]
    
    # 差分ピクセル
    end_point_y = end_point_local[1]
    diff_pixels = zero_line_y - end_point_y
    
    # 差玉計算
    est_diff_balls = diff_pixels * balls_per_pixel

    # 結果として返す画像は、解析に使った「切り抜き画像」を返す
    return int(est_diff_balls), cropped_img

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
    
    config = r'--oem 3 --psm 6 outputbase digits'
    text = pytesseract.image_to_string(mask_inverted, config=config)
    numbers = re.findall(r'\d+', text)
    numbers = [int(n) for n in numbers]
    
    return sum(numbers), numbers

# ---------------------------------------------------------
# メイン画面レイアウト
# ---------------------------------------------------------
col1, col2 = st.columns(2)

# === 左側：画像解析エリア ===
with col1:
    st.markdown("### 📸 画像解析エリア")
    st.markdown("---")
    
    # 注意事項
    st.info("💡 **Hint**: 添付する画像はなるべく **余白の部分をカット（トリミング）** して添付してください。解析精度が向上します。")

    # 1. グラフ画像
    uploaded_graph = st.file_uploader("① グラフ画像をアップロード", type=['jpg', 'png', 'jpeg'], key="graph")
    diff_balls = 0

    if uploaded_graph is not None:
        file_bytes = np.asarray(bytearray(uploaded_graph.read()), dtype=np.uint8)
        img_graph = cv2.imdecode(file_bytes, 1)
        
        # 解析実行（自動トリミング機能付き）
        result, msg_or_img = analyze_graph_final(img_graph)
        
        if result is not None:
            diff_balls = result
            # 切り抜かれた後の画像を表示
            st.image(cv2.cvtColor(msg_or_img, cv2.COLOR_BGR2RGB), caption=f"解析範囲", use_column_width=True)
            st.success(f"推定差玉: {diff_balls} 発")
        else:
            st.error(f"エラー: {msg_or_img}")

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. 履歴画像
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
            
            # 各画像の赤数字を集計
            st_sum, num_list = sum_red_start_counts(img_hist)
            st_spins_auto += st_sum
            all_st_details.extend(num_list)
        
        st.info(f"検出された赤数字 (全{len(all_st_details)}件): {all_st_details}")
        st.success(f"★ 合計ST回転数: {st_spins_auto} 回転")

# === 右側：計算入力エリア ===
with col2:
    st.markdown("### 🔢 データ入力エリア")
    st.markdown("---")

    total_spins = st.number_input("現在の総回転数", min_value=0, value=3000, step=1)
    # 自動集計された合計値が初期値に入る
    st_spins_final = st.number_input("ラッシュ(ST)の回転数", min_value=0, value=st_spins_auto, step=1)

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
        
        # 結果表示
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
