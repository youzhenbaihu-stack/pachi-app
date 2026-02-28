import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract
import re
import os

# ==========================================
# ★設定エリア★
# ==========================================
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
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
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
        st.markdown("""
        <style>.stApp { background-color: #1a1a2e; color: white; }</style>
        <h1 style='text-align: center; color: #FFD700;'>🔒 PRO ANALYZER LOGIN</h1>
        """, unsafe_allow_html=True)
        st.text_input("パスワードを入力", type="password", on_change=password_entered, key="password")
        st.error("パスワードが違います")
        return False
    else:
        return True

if not check_password():
    st.stop()

# ==========================================
# ★★★ デザイン設定 (Dark & Gold) ★★★
# ==========================================
st.markdown("""
    <style>
    /* 全体の背景 */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #ffffff;
    }
    /* タイトル */
    h1 {
        color: #FFD700 !important;
        text-shadow: 0 0 10px #FFD700, 0 0 20px #ff00de;
        font-family: 'Helvetica Neue', sans-serif;
        text-align: center;
        padding-bottom: 20px;
        border-bottom: 2px solid #FFD700;
    }
    /* サイドバー背景 */
    section[data-testid="stSidebar"] {
        background-color: #1a1a2e;
        border-right: 1px solid #FFD700;
    }
    /* 入力フォーム */
    .stNumberInput, .stFileUploader, .stSelectbox {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 10px;
        border: 1px solid rgba(255, 215, 0, 0.3);
    }
    /* ボタン */
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
    /* 文字色 */
    .stMarkdown, p, label, .stInfo, .stCaption {
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
    /* ガイド部分 */
    .streamlit-expanderHeader {
        background-color: #302b63;
        color: #FFD700;
        font-weight: bold;
    }
    
    /* メニュー非表示設定 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# サイドバー
st.sidebar.title("MENU")
mode = st.sidebar.radio("機種タイプを選択", ["① 時短なし (スマパチ・ST機)", "② 時短あり (エヴァ・海など)"])

# タイトル
if mode == "① 時短なし (スマパチ・ST機)":
    st.title("🎰 SITE7 PRO ANALYZER (ST)")
else:
    st.title("🎰 SITE7 PRO ANALYZER (JITAN)")

# ==========================================
# ★★★ 使い方ガイド & サンプル画像 ★★★
# ==========================================
with st.expander("🔰 初めての方へ：使い方と画像の例 (クリックで開く)", expanded=True):
    st.markdown("""
    ### 📝 利用手順
    1. **サイトセブン** で台データの詳細を開きます。
    2. **「スランプグラフ」** と **「大当たり履歴」** のスクリーンショットを撮ります。
       - ※グラフは余白があってもOKですが、なるべくグラフ部分を大きく撮ると精度が上がります。
    3. 下のアップロードエリアに画像をセットします。
    4. 右側のエリアに、データランプの **「総回転数」「大当たり回数」「初当り回数」** を入力します。
    5. **「解析開始」** ボタンを押すと、真の回転率（千円スタート）が表示されます。
    """)
    
    st.markdown("---")
    st.markdown("### 📸 推奨画像サンプル")
    st.caption("以下のような画像を保存してアップロードしてください。")
    
    col_sample1, col_sample2 = st.columns(2)
    
    with col_sample1:
        if os.path.exists("sample_graph.png"):
            st.image("sample_graph.png", caption="【推奨】スランプグラフ", use_column_width=True)
        else:
            st.info("ここにグラフ画像の見本が表示されます (sample_graph.png をアップロードしてください)")
            
    with col_sample2:
        if os.path.exists("sample_history.png"):
            st.image("sample_history.png", caption="【推奨】履歴リスト (赤数字)", use_column_width=True)
        else:
            st.info("ここに履歴画像の見本が表示されます (sample_history.png をアップロードしてください)")

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
    """OCR集計（1と7の誤認識対策版）"""
    height, width = img.shape[:2]
    roi_width = int(width * 0.35) 
    roi = img[:, width - roi_width : width]
    
    # 🌟 改善ポイント1: 画像を2.5倍に拡大してOCRの認識精度を上げる
    roi = cv2.resize(roi, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    
    # 🌟 改善ポイント2: 膨張処理(DILATE)をやめ、メディアンフィルタでノイズ除去する
    # 1と7がくっついて誤認識されるのを防ぐため
    mask = cv2.medianBlur(mask, 3)
    
    mask_inverted = cv2.bitwise_not(mask)
    
    # 🌟 改善ポイント3: 出力を完全に数字(0-9)のみに限定する設定を追加
    config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
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
    st_spins_final = st.number_input("ラッシュ(ST)の回転数合計", min_value=0, value=st_spins_auto, step=1)
    
    jitan_spins = 0
    if mode == "② 時短あり (エヴァ・海など)":
        st.warning("⚠️ 時短モードON")
        jitan_spins = st.number_input("時短中に回した回転数", min_value=0, value=0, step=1)

    # 2. 当たりデータ入力
    st.markdown("#### ▼ 当たりデータ (データ機通りに入力)")
    c_data1, c_data2 = st.columns(2)
    with c_data1:
        total_hits = st.number_input("総大当たり回数", min_value=0, value=0)
    with c_data2:
        first_hits = st.number_input("初当たり回数", min_value=0, value=0)

    # 3. 出玉詳細設定 (LT対応 ＆ RUSH振り分け対応)
    st.markdown("#### ▼ 出玉詳細設定")
    
    has_lt = st.checkbox("⚡ 上位RUSH / ラッキートリガー (LT) を考慮する")
    lt_hits = 0
    lt_payout = 0
    
    if has_lt:
        c_lt1, c_lt2 = st.columns(2)
        with c_lt1:
            lt_hits = st.number_input("上位RUSH(LT)中の当たり回数", min_value=0, value=0)
        with c_lt2:
            lt_payout = st.number_input("上位RUSH(LT)の平均出玉", value=1500, step=10)
    
    # RUSH中の当たり回数を自動計算（総当たり - 初当たり - LT当たり）
    st_hits = total_hits - first_hits - lt_hits
    if st_hits < 0: st_hits = 0
    st.info(f"📊 計算上の通常RUSH中当たり回数: **{st_hits} 回**")

    st.caption("💡 【重要】サイトセブンでは「3000発」は「1500発×2回」とカウントされるため、エヴァやRe:ゼロ等の機種は平均出玉を「1500」のままにしてください。")

    payout_mode = st.radio(
        "通常RUSH中の出玉タイプ", 
        ["① 単一出玉 (オール1500発など)", "② 複数振り分け (甘デジ・ライトミドルなど)"]
    )

    if payout_mode == "① 単一出玉 (オール1500発など)":
        st_payout = st.number_input("RUSH中の平均出玉", value=1500, step=10)
    else:
        c_ratio1, c_ratio2 = st.columns(2)
        with c_ratio1:
            payout1 = st.number_input("出玉A (例: 880)", value=880, step=10)
            ratio1 = st.number_input("割合A (%)", min_value=0.0, max_value=100.0, value=40.0, step=1.0)
        with c_ratio2:
            payout2 = st.number_input("出玉B (例: 330)", value=330, step=10)
            ratio2 = st.number_input("割合B (%)", min_value=0.0, max_value=100.0, value=60.0, step=1.0)
            
        st_payout = (payout1 * (ratio1 / 100)) + (payout2 * (ratio2 / 100))
        st.success(f"📈 算出されたRUSH中の平均出玉: **{st_payout:.1f} 発**")

    st.markdown("---")
    
    # 🌟 NEW: 初当たり出玉のシンプル入力（RUSH突入・非突入を一括計算）
    st.markdown("#### ▼ 初当たり出玉設定")
    first_hit_payout = st.number_input("初当たり1回あたりの出玉 (突入/非突入問わず)", value=330, step=10)
    st.caption(f"※ 初当たり {first_hits} 回 × {first_hit_payout} 発 で自動計算されます。")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # 4. 解析実行
    if st.button("🔥 解析開始 (ANALYZE) 🔥", type="primary"):
        real_spins = total_spins - st_spins_final - jitan_spins
        
        # 🌟 計算ロジックが超シンプルに！
        income_st = st_hits * st_payout               # RUSH中の獲得出玉
        income_lt = lt_hits * lt_payout               # LT中の獲得出玉
        income_first = first_hits * first_hit_payout  # 初当たりの獲得出玉（一括）
        
        total_payout = income_st + income_first + income_lt
        used_balls = total_payout - diff_balls
        
        st.markdown(f"""
        <div style="background-color: rgba(0,0,0,0.5); padding: 20px; border-radius: 10px; border: 2px solid #FFD700; text-align: center;">
            <h3 style="color: #FFD700; margin-bottom: 0;">RESULT</h3>
            <p style="color: #ccc;">実質通常回転数: {real_spins} 回転</p>
            <p style="color: #ccc;">推定投資: {int(used_balls):,}発 ({int(used_balls)*4:,}円)</p>
            <p style="color: #888; font-size: 0.8em;">※理論上の総獲得出玉: {int(total_payout):,}発</p>
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
            st.error("計算エラー：投資がマイナスです。(グラフの読み取り誤差、または出玉設定が多すぎる可能性があります)")
