import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract
import re

# Tesseract OCRの設定
st.set_page_config(page_title="パチンコ回転率アナライザー", page_icon="🎰")

st.title("🎰 究極の回転率アナライザー")
st.markdown("グラフ画像と履歴画像をアップロードして、正確な回転率を算出します。")

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

    # ★修正箇所1：0ライン補正を「0.027」で固定
    # ユーザーには見えない内部計算として処理
    correction_y = int(gh * 0.027) 
    zero_line_y -= correction_y

    # 3. グラフ線検出
    roi = img[gy:gy+gh, gx:gx+gw]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # 緑・紫・オレンジ・赤
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

    # ★修正箇所2：赤線・青丸の描画を削除
    # 計算だけして、画像は元の綺麗なまま返す
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

# --- 1. グラフ画像アップロード ---
st.subheader("① グラフ画像の解析")
uploaded_graph = st.file_uploader("グラフ画像をアップロード", type=['jpg', 'png', 'jpeg'], key="graph")

# ★修正箇所3：スライダー（微調整機能）を削除しました

diff_balls = 0

if uploaded_graph is not None:
    # 画像読み込み
    file_bytes = np.asarray(bytearray(uploaded_graph.read()), dtype=np.uint8)
    img_graph = cv2.imdecode(file_bytes, 1)
    
    # 解析実行（補正値は内部で0.027固定）
    result, msg_or_img = analyze_graph_final(img_graph)
    
    if result is not None:
        diff_balls = result
        # そのままの画像を表示（線なし）
        st.image(cv2.cvtColor(msg_or_img, cv2.COLOR_BGR2RGB), caption=f"解析完了", use_column_width=True)
        st.success(f"推定差玉: {diff_balls} 発")
    else:
        st.error(f"エラー: {msg_or_img}")


# --- 2. 履歴画像アップロード ---
st.subheader("② 履歴リストの解析（ST回転数）")
uploaded_history = st.file_uploader("履歴画像（赤数字）をアップロード（任意）", type=['jpg', 'png', 'jpeg'], key="history")

st_spins_auto = 0
st_details = []

if uploaded_history is not None:
    file_bytes = np.asarray(bytearray(uploaded_history.read()), dtype=np.uint8)
    img_hist = cv2.imdecode(file_bytes, 1)
    
    st_sum, num_list = sum_red_start_counts(img_hist)
    st_spins_auto = st_sum
    st_details = num_list
    
    st.info(f"検出された赤数字: {st_details}")
    st.success(f"自動集計されたST回転数: {st_spins_auto} 回転")


# --- 3. データ入力と計算 ---
st.divider()
st.subheader("③ データ入力と計算")

col1, col2 = st.columns(2)
with col1:
    total_spins = st.number_input("現在の総回転数", min_value=0, value=3000, step=1)
with col2:
    st_spins_final = st.number_input("ST/時短の回転数（自動入力値を修正可）", min_value=0, value=st_spins_auto, step=1)

# 出玉内訳
st.write("▼ 当たり内訳を入力")
c1, c2 = st.columns(2)
with c1:
    count_3000 = st.number_input("上位(3000発) 回数", min_value=0, value=0)
    payout_3000 = st.number_input("上位 出玉/回", value=2800)
with c2:
    count_1500 = st.number_input("通常(1500発) 回数", min_value=0, value=0)
    payout_1500 = st.number_input("通常 出玉/回", value=1400)

c3, c4 = st.columns(2)
with c3:
    count_300 = st.number_input("チャージ(300発) 回数", min_value=0, value=0)
    # チャージ初期値: 280
    payout_300 = st.number_input("チャージ 出玉/回", value=280)

# 計算ボタン
if st.button("回転率を計算する", type="primary"):
    # ロジック
    real_spins = total_spins - st_spins_final
    total_payout = (count_3000 * payout_3000) + (count_1500 * payout_1500) + (count_300 * payout_300)
    used_balls = total_payout - diff_balls
    
    st.markdown("### 📊 判定結果")
    st.write(f"**実質通常回転数**: {real_spins} 回転")
    st.write(f"**総出玉**: {total_payout:,} 発")
    st.write(f"**推定差玉**: {diff_balls:+,} 発")
    st.write(f"**推定投資**: {int(used_balls):,} 発 ({int(used_balls)*4:,} 円相当)")
    
    if used_balls > 0:
        rate = (real_spins / used_balls) * 250
        st.metric(label="1000円あたりの回転数", value=f"{rate:.2f} 回転")
        
        if rate >= 20:
            st.balloons()
            st.success("素晴らしい！文句なしの優秀台です！")
        elif rate <= 15:
            st.error("ボーダー以下の可能性が高いです。撤退を推奨します。")
        else:
            st.warning("ボーダー付近、または微妙なラインです。")
    else:
        st.error("計算エラー：投資がマイナス（勝ちすぎ）です。出玉入力やグラフを確認してください。")
