import streamlit as st
import re

# ==========================================
# ★設定エリア★
# ==========================================
APP_PASSWORD = "777" 
COIN_BASE = 32.0  # 1000円あたりの回転数 (ベース32G)

# ==========================================
# ページ設定
# ==========================================
st.set_page_config(page_title="OKI-DOKI PRO STRATEGY", page_icon="🌺", layout="centered")

# ==========================================
# 🔐 ログイン認証
# ==========================================
def check_password():
    if "password_correct" not in st.session_state:
        st.markdown("<h1 style='text-align: center; color: #ff007f;'>🌺 OKI-DOKI PRO 🌺</h1>", unsafe_allow_html=True)
        pwd = st.text_input("PASSWORD", type="password", key="password_input")
        if pwd == APP_PASSWORD:
            st.session_state["password_correct"] = True
            st.rerun()
        elif pwd:
            st.error("パスワードが違います")
        return False
    return True

if not check_password():
    st.stop()

# ==========================================
# デザイン設定 (黒×ネオンピンク/ゴールド)
# ==========================================
st.markdown("""
    <style>
    .stApp { background-color: #0e0e0e; color: white; }
    h1 { color: #ff007f !important; text-shadow: 0 0 10px #ff007f; text-align: center; font-family: 'Arial Black'; }
    h3 { border-left: 5px solid #FFD700; padding-left: 10px; color: white; }
    .stNumberInput, .stSelectbox, .stTextArea { background-color: #1e1e1e; border-radius: 8px; color: white; }
    .stButton > button {
        background: linear-gradient(90deg, #FFD700, #FDB931);
        color: black; font-weight: bold; border-radius: 25px;
        padding: 15px 30px; font-size: 20px; width: 100%;
        box-shadow: 0 0 10px rgba(255, 215, 0, 0.5);
        border: none;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 20px rgba(255, 215, 0, 1);
        background: linear-gradient(90deg, #ff007f, #ff5e00);
        color: white;
    }
    /* 戦略ボックスのデザイン */
    .strategy-box { 
        padding: 15px; border-radius: 10px; margin-bottom: 15px; 
        border: 2px solid #444; background-color: #1a1a1a;
    }
    .plan-title { font-size: 1.3em; font-weight: bold; margin-bottom: 10px; display: block; }
    .cost-display { font-size: 1.8em; font-weight: bold; }
    .tag { background-color: #333; padding: 2px 8px; border-radius: 4px; font-size: 0.8em; color: #ccc; margin-right: 5px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🌺 沖ドキ完全攻略")

# ==========================================
# 💰 共通計算関数
# ==========================================
def calc_investment(start_g, end_g):
    """指定区間の投資額を計算 (1000円単位切り上げ)"""
    if start_g >= end_g: return 0
    needed = end_g - start_g
    cost = (needed / COIN_BASE) * 1000
    return int(((cost // 1000) + 1) * 1000)

def parse_history(text, current_g):
    """履歴テキストから有利区間消化G数を計算"""
    if not text: return current_g, 0
    nums = re.findall(r'\d+', text)
    bonus_count = len(nums)
    # ボーナス中の消化G数 (BIG/REG平均 60G程度と仮定 ※GOLD/BLACKは重要)
    history_sum = sum([int(n) for n in nums])
    total = history_sum + (bonus_count * 69) + current_g # 69GはBB/RB平均消化
    return total, bonus_count

# ==========================================
# 入力エリア
# ==========================================
model = st.selectbox("機種を選択", [
    "① 沖ドキGOLD (金ドキ/区間)", 
    "② 沖ドキBLACK (黒ドキ/区間)", 
    "③ 沖ドキDUO (テーブル/ゾーン)", 
    "④ 沖ドキゴージャス (深区間)"
])

st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    current_g = st.number_input("現在ゲーム数", min_value=0, value=100, step=10)
with col2:
    is_reset = st.checkbox("朝イチ / リセット", value=False)

# --- 機種別入力 ---
through_count = 0
prev_hit_g = 0
history_input = ""

if "DUO" in model:
    through_count = st.number_input("スルー回数 (天国間)", min_value=0, value=1)
elif "BLACK" in model:
    through_count = st.number_input("スルー回数", min_value=0, value=1)
    prev_hit_g = st.number_input("前回の当選ゲーム数 (1スルー時判定用)", min_value=0, value=100)
    st.caption("※1スルーの狙い目判定に使います")

if model in ["① 沖ドキGOLD (金ドキ/区間)", "② 沖ドキBLACK (黒ドキ/区間)", "④ 沖ドキゴージャス (深区間)"]:
    st.markdown("#### ▼ 有利区間計算")
    history_input = st.text_area("履歴G数 (例: 120 45 320)", height=70, help="スペース区切りで入力")
    
    # リアルタイム計算表示
    if history_input:
        total_yuuri, b_count = parse_history(history_input, current_g)
        st.info(f"📊 推定有利区間消化: **{total_yuuri} G** (ボナ{b_count}回)")
    else:
        st.caption("※有利区間天井狙いの場合は履歴を入力してください")
        total_yuuri = current_g # 入力なしなら現在G数のみ

# ==========================================
# 🔥 判定ロジック & 表示
# ==========================================
st.markdown("<br>", unsafe_allow_html=True)

if st.button("🔥 戦略分析 (ANALYZE) 🔥"):
    st.markdown("---")
    
    plans = [] # 表示する戦略プランのリスト
    
    # ----------------------------------------------------
    # 1. 沖ドキDUO (ゾーン重視 / テーブル管理)
    # ----------------------------------------------------
    if "DUO" in model:
        zone_target = 400
        ceiling_target = 800
        
        # --- ロジック ---
        # A. 3スルー以上 or 2スルー(テーブル優遇)
        if through_count >= 3 or through_count == 2:
            plans.append({
                "title": "👑 スルー天井狙い (問答無用)",
                "color": "#FFD700", # Gold
                "desc": "2スルー以上は期待値が高いです。当たるまで打ち切り推奨。",
                "target_g": ceiling_target,
                "type": "CEILING"
            })
        
        # B. ゾーン狙い (200~400G)
        elif current_g < 400:
            # Plan A: ゾーンまで
            cost_a = calc_investment(current_g, zone_target)
            plans.append({
                "title": "🅰️ プランA：400Gゾーン狙い",
                "color": "#00ff00", # Green
                "desc": "300-400Gの当選率は約80%超。400Gを抜けたらヤメる低リスク戦略。",
                "target_g": zone_target,
                "cost": cost_a,
                "type": "ZONE"
            })
            
            # Plan B: もし抜けた場合の天井リスク
            cost_b = calc_investment(zone_target, ceiling_target)
            total_b = calc_investment(current_g, ceiling_target)
            plans.append({
                "title": "🅱️ プランB：もし抜けた場合 (天井ツッパ)",
                "color": "#ff4444", # Red
                "desc": "400Gを抜けると期待値は下がります。リスク覚悟で追う場合の追加投資額です。",
                "target_g": ceiling_target,
                "cost": total_b,
                "extra_info": f"※ゾーン抜け後、さらに約 {cost_b:,} 円 必要",
                "type": "CEILING"
            })
        
        # C. ゾーン抜け後
        else:
            plans.append({
                "title": "⚠️ 危険領域 (ゾーン抜け)",
                "color": "#ff4444",
                "desc": "現在は400Gを抜けており、スルー回数が少なければ期待値マイナスです。",
                "target_g": ceiling_target,
                "type": "STOP"
            })

    # ----------------------------------------------------
    # 2. 沖ドキGOLD (有利区間 / リセット)
    # ----------------------------------------------------
    elif "GOLD" in model:
        section_target = 2000
        ceiling_target = 999
        
        # A. 有利区間天井狙い
        if total_yuuri >= 2000:
            plans.append({
                "title": "🔥 激アツ！有利区間天井到達",
                "color": "#FF00FF", # Neon Pink
                "desc": "区間2000Gを超えています。ボーナス当選→天国移行まで全ツッパ！",
                "target_g": ceiling_target,
                "type": "CEILING"
            })
        elif total_yuuri >= 1400: # 狙い目
            cost = calc_investment(current_g, ceiling_target) # 一旦ボナ天まで
            plans.append({
                "title": "👑 有利区間狙い (2000Gクロス)",
                "color": "#FFD700",
                "desc": f"有利区間あと {2000 - total_yuuri}G。今のボーナスで踏む可能性大。",
                "target_g": ceiling_target,
                "cost": cost,
                "type": "CEILING"
            })
            
        # B. リセット / ゾーン
        elif is_reset and current_g <= 200:
            cost = calc_investment(current_g, 200)
            plans.append({
                "title": "🔄 リセット・チャンス狙い (200G)",
                "color": "#00ff00",
                "desc": "チャンスモード天井(200G)まで。当たらなければヤメ。",
                "target_g": 200,
                "cost": cost,
                "type": "ZONE"
            })
            
        # C. ゲーム数天井
        elif current_g >= 700:
            plans.append({
                "title": "📈 天井狙い (700G~)",
                "color": "#00ff00",
                "desc": "ゲーム数天井狙い。有利区間が浅くても打てます。",
                "target_g": ceiling_target,
                "type": "CEILING"
            })
        else:
             plans.append({"title": "✋ 様子見推奨 (STOP)", "color": "#777", "desc": "狙い目ラインに達していません。", "type": "STOP"})

    # ----------------------------------------------------
    # 3. 沖ドキBLACK (区間 / スルー別ボーダー)
    # ----------------------------------------------------
    elif "BLACK" in model:
        section_target = 1900
        ceiling_target = 999
        
        # A. 有利区間狙い
        if total_yuuri >= 1900:
             plans.append({
                "title": "⚫ BLACK 有利区間狙い (1900G~)",
                "color": "#FFD700",
                "desc": "黒ドキチャンス！天国まで打ち切り推奨。",
                "target_g": ceiling_target,
                "type": "CEILING"
            })
        
        # B. スルー回数別 天井狙い (BLACK特有)
        else:
            border = 999
            if through_count == 1:
                border = 520 if prev_hit_g <= 150 else 700
                desc = f"1スルー(前回{prev_hit_g}G当選)のボーダー: {border}G"
            elif through_count == 2:
                border = 610
                desc = "2スルーボーダー: 610G"
            elif through_count >= 3:
                border = 580
                desc = "3スルーボーダー: 580G"
            else: # 0スルー
                border = 670 # リセット以外
                desc = "0スルーボーダー: 670G"
                if is_reset: 
                    # リセ0スルーの特別処理
                    if current_g <= 32:
                        plans.append({"title": "🔄 リセット32Gカニ歩き", "color":"#00ff00", "desc":"32Gまで", "target_g":32, "type":"ZONE"})
                        border = 9999 # 重複防止
                    elif 100 <= current_g <= 130:
                        plans.append({"title": "🔄 リセット100Gゾーン", "color":"#00ff00", "desc":"130Gまで", "target_g":130, "type":"ZONE"})
                        border = 9999
                    else:
                        border = 670

            if current_g >= border:
                plans.append({
                    "title": f"📈 {through_count}スルー 天井狙い",
                    "color": "#00ff00",
                    "desc": desc,
                    "target_g": ceiling_target,
                    "type": "CEILING"
                })
            elif border != 9999:
                 plans.append({"title": "✋ まだ早いです", "color": "#777", "desc": f"狙い目は {border}G から。", "type": "STOP"})

    # ----------------------------------------------------
    # 4. 沖ドキゴージャス (深区間)
    # ----------------------------------------------------
    elif "ゴージャス" in model:
        section_target = 2300
        ceiling_target = 999
        
        if total_yuuri >= 2300:
            plans.append({
                "title": "💎 ゴージャス 有利区間狙い",
                "color": "#FFD700",
                "desc": "区間2300G越え。3000G付近の区間切れまでGO！",
                "target_g": ceiling_target,
                "type": "CEILING"
            })
        elif is_reset and current_g <= 200:
             plans.append({
                "title": "🔄 リセット・チャンス狙い",
                "color": "#00ff00",
                "desc": "200G天井まで有効。",
                "target_g": 200,
                "type": "ZONE"
            })
        elif current_g >= 600:
             plans.append({
                "title": "📈 天井狙い (600G~)",
                "color": "#00ff00",
                "desc": "ゴージャスは600GからGO判定。",
                "target_g": ceiling_target,
                "type": "CEILING"
            })
        else:
            plans.append({"title": "✋ STOP", "color": "#777", "desc": "有利区間2300G、または通常600Gから。", "type": "STOP"})

    # ==========================================
    # 結果カードの描画
    # ==========================================
    for plan in plans:
        cost_txt = ""
        if plan.get("type") != "STOP":
            # 投資額計算 (ターゲットG - 現在G)
            if "cost" in plan:
                investment = plan["cost"]
            else:
                investment = calc_investment(current_g, plan["target_g"])
            
            cost_txt = f"¥ {investment:,}"
        else:
            cost_txt = "---"

        # HTMLレンダリング
        st.markdown(f"""
        <div class="strategy-box" style="border-color: {plan['color']};">
            <span class="plan-title" style="color: {plan['color']};">{plan['title']}</span>
            <div style="color: #ddd; margin-bottom: 10px;">{plan['desc']}</div>
            <div style="display:flex; justify-content:space-between; align-items:end;">
                <div>
                    <span class="tag">{plan['type']}</span>
                    Target: <b>{plan.get('target_g', '---')} G</b>
                </div>
                <div style="text-align:right;">
                    投資目安<br>
                    <span class="cost-display" style="color: {plan['color']};">{cost_txt}</span>
                </div>
            </div>
            {'<div style="font-size:12px; color:#aaa; margin-top:5px;">' + plan.get('extra_info', '') + '</div>' if 'extra_info' in plan else ''}
        </div>
        """, unsafe_allow_html=True)
