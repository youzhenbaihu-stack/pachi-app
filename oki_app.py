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
    .alert-text { color: #ff4444; font-size: 0.9em; font-weight: bold; }
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
    current_g = st.number_input("現在ゲーム数", min_value=0, value=0, step=10)
with col2:
    is_reset = st.checkbox("朝イチ / リセット", value=False)

# --- 機種別入力 ---
through_count = 0
prev_hit_g = 0
total_yuuri = current_g 

if "DUO" in model:
    through_count = st.number_input("スルー回数 (天国間)", min_value=0, value=1)
    st.caption("※DUOは 2スルー、4スルー、6スルー以上 が狙い目です")
elif model in ["① 沖ドキGOLD (金ドキ/区間)", "② 沖ドキBLACK (黒ドキ/区間)"]:
    through_count = st.number_input("スルー回数", min_value=0, value=1)
    prev_hit_g = st.number_input("前回の当選ゲーム数 (1スルー時判定用)", min_value=0, value=140)
    st.caption("※朝一1スルーの通常B狙い判定などに使います")

# ----------------------------------------------------
# 👑 ガチ仕様：有利区間精密計算
# ----------------------------------------------------
if model in ["① 沖ドキGOLD (金ドキ/区間)", "② 沖ドキBLACK (黒ドキ/区間)", "④ 沖ドキゴージャス (深区間)"]:
    st.markdown("#### ▼ 【ガチ仕様】有利区間 精密計算")
    st.error("""
    ⚠️ **【重要】履歴入力のルール**
    有利区間は「天国（32G以内の連チャン）」を抜けた時点でリセットされます。
    **必ず『前回の天国抜け以降』のデータだけを入力**してください。
    """)
    
    history_input = st.text_area("履歴G数 (例: 120 45 320)", height=70, help="スペース区切りで通常ゲーム数を入力")
    
    col_b, col_r = st.columns(2)
    with col_b:
        bb_count = st.number_input("BIG回数 (天国抜け後)", min_value=0, value=0)
    with col_r:
        rb_count = st.number_input("REG回数 (天国抜け後)", min_value=0, value=0)

    if history_input or bb_count > 0 or rb_count > 0:
        nums = re.findall(r'\d+', history_input) if history_input else []
        history_sum = sum([int(n) for n in nums])
        
        if "BLACK" in model:
            bb_g = 59
            rb_g = 24
            calc_note = "BIG=59G / REG=24G"
        else:
            bb_g = 70
            rb_g = 30
            calc_note = "BIG=70G / REG=30G"
            
        bonus_sum = (bb_count * bb_g) + (rb_count * rb_g)
        total_yuuri = history_sum + bonus_sum + current_g
        
        st.info(f"📊 **精密・有利区間消化: {total_yuuri} G** \n\n(通常{history_sum}G + ボナ{bonus_sum}G + 現在{current_g}G) ※{calc_note}計算")
        
        if len(nums) != (bb_count + rb_count) and len(nums) > 0:
            st.markdown(f"<div class='alert-text'>⚠️ 警告: 履歴の個数（{len(nums)}個）と、BIG/REGの合計回数（{bb_count + rb_count}回）がズレています！データランプを確認してください。</div>", unsafe_allow_html=True)

# ==========================================
# 🔥 判定ロジック & 表示
# ==========================================
st.markdown("<br>", unsafe_allow_html=True)

if st.button("🔥 戦略分析 (ANALYZE) 🔥"):
    st.markdown("---")
    plans = []
    
    # ----------------------------------------------------
    # 1. 沖ドキDUO
    # ----------------------------------------------------
    if "DUO" in model:
        zone_target = 400
        ceiling_target = 800
        
        if through_count >= 6:
            plans.append({
                "title": "👑 スルー天井ツッパ (絶対GO)",
                "color": "#FFD700",
                "desc": f"現在{through_count}スルー。期待値の塊です。天国に上がるまで全ツッパ推奨！",
                "target_g": ceiling_target,
                "type": "CEILING",
                "action": "天国移行までボーナスを当て続けます。<br><b>【やめどき】</b>ボーナス後32G(or 35G)ヤメ。"
            })
        elif through_count == 4 or through_count == 2:
            table_target = through_count + 1
            plans.append({
                "title": f"🎯 テーブル天井狙い ({through_count}スルー)",
                "color": "#FF00FF",
                "desc": f"規定回数{table_target}回目の振り分けが濃い大チャンス状態です。",
                "target_g": ceiling_target,
                "type": "CEILING",
                "action": "次のボーナス当選まで打ち切ります。<br><b>【やめどき】</b>基本はボーナス後32G(or 35G)ヤメ。<br>⚠️終了画面「夕方」やドキハナチャンス失敗時は次回も期待値が高いため続行。"
            })
        elif current_g < 400:
            plans.append({
                "title": "🅰️ 400G仮天井 ゾーン狙い",
                "color": "#00ff00",
                "desc": "301〜400Gの当選率は80%超え！当たればラッキーの低リスク戦略です。",
                "target_g": zone_target,
                "cost": calc_investment(current_g, zone_target),
                "type": "ZONE",
                "action": "<b>【絶対ルール】400Gまでに当たらなければ「即ヤメ」してください。</b><br>DUOは400G以降を追うと期待値がマイナスになります。<br>当たった場合は天国移行に期待し、ボーナス後32G(or 35G)ヤメ。"
            })
        else:
            plans.append({
                "title": "⚠️ 危険領域 (400G抜け)",
                "color": "#ff4444",
                "desc": f"現在{current_g}G・{through_count}スルー。仮天井を抜け、スルー恩恵も弱い状態です。",
                "target_g": ceiling_target,
                "type": "STOP",
                "action": "ここから800G天井まで追うのはリスクが高すぎます。別の台を探しましょう。"
            })

    # ----------------------------------------------------
    # 2. 沖ドキGOLD
    # ----------------------------------------------------
    elif "GOLD" in model:
        ceiling_target = 999
        
        if total_yuuri >= 2000:
            cost_ceiling = calc_investment(current_g, ceiling_target)
            cost_str = f"""
            <div style="font-size: 0.5em; line-height: 1.2; color: #ccc; margin-top: 5px; font-weight: normal;">最短(即当たり) 〜 最深(天井)</div>
            <div style="line-height: 1.2; color: #FF00FF;">¥ 1,000 〜 ¥ {cost_ceiling:,}</div>
            """
            plans.append({
                "title": "🔥 激アツ！有利区間天井到達",
                "color": "#FF00FF",
                "desc": "すでに区間2000Gを超えています！いつ当たっても次回金ドキの大チャンス！",
                "target_g": ceiling_target,
                "cost_str": cost_str,
                "type": "CEILING",
                "action": "次回ボーナス終了時に有利区間が切れ、<b>金ドキモード（天国以上）へ移行する大チャンス</b>です。大連チャンに期待！<br><b>【やめどき】</b>連チャン終了後、32G回して即ヤメ。"
            })
        elif total_yuuri >= 1400:
            needed_g = 2000 - total_yuuri
            target_cross_g = current_g + needed_g
            
            if target_cross_g < ceiling_target:
                cost_cross = calc_investment(current_g, target_cross_g)
                cost_ceiling = calc_investment(current_g, ceiling_target)
                cost_str = f"""
                <div style="font-size: 0.5em; line-height: 1.2; color: #ccc; margin-top: 5px; font-weight: normal;">最短 ({target_cross_g}G到達)</div>
                <div style="line-height: 1.2; color: #FFD700;">¥ {cost_cross:,}</div>
                <div style="font-size: 0.5em; line-height: 1.2; color: #ccc; margin-top: 5px; font-weight: normal;">最深 (天井)</div>
                <div style="line-height: 1.2; color: #ff4444;">¥ {cost_ceiling:,}</div>
                """
            else:
                cost_str = f"¥ {calc_investment(current_g, ceiling_target):,}"

            plans.append({
                "title": "👑 有利区間狙い (2000Gクロス)",
                "color": "#FFD700",
                "desc": f"有利区間あと {needed_g}G。{target_cross_g}G以降の当選で次回金ドキのチャンス！",
                "target_g": ceiling_target,
                "cost_str": cost_str,
                "type": "CEILING",
                "action": f"まずは最短目標の<b>{target_cross_g}G</b>到達を目指します。もし到達前に当たってしまった場合は区間が切れない可能性があるため、押し引きの判断が必要です。<br><b>【やめどき】</b>天国抜け後32Gヤメ。"
            })
            
        else:
            border = 999
            
            # 朝一1スルー 通常B狙い (実戦ボーダー400G~)
            if through_count == 1 and prev_hit_g <= 150:
                border = 400
                if current_g >= border:
                    cost_ceiling = calc_investment(current_g, ceiling_target)
                    cost_str = f"""
                    <div style="font-size: 0.5em; line-height: 1.2; color: #ccc; margin-top: 5px; font-weight: normal;">最短(即当たり) 〜 最悪(天井リスク)</div>
                    <div style="line-height: 1.2; color: #00ff00;">¥ 1,000 〜 ¥ {cost_ceiling:,}</div>
                    """
                    plans.append({
                        "title": "🔥 朝一1スルー 通常B狙い",
                        "color": "#00ff00",
                        "desc": f"前回{prev_hit_g}G当選。通常B滞在の期待大！道中での早い当たりに期待する立ち回りです。",
                        "target_g": ceiling_target,
                        "cost_str": cost_str,
                        "type": "MODE_B",
                        "action": "ゲーム数天井(999G)リスクはありますが、通常Bの恩恵（早い当たり＆天国移行）を狙います。<br><b>【やめどき】</b>天国抜け後、32G回してヤメ。"
                    })
                else:
                    plans.append({"title": "✋ まだ早いです", "color": "#777", "desc": f"朝一1スルーの通常B狙いは {border}G から。", "type": "STOP"})
            
            # ★追加：GOLDのリセット32Gカニ歩き
            elif is_reset and current_g <= 32:
                plans.append({
                    "title": "🔄 リセット32Gカニ歩き",
                    "color": "#00ff00",
                    "desc": "朝一リセット台はチャンスモード滞在率約40%！32Gだけ回すのが有効。",
                    "target_g": 32,
                    "cost": calc_investment(current_g, 32),
                    "type": "ZONE",
                    "action": "<b>32Gまで回して当たらなければ即ヤメ（カニ歩き）</b>推奨です。<br>当たった場合は次回モードB以上に期待できるため、天国抜け後32Gヤメ。"
                })
            elif is_reset and current_g <= 200:
                plans.append({
                    "title": "🔄 リセット・チャンス狙い (200G)",
                    "color": "#00ff00",
                    "desc": "チャンスモード天井(200G)まで。当たらなければヤメ。",
                    "target_g": 200,
                    "cost": calc_investment(current_g, 200),
                    "type": "ZONE",
                    "action": "チャンスモードの天井(200G)での当選に期待。<br><b>【やめどき】</b>200Gで当たらなければ即ヤメ。当たった場合は32G回してヤメ。"
                })
            elif current_g >= 700:
                plans.append({
                    "title": "📈 天井狙い (700G~)",
                    "color": "#00ff00",
                    "desc": "ゲーム数天井狙い。有利区間が浅くても打てます。",
                    "target_g": ceiling_target,
                    "type": "CEILING",
                    "action": "ゲーム数天井(999G)でのボーナス当選。<br><b>【やめどき】</b>当選後、32G回してヤメ。有利区間が深くない場合は深追い厳禁。"
                })
            else:
                 plans.append({"title": "✋ 様子見推奨 (STOP)", "color": "#777", "desc": "狙い目ラインに達していません。", "type": "STOP"})

    # ----------------------------------------------------
    # 3. 沖ドキBLACK
    # ----------------------------------------------------
    elif "BLACK" in model:
        ceiling_target = 999
        
        # 実戦ボーダー 1500G~ に変更
        if total_yuuri >= 1500:
             cost_ceiling = calc_investment(current_g, ceiling_target)
             cost_str = f"""
             <div style="font-size: 0.5em; line-height: 1.2; color: #ccc; margin-top: 5px; font-weight: normal;">最短(即当たり) 〜 最深(天井)</div>
             <div style="line-height: 1.2; color: #FFD700;">¥ 1,000 〜 ¥ {cost_ceiling:,}</div>
             """
             plans.append({
                "title": "⚫ BLACK 有利区間狙い (1500G~)",
                "color": "#FFD700",
                "desc": "実戦上のボーダー1500Gを超えています。ここから黒ドキチャンス！",
                "target_g": ceiling_target,
                "cost_str": cost_str,
                "type": "CEILING",
                "action": "次回ボーナス当選後、有利区間がリセットされ<b>黒ドキモード（次回天国以上確定）へ移行する大チャンス</b>です。<br><b>【やめどき】</b>連チャン終了後、32G回して即ヤメ。"
            })
        else:
            border = 999
            
            # 朝一1スルー 通常B狙い (実戦ボーダー300G~)
            if through_count == 1 and prev_hit_g <= 150:
                border = 300
                if current_g >= border:
                    cost_ceiling = calc_investment(current_g, ceiling_target)
                    cost_str = f"""
                    <div style="font-size: 0.5em; line-height: 1.2; color: #ccc; margin-top: 5px; font-weight: normal;">最短(即当たり) 〜 最悪(天井リスク)</div>
                    <div style="line-height: 1.2; color: #00ff00;">¥ 1,000 〜 ¥ {cost_ceiling:,}</div>
                    """
                    plans.append({
                        "title": "🔥 朝一1スルー 通常B狙い",
                        "color": "#00ff00",
                        "desc": f"前回{prev_hit_g}G当選。通常B滞在の期待大！道中での早い当たりに期待する立ち回りです。",
                        "target_g": ceiling_target,
                        "cost_str": cost_str,
                        "type": "MODE_B",
                        "action": "ゲーム数天井(999G)リスクはありますが、通常Bの恩恵（早い当たり＆天国移行）を狙います。<br><b>【やめどき】</b>天国抜け後、32G回してヤメ。"
                    })
                else:
                    plans.append({"title": "✋ まだ早いです", "color": "#777", "desc": f"朝一1スルーの通常B狙いは {border}G から。", "type": "STOP"})
                    
            else:
                if through_count == 1:
                    border = 600
                    desc = f"1スルー(前回{prev_hit_g}G当選)のボーダー: {border}G"
                elif through_count == 2:
                    border = 610
                    desc = "2スルーボーダー: 610G"
                elif through_count >= 3:
                    border = 580
                    desc = "3スルーボーダー: 580G"
                else:
                    border = 670
                    desc = "0スルーボーダー: 670G"
                    if is_reset: 
                        if current_g <= 32:
                            plans.append({
                                "title": "🔄 リセット32Gカニ歩き", "color":"#00ff00", "desc":"32Gまで", "target_g":32, "type":"ZONE",
                                "action": "32Gで当たらなければ即ヤメ。当たれば32Gフォロー。"
                            })
                            border = 9999
                        elif 100 <= current_g <= 130:
                            plans.append({
                                "title": "🔄 リセット100Gゾーン", "color":"#00ff00", "desc":"130Gまで", "target_g":130, "type":"ZONE",
                                "action": "130Gで当たらなければ即ヤメ。当たれば32Gフォロー。"
                            })
                            border = 9999
                        else:
                            border = 670

                if current_g >= border and border != 9999:
                    plans.append({
                        "title": f"📈 {through_count}スルー 天井狙い",
                        "color": "#00ff00",
                        "desc": desc,
                        "target_g": ceiling_target,
                        "type": "CEILING",
                        "action": "ゲーム数天井(999G)でボーナス当選。<br><b>【やめどき】</b>天国抜け後、32G回してヤメ。"
                    })
                elif border != 9999:
                     plans.append({"title": "✋ まだ早いです", "color": "#777", "desc": f"狙い目は {border}G から。", "type": "STOP"})

    # ----------------------------------------------------
    # 4. 沖ドキゴージャス
    # ----------------------------------------------------
    elif "ゴージャス" in model:
        ceiling_target = 999
        
        if total_yuuri >= 2300:
            cost_ceiling = calc_investment(current_g, ceiling_target)
            cost_str = f"""
            <div style="font-size: 0.5em; line-height: 1.2; color: #ccc; margin-top: 5px; font-weight: normal;">最短(即当たり) 〜 最深(天井)</div>
            <div style="line-height: 1.2; color: #FFD700;">¥ 1,000 〜 ¥ {cost_ceiling:,}</div>
            """
            plans.append({
                "title": "💎 ゴージャス 有利区間狙い",
                "color": "#FFD700",
                "desc": "すでに区間2300G越え。いつ当たっても区間切れチャンス！",
                "target_g": ceiling_target,
                "cost_str": cost_str,
                "type": "CEILING",
                "action": "有利区間3000G付近の区間切れでの恩恵（ドキドキモード等）に期待。<br><b>【やめどき】</b>天国連チャン終了後、32G回して即ヤメ。"
            })
        
        # ★追加：ゴージャスのリセット32Gカニ歩き
        elif is_reset and current_g <= 32:
             plans.append({
                "title": "🔄 リセット32Gカニ歩き",
                "color": "#00ff00",
                "desc": "朝一リセットはチャンスモード狙いが有効。32Gまで。",
                "target_g": 32,
                "cost": calc_investment(current_g, 32),
                "type": "ZONE",
                "action": "<b>32Gまで回して当たらなければ即ヤメ（カニ歩き）</b>推奨です。<br>当たった場合は次回モードB以上に期待できるため、天国抜け後32Gヤメ。"
            })
        elif is_reset and current_g <= 200:
             plans.append({
                "title": "🔄 リセット・チャンス狙い",
                "color": "#00ff00",
                "desc": "200G天井まで有効。",
                "target_g": 200,
                "type": "ZONE",
                "action": "チャンスモード天井(200G)での当選に期待。<br><b>【やめどき】</b>200Gで当たらなければ即ヤメ。当たった場合は32G回してヤメ。"
            })
        elif current_g >= 600:
             plans.append({
                "title": "📈 天井狙い (600G~)",
                "color": "#00ff00",
                "desc": "ゴージャスは600GからGO判定。",
                "target_g": ceiling_target,
                "type": "CEILING",
                "action": "ゲーム数天井(999G)でボーナス当選。<br><b>【やめどき】</b>当選後、32G回してヤメ。※終了画面「夕方(浴衣)」が出たら天国移行まで全ツッパ！"
            })
        else:
            plans.append({"title": "✋ STOP", "color": "#777", "desc": "有利区間2300G、または通常600Gから。", "type": "STOP"})

    # ==========================================
    # 結果カードの描画
    # ==========================================
    for plan in plans:
        if plan.get("type") != "STOP":
            if "cost_str" in plan:
                cost_txt = plan["cost_str"]
            else:
                investment = plan.get("cost", calc_investment(current_g, plan["target_g"]))
                cost_txt = f"¥ {investment:,}"
        else:
            cost_txt = "---"

        action_html = ""
        if 'action' in plan:
            action_html = f"""<div style="background-color: rgba(255, 255, 255, 0.05); padding: 12px; border-radius: 8px; margin-top: 15px; border-left: 4px solid {plan['color']};"><div style="font-size: 0.9em; font-weight: bold; color: {plan['color']}; margin-bottom: 5px;">▶ 予想される展開・やめどき</div><div style="font-size: 0.9em; line-height: 1.5; color: #eee;">{plan['action']}</div></div>"""

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
            {action_html}
            {'<div style="font-size:12px; color:#aaa; margin-top:10px;">' + plan.get('extra_info', '') + '</div>' if 'extra_info' in plan else ''}
        </div>
        """, unsafe_allow_html=True)
