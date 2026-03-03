import streamlit as st
import re

APP_PASSWORD = "777" 
COIN_BASE = 32.0

st.set_page_config(page_title="OKI-DOKI PRO STRATEGY", page_icon="🌺", layout="centered")

def check_password():
    if "password_correct" not in st.session_state:
        st.markdown("<h1 style='text-align: center; color: #ff007f;'>🌺 OKI-DOKI PRO 🌺</h1>", unsafe_allow_html=True)
        pwd = st.text_input("PASSWORD", type="password", key="password_input")
        if pwd == APP_PASSWORD:
            st.session_state["password_correct"] = True
            st.rerun()
        elif pwd:
            st.error("パスワードが違います")
            
        st.markdown("---")
        st.markdown("<div style='text-align: center; font-size: 14px; color: #aaa; margin-top: 20px;'>※本アプリの利用パスワードはnoteで限定公開しています</div>", unsafe_allow_html=True)
        
        # ▼ 【重要】ご自身のnote記事のURLに書き換えてください ▼
        note_url = "https://note.com/" 
        st.markdown(f"<div style='text-align: center; margin-top: 10px;'><a href='{note_url}' target='_blank' style='background-color: #2cb696; color: white; padding: 10px 20px; border-radius: 5px; text-decoration: none; font-weight: bold;'>📝 note販売ページへ</a></div>", unsafe_allow_html=True)
        
        return False
    return True

if not check_password():
    st.stop()

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

def calc_investment(start_g, end_g):
    if start_g >= end_g: return 0
    needed = end_g - start_g
    cost = (needed / 32.0) * 1000
    return int(((cost // 1000) + 1) * 1000)

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

through_count = 0
prev_hit_g = 0
total_yuuri = current_g 

if "DUO" in model:
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        through_count = st.number_input("スルー回数 (現在何回スルーか)", min_value=0, value=1)
    with col_d2:
        end_screen = st.selectbox("前回の終了画面", ["デフォルト (昼)", "昼画面B (赤い花)", "昼画面C (あたいがふー)", "夕方画面", "夜画面A (シーサー)", "夜画面B (女の子2人)"])
    
    lamp_pattern = st.selectbox("前回の告知ランプ (特殊だった場合)", ["通常", "高速/スロー/同時など (通常B以上示唆)", "337拍子/花だけなど (天国・DUO示唆)", "右のみ/左のみ/カラフルなど (天国以上確定)"])
    
    st.caption("※DUOは【リセット恩恵】【4スルー/9スルー】【終了画面・ランプ示唆】が重要です")

elif model in ["① 沖ドキGOLD (金ドキ/区間)", "② 沖ドキBLACK (黒ドキ/区間)"]:
    through_count = st.number_input("スルー回数", min_value=0, value=1)
    prev_hit_g = st.number_input("前回の当選ゲーム数 (1スルー時判定用)", min_value=0, value=140)
    st.caption("※朝一1スルーの通常B狙い判定などに使います")

if model in ["① 沖ドキGOLD (金ドキ/区間)", "② 沖ドキBLACK (黒ドキ/区間)", "④ 沖ドキゴージャス (深区間)"]:
    st.markdown("#### ▼ 【ガチ仕様】有利区間 精密計算")
    st.error("⚠️ **【重要】履歴入力のルール** 有利区間は「天国（32G以内の連チャン）」を抜けた時点でリセットされます。**必ず『前回の天国抜け以降』のデータだけを入力**してください。")
    
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
            bb_g, rb_g, calc_note = 59, 24, "BIG=59G / REG=24G"
        else:
            bb_g, rb_g, calc_note = 70, 30, "BIG=70G / REG=30G"
            
        bonus_sum = (bb_count * bb_g) + (rb_count * rb_g)
        total_yuuri = history_sum + bonus_sum + current_g
        
        st.info(f"📊 **精密・有利区間消化: {total_yuuri} G** \n\n(通常{history_sum}G + ボナ{bonus_sum}G + 現在{current_g}G) ※{calc_note}計算")

st.markdown("<br>", unsafe_allow_html=True)

if st.button("🔥 戦略分析 (ANALYZE) 🔥"):
    st.markdown("---")
    plans = []
    
    if "DUO" in model:
        ceiling_target = 800
        
        # ① 絶対継続パターンの判定 (画面・ランプ)
        if end_screen in ["夕方画面", "夜画面A (シーサー)"] or "天国以上確定" in lamp_pattern:
            plans.append({"title": "👑 示唆ツッパ (絶対GO)", "color": "#FFD700", "desc": "夕方・夜画面、または強いランプ示唆が出ています。次回ボーナス＋連チャン抜けまで絶対に打ち切ってください！", "target_g": ceiling_target, "type": "CONFIRMED", "action": "次回ボーナス当選まで全ツッパ。<br><b>【やめどき】</b>連チャン抜け後32Gヤメ。"})
        elif end_screen == "夜画面B (女の子2人)":
            plans.append({"title": "🌺 DUOモード確定", "color": "#FF00FF", "desc": "DUOモード滞在が確定しています！大チャンス！", "target_g": ceiling_target, "type": "DUO", "action": "次回ボーナス当選まで全ツッパ。<br><b>【やめどき】</b>連チャン抜け後32Gヤメ。"})
            
        # ② スルー回数天井
        elif through_count >= 9:
            plans.append({"title": "👑 10回目スルー天井ツッパ", "color": "#FFD700", "desc": f"現在{through_count}スルー。規定回数10回目到達で【ドキドキ以上確定＋50%で超ドキドキ】です！", "target_g": ceiling_target, "type": "CEILING", "action": "天国移行まで全ツッパ。<br><b>【やめどき】</b>連チャン抜け後32Gヤメ。"})
        elif through_count == 4:
            plans.append({"title": "🎯 5回目(4スルー)天井狙い", "color": "#FF00FF", "desc": "5回目のボーナス天井選択時は【ドキドキ以上】へ移行します！(リセット時は約25%の高確率で選択)", "target_g": ceiling_target, "type": "CEILING", "action": "次のボーナス当選まで打ち切ります。<br><b>【やめどき】</b>基本はボーナス後32Gヤメ。"})
            
        # ③ リセット恩恵
        elif is_reset and current_g <= 200:
            plans.append({"title": "🅰️ リセット・チャンスモード狙い", "color": "#00ff00", "desc": "設定変更時は約40%でチャンスモード(天井200G)からスタートします。", "target_g": 200, "cost": calc_investment(current_g, 200), "type": "ZONE", "action": "<b>【やめどき】</b>200Gで当たらなければ即ヤメ、または300G仮天井まで様子見。"})
        elif is_reset and current_g <= 300:
             plans.append({"title": "🅰️ リセット・300G仮天井モード狙い", "color": "#00ff00", "desc": "設定変更時は300G仮天井モードへの移行にも期待できます。301G以降の当選率アップ！", "target_g": 300, "cost": calc_investment(current_g, 300), "type": "ZONE", "action": "<b>【やめどき】</b>300G+αで当たらなければヤメ。"})
             
        # ④ ゲーム数ゾーン・天井
        elif current_g >= 550:
            plans.append({"title": "📈 ゲーム数天井狙い", "color": "#FFD700", "desc": "最大天井800G到達時は【のるカナチャンス】発動確定です！700Gでドキハナモードの1段階昇格も確定します。", "target_g": 800, "cost": calc_investment(current_g, 800), "type": "CEILING", "action": "800Gの天井まで打ち切ります。<br><b>【やめどき】</b>ボーナス後32Gヤメ。"})
        elif current_g < 400:
            plans.append({"title": "🅰️ 300~400G 仮天井・ゾーン狙い", "color": "#00ff00", "desc": "通常A/Bの301〜400Gに仮天井あり。また300G到達でドキハナモード昇格抽選も行われます。", "target_g": 400, "cost": calc_investment(current_g, 400), "type": "ZONE", "action": "<b>【絶対ルール】400Gまでに当たらなければ「即ヤメ」してください。</b>"})
        else:
            plans.append({"title": "⚠️ 危険領域 (400G〜550G)", "color": "#ff4444", "desc": "仮天井を抜け、天井まで遠い状態です。500Gでのドキハナ昇格抽選はありますが期待値は低めです。", "target_g": 800, "type": "STOP"})

        # サブ情報の追加
        if end_screen in ["昼画面B (赤い花)", "昼画面C (あたいがふー)"] or "通常B以上" in lamp_pattern:
             plans.append({"title": "💡 通常B以上示唆あり", "color": "#00ffff", "desc": "前回の終了画面やランプで通常B以上の示唆が出ています。次回天国移行率がアップしています。", "type": "INFO"})

    elif "GOLD" in model:
        ceiling_target = 999
        if total_yuuri >= 2000:
            cost_str = f'<div style="color: #FF00FF;">¥ 1,000 〜 ¥ {calc_investment(current_g, ceiling_target):,}</div>'
            plans.append({"title": "🔥 激アツ！有利区間天井到達", "color": "#FF00FF", "desc": "いつ当たっても次回金ドキの大チャンス！", "target_g": ceiling_target, "cost_str": cost_str, "type": "CEILING", "action": "<b>【やめどき】</b>連チャン終了後、32G回して即ヤメ。"})
        elif total_yuuri >= 1400:
            target_cross_g = current_g + (2000 - total_yuuri)
            cost_str = f'<div style="color: #FFD700;">最短: ¥ {calc_investment(current_g, target_cross_g):,}</div>' if target_cross_g < ceiling_target else f"¥ {calc_investment(current_g, ceiling_target):,}"
            plans.append({"title": "👑 有利区間狙い (2000Gクロス)", "color": "#FFD700", "desc": f"有利区間あと {2000 - total_yuuri}G。", "target_g": ceiling_target, "cost_str": cost_str, "type": "CEILING", "action": "<b>【やめどき】</b>天国抜け後32Gヤメ。"})
        else:
            if through_count == 1 and prev_hit_g <= 150 and current_g >= 400:
                cost_str = f'<div style="color: #00ff00;">¥ 1,000 〜 ¥ {calc_investment(current_g, ceiling_target):,}</div>'
                plans.append({"title": "🔥 朝一1スルー 通常B狙い", "color": "#00ff00", "desc": "通常B滞在の期待大！", "target_g": ceiling_target, "cost_str": cost_str, "type": "MODE_B", "action": "<b>【やめどき】</b>天国抜け後、32G回してヤメ。"})
            elif is_reset and current_g <= 32:
                plans.append({"title": "🔄 リセット32Gカニ歩き", "color": "#00ff00", "desc": "朝一リセット台はチャンスモード滞在率約40%！", "target_g": 32, "cost": calc_investment(current_g, 32), "type": "ZONE", "action": "<b>32Gまで回して当たらなければ即ヤメ</b>"})
            elif is_reset and current_g <= 200:
                plans.append({"title": "🔄 リセット・チャンス狙い", "color": "#00ff00", "desc": "200Gまで。", "target_g": 200, "cost": calc_investment(current_g, 200), "type": "ZONE", "action": "<b>【やめどき】</b>200Gで当たらなければ即ヤメ。"})
            elif current_g >= 700:
                plans.append({"title": "📈 天井狙い (700G~)", "color": "#00ff00", "desc": "ゲーム数天井狙い。", "target_g": ceiling_target, "type": "CEILING", "action": "<b>【やめどき】</b>当選後、32G回してヤメ。"})
            else:
                 plans.append({"title": "✋ 様子見推奨 (STOP)", "color": "#777", "desc": "狙い目ラインに達していません。", "type": "STOP"})

    elif "BLACK" in model:
        ceiling_target = 999
        if total_yuuri >= 1500:
             cost_str = f'<div style="color: #FFD700;">¥ 1,000 〜 ¥ {calc_investment(current_g, ceiling_target):,}</div>'
             plans.append({"title": "⚫ BLACK 有利区間狙い (1500G~)", "color": "#FFD700", "desc": "ここから黒ドキチャンス！", "target_g": ceiling_target, "cost_str": cost_str, "type": "CEILING", "action": "<b>【やめどき】</b>連チャン終了後、32G回して即ヤメ。"})
        else:
            if through_count == 1 and prev_hit_g <= 150 and current_g >= 300:
                cost_str = f'<div style="color: #00ff00;">¥ 1,000 〜 ¥ {calc_investment(current_g, ceiling_target):,}</div>'
                plans.append({"title": "🔥 朝一1スルー 通常B狙い", "color": "#00ff00", "desc": "通常B滞在の期待大！", "target_g": ceiling_target, "cost_str": cost_str, "type": "MODE_B", "action": "<b>【やめどき】</b>天国抜け後、32G回してヤメ。"})
            else:
                border = {1:600, 2:610}.get(through_count, 580) if through_count > 0 else 670
                if current_g >= border:
                    plans.append({"title": f"📈 {through_count}スルー 天井狙い", "color": "#00ff00", "desc": f"ボーダー: {border}G", "target_g": ceiling_target, "type": "CEILING", "action": "<b>【やめどき】</b>天国抜け後、32G回してヤメ。"})
                elif is_reset and current_g <= 32:
                    plans.append({"title": "🔄 リセット32Gカニ歩き", "color":"#00ff00", "desc":"32Gまで", "target_g":32, "cost": calc_investment(current_g, 32), "type":"ZONE", "action": "32Gで当たらなければ即ヤメ。"})
                else:
                    plans.append({"title": "✋ まだ早いです", "color": "#777", "desc": f"狙い目には達していません。", "type": "STOP"})

    elif "ゴージャス" in model:
        ceiling_target = 999
        if total_yuuri >= 2300:
            cost_str = f'<div style="color: #FFD700;">¥ 1,000 〜 ¥ {calc_investment(current_g, ceiling_target):,}</div>'
            plans.append({"title": "💎 ゴージャス 有利区間狙い", "color": "#FFD700", "desc": "いつ当たっても区間切れチャンス！", "target_g": ceiling_target, "cost_str": cost_str, "type": "CEILING", "action": "<b>【やめどき】</b>天国連チャン終了後、32G回して即ヤメ。"})
        elif is_reset and current_g <= 32:
             plans.append({"title": "🔄 リセット32Gカニ歩き", "color": "#00ff00", "desc": "32Gまで。", "target_g": 32, "cost": calc_investment(current_g, 32), "type": "ZONE", "action": "<b>32Gで当たらなければ即ヤメ</b>"})
        elif current_g >= 600:
             plans.append({"title": "📈 天井狙い (600G~)", "color": "#00ff00", "desc": "600GからGO判定。", "target_g": ceiling_target, "type": "CEILING", "action": "<b>【やめどき】</b>当選後、32G回してヤメ。"})
        else:
            plans.append({"title": "✋ STOP", "color": "#777", "desc": "有利区間2300G、または通常600Gから。", "type": "STOP"})

    for plan in plans:
        if plan.get("type") not in ["STOP", "INFO"]:
            cost_txt = plan.get("cost_str", f"¥ {plan.get('cost', calc_investment(current_g, plan['target_g'])):,}")
        else:
            cost_txt = "---"

        action_html = f"<div style='background-color: rgba(255, 255, 255, 0.05); padding: 12px; border-radius: 8px; margin-top: 15px; border-left: 4px solid {plan['color']};'><div style='font-size: 0.9em; font-weight: bold; color: {plan['color']}; margin-bottom: 5px;'>▶ 予想される展開・やめどき</div><div style='font-size: 0.9em; line-height: 1.5; color: #eee;'>{plan.get('action', '')}</div></div>" if 'action' in plan else ""

        st.markdown(f"<div class='strategy-box' style='border-color: {plan['color']};'><span class='plan-title' style='color: {plan['color']};'>{plan['title']}</span><div style='color: #ddd; margin-bottom: 10px;'>{plan['desc']}</div><div style='display:flex; justify-content:space-between; align-items:end;'><div><span class='tag'>{plan['type']}</span>Target: <b>{plan.get('target_g', '---')} G</b></div><div style='text-align:right;'>投資目安<br><span class='cost-display' style='color: {plan['color']};'>{cost_txt}</span></div></div>{action_html}</div>", unsafe_allow_html=True)
