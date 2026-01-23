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
    .streamlit-expanderHeader {
        background-color: #302b63;
        color: #FFD700;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# サイドバー
st.sidebar.title("MENU")
mode = st.sidebar.radio("機種タイプを選択", ["① 時短なし (スマパチ・ST機)", "② 時短あり (エヴァ・海など)"])

if mode == "① 時短なし (スマパチ・ST機)":
    st.title("🎰 SITE7 PRO ANALYZER (ST)")
else:
    st.title("🎰 SITE7 PRO ANALYZER (JITAN)")

# ... (ここから先は前回のコードの「with st.expander...」以降をすべて貼り付けてください)
# ※文字数制限のため省略していますが、前回お渡しした「決定版」の続きをそのまま貼り付ければOKです！
# ※「関数定義」～「メイン画面レイアウト」まで全てです。
