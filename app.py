"""ライブ配信チャット分析ツール - Streamlitメインアプリ"""
import streamlit as st
import tempfile
import os
import pickle
import glob
import time
import base64
import sys
import re
import pandas as pd
from datetime import datetime
from typing import Dict, Optional
from utils.csv_processor import (
    load_csv,
    validate_and_process_data,
    extract_questions,
    load_csv_with_elapsed_time
)
from utils.ai_analyzer import analyze_all_comments
from utils.google_sheets import (
    calculate_statistics,
    calculate_question_statistics
)
from utils.api_key_manager import render_api_key_input, get_active_api_key
from config import COMPANIES, DEFAULT_COMPANY, get_company_config


def inject_custom_css():
    """カスタムCSSを注入"""
    css_file_path = os.path.join(os.path.dirname(__file__), "styles", "custom.css")

    if os.path.exists(css_file_path):
        with open(css_file_path, "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def remove_live_name_from_filename(filename: str) -> str:
    """
    ファイル名から「_(ライブ配信の名前)」の部分を削除
    
    Args:
        filename: 元のファイル名
        
    Returns:
        ライブ配信名を削除したファイル名
    """
    # パターン1: _(...) または _(...)（半角括弧のみ）
    filename = re.sub(r'_\s*\([^)]*\)', '', filename)
    # パターン2: （...） または （...）（全角括弧のみ）
    filename = re.sub(r'[_\s　]（[^）]*）', '', filename)
    # パターン3: 全角スペースまたはアンダースコア + 半角開き括弧 + 任意の文字 + 全角閉じ括弧（混在パターン）
    filename = re.sub(r'[_\s　]\([^）]*）', '', filename)
    # パターン4: 全角スペースまたはアンダースコア + 全角開き括弧 + 任意の文字 + 半角閉じ括弧（混在パターン）
    filename = re.sub(r'[_\s　]（[^)]*\)', '', filename)
    # パターン5: 末尾の空白を削除
    filename = filename.strip()
    return filename


def calculate_api_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """
    API使用料金を計算（GPT-4o-mini）
    
    Args:
        prompt_tokens: 入力トークン数
        completion_tokens: 出力トークン数
        
    Returns:
        推定費用（ドル）
    """
    INPUT_COST_PER_MILLION = 0.15  # $0.15 per 1M tokens
    OUTPUT_COST_PER_MILLION = 0.60  # $0.60 per 1M tokens
    
    input_cost = (prompt_tokens / 1_000_000) * INPUT_COST_PER_MILLION
    output_cost = (completion_tokens / 1_000_000) * OUTPUT_COST_PER_MILLION
    
    return input_cost + output_cost


def create_download_link(data: bytes, filename: str, mime_type: str) -> str:
    """
    Base64エンコードしたダウンロードリンクを作成
    
    Args:
        data: ファイルデータ（バイト）
        filename: ファイル名
        mime_type: MIMEタイプ
        
    Returns:
        HTMLリンク文字列
    """
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:{mime_type};base64,{b64}" download="{filename}" style="color: #1f77b4; text-decoration: underline; font-weight: bold;">📥 {filename}</a>'
    return href


def generate_completed_csv(df: pd.DataFrame, stats: Dict) -> str:
    """
    分析結果CSV形式で出力する関数
    
    Args:
        df: データフレーム（配信時間, username, original_text, チャットの属性, チャット感情を含む）
        stats: 統計情報
        
    Returns:
        分析結果CSV文字列
    """
    # 統計情報をCSV形式の文字列として作成
    stats_lines = []
    
    # 1行目: 統計情報,件数
    stats_lines.append("統計情報,件数")
    
    # 2行目: 全コメント件数,{件数}
    stats_lines.append(f"全コメント件数,{stats.get('total_comments', 0)}")
    
    # 空行
    stats_lines.append("")
    
    # 4行目: 属性,件数,,チャット感情別件数,件数,,ユーザーコメント数ランキング,コメント数
    stats_lines.append("属性,件数,,チャット感情別件数,件数,,ユーザーコメント数ランキング,コメント数")
    
    # 属性別件数、感情別件数、ランキングを取得
    from config import CHAT_ATTRIBUTES, CHAT_SENTIMENTS
    attribute_counts = stats.get('attribute_counts', {})
    sentiment_counts = stats.get('sentiment_counts', {})
    
    # すべての属性カテゴリを含む辞書を作成（存在しないものは0）
    all_attribute_counts = {}
    for attr in CHAT_ATTRIBUTES:
        all_attribute_counts[attr] = attribute_counts.get(attr, 0)
    
    # すべての感情カテゴリを含む辞書を作成（存在しないものは0）
    all_sentiment_counts = {}
    for sent in CHAT_SENTIMENTS:
        all_sentiment_counts[sent] = sentiment_counts.get(sent, 0)
    
    # ユーザーコメント数ランキング（上位10名）
    user_counts = {}
    if 'username' in df.columns:
        user_counts = df['username'].value_counts().head(10).to_dict()
    
    # 最大行数を計算（属性、感情、ランキングの最大値）
    max_rows = max(
        len(all_attribute_counts),
        len(all_sentiment_counts),
        len(user_counts),
        1  # 最小1行
    )
    
    # 属性、感情、ランキングをリストに変換（順序保持）
    attr_items = list(all_attribute_counts.items())
    sentiment_items = list(all_sentiment_counts.items())
    user_items = list(user_counts.items())
    
    # データ行を生成（横並び形式）
    for i in range(max_rows):
        # 属性
        if i < len(attr_items):
            attr_name, attr_count = attr_items[i]
            attr_part = f"{attr_name},{attr_count}"
        else:
            attr_part = ","
        
        # 空白列
        empty_col = ""
        
        # チャット感情別件数
        if i < len(sentiment_items):
            sent_name, sent_count = sentiment_items[i]
            sentiment_part = f"{sent_name},{sent_count}"
        else:
            sentiment_part = ","
        
        # 空白列
        empty_col2 = ""
        
        # ユーザーコメント数ランキング
        if i < len(user_items):
            user_name, user_count = user_items[i]
            user_part = f"{user_name},{user_count}"
        else:
            user_part = ","
        
        # 1行に結合
        row = f"{attr_part},,{sentiment_part},,{user_part}"
        stats_lines.append(row)
    
    # 空行を追加
    stats_lines.append("")
    stats_lines.append("")
    
    # コメントデータセクション
    stats_lines.append("コメントデータ")
    
    # 必要な列のみを選択（配信時間, username, original_text, チャットの属性, チャット感情）
    output_columns = ['配信時間', 'username', 'original_text', 'チャットの属性', 'チャット感情']
    available_columns = [col for col in output_columns if col in df.columns]
    
    # データフレームを必要な列のみに絞る
    output_df = df[available_columns].copy()
    
    # 列名を確認し、配信時間がない場合は inserted_at を使用
    if '配信時間' not in output_df.columns and 'inserted_at' in df.columns:
        output_df['配信時間'] = df['inserted_at']
        output_df = output_df[output_columns]
    
    # 配信時間で昇順ソート
    if '配信時間' in output_df.columns:
        # 配信時間をパースしてソート（HH:MM形式、後方互換性のためHH:MM:SSにも対応）
        def parse_time(time_str):
            try:
                parts = str(time_str).split(':')
                if len(parts) >= 3:
                    hours, minutes, seconds = int(parts[0]), int(parts[1]), int(parts[2])
                    return hours * 3600 + minutes * 60 + seconds
                elif len(parts) == 2:
                    hours, minutes = int(parts[0]), int(parts[1])
                    return hours * 3600 + minutes * 60
                return 0
            except (ValueError, IndexError):
                return 0
        
        output_df['_sort_time'] = output_df['配信時間'].apply(parse_time)
        output_df = output_df.sort_values('_sort_time', ascending=True)
        output_df = output_df.drop(columns=['_sort_time'])
    
    # 統計情報をCSV文字列に変換
    stats_csv = "\n".join(stats_lines)
    
    # データフレームをCSV文字列に変換
    data_csv = output_df.to_csv(index=False)
    
    # 統計情報とデータを結合
    combined_csv = stats_csv + "\n" + data_csv
    
    return combined_csv


def add_statistics_to_csv(df: pd.DataFrame, stats: Dict, is_question: bool = False, question_stats: Optional[Dict] = None) -> str:
    """
    CSVに統計情報を追加（グラフ作成しやすいレイアウト）
    
    Args:
        df: データフレーム
        stats: 統計情報
        is_question: 質問CSVかどうか
        question_stats: 質問統計情報（質問CSVの場合のみ）
        
    Returns:
        統計情報が追加されたCSV文字列
    """
    # 統計情報をCSV形式の文字列として作成
    stats_lines = []
    
    if is_question and question_stats:
        # 質問CSV用の統計情報
        stats_lines.append("統計情報")
        stats_lines.append(f"質問コメント件数,{question_stats.get('total_questions', 0)}")
        stats_lines.append(f"質問回答率,{question_stats.get('answer_rate', 0.0):.1f}%")
        stats_lines.append("")  # 空行
        stats_lines.append("質問コメントデータ")
    else:
        # メインCSV用の統計情報
        stats_lines.append("統計情報")
        stats_lines.append(f"全コメント件数,{stats.get('total_comments', 0)}")
        stats_lines.append("")  # 空行
        stats_lines.append("チャットの属性別件数")
        stats_lines.append("属性,件数")
        for attr, count in stats.get('attribute_counts', {}).items():
            stats_lines.append(f"{attr},{count}")
        stats_lines.append("")  # 空行
        stats_lines.append("チャット感情別件数")
        stats_lines.append("感情,件数")
        for sentiment, count in stats.get('sentiment_counts', {}).items():
            stats_lines.append(f"{sentiment},{count}")
        stats_lines.append("")  # 空行
        
        # ユーザーコメント数ランキング（上位10名）
        if 'username' in df.columns:
            user_counts = df['username'].value_counts().head(10)
            stats_lines.append("ユーザーコメント数ランキング")
            stats_lines.append("ユーザー名,コメント数")
            for username, count in user_counts.items():
                stats_lines.append(f"{username},{count}")
            stats_lines.append("")  # 空行
        
        stats_lines.append("コメントデータ")
    
    # 統計情報をCSV文字列に変換
    stats_csv = "\n".join(stats_lines)
    
    # データフレームをCSV文字列に変換
    data_csv = df.to_csv(index=False)
    
    # 統計情報とデータを結合
    combined_csv = stats_csv + "\n" + data_csv
    
    return combined_csv


def format_remaining_time(seconds: float) -> str:
    """
    残り時間（秒）を「あと◯分◯秒」形式に変換
    
    Args:
        seconds: 残り時間（秒）
        
    Returns:
        フォーマットされた残り時間の文字列
    """
    if seconds < 0:
        return "あと0秒"
    
    total_seconds = int(seconds)
    
    # 1時間以上の場合
    if total_seconds >= 3600:
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        return f"あと{hours}時間{minutes}分"
    
    # 1分以上1時間未満の場合
    elif total_seconds >= 60:
        minutes = total_seconds // 60
        secs = total_seconds % 60
        return f"あと{minutes}分{secs}秒"
    
    # 1分未満の場合
    else:
        return f"あと{total_seconds}秒"


def main():
    st.set_page_config(
        page_title="ライブ配信チャット分析ツール",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # カスタムCSSを注入
    inject_custom_css()

    # サイドバー: APIキー設定
    with st.sidebar:
        has_api_key = render_api_key_input()
        st.divider()

    # 機能選択（サイドバー）
    st.sidebar.title("機能選択")
    selected_feature = st.sidebar.radio(
        "使用する機能を選択してください",
        ["コメント分析機能", "質問回答判定機能"],
        index=0
    )
    
    # APIキーが設定されていない場合の警告
    if not has_api_key:
        st.warning("分析を実行するにはAPIキーの設定が必要です。サイドバーからAPIキーを設定してください。")
        st.info("[OpenAI APIキーの取得はこちら](https://platform.openai.com/api-keys)")
        st.stop()

    # 選択された機能に応じてページを表示
    if selected_feature == "質問回答判定機能":
        # 質問回答判定機能のページを表示
        show_question_answer_page()
        return
    
    # 既存のコメント分析機能のページを表示
    show_comment_analysis_page()


def show_comment_analysis_page():
    """コメント分析機能のページを表示"""
    
    # セッションステートの初期化
    if "processed_data" not in st.session_state:
        st.session_state.processed_data = None
    if "analysis_complete" not in st.session_state:
        st.session_state.analysis_complete = False
    if "analysis_save_path" not in st.session_state:
        st.session_state.analysis_save_path = None
    if "analysis_original_df" not in st.session_state:
        st.session_state.analysis_original_df = None
    if "analysis_cancelled" not in st.session_state:
        st.session_state.analysis_cancelled = False
    if "csv_completed_data" not in st.session_state:
        st.session_state.csv_completed_data = None
    if "csv_completed_filename" not in st.session_state:
        st.session_state.csv_completed_filename = None
    if "stats_data" not in st.session_state:
        st.session_state.stats_data = None
    if "question_stats_data" not in st.session_state:
        st.session_state.question_stats_data = None
    if "question_df_data" not in st.session_state:
        st.session_state.question_df_data = None
    if "uploaded_csv_filename" not in st.session_state:
        st.session_state.uploaded_csv_filename = ""
    if "api_usage" not in st.session_state:
        st.session_state.api_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0
        }
    if "selected_company" not in st.session_state:
        st.session_state.selected_company = DEFAULT_COMPANY
    
    # サイドバー: API使用状況（分析完了時のみ表示）
    with st.sidebar:
        if st.session_state.get("analysis_complete") and st.session_state.get("api_usage") and st.session_state.api_usage["total_tokens"] > 0:
            st.divider()
            st.subheader("API使用状況")
            usage = st.session_state.api_usage
            st.metric("使用トークン数", f"{usage['total_tokens']:,}")
            st.write(f"入力: {usage['prompt_tokens']:,} トークン")
            st.write(f"出力: {usage['completion_tokens']:,} トークン")
            st.metric("推定費用", f"${usage['estimated_cost_usd']:.4f}")
            st.caption("モデル: GPT-4o Mini")
    
    st.title("ライブ配信チャット分析ツール")
    
    # CSVファイルアップロード
    st.header("1. CSVファイルのアップロード")
    st.info("💡 CSVファイルをドラッグアンドドロップするか、クリックしてファイルを選択してください。")
    uploaded_file = st.file_uploader(
        "📄 CSVファイルをアップロード（ドラッグ&ドロップ可）",
        type=["csv"],
        help="CSVファイルをドラッグアンドドロップするか、クリックして選択してください。必要な列: guest_id, username, original_text, inserted_at"
    )
    
    if uploaded_file is not None:
        try:
            # アップロードされたファイル名を保存（拡張子なし）
            uploaded_filename = uploaded_file.name
            if uploaded_filename.endswith('.csv'):
                uploaded_filename_base = uploaded_filename[:-4]  # .csvを除去
            else:
                uploaded_filename_base = uploaded_filename
            # ライブ配信名を削除
            uploaded_filename_base = remove_live_name_from_filename(uploaded_filename_base)
            st.session_state.uploaded_csv_filename = uploaded_filename_base
            
            # 一時ファイルに保存
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # CSVを読み込んで処理
            with st.spinner("CSVファイルを読み込んでいます..."):
                # elapsed_timeカラムがあるかどうかをチェック
                try:
                    # まずファイルを読み込んでelapsed_timeカラムがあるかチェック
                    test_df = pd.read_csv(tmp_path, encoding='utf-8-sig', nrows=1)
                    has_elapsed_time = 'elapsed_time' in test_df.columns
                    
                    if has_elapsed_time:
                        # elapsed_timeカラムがある場合は新しい処理を使用
                        df = load_csv_with_elapsed_time(tmp_path)
                    else:
                        # elapsed_timeカラムがない場合は既存の処理を使用
                        df = load_csv(tmp_path)
                        df = validate_and_process_data(df)
                except Exception as e:
                    # エラーが発生した場合は既存の処理にフォールバック
                    st.warning(f"elapsed_timeカラムの検出中にエラーが発生しました。既存の処理を使用します: {str(e)}")
                    df = load_csv(tmp_path)
                    df = validate_and_process_data(df)
                
                st.session_state.processed_data = df
                st.session_state.analysis_complete = False
            
            # データプレビュー
            st.success(f"✓ {len(df)}件のコメントを読み込みました")
            st.subheader("データプレビュー")
            st.dataframe(df.head(10), use_container_width=True)
            
            # 一時ファイルを削除
            os.unlink(tmp_path)
            
        except Exception as e:
            st.error(f"エラー: {str(e)}")
            return
    
    # 企業選択（メインエリアに移動）
    st.header("2. 企業選択")
    company_names = list(COMPANIES.keys())
    
    selected_company = st.selectbox(
        "企業を選択してください",
        company_names,
        index=company_names.index(st.session_state.selected_company) if st.session_state.selected_company in company_names else 0
    )
    
    # 企業選択が変更された場合、セッションステートを更新
    if selected_company != st.session_state.selected_company:
        st.session_state.selected_company = selected_company
        # 分析結果をクリア（企業が変わったら再分析が必要）
        if "analysis_complete" in st.session_state:
            st.session_state.analysis_complete = False
        # 注意: processed_dataは保持する（アップロード済みのCSVデータは残す）
        # 分析結果だけをクリアするため、stats_dataなどもクリア
        if "stats_data" in st.session_state:
            st.session_state.stats_data = None
        if "question_stats_data" in st.session_state:
            st.session_state.question_stats_data = None
        if "question_df_data" in st.session_state:
            st.session_state.question_df_data = None
        if "csv_completed_data" in st.session_state:
            st.session_state.csv_completed_data = None
    
    # 現在の企業設定を取得
    company_config = get_company_config(selected_company)
    st.info(f"**選択中の企業**: {company_config['name']}")
    
    # AI分析
    if st.session_state.processed_data is not None and not st.session_state.analysis_complete:
        st.header("3. AI分析")
        
        df = st.session_state.processed_data.copy()
        
        # 分析途中の結果があるかチェック（PCスリープ対策）
        analysis_resume_available = False
        saved_count = 0
        
        # 保存ファイルのパスを検索（セッションステートにない場合でも検索）
        if not st.session_state.analysis_save_path:
            # 一時ディレクトリから最新の保存ファイルを検索
            save_dir = tempfile.gettempdir()
            save_files = glob.glob(os.path.join(save_dir, "analysis_save_*.pkl"))
            if save_files:
                # 最新のファイルを使用
                latest_file = max(save_files, key=os.path.getmtime)
                st.session_state.analysis_save_path = latest_file
        
        if st.session_state.analysis_save_path and os.path.exists(st.session_state.analysis_save_path):
            try:
                with open(st.session_state.analysis_save_path, 'rb') as f:
                    saved_data = pickle.load(f)
                    if saved_data:
                        if isinstance(saved_data, list):
                            saved_count = len(saved_data)
                        elif isinstance(saved_data, pd.DataFrame):
                            saved_count = len(saved_data)
                        if saved_count > 0:
                            analysis_resume_available = True
            except Exception:
                pass
        
        if analysis_resume_available:
            st.warning(f"⚠️ 分析が途中で中断されました。{saved_count}件の分析結果が保存されています。続きから再開できます。")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("続きから再開", type="primary"):
                    st.session_state.analysis_resume = True
                    # 元のデータフレームを確保
                    if st.session_state.analysis_original_df is None:
                        st.session_state.analysis_original_df = df.copy()
                    st.rerun()
            with col2:
                if st.button("最初から開始"):
                    # 保存ファイルを削除
                    if st.session_state.analysis_save_path and os.path.exists(st.session_state.analysis_save_path):
                        try:
                            os.remove(st.session_state.analysis_save_path)
                        except Exception:
                            pass
                    st.session_state.analysis_resume = False
                    st.session_state.analysis_save_path = None
                    st.session_state.analysis_original_df = None
                    st.rerun()
        
        # 分析開始・中断ボタン
        col1, col2 = st.columns(2)
        with col1:
            start_analysis = st.button("分析を開始", type="primary")
        with col2:
            cancel_analysis = st.button("分析を中断", type="secondary", disabled=st.session_state.analysis_complete)
        
        # 中断ボタンが押された場合
        if cancel_analysis:
            st.session_state.analysis_cancelled = True
            st.warning("⚠️ 分析の中断をリクエストしました。現在処理中のコメントが完了次第、分析が中断されます。")
        
        # 分析開始
        if start_analysis or st.session_state.get("analysis_resume", False):
            # APIキー事前チェック
            try:
                from config import get_openai_api_key
                if not get_openai_api_key():
                    st.error("OpenAI APIキーが未設定です。サイドバーから設定してください。")
                    return
            except Exception:
                st.error("APIキーの取得に失敗しました。再度キーを入力してください。")
                return

            # 中断フラグをリセット
            st.session_state.analysis_cancelled = False
            # トークン使用量をリセット（新しい分析開始時のみ）
            if start_analysis:
                st.session_state.api_usage = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "estimated_cost_usd": 0.0
                }
            # 一時ファイルのパスを設定（PCスリープ対策）
            if not st.session_state.analysis_save_path:
                save_dir = tempfile.gettempdir()
                save_filename = f"analysis_save_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                st.session_state.analysis_save_path = os.path.join(save_dir, save_filename)
            
            # 元のデータフレームを保存（再開時に使用）
            if st.session_state.analysis_original_df is None:
                st.session_state.analysis_original_df = df.copy()
            else:
                df = st.session_state.analysis_original_df.copy()
            
            # プログレスバー
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 開始時刻を記録
            start_time = time.time()
            
            def update_progress(current, total):
                progress = current / total
                progress_bar.progress(progress)
                
                # 経過時間の計算
                elapsed_time = time.time() - start_time
                elapsed_seconds = int(elapsed_time)
                hours = elapsed_seconds // 3600
                minutes = (elapsed_seconds % 3600) // 60
                seconds = elapsed_seconds % 60
                elapsed_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                
                # 予想完了時間の計算
                if current > 0:
                    avg_time_per_item = elapsed_time / current
                    remaining_items = total - current
                    estimated_remaining = avg_time_per_item * remaining_items
                    estimated_str = format_remaining_time(estimated_remaining)
                    
                    status_text.text(
                        f"進行中: {current}/{total} ({progress*100:.1f}%)\n"
                        f"経過時間: {elapsed_str}\n"
                        f"予想完了時間: {estimated_str}"
                    )
                else:
                    status_text.text(f"進行中: {current}/{total} ({progress*100:.1f}%)")
            
            def save_intermediate_results(action, results=None):
                """中間結果を保存（PCスリープ対策）"""
                save_path = st.session_state.analysis_save_path
                
                if action == "save" and results is not None:
                    # 結果を保存
                    try:
                        with open(save_path, 'wb') as f:
                            pickle.dump(results, f)
                    except Exception as e:
                        print(f"保存エラー: {e}")
                elif action == "load":
                    # 保存された結果を読み込む
                    if save_path and os.path.exists(save_path):
                        try:
                            with open(save_path, 'rb') as f:
                                saved_results = pickle.load(f)
                                return saved_results
                        except Exception as e:
                            print(f"読み込みエラー: {e}")
                    return None
                elif action == "clear":
                    # 一時ファイルを削除
                    if save_path and os.path.exists(save_path):
                        try:
                            os.remove(save_path)
                        except Exception:
                            pass
            
            def check_cancel():
                """中断フラグをチェック"""
                return st.session_state.get("analysis_cancelled", False)
            
            try:
                # AI分析実行（統合プロンプト使用：50%高速化）
                with st.spinner("AI分析を実行中です。しばらくお待ちください..."):
                    analysis_result = analyze_all_comments(df, update_progress, save_intermediate_results, check_cancel)
                
                # 分析結果からDataFrameとトークン使用量情報を取得
                if isinstance(analysis_result, dict):
                    analyzed_df = analysis_result["df"]
                    api_usage_info = analysis_result.get("api_usage", {})
                else:
                    # 後方互換性のため、DataFrameが直接返された場合
                    analyzed_df = analysis_result
                    api_usage_info = {}
                
                st.session_state.processed_data = analyzed_df
                st.session_state.analysis_complete = True
                st.session_state.analysis_resume = False
                st.session_state.analysis_original_df = None
                st.session_state.analysis_cancelled = False  # 完了時に中断フラグをクリア
                
                # トークン使用量情報をセッションステートに保存
                if api_usage_info:
                    prompt_tokens = api_usage_info.get("prompt_tokens", 0)
                    completion_tokens = api_usage_info.get("completion_tokens", 0)
                    total_tokens = api_usage_info.get("total_tokens", 0)
                    estimated_cost = calculate_api_cost(prompt_tokens, completion_tokens)
                    
                    st.session_state.api_usage = {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                        "estimated_cost_usd": estimated_cost
                    }
                    
                    # デバッグ情報を出力（トークン使用量が0の場合の原因特定用）
                    if total_tokens == 0:
                        st.warning(f"⚠️ トークン使用量が0です。分析されたコメント数: {len(analyzed_df)}")
                else:
                    # api_usage_infoが空の場合の警告
                    st.warning("⚠️ API使用量情報が取得できませんでした。")
                
                # CSVファイルを自動生成（分析完了時に自動実行）
                try:
                    # デフォルトのファイル名を生成（アップロードされたファイル名を含む）
                    uploaded_filename_base = st.session_state.get("uploaded_csv_filename", "")
                    if uploaded_filename_base:
                        default_file_title = f"コメント分析_{uploaded_filename_base}"
                    else:
                        default_file_title = "コメント分析"
                    
                    # 統計情報を計算（後でCSVに追加するため）
                    temp_stats = calculate_statistics(analyzed_df)
                    
                    # 分析結果CSV形式で出力
                    try:
                        # 分析結果CSV形式で出力
                        completed_csv = generate_completed_csv(analyzed_df, temp_stats)
                        st.session_state.csv_completed_data = completed_csv.encode('utf-8-sig')
                        st.session_state.csv_completed_filename = f"{default_file_title}_分析結果.csv"
                    except Exception as e:
                        # 分析結果CSV生成エラーは無視（後で再生成可能）
                        print(f"分析結果CSV生成エラー: {e}")
                except Exception as e:
                    # CSV生成エラーは無視（後で再生成可能）
                    print(f"CSV自動生成エラー: {e}")
                
                # 統計情報を計算してセッションステートに保存
                question_df = extract_questions(analyzed_df)
                question_df["回答状況"] = "未回答"
                st.session_state.stats_data = calculate_statistics(analyzed_df)
                st.session_state.question_stats_data = calculate_question_statistics(question_df)
                st.session_state.question_df_data = question_df
                
                progress_bar.progress(1.0)
                status_text.text("✓ 分析が完了しました！")
                st.success("分析が完了しました！")
                
                # 通知音を再生
                st.components.v1.html("""
                <script>
                // ビープ音を再生する関数
                function playBeep() {
                    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                    const oscillator = audioContext.createOscillator();
                    const gainNode = audioContext.createGain();
                    
                    oscillator.connect(gainNode);
                    gainNode.connect(audioContext.destination);
                    
                    oscillator.frequency.value = 800; // 周波数（Hz）
                    oscillator.type = 'sine';
                    
                    gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
                    gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.5);
                    
                    oscillator.start(audioContext.currentTime);
                    oscillator.stop(audioContext.currentTime + 0.5);
                }
                
                // 音を再生
                playBeep();
                </script>
                """, height=0)
                
                # 分析結果のプレビュー
                st.subheader("分析結果プレビュー")
                st.dataframe(analyzed_df.head(10), use_container_width=True)
                
            except KeyboardInterrupt:
                # 中断がリクエストされた場合
                st.session_state.analysis_cancelled = True
                st.warning("⚠️ 分析が中断されました。")
                st.info("💡 「続きから再開」ボタンを使用して、中断した箇所から分析を再開できます。")
                # 中断フラグをクリアして、次回の再開時に問題がないようにする
                st.session_state.analysis_cancelled = False
                st.rerun()
            except Exception as e:
                # その他のエラー
                error_message = str(e)
                if "中断" in error_message or "KeyboardInterrupt" in error_message:
                    # 中断関連のエラーの場合
                    st.session_state.analysis_cancelled = True
                    st.warning("⚠️ 分析が中断されました。")
                    st.info("💡 「続きから再開」ボタンを使用して、中断した箇所から分析を再開できます。")
                    st.session_state.analysis_cancelled = False
                    st.rerun()
                else:
                    # その他のエラー
                    st.error(f"分析エラー: {error_message}")
                    st.info("💡 PCがスリープした場合は、ページをリロードして「続きから再開」ボタンを使用してください。")
                    import traceback
                    with st.expander("詳細なエラー情報"):
                        st.code(traceback.format_exc())
                    return
    
    # データ出力
    if st.session_state.analysis_complete and st.session_state.processed_data is not None:
        st.header("3. データ出力")
        
        # 統計情報をセッションステートから取得（なければ計算）
        if st.session_state.stats_data is None:
            df = st.session_state.processed_data.copy()
            question_df = extract_questions(df)
            question_df["回答状況"] = "未回答"
            st.session_state.stats_data = calculate_statistics(df)
            st.session_state.question_stats_data = calculate_question_statistics(question_df)
            st.session_state.question_df_data = question_df
        
        df = st.session_state.processed_data.copy()
        stats = st.session_state.stats_data
        question_stats = st.session_state.question_stats_data
        question_df = st.session_state.question_df_data
        
        # 統計情報を常に表示（エラー時も表示される）
        st.subheader("統計情報")
        
        stat_col1, stat_col2 = st.columns(2)
        
        with stat_col1:
            st.metric("全コメント件数", stats["total_comments"])
            st.write("**チャットの属性別件数**")
            for attr, count in stats["attribute_counts"].items():
                st.write(f"- {attr}: {count}件")
        
        with stat_col2:
            if len(question_df) > 0:
                # 質問統計情報が計算されている場合のみ表示
                if question_stats is not None and "total_questions" in question_stats:
                    st.metric("質問コメント件数", question_stats["total_questions"])
                else:
                    st.metric("質問コメント件数", len(question_df))
            else:
                st.info("質問コメントはありませんでした。")
            
            st.write("**チャット感情別件数**")
            for sentiment, count in stats["sentiment_counts"].items():
                st.write(f"- {sentiment}: {count}件")
        
        st.markdown("---")
        
        # CSVダウンロードリンク（分析完了時に自動生成される）
        st.subheader("📥 CSVファイルをダウンロード")
        
        # ファイル名を変更したい場合の入力欄（オプション）
        uploaded_filename_base = st.session_state.get("uploaded_csv_filename", "")
        if uploaded_filename_base:
            default_file_title = f"コメント分析_{uploaded_filename_base}"
        else:
            default_file_title = "コメント分析"
        
        file_title = st.text_input(
            "ファイル名を変更（拡張子なし、変更しない場合はそのまま）",
            value=default_file_title,
            key="csv_filename_input"
        )
        
        # ファイル名が変更された場合は、CSVファイルを再生成
        if file_title and ("csv_completed_filename" not in st.session_state or 
                          not st.session_state.csv_completed_filename or 
                          file_title not in st.session_state.csv_completed_filename):
            try:
                # 分析結果CSVを再生成
                completed_csv = generate_completed_csv(df, stats)
                st.session_state.csv_completed_data = completed_csv.encode('utf-8-sig')
                st.session_state.csv_completed_filename = f"{file_title}_分析結果.csv"
            except Exception as e:
                st.error(f"CSVファイル生成エラー: {str(e)}")
        
        # 分析結果CSVダウンロードリンク
        if "csv_completed_data" in st.session_state and st.session_state.csv_completed_data:
            download_link = create_download_link(
                st.session_state.csv_completed_data,
                st.session_state.csv_completed_filename,
                "text/csv"
            )
            st.markdown(f"**分析結果CSV**: {download_link}", unsafe_allow_html=True)
        else:
            # 分析結果CSVがまだ生成されていない場合、生成を試みる
            st.info("💡 分析結果CSVファイルを生成中...")
            try:
                completed_csv = generate_completed_csv(df, stats)
                st.session_state.csv_completed_data = completed_csv.encode('utf-8-sig')
                uploaded_filename_base = st.session_state.get("uploaded_csv_filename", "")
                if uploaded_filename_base:
                    default_file_title = f"コメント分析_{uploaded_filename_base}"
                else:
                    default_file_title = "コメント分析"
                st.session_state.csv_completed_filename = f"{default_file_title}_分析結果.csv"
                download_link = create_download_link(
                    st.session_state.csv_completed_data,
                    st.session_state.csv_completed_filename,
                    "text/csv"
                )
                st.markdown(f"**分析結果CSV**: {download_link}", unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"分析結果CSVファイル生成エラー: {str(e)}")
    
    # フッター
    st.markdown("---")
    # フォルダ名を取得
    folder_name = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    st.markdown(
        f"""
        <div style='text-align: center; color: gray;'>
        <p>{folder_name}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def show_question_answer_page():
    """質問回答判定機能のページを表示"""
    st.header("📝 質問回答判定機能")
    st.info("質問CSVに対して回答状況を判定します。文字起こしテキストデータと人間が判定したCSVデータの両方を使用できます。")
    
    # セッションステートの初期化
    if "question_csv_data" not in st.session_state:
        st.session_state.question_csv_data = None
    if "question_answer_result" not in st.session_state:
        st.session_state.question_answer_result = None
    if "question_answer_csv_data" not in st.session_state:
        st.session_state.question_answer_csv_data = None
    
    # ファイルのアップロード
    st.subheader("1. ファイルのアップロード")
    st.info("💡 ファイルをドラッグアンドドロップするか、クリックしてファイルを選択してください。")
    
    # 縦列で並べる
    transcript_file = st.file_uploader(
        "📄 文字起こしテキストファイルをアップロード（ドラッグ&ドロップ可）",
        type=["txt", "csv"],
        key="transcript_upload",
        help="文字起こしテキストファイルをドラッグアンドドロップするか、クリックして選択してください。"
    )
    
    manual_csv_file = st.file_uploader(
        "📊 人間が判定したCSVファイルをアップロード（ドラッグ&ドロップ可）",
        type=["csv"],
        key="manual_csv_upload",
        help="人間が判定したCSVファイルをドラッグアンドドロップするか、クリックして選択してください。"
    )
    
    question_file = st.file_uploader(
        "❓ 質問CSVファイルをアップロード（ドラッグ&ドロップ可）",
        type=["csv"],
        key="question_csv_upload",
        help="質問CSVファイルをドラッグアンドドロップするか、クリックして選択してください。"
    )
    
    # 判定開始ボタン（質問CSVは必須、他の2つはどちらか1つ以上必要）
    if question_file:
        if transcript_file or manual_csv_file:
            if st.button("判定を開始", type="primary"):
                try:
                    # ファイルを一時保存
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_question:
                        tmp_question.write(question_file.getvalue())
                        question_path = tmp_question.name
                    
                    transcript_path = None
                    manual_path = None
                    
                    if transcript_file:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_transcript:
                            tmp_transcript.write(transcript_file.getvalue())
                            transcript_path = tmp_transcript.name
                    
                    if manual_csv_file:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_manual:
                            tmp_manual.write(manual_csv_file.getvalue())
                            manual_path = tmp_manual.name
                    
                    # 判定処理
                    with st.spinner("回答状況を判定中..."):
                        from utils.transcript_parser import parse_transcript
                        from utils.question_answer_matcher import match_questions_with_transcript, match_questions_with_manual_csv
                        
                        # 質問CSVを読み込み
                        question_df = pd.read_csv(question_path, encoding='utf-8-sig')
                        result_df = question_df.copy()
                        
                        # 回答状況列と回答方法列を初期化
                        result_df['回答状況'] = False
                        result_df['回答方法'] = ''
                        
                        # 文字起こしテキストから判定
                        transcript_data = None
                        if transcript_path:
                            # 文字起こしテキストをパース
                            transcript_data = parse_transcript(transcript_path)
                            
                            # パース結果の統計情報を表示
                            if len(transcript_data) > 0:
                                st.success(f"✅ 文字起こしデータのパース成功: {len(transcript_data)}件の回答箇所を抽出しました")
                                
                                # 話者ごとの統計
                                speaker_counts = {}
                                for answer in transcript_data:
                                    speaker = answer.get('speaker', '不明')
                                    speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
                                
                                if speaker_counts:
                                    speaker_info = ", ".join([f"{speaker}: {count}件" for speaker, count in speaker_counts.items()])
                                    st.info(f"📊 話者別の回答数: {speaker_info}")
                            else:
                                st.warning("⚠️ 文字起こしデータから回答箇所を抽出できませんでした。ファイル形式を確認してください。")
                            
                            # サンプルデータのプレビュー（最初の3件）
                            if len(transcript_data) > 0:
                                with st.expander("🔍 文字起こしデータのサンプル（最初の3件）"):
                                    for i, answer in enumerate(transcript_data[:3]):
                                        st.text(f"【{i+1}】話者: {answer.get('speaker', 'N/A')}, 開始時間: {answer.get('start_time', 'N/A')}, 終了時間: {answer.get('end_time', 'N/A')}")
                                        st.text(f"内容: {answer.get('text', '')[:100]}...")
                                        st.text("---")
                            
                            # 質問と回答を照合
                            transcript_result = match_questions_with_transcript(question_df, transcript_data)
                            
                            # 照合結果の統計
                            transcript_matched = transcript_result['回答状況'].sum()
                            st.info(f"✅ 文字起こしテキストからの照合: {transcript_matched}件の質問が回答済みとして判定されました")
                            
                            # 照合失敗の詳細情報を表示
                            if transcript_matched == 0:
                                st.warning("⚠️ 照合が失敗しました。ターミナルのデバッグ出力を確認してください。")
                            
                            # 結果を統合（回答状況がTRUEの場合は上書き、回答方法も更新）
                            for idx in result_df.index:
                                if transcript_result.at[idx, '回答状況']:
                                    result_df.at[idx, '回答状況'] = True
                                    result_df.at[idx, '回答方法'] = transcript_result.at[idx, '回答方法']
                        
                        # 人間が判定したCSVから判定
                        if manual_path:
                            # ヘッダー行を自動検出する関数
                            def detect_manual_csv_header(file_path: str, expected_columns: list = None) -> int:
                                """人間が判定したCSVのヘッダー行を検出"""
                                if expected_columns is None:
                                    # 「回答済」と「回答済み」の両方を検索対象に含める
                                    expected_columns = ['回答済み', '回答済', 'タイムスタンプ', 'ユーザー名', '質問', '回答方法', '回答']
                                
                                try:
                                    with open(file_path, 'r', encoding='utf-8-sig') as f:
                                        lines = f.readlines()
                                    
                                    # 最初の10行をチェック
                                    for row_idx in range(min(10, len(lines))):
                                        line = lines[row_idx].strip()
                                        # タブ区切りとカンマ区切りの両方を試す
                                        for sep in ['\t', ',']:
                                            columns = [col.strip() for col in line.split(sep)]
                                            # 「質問」列が含まれているか確認（最優先）
                                            if '質問' in columns:
                                                # 期待される列名が含まれているかも確認
                                                if any(col in columns for col in expected_columns):
                                                    print(f"DEBUG: ヘッダー行を検出 - 行{row_idx}: {columns}", file=sys.stderr)
                                                    return row_idx
                                except Exception as e:
                                    print(f"DEBUG: ヘッダー検出エラー: {e}", file=sys.stderr)
                                    pass
                                return 0  # 見つからない場合は0行目を返す
                            
                            # ヘッダー行を検出
                            header_row = detect_manual_csv_header(manual_path)
                            st.info(f"DEBUG: 検出されたヘッダー行: {header_row}行目（0始まりなので、実際は{header_row+1}行目）")
                            
                            # 検出されたヘッダー行の実際の内容を表示
                            try:
                                with open(manual_path, 'r', encoding='utf-8-sig') as f:
                                    lines = f.readlines()
                                    if header_row < len(lines):
                                        header_line = lines[header_row].strip()
                                        st.info(f"DEBUG: 検出されたヘッダー行の内容: {header_line}")
                            except Exception as e:
                                st.warning(f"DEBUG: ヘッダー行の内容を読み込めませんでした: {e}")
                            
                            # 区切り文字を自動検出する関数
                            def detect_delimiter(file_path: str, header_row: int) -> str:
                                """CSVファイルの区切り文字を自動検出"""
                                try:
                                    with open(file_path, 'r', encoding='utf-8-sig') as f:
                                        lines = f.readlines()
                                    
                                    if header_row < len(lines):
                                        header_line = lines[header_row].strip()
                                        
                                        # タブとカンマの数をカウント
                                        tab_count = header_line.count('\t')
                                        comma_count = header_line.count(',')
                                        
                                        print(f"DEBUG: 区切り文字検出 - タブ数: {tab_count}, カンマ数: {comma_count}", file=sys.stderr)
                                        
                                        # より多い方を区切り文字として使用
                                        if tab_count > comma_count and tab_count > 0:
                                            print("DEBUG: タブ区切りを検出", file=sys.stderr)
                                            return '\t'
                                        elif comma_count > 0:
                                            print("DEBUG: カンマ区切りを検出", file=sys.stderr)
                                            return ','
                                        
                                        # どちらもない場合は、次のデータ行をチェック
                                        if header_row + 1 < len(lines):
                                            data_line = lines[header_row + 1].strip()
                                            tab_count = data_line.count('\t')
                                            comma_count = data_line.count(',')
                                            
                                            print(f"DEBUG: データ行で区切り文字検出 - タブ数: {tab_count}, カンマ数: {comma_count}", file=sys.stderr)
                                            
                                            if tab_count > comma_count and tab_count > 0:
                                                return '\t'
                                            elif comma_count > 0:
                                                return ','
                                except Exception as e:
                                    print(f"DEBUG: 区切り文字検出エラー: {e}", file=sys.stderr)
                                    pass
                                
                                # デフォルトはカンマ
                                return ','
                            
                            # 区切り文字を自動検出
                            delimiter = detect_delimiter(manual_path, header_row)
                            
                            # 人間が判定したCSVを読み込み
                            try:
                                manual_df = pd.read_csv(
                                    manual_path, 
                                    encoding='utf-8-sig', 
                                    sep=delimiter, 
                                    header=header_row,
                                    skipinitialspace=True,
                                    on_bad_lines='skip'  # 不正な行をスキップ
                                )
                                
                                print(f"DEBUG: CSV読み込み成功。区切り文字: {repr(delimiter)}, 列数: {len(manual_df.columns)}, 行数: {len(manual_df)}", file=sys.stderr)
                                st.info(f"DEBUG: CSV読み込み成功。区切り文字: {repr(delimiter)}, 列数: {len(manual_df.columns)}, 行数: {len(manual_df)}")
                                st.info(f"DEBUG: 列名: {list(manual_df.columns)}")
                                
                                # 列が1つしかない場合は、手動で分割を試みる
                                if len(manual_df.columns) == 1:
                                    st.warning("⚠️ CSVが正しくパースされていません。手動で分割を試みます...")
                                    
                                    # ヘッダー行を読み込んで列名を取得
                                    with open(manual_path, 'r', encoding='utf-8-sig') as f:
                                        lines = f.readlines()
                                        header_line = lines[header_row].strip()
                                        
                                        # カンマとタブの両方で試す
                                        if ',' in header_line:
                                            new_columns = [col.strip() for col in header_line.split(',')]
                                            delimiter = ','
                                        elif '\t' in header_line:
                                            new_columns = [col.strip() for col in header_line.split('\t')]
                                            delimiter = '\t'
                                        else:
                                            # デフォルトはカンマ
                                            new_columns = [col.strip() for col in header_line.split(',')]
                                            delimiter = ','
                                    
                                    print(f"DEBUG: 手動分割を試みます。区切り文字: {repr(delimiter)}, 列数: {len(new_columns)}", file=sys.stderr)
                                    
                                    # データを再読み込み（列名を手動指定）
                                    manual_df = pd.read_csv(
                                        manual_path,
                                        encoding='utf-8-sig',
                                        sep=delimiter,
                                        header=None,
                                        skiprows=header_row + 1,
                                        names=new_columns,
                                        skipinitialspace=True,
                                        on_bad_lines='skip'
                                    )
                                    
                                    print(f"DEBUG: 手動分割成功。列数: {len(manual_df.columns)}, 行数: {len(manual_df)}", file=sys.stderr)
                                    st.info(f"DEBUG: 手動分割成功。列数: {len(manual_df.columns)}, 列名: {list(manual_df.columns)}")
                                
                            except Exception as e:
                                st.error(f"❌ CSV読み込みエラー: {e}")
                                print(f"DEBUG: CSV読み込みエラー: {e}", file=sys.stderr)
                                st.exception(e)
                                manual_df = None  # エラーが発生した場合は後続の処理をスキップ
                            
                            # manual_dfが正常に読み込まれた場合のみ処理を続行
                            if manual_df is None:
                                st.error("❌ CSVの読み込みに失敗したため、人間が判定したCSVからの照合をスキップします。")
                            else:
                                st.info(f"📋 列名: {', '.join(manual_df.columns.tolist())}")
                                
                                # データのプレビューを表示（最初の5行）
                                with st.expander("🔍 読み込まれたデータのプレビュー（最初の5行）"):
                                    st.dataframe(manual_df.head(5), use_container_width=True)
                                
                                # データ検証: 回答済列の値の分布を表示
                                answered_col = None
                                if '回答済' in manual_df.columns:
                                    answered_col = '回答済'
                                elif '回答済み' in manual_df.columns:
                                    answered_col = '回答済み'
                                
                                if answered_col:
                                    value_counts = manual_df[answered_col].value_counts()
                                    st.info(f"📈 回答済列の値の分布: {dict(value_counts)}")
                                    
                                    # TRUEの行の質問テキストを表示（最初の5件）
                                    true_rows = manual_df[manual_df[answered_col].astype(str).str.upper().isin(['TRUE', '1', 'T', 'YES', 'Y'])]
                                    if len(true_rows) > 0:
                                        question_col = None
                                        for col in ['質問', 'original_text', 'コメント', 'text']:
                                            if col in manual_df.columns:
                                                question_col = col
                                                break
                                        if question_col:
                                            st.info(f"✅ 回答済み（TRUE）の質問数: {len(true_rows)}件")
                                            with st.expander("🔍 回答済み（TRUE）の質問のサンプル（最初の5件）"):
                                                st.dataframe(true_rows[[answered_col, question_col]].head(5), use_container_width=True)
                                else:
                                    st.warning("⚠️ 「回答済」または「回答済み」列が見つかりませんでした。")
                                
                                # 質問と回答を照合（文字起こしテキストがアップロードされている場合は渡す）
                                if transcript_data:
                                    st.info(f"📝 文字起こしテキストも参照して照合します（{len(transcript_data)}件の回答箇所）")
                                
                                manual_result = match_questions_with_manual_csv(question_df, manual_df, transcript_data)
                                
                                # 照合結果の統計
                                manual_matched = manual_result['回答状況'].sum()
                                st.info(f"✅ 人間が判定したCSVからの照合: {manual_matched}件の質問が回答済みとして判定されました")
                                
                                # 照合失敗の詳細情報を表示
                                if manual_matched == 0:
                                    st.warning("⚠️ 照合が失敗しました。ターミナルのデバッグ出力を確認してください。")
                                
                                # 照合結果のプレビュー（最初の5件）
                                if manual_matched > 0:
                                    matched_preview = manual_result[manual_result['回答状況']].head(5)
                                    with st.expander("🔍 照合成功した質問のプレビュー（最初の5件）"):
                                        st.dataframe(matched_preview[['回答状況', '回答方法'] + [col for col in matched_preview.columns if col not in ['回答状況', '回答方法']]], use_container_width=True)
                                
                                # 結果を統合（回答状況がTRUEの場合は上書き、回答方法も更新）
                                for idx in result_df.index:
                                    if manual_result.at[idx, '回答状況']:
                                        result_df.at[idx, '回答状況'] = True
                                        # 回答方法が既に設定されている場合は統合（運営コメントを優先）
                                        if manual_result.at[idx, '回答方法']:
                                            result_df.at[idx, '回答方法'] = manual_result.at[idx, '回答方法']
                                        elif not result_df.at[idx, '回答方法']:
                                            result_df.at[idx, '回答方法'] = '運営コメント'
                        
                        # 列の順序を調整（回答状況を一番左列、回答方法を右隣に）
                        cols = ['回答状況', '回答方法'] + [col for col in result_df.columns if col not in ['回答状況', '回答方法']]
                        result_df = result_df[cols]
                        
                        # CSV生成前に、データフレーム全体でnanを処理
                        # すべての列でNaN値を空文字列に変換
                        result_df = result_df.fillna('')
                        
                        # 文字列として"nan"が入っている場合も空文字列に変換（大文字小文字を区別しない）
                        for col in result_df.columns:
                            if result_df[col].dtype == 'object':  # 文字列型の列のみ処理
                                result_df[col] = result_df[col].astype(str).str.strip()
                                nan_strings = ['nan', 'NaN', 'NAN', 'None', 'none', 'NONE', 'null', 'NULL', 'Null']
                                for nan_str in nan_strings:
                                    result_df.loc[result_df[col] == nan_str, col] = ''
                        
                        # 回答方法が空文字列の場合、回答状況をFalseに設定
                        if '回答方法' in result_df.columns and '回答状況' in result_df.columns:
                            empty_mask = result_df['回答方法'] == ''
                            result_df.loc[empty_mask, '回答状況'] = False
                        
                        # 結果をセッションステートに保存
                        st.session_state.question_answer_result = result_df
                        
                        # CSVを生成
                        csv_data = generate_question_answer_csv(result_df)
                        st.session_state.question_answer_csv_data = csv_data.encode('utf-8-sig')
                    
                    # 一時ファイルを削除
                    os.unlink(question_path)
                    if transcript_path:
                        os.unlink(transcript_path)
                    if manual_path:
                        os.unlink(manual_path)
                    
                    st.success("✓ 判定が完了しました！")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"エラー: {str(e)}")
                    import traceback
                    with st.expander("詳細なエラー情報"):
                        st.code(traceback.format_exc())
        else:
            st.warning("⚠️ 文字起こしテキストファイルまたは人間が判定したCSVファイルのいずれかをアップロードしてください。")
    else:
        st.info("💡 質問CSVファイルをアップロードしてください。")
    
    # 結果の表示とダウンロード
    if st.session_state.question_answer_result is not None:
        st.markdown("---")
        st.subheader("3. 結果の確認とダウンロード")
        
        result_df = st.session_state.question_answer_result
        
        # 統計情報の計算
        total_questions = len(result_df)
        answered_count = result_df['回答状況'].sum() if '回答状況' in result_df.columns else 0
        answer_rate = (answered_count / total_questions * 100) if total_questions > 0 else 0.0
        
        # 回答方法別の統計
        answer_method_counts = {}
        if '回答方法' in result_df.columns:
            answered_df_for_stats = result_df[result_df['回答状況']]
            if len(answered_df_for_stats) > 0:
                answer_method_counts = answered_df_for_stats['回答方法'].value_counts().to_dict()
        
        # 統計情報を表示
        col1, col2 = st.columns(2)
        with col1:
            st.metric("質問回答率", f"{answer_rate:.1f}%")
        with col2:
            st.metric("回答件数", f"{answered_count}件")
        
        # 回答方法別の統計を表示
        if answer_method_counts:
            st.subheader("回答方法別の内訳")
            for method, count in answer_method_counts.items():
                st.write(f"- {method}: {count}件")
        
        # 結果のプレビュー
        st.subheader("結果プレビュー")
        
        # タブで回答済みと未回答を分けて表示
        tab1, tab2, tab3 = st.tabs(["すべて", "回答済み", "未回答"])
        
        with tab1:
            st.dataframe(result_df.head(20), use_container_width=True)
        
        with tab2:
            answered_df = result_df[result_df['回答状況']]
            if len(answered_df) > 0:
                st.dataframe(answered_df.head(20), use_container_width=True)
            else:
                st.info("回答済みの質問はありません。")
        
        with tab3:
            unanswered_df = result_df[~result_df['回答状況']]
            if len(unanswered_df) > 0:
                st.dataframe(unanswered_df.head(20), use_container_width=True)
            else:
                st.info("未回答の質問はありません。")
        
        # CSVダウンロードリンク
        if st.session_state.question_answer_csv_data:
            download_link = create_download_link(
                st.session_state.question_answer_csv_data,
                "質問回答まとめ.csv",
                "text/csv"
            )
            st.markdown(f"**📥 ダウンロード**: {download_link}", unsafe_allow_html=True)


def generate_question_answer_csv(df: pd.DataFrame) -> str:
    """
    質問回答判定結果のCSVを生成（統計情報付き）
    
    Args:
        df: 結果データフレーム
        
    Returns:
        CSV文字列
    """
    # inserted_atを配信時間にリネーム
    result_df = df.copy()
    if 'inserted_at' in result_df.columns:
        result_df = result_df.rename(columns={'inserted_at': '配信時間'})
    
    # 回答方法がnanの場合の処理
    if '回答方法' in result_df.columns:
        # まず、NaN値を空文字列に変換
        result_df['回答方法'] = result_df['回答方法'].fillna('')
        
        # 文字列として"nan"が入っている場合も空文字列に変換（大文字小文字を区別しない）
        result_df['回答方法'] = result_df['回答方法'].astype(str).str.strip()
        nan_strings = ['nan', 'NaN', 'NAN', 'None', 'none', 'NONE', 'null', 'NULL', 'Null']
        for nan_str in nan_strings:
            result_df.loc[result_df['回答方法'] == nan_str, '回答方法'] = ''
        
        # 回答方法が空文字列の場合、回答状況をFalseに設定
        if '回答状況' in result_df.columns:
            empty_mask = result_df['回答方法'] == ''
            result_df.loc[empty_mask, '回答状況'] = False
    
    # 列の順序を調整（回答状況をA列、配信時間をB列に）
    if '回答状況' in result_df.columns and '配信時間' in result_df.columns:
        cols = ['回答状況', '配信時間']
        if '回答方法' in result_df.columns:
            cols.append('回答方法')
        # 残りの列を追加
        for col in result_df.columns:
            if col not in cols:
                cols.append(col)
        result_df = result_df[cols]
    
    # 統計情報の計算
    total_questions = len(result_df)
    answered_count = result_df['回答状況'].sum() if '回答状況' in result_df.columns else 0
    answer_rate = (answered_count / total_questions * 100) if total_questions > 0 else 0.0
    
    # 統計情報をCSV形式で作成
    stats_lines = []
    stats_lines.append("統計情報")
    stats_lines.append(f"質問件数,{total_questions}件")
    stats_lines.append(f"回答件数,{answered_count}件")
    stats_lines.append(f"質問回答率,{answer_rate:.1f}%")
    stats_lines.append("")  # 空行
    stats_lines.append("質問データ")
    
    # 統計情報をCSV文字列に変換
    stats_csv = "\n".join(stats_lines)
    
    # データフレームをCSV文字列に変換（修正したresult_dfを使用し、NaN値を空文字列に変換）
    data_csv = result_df.to_csv(index=False, na_rep='')
    
    # 統計情報とデータを結合
    combined_csv = stats_csv + "\n" + data_csv
    
    return combined_csv


if __name__ == "__main__":
    main()


