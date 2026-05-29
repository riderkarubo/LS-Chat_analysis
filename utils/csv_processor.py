"""CSVデータ処理モジュール"""
import pandas as pd
import re
from typing import List
from config import OFFICIAL_GUEST_ID

REQUIRED_COLUMNS = ["guest_id", "username", "original_text", "inserted_at"]


def detect_header_row(file_path: str, required_columns: List[str], max_rows: int = 10) -> int:
    """
    必要な列を含むヘッダー行を検出
    
    Args:
        file_path: CSVファイルのパス
        required_columns: 必要な列名のリスト
        max_rows: 検索する最大行数
    
    Returns:
        ヘッダー行のインデックス（0始まり、見つからない場合は0）
    """
    try:
        # CSVファイルを1行ずつ読み込んで、必要な列が含まれているかチェック
        with open(file_path, 'r', encoding='utf-8') as f:
            for row_idx, line in enumerate(f):
                if row_idx >= max_rows:
                    break
                
                # 行をカンマで分割して列名を取得
                columns = [col.strip() for col in line.split(',')]
                
                # 必要な列がすべて含まれているかチェック
                if all(col in columns for col in required_columns):
                    return row_idx
        
        # 見つからない場合は0行目（1行目）を返す
        return 0
    except Exception:
        # エラーが発生した場合は0行目（1行目）を返す
        return 0


def load_csv(file_path: str) -> pd.DataFrame:
    """
    CSVファイルを読み込む（ヘッダー行を自動検出）
    
    Args:
        file_path: CSVファイルのパス
        
    Returns:
        読み込んだデータフレーム
        
    Raises:
        ValueError: 必要な列が存在しない場合
    """
    # 必要な列を含むヘッダー行を自動検出
    header_row = detect_header_row(file_path, REQUIRED_COLUMNS)
    
    # 検出したヘッダー行を使用してCSVファイルを読み込む
    df = pd.read_csv(file_path, header=header_row)
    
    # 必要な列の存在確認
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"必要な列が見つかりません: {', '.join(missing_columns)}")
    
    # 必要な列のみ抽出
    df = df[REQUIRED_COLUMNS].copy()
    
    # 空の行を削除
    df = df.dropna(subset=["original_text", "inserted_at"])
    
    return df


def validate_and_process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    データの検証と基本処理
    
    Args:
        df: データフレーム
        
    Returns:
        処理済みデータフレーム
        
    Raises:
        ValueError: データが不正な場合
    """
    if df.empty:
        raise ValueError("データが空です。")
    
    # inserted_atをdatetime型に変換
    df["inserted_at"] = pd.to_datetime(df["inserted_at"], errors="coerce")
    
    # 変換に失敗した行を削除
    df = df.dropna(subset=["inserted_at"])
    
    if df.empty:
        raise ValueError("有効な日時データがありません。")
    
    # inserted_atでソート（早い順）
    df = df.sort_values("inserted_at").reset_index(drop=True)
    
    # 相対時間に変換（最初のコメントを00:00:00に）
    df = convert_to_relative_time(df)
    
    return df


def format_time_from_seconds(seconds: float) -> str:
    """
    秒数から配信時間（HH:MM形式）を生成
    
    Args:
        seconds: 経過秒数（float）
        
    Returns:
        配信時間文字列（HH:MM形式）
    """
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    return f"{hours:02d}:{minutes:02d}"


def convert_elapsed_time_to_broadcast_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    elapsed_timeカラムを配信時間（HH:MM形式）に変換
    
    Args:
        df: データフレーム（elapsed_timeカラムを含む）
        
    Returns:
        配信時間カラムが追加されたデータフレーム
    """
    if df.empty:
        return df
    
    # elapsed_timeカラムが存在しない場合はそのまま返す
    if 'elapsed_time' not in df.columns:
        return df
    
    # elapsed_timeがNaNの行を除外
    df = df.dropna(subset=['elapsed_time']).copy()
    
    if df.empty:
        return df
    
    # 最小値を00:00に設定（最初のコメントを00:00にする）
    min_elapsed = df['elapsed_time'].min()
    
    # elapsed_timeを配信時間（HH:MM形式）に変換
    df['配信時間'] = df['elapsed_time'].apply(
        lambda x: format_time_from_seconds(x - min_elapsed)
    )
    
    return df


def convert_to_relative_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    inserted_atを相対時間（00:00:00形式）に変換
    
    Args:
        df: データフレーム
        
    Returns:
        相対時間に変換されたデータフレーム
    """
    if df.empty:
        return df
    
    # 最初のコメントの時刻を取得
    first_time = df["inserted_at"].iloc[0]
    
    # 各コメントの経過時間を計算
    time_deltas = df["inserted_at"] - first_time
    
    # 時:分形式に変換（HH:MM）
    def format_time_delta(td: pd.Timedelta) -> str:
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        return f"{hours:02d}:{minutes:02d}"
    
    df["inserted_at"] = time_deltas.apply(format_time_delta)
    
    return df


def load_csv_with_elapsed_time(file_path: str) -> pd.DataFrame:
    """
    elapsed_timeカラムを含むCSVファイルを読み込む
    
    Args:
        file_path: CSVファイルのパス
        
    Returns:
        読み込んだデータフレーム（配信時間カラムが追加される）
        
    Raises:
        ValueError: 必要な列が存在しない場合
    """
    # CSVファイルを読み込む（すべての列を読み込む）
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    
    # 必要な列の存在確認
    required_columns_for_elapsed = ["username", "original_text"]
    missing_columns = [col for col in required_columns_for_elapsed if col not in df.columns]
    if missing_columns:
        raise ValueError(f"必要な列が見つかりません: {', '.join(missing_columns)}")
    
    # elapsed_timeカラムが存在しない場合のエラーチェック
    if 'elapsed_time' not in df.columns:
        raise ValueError("elapsed_timeカラムが見つかりません。")
    
    # 空の行を削除
    df = df.dropna(subset=["original_text", "elapsed_time"])
    
    # elapsed_timeを配信時間に変換
    df = convert_elapsed_time_to_broadcast_time(df)
    
    return df


def is_question_by_pattern(comment_text: str) -> bool:
    """
    パターンマッチングで質問かどうかを判定
    
    Args:
        comment_text: コメント本文
        
    Returns:
        True（質問）またはFalse（質問ではない）
    """
    if not comment_text or not isinstance(comment_text, str):
        return False
    
    comment_text = comment_text.strip()
    if not comment_text:
        return False
    
    # 情報提供パターンを除外（質問ではない）
    # 「〜となります！」「〜でございます！」「〜です！」「〜ます！」で終わる情報提供文
    information_providing_patterns = [
        r'となります！$', r'となります。$', r'となります$',
        r'でございます！$', r'でございます。$', r'でございます$',
        r'です！$', r'です。$', r'ます！$', r'ます。$',
        r'となります\?$', r'となります\？$',
        r'でございます\?$', r'でございます\？$'
    ]
    
    # 情報提供パターンチェック（先に除外）
    for pattern in information_providing_patterns:
        if re.search(pattern, comment_text):
            return False
    
    # 疑問詞パターン
    question_words = [
        r'何', r'いくら', r'どこ', r'いつ', r'どう', r'なぜ', r'どれ', r'どの', r'誰', r'どちら',
        r'なに', r'いくつ', r'どなた', r'どのくらい', r'どの程度', r'どれくらい',
        r'どんな', r'どのような', r'どういう', r'どうやって', r'なぜ', r'なんで'
    ]
    
    # 疑問符パターン
    question_mark_pattern = r'[？?]'
    
    # 質問パターン（文末）
    question_end_patterns = [
        r'ですか', r'ますか', r'なんですか', r'なんか', r'でしょうか', r'でしょうか',
        r'ですか？', r'ますか？', r'なんですか？', r'でしょうか？',
        r'ですか\?', r'ますか\?', r'なんですか\?', r'でしょうか\?',
        r'か？', r'か\?', r'か。', r'か！'
    ]
    
    # 否定疑問パターン
    negative_question_patterns = [
        r'ないですか', r'ませんか', r'ないですか？', r'ませんか？',
        r'ないですか\?', r'ませんか\?', r'ない？', r'ない\?'
    ]
    
    # 疑問詞チェック
    for word in question_words:
        if re.search(word, comment_text, re.IGNORECASE):
            return True
    
    # 疑問符チェック
    if re.search(question_mark_pattern, comment_text):
        return True
    
    # 質問パターンチェック（文末）
    for pattern in question_end_patterns:
        if re.search(pattern, comment_text):
            return True
    
    # 否定疑問パターンチェック
    for pattern in negative_question_patterns:
        if re.search(pattern, comment_text):
            return True
    
    return False


def is_question_by_ai(comment_text: str) -> bool:
    """
    AI判定で質問かどうかを判定
    
    Args:
        comment_text: コメント本文
        
    Returns:
        True（質問）またはFalse（質問ではない）
    """
    if not comment_text or not isinstance(comment_text, str):
        return False
    
    comment_text = comment_text.strip()
    if not comment_text:
        return False
    
    try:
        from prompts.analysis_prompts import is_question_prompt
        from utils.ai_analyzer import get_openai_client
        
        prompt = is_question_prompt(comment_text)
        client = get_openai_client()
        
        if not client:
            # APIキーが設定されていない場合は簡易判定の結果を返す
            return is_question_by_pattern(comment_text)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_completion_tokens=10,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        raw_response = response.choices[0].message.content.strip()
        
        # 「はい」または「yes」を含む場合は質問
        if "はい" in raw_response or "yes" in raw_response.lower():
            return True
        
        # それ以外は質問ではない
        return False
        
    except Exception as e:
        # エラーが発生した場合は簡易判定の結果を返す
        import sys
        print(f"AI判定エラー: {e}", file=sys.stderr)
        return is_question_by_pattern(comment_text)


def extract_questions(df: pd.DataFrame, attribute_column: str = "チャットの属性") -> pd.DataFrame:
    """
    質問コメントを抽出（公式コメントは除外）
    
    Args:
        df: データフレーム
        attribute_column: チャットの属性の列名
        
    Returns:
        質問コメントのみのデータフレーム（公式コメントは除外）
    """
    if df.empty:
        return df.copy()
    
    # 「商品への質問」はそのまま抽出
    product_questions = df[df[attribute_column] == "商品への質問"].copy()
    
    # 「出演者関連」は質問判定を実施
    performer_comments = df[df[attribute_column] == "出演者関連"].copy()
    performer_questions = pd.DataFrame()
    
    if not performer_comments.empty:
        # 簡易判定を実行
        if 'original_text' in performer_comments.columns:
            question_mask = performer_comments['original_text'].apply(is_question_by_pattern)
            performer_questions = performer_comments[question_mask].copy()
            
            # 不明確なケース（簡易判定でFalseだが疑問符がある場合など）はAI判定
            uncertain_mask = ~question_mask & performer_comments['original_text'].str.contains(r'[？?]', na=False, regex=True)
            if uncertain_mask.any():
                uncertain_comments = performer_comments[uncertain_mask].copy()
                # AI判定を実行
                ai_question_mask = uncertain_comments['original_text'].apply(is_question_by_ai)
                ai_questions = uncertain_comments[ai_question_mask].copy()
                if not ai_questions.empty:
                    performer_questions = pd.concat([performer_questions, ai_questions], ignore_index=True)
    
    # 質問コメントを結合
    question_df = pd.concat([product_questions, performer_questions], ignore_index=True)
    
    if question_df.empty:
        return question_df
    
    # 公式コメントを除外
    # 1. user_typeが"moderator"の行を除外
    if 'user_type' in question_df.columns:
        question_df = question_df[question_df['user_type'].astype(str).str.strip().str.lower() != 'moderator'].copy()
    
    # 2. user_idが存在し、値が空でない行を除外
    if 'user_id' in question_df.columns:
        # user_idがNaN/Noneでなく、空文字列でない行を除外
        mask = question_df['user_id'].isna() | (question_df['user_id'].astype(str).str.strip() == '')
        question_df = question_df[mask].copy()
    
    # 後方互換性のため、guest_idによる判定も残す（将来的に削除予定）
    if 'guest_id' in question_df.columns:
        question_df = question_df[question_df['guest_id'].astype(str).str.strip() != OFFICIAL_GUEST_ID].copy()
    
    # usernameが"マツキヨココカラSTAFF"の行を除外
    if 'username' in question_df.columns:
        question_df = question_df[question_df['username'].astype(str).str.strip() != 'マツキヨココカラSTAFF'].copy()
    
    return question_df.reset_index(drop=True)


