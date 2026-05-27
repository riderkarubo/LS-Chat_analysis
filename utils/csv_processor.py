"""CSVデータ処理モジュール"""
import pandas as pd
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


def extract_questions(df: pd.DataFrame, attribute_column: str = "チャットの属性") -> pd.DataFrame:
    """
    質問コメントを抽出（公式コメントは除外）
    
    Args:
        df: データフレーム
        attribute_column: チャットの属性の列名
        
    Returns:
        質問コメントのみのデータフレーム（公式コメントは除外）
    """
    question_attributes = [
        "00商品への質問",
        "04出演者関連"
    ]
    
    # 質問コメントを抽出
    question_df = df[df[attribute_column].isin(question_attributes)].copy()
    
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


