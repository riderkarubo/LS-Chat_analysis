"""質問と回答の照合モジュール"""
import pandas as pd
from typing import List, Dict, Optional
import time
from collections import deque
import threading
from config import OPENAI_API_KEY
from prompts.analysis_prompts import get_question_answer_match_prompt

# OpenAIクライアントとレート制限モニターの初期化
client = None
_rate_limit_monitor = None

# OpenAIクライアントの初期化
if OPENAI_API_KEY:
    try:
        import openai
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        # レート制限モニターの初期化（ai_analyzerと同じ設定）
        class RateLimitMonitor:
            """レート制限を監視するクラス（OpenAIのレート制限に対応）"""
            def __init__(self, max_requests_per_minute: int = 480):
                self.max_requests = max_requests_per_minute
                self.request_times = deque()
                self.lock = threading.Lock()
            
            def wait_if_needed(self):
                """必要に応じて待機（レート制限を超えないように）"""
                with self.lock:
                    now = time.time()
                    # 1分以上前のリクエストを削除
                    while self.request_times and (now - self.request_times[0]) > 60:
                        self.request_times.popleft()
                    
                    # レート制限に近づいている場合は待機
                    if len(self.request_times) >= self.max_requests:
                        oldest_time = self.request_times[0]
                        wait_time = 60 - (now - oldest_time) + 1  # 1秒のマージン
                        if wait_time > 0:
                            time.sleep(wait_time)
                            # 再度クリーンアップ
                            now = time.time()
                            while self.request_times and (now - self.request_times[0]) > 60:
                                self.request_times.popleft()
                    
                    # 現在のリクエスト時刻を記録
                    self.request_times.append(time.time())
        
        _rate_limit_monitor = RateLimitMonitor(max_requests_per_minute=480)
    except Exception as e:
        import sys
        print(f"DEBUG: OpenAIクライアントの初期化エラー: {e}", file=sys.stderr)
        client = None
        _rate_limit_monitor = None


def match_questions_with_transcript(question_df: pd.DataFrame, transcript_data: List[Dict]) -> pd.DataFrame:
    """
    質問CSVと文字起こしテキストを照合して回答状況を判定
    
    Args:
        question_df: 質問CSVのデータフレーム
        transcript_data: 文字起こしテキストのパース結果
        
    Returns:
        回答状況列と回答方法列が追加されたデータフレーム
    """
    result_df = question_df.copy()
    
    # 回答状況列と回答方法列を初期化
    result_df['回答状況'] = False
    result_df['回答方法'] = ''
    
    # 質問テキストの列名を特定（original_text列を探す）
    question_column = None
    for col in ['original_text', '質問', 'コメント', 'text']:
        if col in result_df.columns:
            question_column = col
            break
    
    if question_column is None:
        raise ValueError("質問テキストの列が見つかりません。original_text列が必要です。")
    
    # デバッグ情報
    import sys
    print(f"DEBUG: 質問数: {len(result_df)}", file=sys.stderr)
    print(f"DEBUG: 文字起こしデータ数: {len(transcript_data)}", file=sys.stderr)
    print(f"DEBUG: 質問列名: {question_column}", file=sys.stderr)
    
    matched_count = 0
    
    # 各質問に対して回答を検索
    for idx, row in result_df.iterrows():
        question_text = str(row[question_column]).strip()
        
        if not question_text or question_text == 'nan':
            continue
        
        # 文字起こしテキストから回答を検索（部分一致）
        for answer in transcript_data:
            answer_text = answer.get('text', '').strip()
            
            if not answer_text:
                continue
            
            # 双方向の照合を試す（質問→回答、回答→質問）
            # 部分一致で照合（質問の主要なキーワードが回答に含まれているか）
            if is_question_answered(question_text, answer_text):
                result_df.at[idx, '回答状況'] = True
                result_df.at[idx, '回答方法'] = '出演者'
                matched_count += 1
                print(f"DEBUG: 照合成功 - 質問[{idx}]: {question_text[:50]}... <-> 回答: {answer_text[:50]}...", file=sys.stderr)
                break
    
    print(f"DEBUG: 照合成功数: {matched_count}/{len(result_df)}", file=sys.stderr)
    
    # 列の順序を調整（回答状況を一番左列、回答方法を右隣に）
    cols = ['回答状況', '回答方法'] + [col for col in result_df.columns if col not in ['回答状況', '回答方法']]
    result_df = result_df[cols]
    
    return result_df


def match_questions_with_manual_csv(question_df: pd.DataFrame, manual_df: pd.DataFrame, transcript_data: List[Dict] = None) -> pd.DataFrame:
    """
    質問CSVと人間が判定したCSVを照合して回答状況を判定
    
    Args:
        question_df: 質問CSVのデータフレーム
        manual_df: 人間が判定したCSVのデータフレーム
        transcript_data: 文字起こしテキストのパース結果（オプショナル）
        
    Returns:
        回答状況列と回答方法列が追加されたデータフレーム
    """
    result_df = question_df.copy()
    
    # 回答状況列と回答方法列を初期化
    result_df['回答状況'] = False
    result_df['回答方法'] = ''
    
    # 質問テキストの列名を特定
    question_column = None
    for col in ['original_text', '質問', 'コメント', 'text']:
        if col in result_df.columns:
            question_column = col
            break
    
    if question_column is None:
        raise ValueError("質問テキストの列が見つかりません。original_text列が必要です。")
    
    # 人間が判定したCSVの列名を特定（「質問」という項目名で検索）
    manual_question_column = None
    
    # デバッグ: 実際の列名を確認
    import sys
    print(f"DEBUG: manual_df.columns = {list(manual_df.columns)}", file=sys.stderr)
    print(f"DEBUG: manual_df.shape = {manual_df.shape}", file=sys.stderr)
    print(f"DEBUG: 列名の詳細（repr）: {[repr(col) for col in manual_df.columns]}", file=sys.stderr)
    
    # 列名を検索（空白を除去して比較、完全一致を優先）
    for col in manual_df.columns:
        col_stripped = col.strip()
        # 「質問」と完全一致する列を優先
        if col_stripped == '質問':
            manual_question_column = col
            print(f"DEBUG: 「質問」列を発見（完全一致）: {repr(col)}", file=sys.stderr)
            break
    
    # 完全一致が見つからない場合は、部分一致で検索
    if manual_question_column is None:
        for col in manual_df.columns:
            col_stripped = col.strip()
            if '質問' in col_stripped:
                manual_question_column = col
                print(f"DEBUG: 「質問」列を発見（部分一致）: {repr(col)}", file=sys.stderr)
                break
    
    # 「質問」が見つからない場合は、他の候補も試す
    if manual_question_column is None:
        possible_column_names = ['original_text', 'コメント', 'text']
        for col in manual_df.columns:
            col_stripped = col.strip()
            if col_stripped in possible_column_names:
                manual_question_column = col
                print(f"DEBUG: 代替列名を使用: {repr(col)}", file=sys.stderr)
                break
    
    if manual_question_column is None:
        # エラーメッセージに実際の列名を表示
        available_columns = ', '.join([f'"{col}"' for col in manual_df.columns])
        print(f"DEBUG: エラー - 利用可能な列: {available_columns}", file=sys.stderr)
        raise ValueError(f"人間が判定したCSVに「質問」という項目名の列が見つかりません。利用可能な列: {available_columns}")
    
    # 回答済列と回答方法列を特定
    answered_column = None
    # 「回答済み」も検索対象に追加
    for col in ['回答済み', '回答済', 'answered', '回答状況']:
        if col in manual_df.columns:
            answered_column = col
            print(f"DEBUG: 回答済列を発見: {repr(col)}", file=sys.stderr)
            break
    
    answer_method_column = None
    for col in ['回答方法', 'answer_method', '方法']:
        if col in manual_df.columns:
            answer_method_column = col
            break
    
    # 人間が判定したCSVで回答済みの質問をマップ（重複対応）
    answered_questions_map = {}  # {質問テキスト: (回答状況, 回答方法)}
    
    print(f"DEBUG: 人間が判定したCSVの行数: {len(manual_df)}", file=sys.stderr)
    
    answered_count_in_manual = 0
    
    for manual_idx, manual_row in manual_df.iterrows():
        manual_question = str(manual_row[manual_question_column]).strip()
        
        if not manual_question or manual_question == 'nan':
            continue
        
        # 回答済列がTRUEの場合のみ処理（大文字小文字を区別しない）
        if answered_column:
            answered_value = str(manual_row[answered_column]).strip().upper()
            # TRUE/1のすべての形式に対応（大文字小文字を区別しない）
            if answered_value in ['TRUE', '1', 'T', 'YES', 'Y']:
                # 回答方法を取得
                method = '運営コメント'
                if answer_method_column:
                    method_value = str(manual_row[answer_method_column]).strip()
                    if method_value:
                        method = method_value
                
                # マップに追加（重複している場合は上書き）
                answered_questions_map[manual_question] = (True, method)
                answered_count_in_manual += 1
                print(f"DEBUG: 回答済み質問を追加 - {manual_question[:50]}... (回答方法: {method})", file=sys.stderr)
    
    print(f"DEBUG: 人間が判定したCSVで回答済みの質問数: {answered_count_in_manual}", file=sys.stderr)
    print(f"DEBUG: 回答済み質問マップのサイズ: {len(answered_questions_map)}", file=sys.stderr)
    
    # デバッグ: 回答済み質問マップの内容を表示（最初の5件）
    if answered_questions_map:
        print("DEBUG: 回答済み質問マップのサンプル（最初の5件）:", file=sys.stderr)
        for i, (q, (a, m)) in enumerate(list(answered_questions_map.items())[:5]):
            print(f"  [{i+1}] 質問: {q[:50]}... (回答状況: {a}, 回答方法: {m})", file=sys.stderr)
    
    matched_count = 0
    
    # 各質問に対して回答を検索
    for idx, row in result_df.iterrows():
        question_text = str(row[question_column]).strip()
        
        if not question_text or question_text == 'nan':
            continue
        
        # デバッグ: 照合前の質問テキストを表示（最初の10件のみ）
        if matched_count < 10:
            print(f"DEBUG: 照合開始 - 質問CSV[{idx}]: {question_text}", file=sys.stderr)
        
        # 人間が判定したCSVの質問と照合
        for manual_question, (answered, method) in answered_questions_map.items():
            # 完全一致を最優先でチェック
            if question_text == manual_question:
                result_df.at[idx, '回答状況'] = answered
                result_df.at[idx, '回答方法'] = method
                matched_count += 1
                print(f"DEBUG: 照合成功（完全一致） - 質問CSV[{idx}]: {question_text[:50]}... <-> 人間判定CSV: {manual_question[:50]}...", file=sys.stderr)
                break
            
            # 双方向の照合を試す（質問→回答、回答→質問）
            # 部分一致で照合
            if is_question_answered(question_text, manual_question):
                result_df.at[idx, '回答状況'] = answered
                result_df.at[idx, '回答方法'] = method
                matched_count += 1
                print(f"DEBUG: 照合成功（部分一致） - 質問CSV[{idx}]: {question_text[:50]}... <-> 人間判定CSV: {manual_question[:50]}...", file=sys.stderr)
                break
    
    print(f"DEBUG: 照合成功数: {matched_count}/{len(result_df)}", file=sys.stderr)
    
    # FALSEで回答方法が空白のものも文字起こしテキストと照合
    if transcript_data:
        false_blank_count = 0
        transcript_match_count = 0
        
        # FALSEで回答方法が空白の行を抽出
        false_blank_rows = []
        for manual_idx, manual_row in manual_df.iterrows():
            manual_question = str(manual_row[manual_question_column]).strip()
            
            if not manual_question or manual_question == 'nan':
                continue
            
            # 回答済列がFALSEで回答方法が空白の場合
            if answered_column:
                answered_value = str(manual_row[answered_column]).strip().upper()
                if answered_value in ['FALSE', '0', 'F', 'NO', 'N']:
                    method_value = ''
                    if answer_method_column:
                        method_value = str(manual_row[answer_method_column]).strip()
                        if method_value in ['', 'nan', 'NaN']:
                            method_value = ''
                    
                    if not method_value:
                        false_blank_rows.append(manual_question)
                        false_blank_count += 1
        
        print(f"DEBUG: FALSEで回答方法が空白の行数: {false_blank_count}", file=sys.stderr)
        
        # これらの行の質問テキストを文字起こしテキストと照合
        if false_blank_rows and transcript_data:
            for idx, row in result_df.iterrows():
                question_text = str(row[question_column]).strip()
                
                if not question_text or question_text == 'nan':
                    continue
                
                # 既にTRUEになっている場合はスキップ
                if result_df.at[idx, '回答状況']:
                    continue
                
                # FALSEで回答方法が空白の質問と照合
                for false_blank_question in false_blank_rows:
                    # 質問テキストが一致するか確認
                    if question_text == false_blank_question or is_question_answered(question_text, false_blank_question):
                        # 文字起こしテキストと照合
                        for answer in transcript_data:
                            answer_text = answer.get('text', '').strip()
                            
                            if not answer_text:
                                continue
                            
                            # 文字起こしテキストと照合
                            if is_question_answered(question_text, answer_text):
                                result_df.at[idx, '回答状況'] = True
                                result_df.at[idx, '回答方法'] = '出演者'
                                transcript_match_count += 1
                                print(f"DEBUG: 文字起こしテキストとの照合成功 - 質問[{idx}]: {question_text[:50]}...", file=sys.stderr)
                                break
                        
                        # 1つの質問に対して1つのマッチが見つかったら終了
                        if result_df.at[idx, '回答状況']:
                            break
        
        print(f"DEBUG: 文字起こしテキストとの照合成功数: {transcript_match_count}", file=sys.stderr)
    
    # 列の順序を調整（回答状況を一番左列、回答方法を右隣に）
    cols = ['回答状況', '回答方法'] + [col for col in result_df.columns if col not in ['回答状況', '回答方法']]
    result_df = result_df[cols]
    
    return result_df


def is_question_answered_with_ai(question_text: str, answer_text: str) -> Optional[bool]:
    """
    AI分析を使用して質問が回答されたかを判定
    
    Args:
        question_text: 質問テキスト
        answer_text: 回答テキスト
        
    Returns:
        回答された場合True、回答されていない場合False、エラー時はNone
    """
    if not client:
        return None
    
    if not question_text or not answer_text:
        return None
    
    try:
        # レート制限の監視
        if _rate_limit_monitor:
            _rate_limit_monitor.wait_if_needed()
        
        # プロンプトを生成
        prompt = get_question_answer_match_prompt(question_text, answer_text)
        
        # OpenAI APIを呼び出し
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_completion_tokens=10,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1  # 一貫性を重視
        )
        
        # レスポンスをパース
        result = response.choices[0].message.content.strip().upper()
        
        # YES/NOを判定
        if "YES" in result or "はい" in result or "回答" in result:
            return True
        elif "NO" in result or "いいえ" in result or "未回答" in result:
            return False
        else:
            # 不明な場合はFalseを返す（安全側に倒す）
            return False
            
    except Exception as e:
        import sys
        print(f"DEBUG: AI分析エラー: {e}", file=sys.stderr)
        return None


def is_question_answered(question_text: str, answer_text: str, threshold: float = 0.1) -> bool:
    """
    質問が回答されたかを部分一致で判定（改善版）
    
    Args:
        question_text: 質問テキスト
        answer_text: 回答テキスト
        threshold: 一致率の閾値（0.0-1.0、デフォルト0.1に変更）
        
    Returns:
        回答された場合True
    """
    import re
    
    if not question_text or not answer_text:
        return False
    
    # 絵文字を除去（Unicode範囲を使用）
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    
    question_text = emoji_pattern.sub('', str(question_text)).strip()
    answer_text = emoji_pattern.sub('', str(answer_text)).strip()
    
    # 空文字列のチェック
    if not question_text or not answer_text or question_text == 'nan' or answer_text == 'nan':
        return False
    
    # 完全一致
    if question_text == answer_text:
        return True
    
    # 一方がもう一方に完全に含まれている場合（より柔軟に）
    if question_text in answer_text or answer_text in question_text:
        return True
    
    # 短い質問（3文字以下）の場合は、完全一致または部分一致のみで判定
    if len(question_text) <= 3:
        # 質問の主要なキーワードが回答に含まれているかチェック
        if question_text in answer_text:
            return True
        # 回答の主要なキーワードが質問に含まれているかチェック
        if len(answer_text) <= 10 and answer_text in question_text:
            return True
        return False
    
    # 長い方のテキストの一部が短い方に含まれている場合もチェック
    if len(question_text) > len(answer_text):
        # 質問が長い場合、回答が質問に含まれているかチェック
        if answer_text in question_text:
            return True
    else:
        # 回答が長い場合、質問が回答に含まれているかチェック
        if question_text in answer_text:
            return True
    
    # 日本語テキストの場合、文字単位で照合
    # 日本語文字（ひらがな、カタカナ、漢字）と英数字を抽出
    question_chunks = []
    # 1文字以上の連続する日本語文字列を抽出（Unicode範囲を使用）
    # ひらがな: \u3040-\u309F, カタカナ: \u30A0-\u30FF, 漢字: \u4E00-\u9FAF
    japanese_pattern = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF\w]+')
    question_chunks = japanese_pattern.findall(question_text)
    
    # 日本語文字列が見つからない場合は、スペースや句読点で分割
    if not question_chunks:
        question_chunks = [w.strip() for w in re.split(r'[\s、。，．]+', question_text) if len(w.strip()) >= 1]
    
    # それでも見つからない場合は、1文字以上の文字列を抽出
    if not question_chunks:
        question_chunks = [w.strip() for w in re.split(r'[\s、。，．]+', question_text) if len(w.strip()) >= 1]
    
    if not question_chunks:
        return False
    
    # 回答テキストに質問のキーワードが含まれているかチェック
    matched_chunks = 0
    for chunk in question_chunks:
        if chunk in answer_text:
            matched_chunks += 1
    
    # 一致率を計算
    match_ratio = matched_chunks / len(question_chunks) if question_chunks else 0
    
    # 文字列マッチングで閾値を超える場合はTrueを返す
    if match_ratio >= threshold:
        return True
    
    # 文字列マッチングで判定できない場合（閾値未満）はAI分析を試す
    # ただし、短いテキスト（3文字以下）の場合はAI分析をスキップ
    if len(question_text) > 3:
        ai_result = is_question_answered_with_ai(question_text, answer_text)
        if ai_result is not None:
            return ai_result
    
    # AI分析が失敗した場合や短いテキストの場合は、文字列マッチングの結果を返す
    return False

