"""文字起こしテキストパーサーモジュール"""
import re
import sys
from typing import List, Dict


def parse_transcript(file_path: str) -> List[Dict]:
    """
    文字起こしテキストファイルをパースして回答箇所を抽出（行単位で処理）
    
    Args:
        file_path: 文字起こしテキストファイルのパス
        
    Returns:
        回答箇所のリスト（各要素はタイムコード、話者名、発言内容を含む辞書）
    """
    answers = []
    
    # タイムコードのパターン（例: 00:00:00:01 - 00:00:11:22）
    time_pattern = re.compile(r'(\d{2}:\d{2}:\d{2}:\d{2})\s*-\s*(\d{2}:\d{2}:\d{2}:\d{2})')
    
    # 話者名のパターン（例: 話者 1）
    speaker_pattern = re.compile(r'話者\s*\d+')
    
    current_timecode = None
    current_speaker = None
    current_text_lines = []
    skipped_count = 0
    invalid_count = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        total_lines = len(lines)
        print(f"DEBUG [文字起こしパース] ファイルを読み込みました: {total_lines}行", file=sys.stderr)
        
        # 行単位で処理
        for line_num, line in enumerate(lines, start=1):
            line = line.strip()
            
            # 空行の場合は次の行へ
            if not line:
                # 空行で区切られたセクションの終了を検出
                if current_timecode and current_speaker and current_text_lines:
                    # データ検証: タイムコード、話者、テキストがすべて存在するか確認
                    text_content = '\n'.join(current_text_lines).strip()
                    if _validate_answer_data(current_timecode, current_speaker, text_content):
                        answers.append({
                            'start_time': current_timecode[0],
                            'end_time': current_timecode[1],
                            'speaker': current_speaker,
                            'text': text_content
                        })
                    else:
                        skipped_count += 1
                        print(f"警告: 行{line_num}付近のデータが無効のためスキップしました（タイムコード: {current_timecode[0] if current_timecode else 'なし'}, 話者: {current_speaker or 'なし'}, テキスト: {'あり' if text_content else 'なし'}）", file=sys.stderr)
                    
                    # 状態をリセット
                    current_timecode = None
                    current_speaker = None
                    current_text_lines = []
                continue
            
            # タイムコード行を検出
            time_match = time_pattern.match(line)
            if time_match:
                # 前の回答を保存（空行で区切られていない場合）
                if current_timecode and current_speaker and current_text_lines:
                    text_content = '\n'.join(current_text_lines).strip()
                    if _validate_answer_data(current_timecode, current_speaker, text_content):
                        answers.append({
                            'start_time': current_timecode[0],
                            'end_time': current_timecode[1],
                            'speaker': current_speaker,
                            'text': text_content
                        })
                    else:
                        skipped_count += 1
                        print(f"警告: 行{line_num}付近のデータが無効のためスキップしました（タイムコード: {current_timecode[0] if current_timecode else 'なし'}, 話者: {current_speaker or 'なし'}, テキスト: {'あり' if text_content else 'なし'}）", file=sys.stderr)
                
                # 新しいタイムコードを設定
                current_timecode = (time_match.group(1), time_match.group(2))
                current_speaker = None
                current_text_lines = []
                continue
            
            # 話者行を検出（タイムコードが設定されている場合のみ）
            if current_timecode and not current_speaker:
                speaker_match = speaker_pattern.match(line)
                if speaker_match:
                    current_speaker = speaker_match.group(0)
                    continue
            
            # テキスト行（タイムコードと話者が設定されている場合）
            if current_timecode and current_speaker:
                current_text_lines.append(line)
                continue
            
            # 予期しない行の場合はスキップ
            invalid_count += 1
            if invalid_count <= 10:  # 最初の10件のみ警告を表示
                print(f"警告: 行{line_num}は予期しない形式です: {line[:50]}...", file=sys.stderr)
        
        # 最後の回答を保存（ファイル終端の場合）
        if current_timecode and current_speaker and current_text_lines:
            text_content = '\n'.join(current_text_lines).strip()
            if _validate_answer_data(current_timecode, current_speaker, text_content):
                answers.append({
                    'start_time': current_timecode[0],
                    'end_time': current_timecode[1],
                    'speaker': current_speaker,
                    'text': text_content
                })
            else:
                skipped_count += 1
                print(f"警告: ファイル終端のデータが無効のためスキップしました（タイムコード: {current_timecode[0] if current_timecode else 'なし'}, 話者: {current_speaker or 'なし'}, テキスト: {'あり' if text_content else 'なし'}）", file=sys.stderr)
        
        # デバッグ情報を出力
        print(f"DEBUG [文字起こしパース] パース完了: {len(answers)}件の回答を抽出しました", file=sys.stderr)
        print(f"DEBUG [文字起こしパース] スキップされたデータ: {skipped_count}件", file=sys.stderr)
        print(f"DEBUG [文字起こしパース] 予期しない行: {invalid_count}件", file=sys.stderr)
        
        # 話者ごとの統計
        speaker_counts = {}
        for answer in answers:
            speaker = answer['speaker']
            speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
        
        print(f"DEBUG [文字起こしパース] 話者ごとの件数:", file=sys.stderr)
        for speaker, count in sorted(speaker_counts.items()):
            print(f"  {speaker}: {count}件", file=sys.stderr)
        
        # サンプルデータを出力
        if len(answers) > 0:
            print(f"DEBUG [文字起こしパース] 最初の3件のサンプル:", file=sys.stderr)
            for i, answer in enumerate(answers[:3], start=1):
                print(f"  [{i}] 話者: {answer['speaker']}, 開始時間: {answer['start_time']}, 終了時間: {answer['end_time']}, テキスト長: {len(answer['text'])}文字", file=sys.stderr)
                print(f"      テキスト: {answer['text'][:100]}...", file=sys.stderr)
        
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません: {file_path}", file=sys.stderr)
        raise
    except Exception as e:
        print(f"エラー: ファイルのパース中にエラーが発生しました: {str(e)}", file=sys.stderr)
        raise
    
    return answers


def _validate_answer_data(timecode: tuple, speaker: str, text: str) -> bool:
    """
    回答データの検証（タイムコード、話者、テキストの存在確認）
    
    Args:
        timecode: タイムコードのタプル（開始時間、終了時間）
        speaker: 話者名
        text: テキスト内容
        
    Returns:
        検証が成功した場合True
    """
    # タイムコードの検証
    if not timecode or len(timecode) != 2:
        return False
    if not timecode[0] or not timecode[1]:
        return False
    
    # 話者の検証
    if not speaker or not speaker.strip():
        return False
    
    # テキストの検証
    if not text or not text.strip():
        return False
    
    return True


def timecode_to_seconds(timecode: str) -> int:
    """
    タイムコード（HH:MM:SS:FF形式）を秒に変換
    
    Args:
        timecode: タイムコード文字列（例: "00:00:05:30"）
        
    Returns:
        秒数
    """
    parts = timecode.split(':')
    if len(parts) == 4:
        hours, minutes, seconds, frames = map(int, parts)
        # フレームを秒に変換（30fpsを仮定）
        total_seconds = hours * 3600 + minutes * 60 + seconds + frames / 30.0
        return int(total_seconds)
    elif len(parts) == 3:
        hours, minutes, seconds = map(int, parts)
        return hours * 3600 + minutes * 60 + seconds
    return 0

