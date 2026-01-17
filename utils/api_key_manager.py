"""APIキー管理モジュール

ユーザーが入力したAPIキーを管理するモジュール。
セッションステートとブラウザのローカルストレージを使用。
"""
import os
import base64
import time
import streamlit as st
from typing import Optional

# streamlit-local-storageのインポート
try:
    from streamlit_local_storage import LocalStorage
    LOCAL_STORAGE_AVAILABLE = True
except ImportError:
    LOCAL_STORAGE_AVAILABLE = False

# ローカルストレージのキー名
API_KEY_STORAGE_KEY = "chat_analysis_openai_api_key"
_cached_api_key: Optional[str] = None


def _get_local_storage():
    """LocalStorageインスタンスを取得"""
    if LOCAL_STORAGE_AVAILABLE:
        return LocalStorage()
    return None


def _encode_key(api_key: str) -> str:
    """APIキーをBase64エンコード（軽い難読化）"""
    return base64.b64encode(api_key.encode()).decode()


def _decode_key(encoded_key: str) -> str:
    """Base64エンコードされたAPIキーをデコード"""
    return base64.b64decode(encoded_key.encode()).decode()


def _set_cached_api_key(api_key: str):
    """環境変数とキャッシュにAPIキーを保持"""
    global _cached_api_key
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        _cached_api_key = api_key

def save_api_key_to_storage(api_key: str) -> bool:
    """
    APIキーをローカルストレージに保存

    Args:
        api_key: OpenAI APIキー

    Returns:
        保存成功時True
    """
    local_storage = _get_local_storage()
    if local_storage:
        try:
            encoded_key = _encode_key(api_key)
            local_storage.setItem(API_KEY_STORAGE_KEY, encoded_key)
            time.sleep(0.1)  # ストレージへの書き込み待機
            _set_cached_api_key(api_key)
            return True
        except Exception as e:
            st.warning(f"APIキーの保存に失敗しました: {e}")
            return False
    return False


def load_api_key_from_storage() -> Optional[str]:
    """
    ローカルストレージからAPIキーを読み込み

    Returns:
        保存されたAPIキー、なければNone
    """
    local_storage = _get_local_storage()
    if local_storage:
        try:
            encoded_key = local_storage.getItem(API_KEY_STORAGE_KEY)
            if encoded_key:
                decoded = _decode_key(encoded_key)
                _set_cached_api_key(decoded)
                return decoded
        except Exception:
            pass
    return None


def delete_api_key_from_storage() -> bool:
    """
    ローカルストレージからAPIキーを削除

    Returns:
        削除成功時True
    """
    local_storage = _get_local_storage()
    if local_storage:
        try:
            local_storage.deleteItem(API_KEY_STORAGE_KEY)
            return True
        except Exception:
            try:
                # deleteItemがない場合は空文字をセット
                local_storage.setItem(API_KEY_STORAGE_KEY, "")
                return True
            except Exception:
                pass
    return False


def validate_api_key(api_key: str) -> bool:
    """
    APIキーの形式を簡易検証

    Args:
        api_key: 検証するAPIキー

    Returns:
        有効な形式の場合True
    """
    if not api_key:
        return False
    # OpenAI APIキーは"sk-"で始まる
    return api_key.startswith("sk-") and len(api_key) > 20


def get_active_api_key() -> Optional[str]:
    """
    有効なAPIキーを取得

    優先順位:
    1. セッションステート（ユーザー入力）
    2. ローカルストレージ（記憶されたキー）
    3. 環境変数（フォールバック）

    Returns:
        有効なAPIキー、なければNone
    """
    # 0. キャッシュ
    if _cached_api_key and validate_api_key(_cached_api_key):
        return _cached_api_key

    # 1. 環境変数（スレッドからも参照可能）
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key and validate_api_key(env_key):
        _set_cached_api_key(env_key)
        return env_key

    # 2. セッションステート
    try:
        if "user_api_key" in st.session_state and st.session_state.user_api_key:
            if validate_api_key(st.session_state.user_api_key):
                _set_cached_api_key(st.session_state.user_api_key)
                return st.session_state.user_api_key
    except Exception:
        pass

    # 3. ローカルストレージ
    stored_key = load_api_key_from_storage()
    if stored_key and validate_api_key(stored_key):
        try:
            st.session_state.user_api_key = stored_key
        except Exception:
            pass
        _set_cached_api_key(stored_key)
        return stored_key

    return None


def mask_api_key(api_key: str) -> str:
    """
    APIキーをマスク表示用に変換

    Args:
        api_key: APIキー

    Returns:
        マスクされたAPIキー（例: sk-...xxxx）
    """
    if not api_key or len(api_key) < 10:
        return "***"
    return f"{api_key[:7]}...{api_key[-4:]}"


def render_api_key_input() -> bool:
    """
    サイドバーにAPIキー入力UIをレンダリング

    Returns:
        有効なAPIキーがあればTrue
    """
    st.subheader("API設定")

    # セッションステートの初期化
    if "user_api_key" not in st.session_state:
        st.session_state.user_api_key = None
    if "api_key_saved_to_storage" not in st.session_state:
        st.session_state.api_key_saved_to_storage = False

    # 現在のAPIキー状態を確認
    current_key = get_active_api_key()
    env_key_exists = os.getenv("OPENAI_API_KEY") is not None
    user_key_exists = st.session_state.user_api_key is not None

    # APIキーが設定されている場合
    if current_key:
        # ユーザー入力のキーか環境変数のキーか判定
        if user_key_exists:
            st.success("APIキーが設定されています")
            st.text(f"現在のキー: {mask_api_key(current_key)}")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("変更", use_container_width=True):
                    st.session_state.show_api_key_form = True
            with col2:
                if st.button("削除", use_container_width=True):
                    st.session_state.user_api_key = None
                    delete_api_key_from_storage()
                    st.session_state.api_key_saved_to_storage = False
                    st.rerun()

            # 変更フォームの表示
            if st.session_state.get("show_api_key_form", False):
                _render_api_key_form()
        else:
            # 環境変数から読み込まれたキー
            st.success("システムAPIキーが設定されています")

            with st.expander("独自のAPIキーを使用する"):
                _render_api_key_form()
    else:
        # APIキーが未設定
        st.warning("APIキーが設定されていません")
        _render_api_key_form()

    return current_key is not None


def _render_api_key_form():
    """APIキー入力フォームをレンダリング"""

    # 表示/非表示トグル（デフォルト: 表示）でChromeのパスワード候補を抑制
    api_key_hidden = st.checkbox(
        "入力を隠す（伏字）",
        value=False,
        key="api_key_hide_checkbox"
    )
    input_type = "password" if api_key_hidden else "default"

    api_key_input = st.text_input(
        "OpenAI APIキー",
        type=input_type,
        placeholder="sk-...",
        help="OpenAIのAPIキーを入力してください",
        key="api_key_input_field"
    )

    # 記憶するチェックボックス
    remember_key = st.checkbox(
        "このブラウザに記憶する",
        value=False,
        help="チェックを入れると、このブラウザに保存されます",
        key="remember_api_key_checkbox"
    )

    # 警告メッセージ（記憶する場合）
    if remember_key:
        st.warning(
            "**セキュリティに関する注意**\n\n"
            "APIキーはこのブラウザのローカルストレージに保存されます。\n"
            "- 共有PCでは使用しないでください\n"
            "- キーは軽い難読化のみで保存されます"
        )

    # 設定ボタン
    if st.button("APIキーを設定", type="primary", use_container_width=True):
        if api_key_input:
            if validate_api_key(api_key_input):
                # セッションステートに保存
                st.session_state.user_api_key = api_key_input
                st.session_state.show_api_key_form = False
                _set_cached_api_key(api_key_input)

                # ローカルストレージに保存（オプション）
                if remember_key:
                    if save_api_key_to_storage(api_key_input):
                        st.session_state.api_key_saved_to_storage = True
                        st.success("APIキーを設定し、ブラウザに保存しました")
                    else:
                        st.success("APIキーを設定しました（ブラウザへの保存は失敗）")
                else:
                    st.success("APIキーを設定しました（セッション中のみ有効）")

                time.sleep(0.5)
                st.rerun()
            else:
                st.error("無効なAPIキー形式です。'sk-'で始まる正しいキーを入力してください。")
        else:
            st.error("APIキーを入力してください。")

    # APIキー取得リンク
    st.caption(
        "[OpenAI APIキーの取得はこちら](https://platform.openai.com/api-keys)"
    )
