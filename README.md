# Whisper Multi-File Audio Transcription

OpenAIのWhisper large-v3モデルを使用して、指定したフォルダ内のすべての音声ファイルを一括で音声認識（文字起こし）するPythonスクリプトです。

## 特徴

- 複数の音声ファイルを一括処理
- GPU（CUDA）利用時の自動最適化
- 複数の音声形式に対応（mp3, wav, m4a, flac, aac）
- 環境変数による設定管理
- **バッチ処理**によるパフォーマンス向上（グループ内同時処理）
- **エラーハンドリング**で破損ファイルにも対応
- **結果保存**（JSON形式）
- **タイムスタンプ**情報の取得
- **多言語対応**と英語翻訳機能
- **進捗表示**と統計情報

## 必要な環境

- Python 3.8以上
- CUDA対応GPU（推奨、CPUでも動作可能）

## インストール

1. リポジトリをクローン：
```bash
git clone <repository-url>
cd whisper-multifile
```

2. 依存関係をインストール（uvを使用）：
```bash
uv sync
```

## 設定

1. `.env`ファイルを作成：
```bash
cp .env.example .env
```

2. `.env`ファイルを編集して設定：
```bash
# 必須設定
AUDIO_FOLDER_PATH=/path/to/your/audio/folder

# オプション設定
OUTPUT_FILE=transcription_results.json
BATCH_SIZE=4                    # バッチサイズ（同時処理ファイル数）
CHUNK_LENGTH_S=30              # 長時間音声のチャンク長（秒）
LANGUAGE=auto                  # 言語（auto/japanese/english等）
TASK=transcribe                # transcribe（音声認識）/translate（英語翻訳）
RETURN_TIMESTAMPS=true         # タイムスタンプ情報を含める
```

## 使用方法

```bash
uv run python main.py
```

## 対応ファイル形式

- MP3 (.mp3)
- WAV (.wav)
- M4A (.m4a)
- FLAC (.flac)
- AAC (.aac)

## 出力例

### コンソール出力
```
設定:
- 音声フォルダ: /path/to/audio
- 出力ファイル: transcription_results.json
- バッチサイズ: 4 （グループ内同時処理ファイル数）
- チャンク長: 30秒
- 言語: auto
- タスク: transcribe
- タイムスタンプ: true
--------------------------------------------------
デバイス: cuda:0
モデルを読み込み中...
モデル読み込み完了
処理対象ファイル数: 12
処理開始: 2024-01-15 14:30:15
バッチ処理モードで実行中...
バッチ処理中 (1-4/12)
バッチ処理中 (5-8/12)
バッチ処理中 (9-12/12)
==================================================
処理完了
処理時間: 0:01:23.456789
成功: 12ファイル
失敗: 0ファイル
結果を保存しました: transcription_results.json
リソースをクリーンアップしました
```

### JSON出力ファイル
```json
{
  "metadata": {
    "processing_date": "2024-01-15T14:30:15",
    "duration_seconds": 83.456789,
    "total_files": 12,
    "successful": 12,
    "failed": 0,
    "settings": {
      "model": "openai/whisper-large-v3",
      "language": "auto",
      "task": "transcribe",
      "batch_size": 4,
      "chunk_length_s": 30,
      "return_timestamps": true
    }
  },
  "results": [
    {
      "file": "audio1.mp3",
      "status": "success",
      "text": "こんにちは、これはテスト音声です。",
      "chunks": [
        {
          "timestamp": [0.0, 2.5],
          "text": "こんにちは、これはテスト音声です。"
        }
      ]
    }
  ]
}
```

## 技術仕様

- **モデル**: OpenAI Whisper Large V3
- **フレームワーク**: Hugging Face Transformers
- **GPU最適化**: CUDA利用時はfloat16精度を使用
- **メモリ効率**: low_cpu_mem_usage=True設定
- **バッチ処理**: グループ内複数ファイル同時処理対応
- **長時間音声**: チャンク分割による効率的処理
- **エラー処理**: 堅牢なエラーハンドリング機能

## トラブルシューティング

### 環境変数エラー
```
エラー: 環境変数 'AUDIO_FOLDER_PATH' が設定されていません。
```
→ `.env`ファイルに`AUDIO_FOLDER_PATH`が正しく設定されているか確認してください。

### フォルダが見つからない
```
フォルダが存在しません: /path/to/folder
```
→ 指定したパスが正しく、フォルダが存在することを確認してください。

### 音声ファイルが見つからない
```
音声ファイルが見つかりませんでした。
```
→ 指定したフォルダに対応形式の音声ファイルがあることを確認してください。

## パフォーマンスについて

### バッチ処理の仕組み
- **BATCH_SIZE=4**の場合：4ファイルずつグループ化
- **グループ内**：4ファイルを同時並列処理
- **グループ間**：順次処理（Group1 → Group2 → Group3...）

### 推奨設定
- **GPU使用時**: BATCH_SIZE=4-8（VRAMに応じて調整）
- **CPU使用時**: BATCH_SIZE=2-4（CPUコア数に応じて調整）
- **大容量ファイル**: BATCH_SIZEを小さく、CHUNK_LENGTH_Sを30に設定