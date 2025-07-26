import torch
import os
import glob
import json
from datetime import datetime
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from dotenv import load_dotenv

def get_audio_files(folder_path):
    """指定フォルダ内のすべての音声ファイルを取得"""
    if not folder_path or not os.path.exists(folder_path):
        print(f"フォルダが存在しません: {folder_path}")
        return []
    
    audio_extensions = ['*.mp3', '*.wav', '*.m4a', '*.flac', '*.aac']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(glob.glob(os.path.join(folder_path, ext)))
        audio_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
    
    return sorted(audio_files)

def save_results_to_file(results, output_path):
    """結果をJSONファイルに保存"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"結果を保存しました: {output_path}")
    except Exception as e:
        print(f"結果の保存に失敗しました: {e}")

def process_audio_file(pipe, audio_file, generate_kwargs):
    """単一の音声ファイルを処理（エラーハンドリング付き）"""
    try:
        result = pipe(audio_file, **generate_kwargs)
        return {
            "file": os.path.basename(audio_file),
            "status": "success",
            "text": result["text"],
            "chunks": result.get("chunks", []) if "chunks" in result else None
        }
    except Exception as e:
        print(f"エラー - {os.path.basename(audio_file)}: {str(e)}")
        return {
            "file": os.path.basename(audio_file),
            "status": "error",
            "error": str(e),
            "text": None,
            "chunks": None
        }

def process_batch(pipe, audio_files, batch_size, generate_kwargs):
    """バッチ処理で音声ファイルを処理"""
    all_results = []
    
    for i in range(0, len(audio_files), batch_size):
        batch = audio_files[i:i + batch_size]
        print(f"バッチ処理中 ({i+1}-{min(i+batch_size, len(audio_files))}/{len(audio_files)})")
        
        try:
            # バッチ処理
            results = pipe(batch, batch_size=len(batch), **generate_kwargs)
            
            # 結果を整理
            for j, result in enumerate(results):
                all_results.append({
                    "file": os.path.basename(batch[j]),
                    "status": "success",
                    "text": result["text"],
                    "chunks": result.get("chunks", []) if "chunks" in result else None
                })
                
        except Exception as e:
            print(f"バッチ処理エラー: {e}")
            # バッチ処理に失敗した場合は個別処理にフォールバック
            print("個別処理にフォールバックします...")
            for audio_file in batch:
                result = process_audio_file(pipe, audio_file, generate_kwargs)
                all_results.append(result)
    
    return all_results

# .envファイルを読み込み
load_dotenv()

# 設定の読み込み
folder_path = os.getenv('AUDIO_FOLDER_PATH')
output_file = os.getenv('OUTPUT_FILE', 'transcription_results.json')
batch_size = int(os.getenv('BATCH_SIZE', '4'))
chunk_length_s = int(os.getenv('CHUNK_LENGTH_S', '30'))
language = os.getenv('LANGUAGE', 'auto')
task = os.getenv('TASK', 'transcribe')  # transcribe or translate
return_timestamps = os.getenv('RETURN_TIMESTAMPS', 'true').lower() == 'true'

if not folder_path:
    print("エラー: 環境変数 'AUDIO_FOLDER_PATH' が設定されていません。")
    print(".envファイルにAUDIO_FOLDER_PATH=/path/to/audio/folderを設定してください。")
    exit(1)

print("設定:")
print(f"- 音声フォルダ: {folder_path}")
print(f"- 出力ファイル: {output_file}")
print(f"- バッチサイズ: {batch_size}")
print(f"- チャンク長: {chunk_length_s}秒")
print(f"- 言語: {language}")
print(f"- タスク: {task}")
print(f"- タイムスタンプ: {return_timestamps}")
print("-" * 50)

# デバイス設定
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

print(f"デバイス: {device}")
print("モデルを読み込み中...")

try:
    # モデル読み込み
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    # パイプライン作成（チャンク設定付き）
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=chunk_length_s,
        torch_dtype=torch_dtype,
        device=device,
    )
    
    print("モデル読み込み完了")
    
except Exception as e:
    print(f"モデル読み込みエラー: {e}")
    exit(1)

# 音声ファイル一覧を取得
audio_files = get_audio_files(folder_path)

if not audio_files:
    print("音声ファイルが見つかりませんでした。")
    exit(1)

print(f"処理対象ファイル数: {len(audio_files)}")

# 生成パラメータを設定
generate_kwargs = {}
if language != 'auto':
    generate_kwargs["language"] = language
if task == 'translate':
    generate_kwargs["task"] = "translate"
if return_timestamps:
    generate_kwargs["return_timestamps"] = True

# 処理開始
start_time = datetime.now()
print(f"処理開始: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

try:
    if batch_size > 1 and len(audio_files) > 1:
        # バッチ処理
        print("バッチ処理モードで実行中...")
        results = process_batch(pipe, audio_files, batch_size, generate_kwargs)
    else:
        # 個別処理
        print("個別処理モードで実行中...")
        results = []
        for i, audio_file in enumerate(audio_files, 1):
            print(f"処理中 ({i}/{len(audio_files)}): {os.path.basename(audio_file)}")
            result = process_audio_file(pipe, audio_file, generate_kwargs)
            results.append(result)
            print(f"完了: {result['status']}")
            if result['status'] == 'success':
                print(f"テキスト: {result['text'][:100]}...")
            print("-" * 30)

    # 結果をまとめて表示
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = len(results) - successful
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 50)
    print("処理完了")
    print(f"処理時間: {duration}")
    print(f"成功: {successful}ファイル")
    print(f"失敗: {failed}ファイル")
    
    # 結果をファイルに保存
    output_data = {
        "metadata": {
            "processing_date": start_time.isoformat(),
            "duration_seconds": duration.total_seconds(),
            "total_files": len(audio_files),
            "successful": successful,
            "failed": failed,
            "settings": {
                "model": model_id,
                "language": language,
                "task": task,
                "batch_size": batch_size,
                "chunk_length_s": chunk_length_s,
                "return_timestamps": return_timestamps
            }
        },
        "results": results
    }
    
    save_results_to_file(output_data, output_file)
    
except Exception as e:
    print(f"処理中に予期しないエラーが発生しました: {e}")
    exit(1)

finally:
    # メモリクリーンアップ
    if 'model' in locals():
        del model
    if 'pipe' in locals():
        del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("リソースをクリーンアップしました")
