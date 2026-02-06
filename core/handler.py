"""
这个脚本会遍历每条音频，在每个单词实例内部找所有相邻音素对；对每个相邻音素对，如果区间足够长，就生成一组固定长度、重叠的滑动窗口，
并对每个窗口调用 core.core_main() 提取特征，最后把所有窗口的结果写成 CSV。
"""
import logging
import os
os.environ["NUMBA_CUDA_LOG_LEVEL"] = "WARNING"
import sys
import numpy as np
import pandas as pd
import librosa
import core
logging.getLogger("numba.cuda.cudadrv.driver").setLevel(logging.WARNING)
logging.getLogger("numba.cuda.cudadrv.nvvm").setLevel(logging.WARNING)

from tqdm import tqdm
import json
from datetime import datetime

#Global variables
WINDOW_SIZE = 565
OVERLAP = 115

def load_progress(data_name):
    """加载处理进度"""
    progress_file = f'../output/{data_name}/progress.json'
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return set(json.load(f).get('completed_files', []))
    return set()

def save_progress(data_name, completed_files):
    """保存处理进度"""
    progress_file = f'../output/{data_name}/progress.json'
    os.makedirs(os.path.dirname(progress_file), exist_ok=True)
    with open(progress_file, 'w') as f:
        json.dump({
        'completed_files': list(completed_files),
        'last_updated': datetime.now().isoformat(),
        'total_completed': len(completed_files)
        }, f, indent=2)

def is_file_completed(filepath, data_name, df):
    """检查文件是否已经处理完成"""
    try:
        current_speaker = df[df.filepath == filepath]['speaker_id'].iloc[0]
    except:
        current_speaker = os.path.basename(filepath).split('_')[0]

    expected_output = f'../output/{data_name}/{current_speaker}/{current_speaker}_{os.path.basename(filepath)}.csv'

    if os.path.exists(expected_output) and os.path.getsize(expected_output) > 0:
        try:
            return not pd.read_csv(expected_output).empty
        except:
            return False
    return False

def df_read_csv(path):
    """Function to read in the csv containing all the necessary information for the
    acoustic core functionalitry.
    """
    df_read = pd.read_csv(path, sep=',',
            dtype={
                'start_word' : int,
                'end_word': int,
                'word': str,
                'sample_id': str,
                'speaker_id': str,
                'start_phoneme': int,
                'end_phoneme': int,
                'arpabet': str,
                'ipa': str,
                'filename': str,
                'index_phoneme': int
       })
    print("Read csv done...")
    return df_read

def construct_uniform_windows_ph(start, div, end):
    """
    基于双音素边界构建对称滑窗：
    - 始终包含一个以边界为中心的窗口（必要时向内收缩以贴合区间）。
    - 左右按 step=W-OVERLAP 对称扩展，直到覆盖 [start, end]。
    - 区间过短(<WINDOW_SIZE) 时返回空列表（保持原有“跳过”策略）。
    """
    W = WINDOW_SIZE
    step = W - OVERLAP
    seg_len = end - start

    # 区间短于一个窗口：返回空列表，保持原始“跳过”行为
    if seg_len <= W:
        return []

    # 初始中心窗，优先居中在分界处，再做边界收缩
    center = div - W // 2
    center = max(center, start)
    center = min(center, end - W)
    windows = [(int(center), int(center + W))]

    # 向左扩展
    left = center - step
    while left >= start:
        windows.insert(0, (int(left), int(left + W)))
        left -= step

    # 向右扩展
    right = center + step
    while right + W <= end:
        windows.append((int(right), int(right + W)))
        right += step

    return windows

def setup_logger(data_name):
    """配置日志记录器"""
    log_file = f'../output/{data_name}/process.log'
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )
    return logging.getLogger()

def bigram_multi(df_audio, data_name):
    """批量处理音频文件，执行双音素分析"""
    logger = setup_logger(data_name)
    completed_files = load_progress(data_name)
    all_files = df_audio.filepath.unique()
    
    # 只处理未完成的文件
    remaining = [f for f in all_files if f not in completed_files]
    logger.info(f"Total: {len(all_files)}, Remaining: {len(remaining)}")

    if not remaining:
        logger.info("所有文件都已处理完成！")
        return

    verified = set(completed_files)
    
    for path in tqdm(remaining):
        try:
            # 再次检查
            if is_file_completed(path, data_name, df_audio):
                verified.add(path)
                continue

            # sr=None preserves the native sampling rate (expected 16kHz)
            # mono=False loads all channels (shape: [channels, samples])
            curr_audio, fs = librosa.load(path, sr=None)
            file_df = df_audio[df_audio.filepath == path]
            
            if file_df.empty:
                logger.warning(f"Skipping {path}: No metadata found")
                continue
                
            current_speaker = file_df['speaker_id'].iloc[0]
            output_file = f'../output/{data_name}/{current_speaker}/{current_speaker}_{os.path.basename(path)}.csv'
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            all_data = []
            fail_count = 0

            # 按单词分组，确保 bigram 不跨词
            # 改进：增加 start_word, end_word 以唯一标识单词实例（避免同句同词合并）
            for _, word_df in file_df.groupby(['speaker_id', 'sample_id', 'start_word', 'end_word', 'word']):
                # 在单词内部寻找相邻对，使用原版 index 逻辑确保鲁棒性
                max_idx = word_df.index_phoneme.max()
                if pd.isna(max_idx): continue
                
                for i in range(int(max_idx)):
                    # 显式按 index 获取，处理重复/缺失情况
                    first_rows = word_df[word_df.index_phoneme == i]
                    second_rows = word_df[word_df.index_phoneme == i + 1]

                    if first_rows.empty or second_rows.empty:
                        continue
                        
                    first = first_rows.iloc[0]
                    second = second_rows.iloc[0]

                    try:
                        label = f"{first['ipa']} -- {second['ipa']}"
                        windows = construct_uniform_windows_ph(first['start_phoneme'], first['end_phoneme'], second['end_phoneme'])
                        
                        core_meta = {'FS':fs, 'sex':first['sex']}
                        audio_len = len(curr_audio)

                        for win_idx, win in enumerate(windows):
                            # Check bounds
                            if win[0] < 0 or win[1] > audio_len:
                                logger.warning(f"Window out of bounds in {path}: {win}")
                                continue

                            try:
                                acoustic_data, _ = core.core_main(curr_audio[win[0]:win[1]], label, core_meta)

                                all_data.append({
                                    **acoustic_data,
                                    'filepath': path,
                                    'speaker_id': current_speaker,
                                    'start_bigram': int(first['start_phoneme']),
                                    'end_bigram': int(second['end_phoneme']),
                                    'window_start': int(win[0]),
                                    'window_end': int(win[1]),
                                    'window_index': win_idx,
                                    'sex': first['sex']
                                })
                            except Exception as e:
                                fail_count += 1
                                logger.exception(
                                    f"Core extraction failed on window {win_idx} ({label}) in {path} "
                                    f"[{win[0]}:{win[1]}]"
                                )
                                continue
                    except Exception:
                        logger.exception(f"Error processing bigram pair {i} in {path}")

            if all_data:
                pd.DataFrame(all_data).to_csv(output_file, index=False)
                verified.add(path)
                save_progress(data_name, verified)
            else:
                logger.warning(f"No valid data extracted for {path} (fail_count={fail_count})")

        except Exception:
            logger.exception(f"Critical error processing file {path}")

    logger.info("All processing done.")

def main():

    # data_name = 'TIMIT'
    data_name = 'TIMIT_generated'

    # csv_path = '../metadata/timit.csv'
    csv_path = '../metadata/timit_generated.csv'

    bigram_multi(df_read_csv(csv_path), data_name)


if __name__ == "__main__":
    main()
