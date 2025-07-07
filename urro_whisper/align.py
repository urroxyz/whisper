import numpy as np
import torch
import logging
import gzip
import base64
import traceback
import re
import math
import unicodedata
from typing import List, Dict, Tuple, Optional, Union

# alignment specific imports
try:
    import dtw
    from scipy.ndimage import median_filter
except ImportError:
    raise ImportError("please install alignment dependencies: `pip install dtw-python scipy`")

# icu tokenizer import for better word splitting
try:
    import icu
    from icu import Locale, BreakIterator
except ImportError:
    raise ImportError("pyicu is required for word splitting. install with `pip install pyicu`")

logger = logging.getLogger("urro_whisper")  # consistent logger name
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # set a default level if not configured elsewhere
    logger.setLevel(logging.WARNING)

# constants
# define audio constants used in timestamp calculations
SAMPLE_RATE = 16000
HOP_LENGTH = 160  # from whisper feature extraction
AUDIO_SAMPLES_PER_TOKEN = HOP_LENGTH * 2  # each token corresponds to 2 feature frames
AUDIO_TIME_PER_TOKEN = AUDIO_SAMPLES_PER_TOKEN / SAMPLE_RATE  # time duration of one token
DEFAULT_HEAD_DIM = 64  # standard head dimension for whisper models

# constants for sliding window alignment
ALIGNMENT_WINDOW_SIZE_SAMPLES = 30 * SAMPLE_RATE  # 30 seconds
ALIGNMENT_OVERLAP_SAMPLES = 15 * SAMPLE_RATE  # 15 seconds overlap
MIN_ALIGNMENT_WINDOW_SAMPLES = int(5.0 * SAMPLE_RATE)  # minimum 5s window

# alignment head data (precomputed masks)
_ALIGNMENT_HEADS = {
    # standard models - compressed boolean masks indicating which attention heads are good for alignment
    "tiny.en": b"ABzY8J1N>@0{>%R00Bk>$p{7v037`oCl~+#00",
    "tiny": b"ABzY8bu8Lr0{>%RKn9Fp%m@SkK7Kt=7ytkO",
    "base.en": b"ABzY8;40c<0{>%RzzG;p*o+Vo09|#PsxSZm00",
    "base": b"ABzY8KQ!870{>%RzyTQH3`Q^yNP!>##QT-<FaQ7m",
    "small.en": b"ABzY8>?_)10{>%RpeA61k&I|OI3I$65C{;;pbCHh0B{qLQ;+}v00",
    "small": b"ABzY8DmU6=0{>%Rpa?J`kvJ6qF(V^F86#Xh7JUGMK}P<N0000",
    "medium.en": b"ABzY8usPae0{>%R7<zz_OvQ{)4kMa0BMw6u5rT}kRKX;$NfYBv00*Hl@qhsU00",
    "medium": b"ABzY8B0Jh+0{>%R7}kK1fFL7w6%<-Pf*t^=N)Qr&0RR9",
    "large-v1": b"ABzY8r9j$a0{>%R7#4sLmoOs{s)o3~84-RPdcFk!JR<kSfC2yj",
    "large-v2": b'ABzY8zd+h!0{>%R7=D0pU<_bnWW*tkYAhobTNnu$jnkEkXqp)j;w1Tzk)UH3X%SZd&fFZ2fC2yj',
    "large-v3": b"ABzY8gWO1E0{>%R7(9S+Kn!D~%ngiGaR?*L!iJG9p-nab0JQ=-{D1-g00",
    # aliases
    "large": b"ABzY8gWO1E0{>%R7(9S+Kn!D~%ngiGaR?*L!iJG9p-nab0JQ=-{D1-g00",  # large defaults to v3
    # turbo models (distil-large-v3)
    "large-v3-turbo": b"ABzY8j^C+e0{>%RARaKHP%t(lGR*)0g!tONPyhe`",
    "turbo": b"ABzY8j^C+e0{>%RARaKHP%t(lGR*)0g!tONPyhe`",
}


def _get_alignment_heads(model_name, num_layers, num_heads):
    """retrieves and validates the alignment head mask for a given model configuration"""
    if model_name not in _ALIGNMENT_HEADS:
        logger.warning(f"alignment heads not found for model '{model_name}', using default behavior (all heads averaged). timestamps might be less accurate")
        return None  # fallback: average all heads later

    dump = _ALIGNMENT_HEADS[model_name]
    try:
        # decompress and decode the mask
        array = np.frombuffer(gzip.decompress(base64.b85decode(dump)), dtype=bool).copy()
        expected_size = num_layers * num_heads
        if array.size != expected_size:
            # this indicates a mismatch between the compiled mask and the model's reported layers/heads
            logger.warning(f"alignment head data size mismatch for {model_name}. expected {expected_size} ({num_layers}x{num_heads}), got {array.size}. using none")
            return None
        mask = torch.from_numpy(array).reshape(num_layers, num_heads)
        alignment_heads = mask.to_sparse()  # convert to sparse tensor for easier indexing
        logger.info(f"loaded alignment heads for {model_name} with shape: ({num_layers}, {num_heads})")
        return alignment_heads
    except Exception as e:
        logger.error(f"error processing alignment heads for {model_name}: {e}. using none")
        return None


def split_tokens_with_icu(text_tokens: list, tokenizer, lang='en'):
    """
    splits tokens into words using icu word boundaries and maps tokens to words.
    filters out tokens/words that don't contain letters for alignment purposes.
    """
    # decode the relevant text tokens for icu processing
    try:
        decoded_text = tokenizer.decode(text_tokens).strip()
    except Exception as e:
        logger.error(f"failed to decode text tokens: {e}")
        raise RuntimeError(f"token decoding failed: {e}")
    
    if not decoded_text:
        return [], [], []

    try:
        locale = Locale(lang)  # create icu locale for the specified language
        breaker = BreakIterator.createWordInstance(locale)  # create word boundary iterator
    except Exception as e:
        logger.error(f"failed to create icu breakiterator for lang='{lang}', error: {e}")
        raise RuntimeError(f"icu breakiterator creation failed: {e}")

    breaker.setText(decoded_text)

    words = []  # list of identified words
    word_tokens_list = []  # list of token id lists, one per word
    word_indices_list = []  # list of original token index lists, one per word

    token_spans = []  # stores (start_offset, end_offset) in decoded_text for each *letter-containing* token
    running_offset = 0  # track position in decoded_text
    filtered_indices_map = {}  # map original token index -> filtered token index (for letter tokens only)
    filtered_text_tokens = []  # list of token ids containing letters
    filtered_original_indices = []  # original indices corresponding to filtered_text_tokens
    current_filtered_idx = 0

    # first pass: identify letter tokens and their approximate spans in the decoded text
    for original_idx, token_id in enumerate(text_tokens):
        try:
            token_str = tokenizer.decode([token_id])
        except Exception as e:
            logger.error(f"failed to decode token {token_id}: {e}")
            raise RuntimeError(f"token decoding failed: {e}")
        
        token_str_strip = token_str.lstrip()  # use lstrip for finding offset, but keep original token_str

        # inlined logic to check for letter
        is_letter = False
        for char in token_str_strip:
            if unicodedata.category(char).startswith('L'):
                is_letter = True
                break
        
        # only consider tokens with letters for alignment mapping
        if is_letter:
            filtered_text_tokens.append(token_id)
            filtered_original_indices.append(original_idx)
            filtered_indices_map[original_idx] = current_filtered_idx
            current_filtered_idx += 1

            # find the token's position in the decoded string
            try:
                # search for the exact token string first
                offset = decoded_text.index(token_str, running_offset)
                # crude check for large jump, might indicate error due to normalization/whitespace differences
                if offset > running_offset + 10:
                    # try finding the stripped version if exact match jumped too far
                    offset_strip = decoded_text.find(token_str_strip, running_offset)
                    if offset_strip != -1 and abs(offset_strip - running_offset) < abs(offset - running_offset):
                        offset = offset_strip
                        token_len = len(token_str_strip)
                    else:
                        token_len = len(token_str)  # stick with original if stripped wasn't better
                else:
                    token_len = len(token_str)
            except ValueError:  # if exact token not found
                try:  # try finding the stripped version
                    offset = decoded_text.index(token_str_strip, running_offset)
                    token_len = len(token_str_strip)
                except ValueError:  # if still not found, log warning and approximate
                    logger.warning(f"could not accurately map token '{token_str}' (id: {token_id}) to text offset, using approximate position")
                    offset = running_offset  # place it at the current tracked offset
                    token_len = len(token_str_strip)  # use stripped length as best guess

            token_spans.append((offset, offset + token_len))
            running_offset = offset + token_len  # advance offset
        else:
            # even for non-letter tokens, advance the running offset roughly
            try:
                offset = decoded_text.index(token_str, running_offset)
                running_offset = offset + len(token_str)
            except ValueError:
                try:  # try stripped version if original fails
                    offset = decoded_text.index(token_str_strip, running_offset)
                    running_offset = offset + len(token_str_strip)
                except ValueError:
                    pass  # ignore if token can't be found at all
            continue  # skip adding non-letter tokens to spans

    # second pass: iterate through icu word boundaries and map tokens to words
    start = breaker.first()
    for end in breaker:
        word = decoded_text[start:end]
        word_stripped = word.strip()

        # skip whitespace or words without letters (e.g., punctuation-only words)
        is_letter_word = False
        for char in word_stripped:
            if unicodedata.category(char).startswith('L'):
                is_letter_word = True
                break
        
        if not word_stripped or not is_letter_word:
            start = end
            continue

        current_word_token_original_indices = []
        word_start = start  # character offset of the word start
        word_end = end  # character offset of the word end

        # find which *letter-containing* tokens overlap with this word's span
        for i, (tok_start, tok_end) in enumerate(token_spans):
            # simple overlap check: token starts before word ends and token ends after word starts
            if tok_start < word_end and tok_end > word_start:
                # get the original index of this token (relative to the input text_tokens list)
                original_token_idx = filtered_original_indices[i]
                current_word_token_original_indices.append(original_token_idx)

        # if we found tokens belonging to this word
        if current_word_token_original_indices:
            words.append(word_stripped)  # store the cleaned word
            # get the original token indices that are also *letter-containing* tokens
            relative_indices = [idx for idx in current_word_token_original_indices if idx in filtered_indices_map]
            # store the original indices (relative to input text_tokens) and the corresponding token ids
            word_indices_list.append(relative_indices)
            word_tokens_list.append([text_tokens[i] for i in relative_indices])

        start = end  # move to the next word boundary

    return words, word_tokens_list, word_indices_list


def calculate_sliding_windows_for_alignment(
    total_samples: int, 
    transcript_tokens: List[int],
    window_size_samples: int = ALIGNMENT_WINDOW_SIZE_SAMPLES,
    overlap_samples: int = ALIGNMENT_OVERLAP_SAMPLES,
    verbose: bool = False
) -> List[Tuple[int, int, int, int]]:
    """
    calculate sliding windows for alignment mode.
    returns a list of (start_sample, end_sample, token_start_idx, token_end_idx) tuples.
    """
    if verbose:
        logger.info(f"calculating sliding windows for alignment: total_samples={total_samples}, total_tokens={len(transcript_tokens)}")
        logger.info(f"window_size={window_size_samples/SAMPLE_RATE:.1f}s, overlap={overlap_samples/SAMPLE_RATE:.1f}s")
    
    windows = []
    current_start = 0
    total_tokens = len(transcript_tokens)
    
    while current_start < total_samples:
        current_end = min(current_start + window_size_samples, total_samples)
        
        # calculate which tokens correspond to this audio segment
        # use proportional distribution based on audio duration
        progress_start = current_start / total_samples
        progress_end = current_end / total_samples
        
        token_start_idx = int(progress_start * total_tokens)
        token_end_idx = min(int(progress_end * total_tokens), total_tokens)
        
        # ensure we don't have empty token ranges
        if token_start_idx >= total_tokens:
            if verbose:
                logger.info(f"breaking early: token_start_idx ({token_start_idx}) >= total_tokens ({total_tokens})")
            break
            
        if token_end_idx <= token_start_idx:
            token_end_idx = min(token_start_idx + 1, total_tokens)
        
        window_duration = (current_end - current_start) / SAMPLE_RATE
        token_count = token_end_idx - token_start_idx
        
        if verbose:
            logger.info(f"window {len(windows)+1}: audio=[{current_start}-{current_end}] ({window_duration:.1f}s), tokens=[{token_start_idx}-{token_end_idx}] ({token_count} tokens)")
        
        windows.append((current_start, current_end, token_start_idx, token_end_idx))
        
        # move to next window with overlap
        next_start = current_start + (window_size_samples - overlap_samples)
        
        # if we've reached the end, break
        if current_end >= total_samples:
            if verbose:
                logger.info(f"reached end of audio at window {len(windows)}")
            break
        
        # skip if next window would be too small
        if (total_samples - next_start) < MIN_ALIGNMENT_WINDOW_SAMPLES:
            if verbose:
                logger.info(f"skipping final small window: remaining={total_samples - next_start} samples ({(total_samples - next_start)/SAMPLE_RATE:.1f}s)")
            break
        
        current_start = next_start
    
    if verbose:
        logger.info(f"calculated {len(windows)} sliding windows for alignment")
    
    return windows


def merge_overlapping_alignment_results(
    window_results: List[List[Dict]], 
    window_info: List[Tuple[int, int, int, int]],
    verbose: bool = False
) -> List[Dict]:
    """
    robustly merge overlapping alignment results using a two-pass grouping and selection method.
    """
    if verbose:
        logger.info(f"merging alignment results from {len(window_results)} windows")
    
    # pass 1: collect all timed words from all windows
    all_words_with_window = []
    for window_idx, (words, (start_sample, end_sample, _, _)) in enumerate(zip(window_results, window_info)):
        window_start_time = start_sample / SAMPLE_RATE
        window_end_time = end_sample / SAMPLE_RATE
        
        for word in words:
            if word.get('start') is not None and word.get('end') is not None:
                word_with_window = word.copy()
                word_with_window['_window_idx'] = window_idx
                word_with_window['_window_start_time'] = window_start_time
                word_with_window['_window_end_time'] = window_end_time
                all_words_with_window.append(word_with_window)
    
    if not all_words_with_window:
        if verbose:
            logger.info("no timed words found across all windows to merge.")
        return []

    # sort all words by start time to process them chronologically
    all_words_with_window.sort(key=lambda x: x['start'])
    if verbose:
        logger.info(f"collected and sorted {len(all_words_with_window)} timed words from all windows")

    # pass 2: group overlapping words and select the best from each group
    word_candidate_groups = []
    if all_words_with_window:
        # start with the first word in its own group
        word_candidate_groups.append([all_words_with_window[0]])

        for word in all_words_with_window[1:]:
            # get the latest end time from the last group
            last_group = word_candidate_groups[-1]
            last_group_max_end = max(w['end'] for w in last_group)
            
            # if the current word starts after the last group ended, it's a new group
            if word['start'] >= last_group_max_end:
                word_candidate_groups.append([word])
            else:
                # otherwise, it overlaps with the last group, so add it
                last_group.append(word)

    if verbose:
        logger.info(f"formed {len(word_candidate_groups)} candidate groups for merging.")

    # pass 3: select the best candidate from each group
    merged_words = []
    for group in word_candidate_groups:
        if not group:
            continue
        
        # choose the best word from the group based on centrality
        best_word = group[0]
        if len(group) > 1:
            best_centrality = float('inf')
            for word in group:
                word_center = (word['start'] + word['end']) / 2
                window_center = (word['_window_start_time'] + word['_window_end_time']) / 2
                centrality = abs(word_center - window_center)
                
                if centrality < best_centrality:
                    best_centrality = centrality
                    best_word = word
        
        # clean up window metadata and add to the final list
        final_word = {k: v for k, v in best_word.items() if not k.startswith('_')}
        merged_words.append(final_word)

    if verbose:
        logger.info(f"merged to {len(merged_words)} final words")
    
    return merged_words


def perform_sliding_window_alignment(
    chunk_tokens: List[int],
    text_token_ids: List[int],
    hidden_states: np.ndarray,
    decoder_sess,
    tokenizer,
    model: str,
    language: str,
    n_frames: int,
    hidden_dim: int,
    attn_names: List[str],
    ids_name: str,
    states_name: str,
    model_uses_kv_cache_structure: bool,
    initial_past_kv_state: Optional[Dict],
    requires_use_cache_branch: bool,
    prompt_len: int,
    window_start_time_global: float,
    verbose: bool = False
) -> List[Dict]:
    """
    perform alignment for a single sliding window.
    """
    if verbose:
        logger.info(f"performing alignment for {len(text_token_ids)} text tokens")
    
    # get alignment heads
    size_map = {384: "tiny", 512: "base", 768: "small", 1024: "medium", 1280: "large"}
    model_base = size_map.get(hidden_dim, model.split("-")[0].split(".")[0])
    model_name_for_align = (f"{model_base}.en" if (language == "en" and f"{model_base}.en" in _ALIGNMENT_HEADS) else model_base)
    if model in _ALIGNMENT_HEADS:
        model_name_for_align = model
    
    # get model architecture info
    num_layers = 0
    max_layer_idx = -1
    for name in attn_names:
        match = re.search(r"\.(\d+)", name)
        if match:
            max_layer_idx = max(max_layer_idx, int(match.group(1)))
    num_layers = max_layer_idx + 1
    if num_layers == 0:
        layer_map = {"tiny": 4, "base": 6, "small": 12, "medium": 24, "large": 32}
        num_layers = layer_map.get(model_base, 6)
    
    num_heads = hidden_dim // DEFAULT_HEAD_DIM
    if num_heads == 0:
        num_heads = 8
    
    align_heads = _get_alignment_heads(model_name_for_align, num_layers, num_heads)
    
    if align_heads is None:
        logger.warning(f"could not load alignment heads for {model_name_for_align}")
        return []
    
    # extract cross-attentions
    layer_attentions = [[] for _ in range(num_layers)]
    align_failed = False
    align_tokens_input = chunk_tokens[:prompt_len]
    
    for token_idx, token_id in enumerate(text_token_ids):
        if verbose and token_idx % 10 == 0:
            logger.info(f"extracting attention for token {token_idx + 1}/{len(text_token_ids)}")
        
        align_ids_np = np.array([align_tokens_input], dtype=np.int64)
        align_inputs = {ids_name: align_ids_np, states_name: hidden_states.astype(np.float32)}
        if model_uses_kv_cache_structure and initial_past_kv_state is not None:
            align_inputs.update(initial_past_kv_state)
            if requires_use_cache_branch:
                align_inputs["use_cache_branch"] = np.array([False], dtype=bool)
        
        align_req_outputs = attn_names
        try:
            align_outs = decoder_sess.run(align_req_outputs, align_inputs)
            for layer_idx, att_tensor in enumerate(align_outs):
                layer_attentions[layer_idx].append(att_tensor[0, :, -1, :])
        except Exception as e_inner:
            align_failed = True
            logger.error(f"alignment pass failed for token {token_id}: {e_inner}")
            break
        
        align_tokens_input.append(token_id)
    
    # perform dtw alignment
    if not align_failed:
        expected_len = len(text_token_ids)
        if all(len(lst) == expected_len for lst in layer_attentions):
            try:
                attentions = [np.stack(layer_atts, axis=1)[np.newaxis, :, :, :] for layer_atts in layer_attentions]
                if verbose:
                    logger.info(f"successfully extracted attentions")
            except Exception as e_stack:
                logger.error(f"stacking attentions failed: {e_stack}")
                attentions = []
        else:
            logger.warning(f"attention length mismatch")
            attentions = []
        
        if attentions:
            if verbose:
                logger.info(f"performing dtw alignment")
            try:
                raw_window_words = perform_word_alignment(
                    full_token_sequence=chunk_tokens,
                    generated_text_tokens_ids=text_token_ids,
                    cross_attentions_list=attentions,
                    tokenizer=tokenizer,
                    alignment_heads=align_heads,
                    model_n_text_layers=num_layers,
                    n_frames_feature=n_frames,
                    language=language,
                    medfilt_width=7,
                    qk_scale=1.0,
                    debug=verbose
                )
                
                if raw_window_words:
                    if verbose:
                        logger.info(f"dtw successful, got {len(raw_window_words)} words")
                    
                    # convert to global timestamps
                    window_words = []
                    
                    for wt in raw_window_words:
                        start_rel_window = wt.get('start')
                        end_rel_window = wt.get('end')
                        start_glob, end_glob = None, None
                        
                        if start_rel_window is not None:
                            start_glob = round(start_rel_window + window_start_time_global, 3)
                        if end_rel_window is not None:
                            end_glob = round(end_rel_window + window_start_time_global, 3)
                            if start_glob is not None and end_glob < start_glob:
                                end_glob = start_glob
                        
                        if wt.get('word'):
                            window_words.append({
                                "text": wt['word'],
                                "start": start_glob,
                                "end": end_glob
                            })
                    
                    if verbose:
                        logger.info(f"converted to {len(window_words)} global-timed words")
                    return window_words
                else:
                    logger.warning(f"dtw returned empty result")
            except Exception as e_dtw:
                logger.error(f"dtw alignment failed: {e_dtw}")
        else:
            logger.warning(f"no attentions available for dtw")
    else:
        logger.warning(f"alignment failed due to attention extraction failure")
    
    return []


def perform_chunk_alignment(
    chunk_tokens: List[int],
    text_token_ids: List[int],
    hidden_states: np.ndarray,
    decoder_sess,
    tokenizer,
    model: str,
    language: str,
    n_frames: int,
    hidden_dim: int,
    attn_names: List[str],
    ids_name: str,
    states_name: str,
    model_uses_kv_cache_structure: bool,
    initial_past_kv_state: Optional[Dict],
    requires_use_cache_branch: bool,
    prompt_len: int,
    start_sample: int,
    context_dur: float,
    verbose: bool = False
) -> List[Dict]:
    """
    perform alignment for a single chunk in batch mode.
    """
    # get alignment heads
    size_map = {384: "tiny", 512: "base", 768: "small", 1024: "medium", 1280: "large"}
    model_base = size_map.get(hidden_dim, model.split("-")[0].split(".")[0])
    model_name_for_align = (f"{model_base}.en" if (language == "en" and f"{model_base}.en" in _ALIGNMENT_HEADS) else model_base)
    if model in _ALIGNMENT_HEADS:
        model_name_for_align = model

    num_layers = 0
    max_layer_idx = -1
    for name in attn_names:
        match = re.search(r"\.(\d+)", name)
        if match:
            max_layer_idx = max(max_layer_idx, int(match.group(1)))
    num_layers = max_layer_idx + 1
    if num_layers == 0:
        layer_map = {"tiny": 4, "base": 6, "small": 12, "medium": 24, "large": 32}
        num_layers = layer_map.get(model_base, 6)

    num_heads = hidden_dim // DEFAULT_HEAD_DIM
    if num_heads == 0:
        num_heads = 8

    align_heads = _get_alignment_heads(model_name_for_align, num_layers, num_heads)

    if align_heads is None:
        logger.warning(f"could not load alignment heads for {model_name_for_align}")
        return []

    if verbose:
        logger.info(f"extracting cross-attentions for alignment...")

    # attention extraction pass
    layer_attentions = [[] for _ in range(num_layers)]
    align_failed = False
    align_tokens_input = chunk_tokens[:prompt_len]

    for token_idx, token_id in enumerate(text_token_ids):
        align_ids_np = np.array([align_tokens_input], dtype=np.int64)
        align_inputs = {ids_name: align_ids_np, states_name: hidden_states.astype(np.float32)}
        if model_uses_kv_cache_structure and initial_past_kv_state is not None:
            align_inputs.update(initial_past_kv_state)
            if requires_use_cache_branch:
                align_inputs["use_cache_branch"] = np.array([False], dtype=bool)
        align_req_outputs = attn_names
        try:
            align_outs = decoder_sess.run(align_req_outputs, align_inputs)
            for layer_idx, att_tensor in enumerate(align_outs):
                layer_attentions[layer_idx].append(att_tensor[0, :, -1, :])
        except Exception as e_inner:
            align_failed = True
            logger.error(f"alignment pass failed token {token_id}: {e_inner}")
            break
        align_tokens_input.append(token_id)

    # dtw alignment
    if not align_failed:
        expected_len = len(text_token_ids)
        if all(len(lst) == expected_len for lst in layer_attentions):
            try:
                attentions = [np.stack(layer_atts, axis=1)[np.newaxis, :, :, :] for layer_atts in layer_attentions]
                if verbose:
                    logger.info(f"successfully extracted attentions.")
            except Exception as e_stack:
                logger.error(f"stacking attentions failed: {e_stack}")
                attentions = []
        else:
            logger.warning(f"attention length mismatch. alignment aborted.")
            attentions = []

        if attentions and align_heads is not None:
            if verbose:
                logger.info(f"performing dtw alignment")
            try:
                raw_chunk_words = perform_word_alignment(
                    full_token_sequence=chunk_tokens,
                    generated_text_tokens_ids=text_token_ids,
                    cross_attentions_list=attentions,
                    tokenizer=tokenizer,
                    alignment_heads=align_heads,
                    model_n_text_layers=num_layers,
                    n_frames_feature=n_frames,
                    language=language,
                    medfilt_width=7,
                    qk_scale=1.0,
                    debug=verbose
                )
                if raw_chunk_words:
                    if verbose:
                        logger.info(f"dtw successful.")
                    chunk_start_time_global = start_sample / SAMPLE_RATE
                    chunk_words = []
                    for wt in raw_chunk_words:
                        start_rel_segment = wt.get('start')
                        end_rel_segment = wt.get('end')
                        start_glob, end_glob = None, None
                        if start_rel_segment is not None:
                            if start_rel_segment >= context_dur:
                                start_glob = round(max(0, start_rel_segment - context_dur) + chunk_start_time_global, 3)
                            else:
                                continue  # skip word in context
                        if end_rel_segment is not None:
                            end_glob = round(max(0, end_rel_segment - context_dur) + chunk_start_time_global, 3)
                            if start_glob is not None and end_glob < start_glob:
                                end_glob = start_glob
                        if wt.get('word'):
                            chunk_words.append({"text": wt['word'], "start": start_glob, "end": end_glob})
                    return chunk_words
                else:
                    logger.warning(f"dtw alignment failed or returned empty list.")
            except Exception as e_dtw:
                logger.error(f"error during perform_word_alignment call: {e_dtw}\n{traceback.format_exc()}")
        else:  # attentions empty or heads missing
            if not align_heads:
                logger.warning("alignment skipped (heads unavailable).")
            else:
                logger.warning("alignment skipped (attentions empty/stack failed).")
    else:  # align_failed was true
        logger.warning(f"alignment not performed due to attention extraction failure.")

    return []


def perform_word_alignment(
    full_token_sequence,  # includes prompt, prefix, generated tokens
    generated_text_tokens_ids,  # only the generated text tokens (no specials)
    cross_attentions_list,  # list of numpy arrays [batch, heads, seq_len, key_len] per layer
    tokenizer,
    alignment_heads,  # sparse tensor mask from _get_alignment_heads
    model_n_text_layers,  # number of decoder layers
    n_frames_feature,  # number of audio frames in the encoder output (mel spectrogram time dim)
    medfilt_width=7,  # median filter width for attention smoothing
    qk_scale=1.0,  # scaling factor for attention before softmax (usually 1.0)
    debug=False,  # more verbose logging during alignment
    language='en'  # language for word splitting
):
    """
    performs dtw alignment between text tokens and audio frames using cross-attentions.
    returns a list of dictionaries, each containing 'word', 'start', 'end'.
    """
    if not cross_attentions_list:
        logger.warning("no cross-attentions provided for alignment, cannot perform dtw alignment")
        return []
    if not generated_text_tokens_ids:
        logger.warning("no generated text tokens provided for alignment")
        return []

    # attention processing
    if len(cross_attentions_list) != model_n_text_layers:
        logger.warning(f"expected {model_n_text_layers} cross-attention layers, but got {len(cross_attentions_list)}, using available layers")
        model_n_text_layers = len(cross_attentions_list)  # adjust layer count to match data

    # stack attentions: list of (batch, heads, seq_len, key_len) -> (layers, batch, heads, seq_len, key_len)
    try:
        # convert numpy arrays from onnx output to torch tensors
        relevant_attentions = torch.stack([torch.from_numpy(att) for att in cross_attentions_list])
    except Exception as e:
        logger.error(f"error stacking attention tensors: {e}")
        return []

    # handle batch dimension (should ideally be 1)
    if relevant_attentions.shape[1] == 1:
        weights = relevant_attentions.squeeze(1)  # shape: (layers, heads, seq_len, key_len/n_frames)
    elif relevant_attentions.shape[1] > 1:
        logger.warning(f"attention batch size is {relevant_attentions.shape[1]}, expected 1, using first batch element")
        weights = relevant_attentions[:, 0, :, :, :]
    else:
        logger.error(f"invalid attention batch size: {relevant_attentions.shape[1]}")
        return []

    # sequence length from attention tensors (should match generated_text_tokens_ids)
    seq_len_att = weights.shape[2]
    num_text_tokens = len(generated_text_tokens_ids)

    # sanity check and potential trimming if lengths mismatch (can happen with beam search?)
    if seq_len_att != num_text_tokens:
        logger.warning(f"attention sequence length ({seq_len_att}) differs from generated text token count ({num_text_tokens}) for alignment, using {min(seq_len_att, num_text_tokens)} tokens")
        min_len = min(seq_len_att, num_text_tokens)
        # trim attention sequence length to match the shorter one
        weights = weights[:, :, :min_len, :]
        # don't trim generated_text_tokens_ids here; split_tokens will handle the full list later
        # update the effective sequence length used
        seq_len_att = min_len
        if seq_len_att == 0:
            logger.error("no text tokens remain after adjusting for attention length mismatch")
            return []

    # apply alignment heads mask if available and valid
    if alignment_heads is not None and alignment_heads.shape[0] == model_n_text_layers and alignment_heads.shape[1] == weights.shape[1]:
        selected_weights = []
        # get [layer, head] indices from the sparse mask
        head_indices = alignment_heads.indices().T.tolist()
        for layer_idx, head_idx in head_indices:
            # ensure indices are within the bounds of the actual weights tensor
            if layer_idx < weights.shape[0] and head_idx < weights.shape[1]:
                # select weights for this specific head: [seq_len_att, n_frames_feature]
                selected_weights.append(weights[layer_idx, head_idx, :, :n_frames_feature])
            else:
                logger.warning(f"alignment head index ({layer_idx}, {head_idx}) out of bounds for weights shape {weights.shape[:2]}")

        if not selected_weights:
            # if filtering resulted in no heads (e.g., all out of bounds), fallback to averaging all
            logger.warning("no valid alignment heads found/selected after filtering, averaging all heads/layers")
            weights = weights[:, :, :, :n_frames_feature].mean(dim=(0, 1))  # average across layers and heads -> [seq_len_att, n_frames_feature]
        else:
            # average the weights from the selected heads
            weights = torch.stack(selected_weights).mean(dim=0)  # shape: [seq_len_att, n_frames_feature]
    else:  # alignment heads not specified, invalid, or shape mismatch
        if alignment_heads is not None:  # log if mismatch occurred
            logger.warning(f"alignment head shape mismatch ({alignment_heads.shape} vs layers={model_n_text_layers}, heads={weights.shape[1]}), averaging all heads/layers")
        else:  # log if no heads were provided (this is expected if using default)
            logger.info("no alignment heads specified, averaging all heads/layers")
        # average across all layers and heads
        weights = weights[:, :, :, :n_frames_feature].mean(dim=(0, 1))  # shape: [seq_len_att, n_frames_feature]

    # validate weights shape before proceeding to dtw
    if weights.ndim != 2 or weights.shape[0] != seq_len_att or weights.shape[1] > n_frames_feature:
        logger.error(f"unexpected attention weights shape after processing: {weights.shape}, expected ({seq_len_att}, <= {n_frames_feature}), cannot proceed with dtw")
        return []

    # convert final weights to numpy for filtering and dtw
    text_token_weights = weights.float().cpu().numpy()

    if text_token_weights.shape[0] == 0:
        logger.warning("no attention weights found for text tokens after processing, cannot align")
        return []

    # apply median filter along the audio frame axis to smooth attention spikes
    if medfilt_width > 0 and text_token_weights.shape[1] > medfilt_width:
        if debug:
            logger.info(f"applying median filter with width {medfilt_width}")
        text_token_weights = median_filter(text_token_weights, (1, medfilt_width))  # filter each token's attention independently

    # apply softmax scaling (temperature scaling) along the audio frame axis
    if debug:
        logger.info(f"applying softmax scaling with qk_scale={qk_scale}")
    # multiplying by qk_scale increases/decreases the peakiness of the distribution
    text_token_weights = torch.from_numpy(text_token_weights * qk_scale).softmax(dim=-1).numpy()

    # prepare cost matrix for dtw (use negative weights: higher attention = lower cost)
    # cost_matrix shape: [num_text_tokens, num_audio_frames]
    cost_matrix = -text_token_weights

    # dtw alignment
    try:
        if debug:
            logger.info(f"running dtw on cost matrix shape: {cost_matrix.shape}")
        # dtw aligns text tokens (query, index1) to audio frames (template, index2)
        alignment = dtw.dtw(cost_matrix.astype(np.double),  # requires float64
                            keep_internals=False,  # don't need intermediate matrices
                            step_pattern=dtw.stepPattern.symmetric1)  # standard step pattern
        if debug:
            logger.info("dtw finished")
    except Exception as e:
        logger.error(f"error during dtw: {e}, cannot generate word timestamps")
        if debug:
            traceback.print_exc()
        return []  # return empty list on dtw failure

    # extract the alignment path indices
    path_token_indices = alignment.index1  # indices into the text tokens used in dtw (0 to seq_len_att-1)
    path_frame_indices = alignment.index2  # indices into the audio frames (0 to n_frames_feature-1)

    # find frame boundaries where the aligned token index changes
    # np.diff finds changes, > 0 means the token index increased
    token_change_indices = np.where(np.diff(path_token_indices) > 0)[0] + 1
    # get the frame index corresponding to each token boundary
    # pad with frame 0 at the start to represent the beginning of the first token
    token_boundaries_frames = np.pad(path_frame_indices[token_change_indices], (1, 0), constant_values=0)

    # validate boundary count - should be #tokens + 1
    expected_boundaries = seq_len_att + 1  # number of tokens used in cost matrix + 1
    if len(token_boundaries_frames) != expected_boundaries:
        logger.warning(f"dtw boundary count ({len(token_boundaries_frames)}) mismatch with expected token count ({seq_len_att}), adjusting boundaries")
        # attempt to fix boundary array length if it's too short or too long
        if len(token_boundaries_frames) < expected_boundaries:
            diff = expected_boundaries - len(token_boundaries_frames)
            # pad with the last known frame index (or max frames if empty)
            last_val = token_boundaries_frames[-1] if len(token_boundaries_frames) > 0 else n_frames_feature
            token_boundaries_frames = np.pad(token_boundaries_frames, (0, diff), constant_values=last_val)
        else:  # too long
            token_boundaries_frames = token_boundaries_frames[:expected_boundaries]  # truncate
        # ensure the very last boundary doesn't exceed the number of frames
        token_boundaries_frames[-1] = min(token_boundaries_frames[-1], n_frames_feature)

    # map token boundaries to word boundaries
    # split the *original* generated text tokens (before potential length mismatch trim) into words
    # this uses the icu/fallback splitter to get word strings and their corresponding original token indices
    words, word_tokens_list, word_indices_list_rel = split_tokens_with_icu(
        generated_text_tokens_ids[:], tokenizer, lang=language  # pass a copy
    )
    if debug:
        logger.info(f"split into {len(words)} words using icu")

    # calculate anchor time
    # whispers predicts timestamps relative to the start of the *audio segment fed to the encoder*
    # however, it sometimes includes timestamp tokens from the prompt/prefix in its output
    # we need to find the time offset represented by the *last* timestamp token *before* the actual generated text starts
    timestamp_boundaries = {}  # map token index in full_token_sequence to its time value
    current_ts = 0.0
    prompt_len = 0  # heuristic length of the initial prompt (<|sot|><|lang|><|task|><|startoflm|>...)
    found_task_token = False

    # find end of prompt (heuristic: first special or timestamp token after sot)
    # also find all timestamp tokens and their values
    for i, token_id in enumerate(full_token_sequence):
        is_special = token_id >= tokenizer.sot
        is_timestamp = token_id >= tokenizer.timestamp_begin and token_id < tokenizer.eot  # eot is not a timestamp

        # assume prompt ends *before* the first task or timestamp token is encountered
        if not found_task_token and (token_id == tokenizer.task or token_id == tokenizer.transcribe or token_id == tokenizer.translate or is_timestamp):
            prompt_len = i  # index *before* this token
            found_task_token = True

        # store timestamp value if found
        if is_timestamp:
            current_ts = round((token_id - tokenizer.timestamp_begin) * AUDIO_TIME_PER_TOKEN, 3)  # use 3 decimal places
            timestamp_boundaries[i] = current_ts

    # find the index of the first *actual text* token after the initial prompt and forced prefix
    first_gen_text_token_original_idx = -1
    # search starts after the estimated prompt length plus the known prefix length (which was forced)
    # note: prefix length calculation needs care if prefix itself contained special tokens
    search_start_index = prompt_len  # start searching right after the determined prompt
    for i in range(search_start_index, len(full_token_sequence)):
        token_id = full_token_sequence[i]
        # check if it's a regular text token (not special, not timestamp)
        if token_id < tokenizer.sot and token_id < tokenizer.timestamp_begin:
            first_gen_text_token_original_idx = i
            break  # found the first one

    # determine the anchor time based on the last timestamp before the first generated text token
    anchor_time = 0.0
    if first_gen_text_token_original_idx != -1:
        # find indices of timestamp tokens that occurred *before* the first text token
        relevant_ts_indices = sorted([idx for idx in timestamp_boundaries if idx < first_gen_text_token_original_idx])
        if relevant_ts_indices:
            last_ts_index = relevant_ts_indices[-1]
            anchor_time = timestamp_boundaries[last_ts_index]
            if debug:
                try:
                    timestamp_token_text = tokenizer.decode([full_token_sequence[last_ts_index]])
                    logger.info(f"found timestamp token {full_token_sequence[last_ts_index]} ({timestamp_token_text}) at index {last_ts_index} ({anchor_time:.3f}s) before first text token index {first_gen_text_token_original_idx}")
                except Exception as e_decode:
                    logger.info(f"found timestamp token {full_token_sequence[last_ts_index]} at index {last_ts_index} ({anchor_time:.3f}s) before first text token index {first_gen_text_token_original_idx}")
        else:  # no timestamp found before text started
            if debug:
                logger.info("no timestamp token found before the first generated text token, using anchor time 0.0s")
    else:  # couldn't find the first generated text token (e.g., only special tokens generated)
        if debug:
            logger.info("could not determine the first generated text token index, using anchor time 0.0s")

    logger.info(f"timestamp anchor time set to: {anchor_time:.3f}s")

    # generate word timestamps
    word_timestamps_aligned = []
    # track which original token indices (relative to generated_text_tokens_ids) were successfully aligned to a word
    aligned_token_original_indices = set()

    for i, word in enumerate(words):
        # word_indices_list_rel contains indices relative to generated_text_tokens_ids
        relative_indices_for_word = word_indices_list_rel[i]
        if not relative_indices_for_word:
            continue  # skip if word somehow has no tokens

        # filter these indices to only include those that were actually used in the dtw alignment
        # (i.e., indices from 0 up to seq_len_att - 1)
        dtw_indices_for_word = [idx for idx in relative_indices_for_word if idx < seq_len_att]

        if not dtw_indices_for_word:
            # this word consists entirely of tokens that were outside the dtw range (e.g., due to length mismatch)
            if debug:
                logger.warning(f"word '{word}' consists of tokens outside the dtw alignment range (0-{seq_len_att-1}), skipping timing")
            continue

        # find the start/end index for this word within the dtw-aligned tokens
        start_dtw_idx = min(dtw_indices_for_word)
        # end index is inclusive for the last token, so add 1 for slicing the boundaries array
        end_dtw_idx = max(dtw_indices_for_word) + 1

        # get corresponding frame boundaries from the dtw result
        start_frame = token_boundaries_frames[start_dtw_idx]
        # ensure end index doesn't go out of bounds for the boundaries array
        end_frame = token_boundaries_frames[end_dtw_idx] if end_dtw_idx < len(token_boundaries_frames) else n_frames_feature

        # convert frame boundaries to time relative to the start of the dtw alignment window
        start_time_rel = round(start_frame * AUDIO_TIME_PER_TOKEN, 3)
        end_time_rel = round(end_frame * AUDIO_TIME_PER_TOKEN, 3)

        # add the calculated anchor time to get the final timestamp relative to the segment start
        adjusted_start = round(anchor_time + start_time_rel, 3)
        adjusted_end = round(anchor_time + end_time_rel, 3)
        # ensure end time is not before start time after adjustments
        adjusted_end = max(adjusted_start, adjusted_end)

        word_timestamps_aligned.append({
            "word": word,
            "start": adjusted_start,
            "end": adjusted_end,
            "tokens": word_tokens_list[i],  # store token ids for this word (mainly for debug)
            "token_indices": relative_indices_for_word  # store original relative indices (mainly for debug/reconstruction)
        })
        # record the original indices (relative to generated_text_tokens_ids) covered by this word
        aligned_token_original_indices.update(relative_indices_for_word)

    # reconstruct output including potentially skipped tokens
    # the alignment process focuses on letter-containing tokens/words.
    # this step reconstructs the full sequence, adding back words/tokens that were skipped
    # (e.g., punctuation, words without letters) with none timestamps.
    output_with_skipped = []
    word_iter = iter(word_timestamps_aligned)  # iterator over successfully aligned words
    current_word_entry = next(word_iter, None)

    # iterate through the *original* list of generated text tokens
    for idx, token_id in enumerate(generated_text_tokens_ids):
        try:
            token_str = tokenizer.decode([token_id]).strip()
        except Exception as e:
            logger.error(f"failed to decode token {token_id}: {e}")
            continue

        if idx in aligned_token_original_indices:
            # if this token index belongs to an aligned word, add the word entry
            # only add the word entry when encountering the *first* token of that word to avoid duplicates
            if current_word_entry and current_word_entry["token_indices"] and idx == current_word_entry["token_indices"][0]:
                output_with_skipped.append({
                    "word": current_word_entry["word"],
                    "start": current_word_entry["start"],
                    "end": current_word_entry["end"],
                })
                current_word_entry = next(word_iter, None)  # advance to the next aligned word
        elif token_str and token_id < tokenizer.sot:  # add skipped *non-special* tokens back as words with no timestamps
            output_with_skipped.append({
                "word": token_str,
                "start": None,
                "end": None,
            })

    if not output_with_skipped and debug:
        logger.warning("dtw alignment resulted in an empty timestamp list after reconstruction")

    return output_with_skipped


def format_timestamp(seconds):
    """formats seconds into hh:mm:ss.ms string"""
    if seconds is None:
        return "n/a"
    milliseconds = round(seconds * 1000)
    ss = milliseconds // 1000
    ms = milliseconds % 1000
    mm = ss // 60
    ss %= 60
    hh = mm // 60
    mm %= 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}"
