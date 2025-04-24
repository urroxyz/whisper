import logging
import numpy as np
import re
import traceback
import math
from typing import Dict, Generator, Union, Optional, List

from .audio.load import load_audio_and_resample, calculate_mel_for_segment
from .model.download import get_onnx_model_paths
from .model.load import load_onnx_models, get_tokenizer_for_model
from .align import (
    perform_word_alignment,
    _get_alignment_heads,
    _ALIGNMENT_HEADS,
    SAMPLE_RATE,
)

logger = logging.getLogger("urro_whisper")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.WARNING) # default level

# constants for chunking
CHUNK_LENGTH_SAMPLES = 30 * SAMPLE_RATE  # 30 seconds * 16000 hz
CONTEXT_WORDS = 5  # number of words for textual context
MAX_CONTEXT_SEARCH_WORDS = 15 # how far back to look for valid timestamps for audio context
MIN_CHUNK_SAMPLES = int(0.2 * SAMPLE_RATE) # minimum audio length (e.g., 0.2s) to process a chunk
DEFAULT_HEAD_DIM = 64 # standard head dimension for whisper models

# <<< define the main function >>>
def whisperer(
    model: str,
    audio: Union[str, np.ndarray], # path or numpy array
    language: str = "en",
    delimiter: str = " -", # initial delimiter only, critical for speaker segmentation
    prompt: Optional[str] = None, # user prompt
    onnx_encoder: Optional[str] = None,
    onnx_decoder: Optional[str] = None,
    verbose: bool = False,
    stream: bool = False, # enable streaming
    max_tokens: int = 448, # max total sequence length limit for decoder, including prompt/prefix
    onnx_providers: Optional[List[str]] = None,
    exclude_providers_on_error: List[str] = ['CoreMLExecutionProvider'] # default excludes coreml
) -> Union[Dict, Generator[str, None, None]]: # return type depends on 'stream'
    """
    transcribes audio using whisper onnx models with chunking, context handling,
    optional user prompt, and forced alignment (if available).

    if stream=true, yields decoded text tokens one by one as a generator.
    the yielded tokens are only the textual content (special tokens like timestamps are skipped).
    the final dictionary containing word timestamps is not available in streaming mode.

    otherwise (stream=false), returns a dictionary with the full transcript,
    word timings (if alignment successful), token ids, and language.

    handles models requiring dummy past_key_values inputs (e.g., large-v3-turbo).
    """
    # --- initial setup common to both modes ---
    task = "transcribe" # hardcode task

    if verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)
    logger.info(f"starting transcription for: {audio if isinstance(audio, str) else 'numpy array'}")
    logger.info(f"using model shorthand: {model}, language: {language}")
    if prompt:
        logger.info(f"using user prompt: '{prompt}'")

    # load and prepare audio
    if isinstance(audio, str):
        audio_data, actual_sr = load_audio_and_resample(audio, verbose=verbose)
    elif isinstance(audio, np.ndarray):
        audio_data = audio.astype(np.float32)
        actual_sr = SAMPLE_RATE # assume 16khz
        logger.info("using provided numpy array as audio input")
        if audio.ndim > 1:
             logger.warning("input numpy array has >1 dimension, averaging channels")
             audio_data = audio_data.mean(axis=-1)
    else:
        raise TypeError("input 'audio' must be a file path (str) or a numpy array")

    if actual_sr != SAMPLE_RATE:
        logger.error(f"audio data sample rate ({actual_sr}) must be {SAMPLE_RATE}")
        raise ValueError(f"incorrect audio sample rate: {actual_sr}")

    total_samples = len(audio_data)
    duration = total_samples / SAMPLE_RATE
    logger.info(f"total audio duration: {duration:.2f}s")

    # load models and tokenizer
    encoder_path, decoder_path = get_onnx_model_paths(
        model, onnx_encoder, onnx_decoder, verbose=verbose
    )
    encoder_sess, decoder_sess = load_onnx_models(
        encoder_path, decoder_path, onnx_providers=onnx_providers, verbose=verbose,
        exclude_providers=exclude_providers_on_error
    )
    tokenizer, is_multilingual = get_tokenizer_for_model(model, language, task)
    eos_token = tokenizer.eot
    sot_token = tokenizer.sot
    sot_prev_token = tokenizer.sot_prev
    no_timestamps_token = tokenizer.no_timestamps
    timestamp_begin_token_id = tokenizer.timestamp_begin

    try:
        timestamp_end_token_id = tokenizer.encode("<|30.00|>", allowed_special="all")[0]
        if verbose: logger.info(f"identified timestamp end token id: {timestamp_end_token_id} (<|30.00|>)")
    except Exception as e_ts_end:
        logger.warning(f"could not encode '<|30.00|>' token: {e_ts_end}. timestamp filtering might be incomplete.")
        timestamp_end_token_id = sot_token - 1
        if verbose: logger.info(f"using fallback timestamp end token id: {timestamp_end_token_id}")

    n_mels_required = 128 if "large-v3" in model else 80

    # decoder model capabilities checks
    decoder_input_names = [inp.name for inp in decoder_sess.get_inputs()]
    decoder_output_names = [out.name for out in decoder_sess.get_outputs()]
    required_kv_cache_inputs = [name for name in decoder_input_names if name.startswith("past_key_values")]
    requires_use_cache_branch = "use_cache_branch" in decoder_input_names
    model_uses_kv_cache_structure = bool(required_kv_cache_inputs)
    attn_names = [name for name in decoder_output_names if "cross_attentions" in name]
    can_align = bool(attn_names)
    ids_name = "input_ids"
    states_name = "encoder_hidden_states"
    logits_name = "logits"

    if model_uses_kv_cache_structure and verbose:
        logger.info("decoder model requires 'past_key_values' inputs.")
        if requires_use_cache_branch: logger.info("decoder model requires 'use_cache_branch' input.")
    if can_align and verbose: logger.info("decoder outputs cross-attentions, alignment will be attempted.")
    elif not can_align and verbose: logger.info("decoder does not output cross-attentions, alignment will be skipped.")

    # speaker diarization compatibility check
    is_speaker_diarization_delimiter = isinstance(delimiter, str) and ("speaker 1" in delimiter.lower() or "person 1" in delimiter.lower())
    is_large_or_medium_model = "medium" in model.lower() or "large" in model.lower()
    if is_speaker_diarization_delimiter and not is_large_or_medium_model:
        logger.warning("speaker diarization with `speaker()` and `person()` is only supported by model sizes medium, large-v1, large-v2, and large-v3")

    # --- end of initial setup ---


    # <<< conditional execution based on 'stream' flag >>>
    if stream:
        # define a nested generator function for streaming logic
        def _perform_streaming_transcription():
            logger.info("streaming output enabled")
            # initialize state variables for the loop (within generator scope)
            offset_samples_stream = 0
            chunk_index_stream = 0
            prefix_stream = delimiter
            context_stream = np.array([], dtype=np.float32)
            hidden_dim_stream = None # to store hidden dim after first chunk

            while offset_samples_stream < total_samples:
                # --- initialize per-iteration variables ---
                mel = None; enc_out = None; hidden_states = None; chunk_tokens = None
                decoding_error = None; text_token_ids = []
                initial_past_kv_state = None
                num_layers_kv = 0; num_heads_kv = 0; head_dim_kv = DEFAULT_HEAD_DIM
                chunk_words_for_context = [] # only needed for context prep in streaming

                chunk_index_stream += 1
                start_sample = offset_samples_stream
                end_sample = min(offset_samples_stream + CHUNK_LENGTH_SAMPLES, total_samples)
                chunk_duration = (end_sample - start_sample) / SAMPLE_RATE

                if (end_sample - start_sample) < MIN_CHUNK_SAMPLES and chunk_index_stream > 1:
                    logger.info(f"skipping very short final segment ({chunk_duration:.2f}s)")
                    break

                logger.info(f"processing chunk {chunk_index_stream}: samples {start_sample}-{end_sample} ({chunk_duration:.2f}s)")

                # prepare audio segment
                chunk_audio = audio_data[start_sample:end_sample]
                segment = np.concatenate([context_stream, chunk_audio])
                segment_dur = len(segment) / SAMPLE_RATE
                context_dur_stream = len(context_stream) / SAMPLE_RATE # use stream context duration

                if len(segment) < MIN_CHUNK_SAMPLES:
                    logger.warning(f"chunk {chunk_index_stream}: segment audio too short ({segment_dur:.2f}s), skipping")
                    offset_samples_stream = end_sample
                    context_stream = np.array([], dtype=np.float32)
                    prefix_stream = delimiter
                    # cleanup optional
                    del mel, enc_out, hidden_states, chunk_tokens, decoding_error, text_token_ids, initial_past_kv_state
                    continue

                # feature extraction
                try:
                    mel = calculate_mel_for_segment(segment, model, n_mels_required, verbose=verbose)
                except Exception as e:
                    logger.error(f"mel calculation failed for chunk {chunk_index_stream}: {e}\n{traceback.format_exc()}")
                    offset_samples_stream = end_sample; context_stream = np.array([], dtype=np.float32); prefix_stream = delimiter
                    del mel
                    continue

                # encoder run
                enc_input_name = encoder_sess.get_inputs()[0].name
                try:
                    enc_out = encoder_sess.run(None, {enc_input_name: mel.astype(np.float32)})
                    hidden_states = enc_out[0]
                    # store hidden_dim on first successful chunk if not already set
                    if hidden_dim_stream is None: hidden_dim_stream = hidden_states.shape[-1]
                except Exception as e:
                     logger.error(f"onnx encoder failed for chunk {chunk_index_stream}: {e}\n{traceback.format_exc()}")
                     offset_samples_stream = end_sample; context_stream = np.array([], dtype=np.float32); prefix_stream = delimiter
                     del mel, enc_out, hidden_states
                     continue

                # --- initialize dummy kv state if needed (using stored hidden_dim) ---
                if model_uses_kv_cache_structure and hidden_dim_stream is not None:
                    try:
                        max_layer_idx_kv = -1
                        for name in required_kv_cache_inputs:
                            match = re.search(r"past_key_values\.(\d+)\.", name)
                            if match: max_layer_idx_kv = max(max_layer_idx_kv, int(match.group(1)))
                        num_layers_kv = max_layer_idx_kv + 1

                        if hidden_dim_stream % DEFAULT_HEAD_DIM == 0:
                            num_heads_kv = hidden_dim_stream // DEFAULT_HEAD_DIM
                            head_dim_kv = DEFAULT_HEAD_DIM
                        else:
                            logger.warning(f"cannot accurately determine num_heads from hidden_dim {hidden_dim_stream}. guessing.")
                            layer_to_heads = {4: 6, 6: 8, 12: 12, 24: 16, 32: 20}
                            num_heads_kv = layer_to_heads.get(num_layers_kv, 8)
                            head_dim_kv = DEFAULT_HEAD_DIM

                        if num_layers_kv > 0 and num_heads_kv > 0:
                            initial_past_kv_state = {}
                            kv_shape = (1, num_heads_kv, 0, head_dim_kv)
                            for name in required_kv_cache_inputs:
                                initial_past_kv_state[name] = np.empty(kv_shape, dtype=np.float32)
                            if verbose: logger.info(f"chunk {chunk_index_stream}: initialized dummy kv state.")
                        else:
                            logger.error(f"chunk {chunk_index_stream}: failed to determine valid dimensions for dummy kv state. disabling kv.")
                            initial_past_kv_state = None # ensure it's none if invalid

                    except Exception as e_kv_init:
                        logger.error(f"chunk {chunk_index_stream}: error initializing dummy kv state: {e_kv_init}. disabling kv.")
                        initial_past_kv_state = None


                # --- prepare decoder prompt ---
                prompt_tokens = []
                if prompt and isinstance(prompt, str):
                    try:
                        if sot_prev_token is not None:
                            encoded_prompt = tokenizer.encode(prompt.strip(), allowed_special="all")
                            if encoded_prompt: prompt_tokens = [sot_prev_token] + encoded_prompt
                        else: logger.warning("tokenizer missing 'sot_prev', cannot add user prompt marker.")
                    except Exception as e: logger.error(f"failed to encode user prompt: {e}")

                standard_prompt_tokens = [sot_token]
                if is_multilingual:
                    try: standard_prompt_tokens.append(tokenizer.encode(f"<|{language}|>", allowed_special="all")[0])
                    except Exception: pass
                try: standard_prompt_tokens.append(tokenizer.encode(f"<|{task}|>", allowed_special="all")[0])
                except Exception as e: logger.error(f"failed encode task: {e}"); raise

                include_timestamps = no_timestamps_token is None or no_timestamps_token not in prompt_tokens
                if include_timestamps: standard_prompt_tokens.append(timestamp_begin_token_id)
                elif no_timestamps_token is not None: standard_prompt_tokens.append(no_timestamps_token)
                else: standard_prompt_tokens.append(timestamp_begin_token_id) # fallback if disable token missing

                prefix_tokens = []
                try:
                    current_prefix_str = prefix_stream if isinstance(prefix_stream, str) else ""
                    encoded_prefix = tokenizer.encode(current_prefix_str, allowed_special="all")
                    if encoded_prefix: prefix_tokens = encoded_prefix
                except Exception as e: logger.error(f"failed encode prefix: {e}")

                chunk_tokens = prompt_tokens + standard_prompt_tokens + prefix_tokens
                prompt_len = len(chunk_tokens)
                max_new_tokens = max(0, max_tokens - prompt_len)

                if verbose:
                    logger.info(f"chunk {chunk_index_stream}: initial prompt tokens ({prompt_len}): {chunk_tokens}")
                    logger.info(f"chunk {chunk_index_stream}: starting generation (max new={max_new_tokens})")

                decoding_error = None
                new_tokens_stream = [] # store generated token ids for context

                # --- generation loop ---
                try:
                    if max_new_tokens > 0:
                        current_gen_tokens = chunk_tokens[:]
                        for step in range(max_new_tokens):
                            input_ids_np = np.array([current_gen_tokens], dtype=np.int64)
                            decoder_inputs = {ids_name: input_ids_np, states_name: hidden_states.astype(np.float32)}
                            if model_uses_kv_cache_structure and initial_past_kv_state is not None:
                                decoder_inputs.update(initial_past_kv_state)
                                if requires_use_cache_branch: decoder_inputs["use_cache_branch"] = np.array([False], dtype=bool)

                            try:
                                decoder_outputs = decoder_sess.run([logits_name], decoder_inputs)
                            except Exception as e_dec_run:
                                if "are missing from input feed" in str(e_dec_run):
                                    error_msg = getattr(e_dec_run, 'args', [''])[0] if isinstance(e_dec_run, (ValueError, RuntimeError)) else ''
                                    logger.error(f"decoder run failed (step {step+1}) - missing inputs! required: {error_msg}.")
                                else: logger.error(f"decoder run failed (step {step+1}): {e_dec_run}")
                                raise e_dec_run

                            logits = decoder_outputs[0]
                            next_token_logits = logits[0, -1, :]
                            next_token = int(np.argmax(next_token_logits))

                            if next_token == eos_token:
                                if verbose: logger.info(f"chunk {chunk_index_stream}: eos token detected at step {step+1}")
                                break

                            current_gen_tokens.append(next_token)
                            new_tokens_stream.append(next_token) # add to list for context prep

                            # <<< yield token text >>>
                            is_timestamp = timestamp_begin_token_id <= next_token <= timestamp_end_token_id
                            if next_token < sot_token and not is_timestamp:
                                try:
                                    token_text = tokenizer.decode([next_token])
                                    if token_text and not token_text.startswith("�"):
                                         yield token_text #<<< the yield happens here >>>
                                except Exception as e_stream_decode:
                                     logger.warning(f"failed to decode token {next_token} for streaming: {e_stream_decode}")

                        if step == max_new_tokens - 1 and next_token != eos_token:
                            logger.info(f"chunk {chunk_index_stream}: reached max generated token limit ({max_new_tokens})")
                    else:
                         logger.warning(f"chunk {chunk_index_stream}: prompt length meets or exceeds max_tokens, no tokens generated")

                except Exception as e:
                    decoding_error = e
                    logger.error(f"error during decoder generation loop for chunk {chunk_index_stream}: {decoding_error}\n{traceback.format_exc()}")


                # --- context preparation (simplified for streaming, no alignment needed) ---
                text_token_ids_stream = [t for t in new_tokens_stream if t < sot_token and not (timestamp_begin_token_id <= t <= timestamp_end_token_id)]
                chunk_raw_text_stream = ""
                if text_token_ids_stream:
                    try:
                        chunk_raw_text_stream = tokenizer.decode(text_token_ids_stream).strip()
                        # create basic word list just for text context prefix
                        words_only = chunk_raw_text_stream.split()
                        chunk_words_for_context = [{"text": w} for w in words_only]
                    except Exception as e_decode:
                         logger.error(f"failed decode text tokens for stream context: {e_decode}")

                context_stream = np.array([], dtype=np.float32) # audio context not used between chunks in basic stream
                next_prefix_stream = delimiter

                if chunk_words_for_context:
                    context_word_data = chunk_words_for_context[-CONTEXT_WORDS:]
                    context_text_parts = [wt['text'] for wt in context_word_data]

                    current_delimiter = delimiter
                    potential_delimiters = {"-", ">>", "<"}
                    if isinstance(delimiter, str) and delimiter.strip(): potential_delimiters.add(delimiter.strip())
                    if is_speaker_diarization_delimiter: potential_delimiters.update(["speaker 1:", "person 1:", "speaker 2:", "person 2:"])

                    prefix_search_limit = min(len(chunk_words_for_context), MAX_CONTEXT_SEARCH_WORDS * 2)
                    found_delimiter = False
                    for wt in reversed(chunk_words_for_context[-prefix_search_limit:]):
                        word_text_stripped = wt.get('text', '').strip()
                        if word_text_stripped in potential_delimiters:
                            current_delimiter = word_text_stripped + " "
                            found_delimiter = True; break

                    if context_text_parts:
                         last_word = context_text_parts[-1]
                         clean_last_word = last_word.strip(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""")
                         if clean_last_word: context_text_parts[-1] = clean_last_word
                         elif len(context_text_parts) > 1: context_text_parts.pop()

                    context_text_joined = " ".join(context_text_parts)
                    next_prefix_stream = (current_delimiter + context_text_joined).strip()
                    if not next_prefix_stream: next_prefix_stream = delimiter
                else:
                     if decoding_error is None: logger.warning(f"chunk {chunk_index_stream}: no words for context")
                     next_prefix_stream = delimiter

                prefix_stream = next_prefix_stream # set prefix for the next iteration
                offset_samples_stream = end_sample # advance offset

                # cleanup per-iteration vars
                del mel, enc_out, hidden_states, chunk_tokens, decoding_error
                del text_token_ids, initial_past_kv_state, new_tokens_stream, chunk_words_for_context

                if verbose: logger.info(f"--- end of stream chunk {chunk_index_stream} ---")
            # end of while loop for streaming
            logger.info("streaming finished.")
            # generator implicitly ends here

        # return the generator *iterator* created by calling the nested function
        return _perform_streaming_transcription()

    else: # stream=false, execute non-streaming logic
        logger.info("streaming output disabled, performing batch transcription.")
        # --- batch mode: initialize accumulators ---
        all_words = [] # holds word timestamp dicts
        raw_chunk_texts = [] # holds raw text strings per chunk
        all_tokens = [] # holds all generated token ids
        offset_samples = 0
        chunk_index = 0
        prefix = delimiter
        context = np.array([], dtype=np.float32)
        hidden_dim = None # store hidden dim after first chunk

        # --- batch mode: main loop (no yield here) ---
        while offset_samples < total_samples:
            # --- initialize per-iteration variables ---
            mel = None; enc_out = None; hidden_states = None; chunk_tokens = None
            decoding_error = None; output_tokens = None; text_token_ids = []
            chunk_words = []; align_trace = None; attentions = []
            layer_attentions = []; align_outs = None; new_tokens = []
            chunk_raw_text = ""
            initial_past_kv_state = None
            num_layers_kv = 0; num_heads_kv = 0; head_dim_kv = DEFAULT_HEAD_DIM

            chunk_index += 1
            start_sample = offset_samples
            end_sample = min(offset_samples + CHUNK_LENGTH_SAMPLES, total_samples)
            chunk_duration = (end_sample - start_sample) / SAMPLE_RATE

            if (end_sample - start_sample) < MIN_CHUNK_SAMPLES and chunk_index > 1:
                logger.info(f"skipping very short final segment ({chunk_duration:.2f}s)")
                break

            logger.info(f"processing chunk {chunk_index}: samples {start_sample}-{end_sample} ({chunk_duration:.2f}s)")

            # prepare audio segment
            chunk_audio = audio_data[start_sample:end_sample]
            segment = np.concatenate([context, chunk_audio])
            segment_dur = len(segment) / SAMPLE_RATE
            context_dur = len(context) / SAMPLE_RATE # use batch context duration

            if len(segment) < MIN_CHUNK_SAMPLES:
                logger.warning(f"chunk {chunk_index}: segment audio too short ({segment_dur:.2f}s), skipping")
                offset_samples = end_sample
                context = np.array([], dtype=np.float32)
                prefix = delimiter
                # cleanup optional
                del mel, enc_out, hidden_states, chunk_tokens, decoding_error, output_tokens, text_token_ids, chunk_words, align_trace, attentions, layer_attentions, align_outs, new_tokens, chunk_raw_text, initial_past_kv_state
                continue

            # feature extraction
            try:
                mel = calculate_mel_for_segment(segment, model, n_mels_required, verbose=verbose)
            except Exception as e:
                logger.error(f"mel calculation failed for chunk {chunk_index}: {e}\n{traceback.format_exc()}")
                offset_samples = end_sample; context = np.array([], dtype=np.float32); prefix = delimiter
                del mel
                continue

            # encoder run
            enc_input_name = encoder_sess.get_inputs()[0].name
            try:
                enc_out = encoder_sess.run(None, {enc_input_name: mel.astype(np.float32)})
                hidden_states = enc_out[0]
                n_frames = hidden_states.shape[1] # needed for alignment
                if hidden_dim is None: hidden_dim = hidden_states.shape[-1] # store hidden dim
            except Exception as e:
                 logger.error(f"onnx encoder failed for chunk {chunk_index}: {e}\n{traceback.format_exc()}")
                 offset_samples = end_sample; context = np.array([], dtype=np.float32); prefix = delimiter
                 del mel, enc_out, hidden_states
                 continue

            # --- initialize dummy kv state if needed (using stored hidden_dim) ---
            if model_uses_kv_cache_structure and hidden_dim is not None:
                try:
                    max_layer_idx_kv = -1
                    for name in required_kv_cache_inputs:
                        match = re.search(r"past_key_values\.(\d+)\.", name)
                        if match: max_layer_idx_kv = max(max_layer_idx_kv, int(match.group(1)))
                    num_layers_kv = max_layer_idx_kv + 1

                    if hidden_dim % DEFAULT_HEAD_DIM == 0:
                        num_heads_kv = hidden_dim // DEFAULT_HEAD_DIM
                        head_dim_kv = DEFAULT_HEAD_DIM
                    else:
                        logger.warning(f"cannot accurately determine num_heads from hidden_dim {hidden_dim}. guessing.")
                        layer_to_heads = {4: 6, 6: 8, 12: 12, 24: 16, 32: 20}
                        num_heads_kv = layer_to_heads.get(num_layers_kv, 8)
                        head_dim_kv = DEFAULT_HEAD_DIM

                    if num_layers_kv > 0 and num_heads_kv > 0:
                        initial_past_kv_state = {}
                        kv_shape = (1, num_heads_kv, 0, head_dim_kv)
                        for name in required_kv_cache_inputs:
                            initial_past_kv_state[name] = np.empty(kv_shape, dtype=np.float32)
                        if verbose: logger.info(f"chunk {chunk_index}: initialized dummy kv state.")
                    else:
                        logger.error(f"chunk {chunk_index}: failed to determine valid dimensions for dummy kv state. disabling kv.")
                        initial_past_kv_state = None # ensure it's none if invalid

                except Exception as e_kv_init:
                    logger.error(f"chunk {chunk_index}: error initializing dummy kv state: {e_kv_init}. disabling kv.")
                    initial_past_kv_state = None

            # --- prepare decoder prompt (same as streaming) ---
            prompt_tokens = []
            if prompt and isinstance(prompt, str):
                try:
                    if sot_prev_token is not None:
                        encoded_prompt = tokenizer.encode(prompt.strip(), allowed_special="all")
                        if encoded_prompt: prompt_tokens = [sot_prev_token] + encoded_prompt
                    else: logger.warning("tokenizer missing 'sot_prev', cannot add user prompt marker.")
                except Exception as e: logger.error(f"failed to encode user prompt: {e}")

            standard_prompt_tokens = [sot_token]
            if is_multilingual:
                try: standard_prompt_tokens.append(tokenizer.encode(f"<|{language}|>", allowed_special="all")[0])
                except Exception: pass
            try: standard_prompt_tokens.append(tokenizer.encode(f"<|{task}|>", allowed_special="all")[0])
            except Exception as e: logger.error(f"failed encode task: {e}"); raise

            include_timestamps = no_timestamps_token is None or no_timestamps_token not in prompt_tokens
            if include_timestamps: standard_prompt_tokens.append(timestamp_begin_token_id)
            elif no_timestamps_token is not None: standard_prompt_tokens.append(no_timestamps_token)
            else: standard_prompt_tokens.append(timestamp_begin_token_id)

            prefix_tokens = []
            try:
                current_prefix_str = prefix if isinstance(prefix, str) else ""
                encoded_prefix = tokenizer.encode(current_prefix_str, allowed_special="all")
                if encoded_prefix: prefix_tokens = encoded_prefix
            except Exception as e: logger.error(f"failed encode prefix: {e}")

            chunk_tokens = prompt_tokens + standard_prompt_tokens + prefix_tokens
            prompt_len = len(chunk_tokens)
            max_new_tokens = max(0, max_tokens - prompt_len)

            if verbose:
                logger.info(f"chunk {chunk_index}: initial prompt tokens ({prompt_len}): {chunk_tokens}")
                logger.info(f"chunk {chunk_index}: starting generation (max new={max_new_tokens})")

            decoding_error = None
            new_tokens = [] # store generated token ids for this chunk

            # --- generation loop (no yield here) ---
            try:
                if max_new_tokens > 0:
                    current_gen_tokens = chunk_tokens[:]
                    for step in range(max_new_tokens):
                        input_ids_np = np.array([current_gen_tokens], dtype=np.int64)
                        decoder_inputs = {ids_name: input_ids_np, states_name: hidden_states.astype(np.float32)}
                        if model_uses_kv_cache_structure and initial_past_kv_state is not None:
                            decoder_inputs.update(initial_past_kv_state)
                            if requires_use_cache_branch: decoder_inputs["use_cache_branch"] = np.array([False], dtype=bool)

                        try:
                            decoder_outputs = decoder_sess.run([logits_name], decoder_inputs)
                        except Exception as e_dec_run:
                            if "are missing from input feed" in str(e_dec_run):
                                error_msg = getattr(e_dec_run, 'args', [''])[0] if isinstance(e_dec_run, (ValueError, RuntimeError)) else ''
                                logger.error(f"decoder run failed (step {step+1}) - missing inputs! required: {error_msg}.")
                            else: logger.error(f"decoder run failed (step {step+1}): {e_dec_run}")
                            raise e_dec_run

                        logits = decoder_outputs[0]
                        next_token_logits = logits[0, -1, :]
                        next_token = int(np.argmax(next_token_logits))

                        if next_token == eos_token:
                            if verbose: logger.info(f"chunk {chunk_index}: eos token detected at step {step+1}")
                            break

                        current_gen_tokens.append(next_token)
                        new_tokens.append(next_token) # add to list for processing/accumulation

                    if step == max_new_tokens - 1 and next_token != eos_token:
                        logger.info(f"chunk {chunk_index}: reached max generated token limit ({max_new_tokens})")

                    # update chunk_tokens to include generated ones for alignment context
                    chunk_tokens = current_gen_tokens[:]

                else:
                    logger.warning(f"chunk {chunk_index}: prompt length meets or exceeds max_tokens, no tokens generated")

            except Exception as e:
                decoding_error = e
                logger.error(f"error during decoder generation loop for chunk {chunk_index}: {decoding_error}\n{traceback.format_exc()}")


            # --- batch post-processing, alignment, and context prep ---
            all_tokens.extend(new_tokens) # accumulate token ids

            chunk_words = [] # reset for this chunk
            text_token_ids = []
            chunk_raw_text = ""

            try:
                output_tokens = new_tokens
                text_token_ids = [t for t in output_tokens if t < sot_token and not (timestamp_begin_token_id <= t <= timestamp_end_token_id)]

                # decode raw text for this chunk
                if text_token_ids:
                    try:
                        chunk_raw_text = tokenizer.decode(text_token_ids).strip()
                        if verbose: logger.info(f"chunk {chunk_index} raw decoded text: '{chunk_raw_text}'")
                        if chunk_raw_text:
                            raw_chunk_texts.append(chunk_raw_text) # accumulate raw text
                    except Exception as e_decode:
                        logger.error(f"failed to decode text tokens for chunk {chunk_index}: {e_decode}")

                if not text_token_ids and decoding_error is None:
                     logger.warning(f"chunk {chunk_index}: no text tokens generated after prompt")

                # perform alignment if possible and needed
                if text_token_ids and can_align and include_timestamps and hidden_dim is not None:
                    # (alignment logic is complex and remains the same as before)
                    size_map = {384: "tiny", 512: "base", 768: "small", 1024: "medium", 1280: "large"}
                    model_base = size_map.get(hidden_dim, model.split("-")[0].split(".")[0])
                    model_name_for_align = (f"{model_base}.en" if (language == "en" and f"{model_base}.en" in _ALIGNMENT_HEADS) else model_base)
                    if model in _ALIGNMENT_HEADS: model_name_for_align = model

                    num_layers = 0; max_layer_idx = -1
                    for name in attn_names:
                        match = re.search(r"\.(\d+)", name);
                        if match: max_layer_idx = max(max_layer_idx, int(match.group(1)))
                    num_layers = max_layer_idx + 1
                    if num_layers == 0:
                        layer_map = {"tiny": 4, "base": 6, "small": 12, "medium": 24, "large": 32};
                        num_layers = layer_map.get(model_base, 6)

                    num_heads = num_heads_kv if num_heads_kv > 0 else hidden_dim // DEFAULT_HEAD_DIM
                    if num_heads == 0: num_heads = 8

                    align_heads = _get_alignment_heads(model_name_for_align, num_layers, num_heads)

                    if align_heads is None: logger.warning(f"could not load alignment heads for {model_name_for_align}")
                    if verbose: logger.info(f"chunk {chunk_index}: extracting cross-attentions for alignment...")

                    # attention extraction pass
                    layer_attentions = [[] for _ in range(num_layers)]
                    align_failed = False
                    align_tokens_input = chunk_tokens[:prompt_len]

                    for token_idx, token_id in enumerate(text_token_ids):
                        align_ids_np = np.array([align_tokens_input], dtype=np.int64)
                        align_inputs = {ids_name: align_ids_np, states_name: hidden_states.astype(np.float32)}
                        if model_uses_kv_cache_structure and initial_past_kv_state is not None:
                            align_inputs.update(initial_past_kv_state)
                            if requires_use_cache_branch: align_inputs["use_cache_branch"] = np.array([False], dtype=bool)
                        align_req_outputs = attn_names
                        try:
                            align_outs = decoder_sess.run(align_req_outputs, align_inputs)
                            for layer_idx, att_tensor in enumerate(align_outs):
                                layer_attentions[layer_idx].append(att_tensor[0, :, -1, :])
                        except Exception as e_inner:
                            align_failed = True; logger.error(f"alignment pass failed token {token_id}: {e_inner}"); break
                        align_tokens_input.append(token_id)

                    # dtw alignment
                    if not align_failed:
                        expected_len = len(text_token_ids)
                        if all(len(lst) == expected_len for lst in layer_attentions):
                            try:
                                attentions = [np.stack(layer_atts, axis=1)[np.newaxis, :, :, :] for layer_atts in layer_attentions]
                                if verbose: logger.info(f"successfully extracted attentions.")
                            except Exception as e_stack: logger.error(f"stacking attentions failed: {e_stack}"); attentions = []
                        else: logger.warning(f"attention length mismatch. alignment aborted."); attentions = []

                        if attentions and align_heads is not None:
                            if verbose: logger.info(f"chunk {chunk_index}: performing dtw alignment")
                            try:
                                raw_chunk_words = perform_word_alignment(
                                    full_token_sequence=chunk_tokens, generated_text_tokens_ids=text_token_ids,
                                    cross_attentions_list=attentions, tokenizer=tokenizer, alignment_heads=align_heads,
                                    model_n_text_layers=num_layers, n_frames_feature=n_frames, language=language,
                                    medfilt_width=7, qk_scale=1.0, debug=verbose )
                                if raw_chunk_words:
                                    if verbose: logger.info(f"chunk {chunk_index}: dtw successful.")
                                    chunk_start_time_global = start_sample / SAMPLE_RATE
                                    temp_chunk_words = []
                                    for wt in raw_chunk_words:
                                        start_rel_segment = wt.get('start'); end_rel_segment = wt.get('end')
                                        start_glob, end_glob = None, None
                                        if start_rel_segment is not None:
                                            if start_rel_segment >= context_dur:
                                                start_glob = round(max(0, start_rel_segment - context_dur) + chunk_start_time_global, 3)
                                            else: continue # skip word in context
                                        if end_rel_segment is not None:
                                            end_glob = round(max(0, end_rel_segment - context_dur) + chunk_start_time_global, 3)
                                            if start_glob is not None and end_glob < start_glob: end_glob = start_glob
                                        if wt.get('word'):
                                            temp_chunk_words.append({"text": wt['word'], "start": start_glob, "end": end_glob})
                                    chunk_words = temp_chunk_words # assign aligned words
                                else: logger.warning(f"chunk {chunk_index}: dtw alignment failed or returned empty list.")
                            except Exception as e_dtw:
                                logger.error(f"error during perform_word_alignment call: {e_dtw}\n{traceback.format_exc()}")
                                chunk_words = []
                        else: # attentions empty or heads missing
                             if not align_heads: logger.warning("alignment skipped (heads unavailable).")
                             else: logger.warning("alignment skipped (attentions empty/stack failed).")
                    else: # align_failed was true
                        logger.warning(f"alignment not performed due to attention extraction failure.")

                elif text_token_ids and not can_align:
                     logger.info(f"chunk {chunk_index}: alignment not possible (model lacks cross-attentions).")
                elif text_token_ids and not include_timestamps:
                     logger.info(f"chunk {chunk_index}: alignment skipped (timestamps disabled).")

                # fallback context prep if alignment failed but we have text
                if not chunk_words and chunk_raw_text:
                     words_only = chunk_raw_text.split()
                     chunk_words = [{"text": w, "start": None, "end": None} for w in words_only]
                     logger.info(f"chunk {chunk_index}: using basic text split for context prep.")

            except Exception as e_post:
                logger.error(f"error during post-processing/alignment for chunk {chunk_index}: {e_post}\n{traceback.format_exc()}")
                chunk_words = []

            # accumulate timed words for final result
            valid_timed_words = [wt for wt in chunk_words if wt.get('start') is not None]
            all_words.extend(valid_timed_words)

            # prepare context for the *next* chunk (uses batch variables `context`, `prefix`)
            context = np.array([], dtype=np.float32) # reset audio context
            next_prefix = delimiter # default next prefix

            if chunk_words:
                # text context
                context_word_data = [wt for wt in chunk_words if wt.get('text')][-CONTEXT_WORDS:]
                context_text_parts = [wt['text'] for wt in context_word_data]

                # find last delimiter
                current_delimiter = delimiter
                potential_delimiters = {"-", ">>", "<"}
                if isinstance(delimiter, str) and delimiter.strip(): potential_delimiters.add(delimiter.strip())
                if is_speaker_diarization_delimiter: potential_delimiters.update(["speaker 1:", "person 1:", "speaker 2:", "person 2:"])
                prefix_search_limit = min(len(chunk_words), MAX_CONTEXT_SEARCH_WORDS * 2)
                found_delimiter = False
                for wt in reversed(chunk_words[-prefix_search_limit:]):
                    word_text_stripped = wt.get('text', '').strip()
                    if word_text_stripped in potential_delimiters:
                        current_delimiter = word_text_stripped + " "; found_delimiter = True; break

                # clean text context
                if context_text_parts:
                    last_word = context_text_parts[-1]
                    clean_last_word = last_word.strip(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""")
                    if clean_last_word: context_text_parts[-1] = clean_last_word
                    elif len(context_text_parts) > 1: context_text_parts.pop()

                context_text_joined = " ".join(context_text_parts)
                next_prefix = (current_delimiter + context_text_joined).strip()
                if not next_prefix: next_prefix = delimiter

                # audio context (requires timestamps)
                first_ctx_word = None; last_ctx_word = None
                ts_search_limit = min(len(chunk_words), MAX_CONTEXT_SEARCH_WORDS)
                ts_ctx_words = [wt for wt in chunk_words[-ts_search_limit:] if wt.get('start') is not None and wt.get('end') is not None]
                if ts_ctx_words:
                    last_ctx_word = ts_ctx_words[-1]
                    start_index_in_filtered = max(0, len(ts_ctx_words) - CONTEXT_WORDS)
                    first_ctx_word = ts_ctx_words[start_index_in_filtered]

                if first_ctx_word and last_ctx_word:
                    try:
                        ctx_start_time_global = float(first_ctx_word['start'])
                        ctx_end_time_global = float(last_ctx_word['end'])
                        if ctx_end_time_global < ctx_start_time_global: ctx_end_time_global = ctx_start_time_global

                        ctx_start_sample = max(0, math.floor(ctx_start_time_global * SAMPLE_RATE))
                        ctx_end_sample = min(total_samples, math.ceil(ctx_end_time_global * SAMPLE_RATE))

                        if ctx_end_sample > ctx_start_sample:
                            context = audio_data[ctx_start_sample:ctx_end_sample]
                            max_context_s = 5.0
                            if len(context) > max_context_s * SAMPLE_RATE:
                                context = context[-int(max_context_s * SAMPLE_RATE):]
                            prefix = next_prefix # set text prefix
                            if verbose: logger.info(f"chunk {chunk_index}: prepared context audio ({len(context)/SAMPLE_RATE:.2f}s) and prefix '{prefix}'")
                        else:
                            logger.warning(f"empty context audio slice calculated, resetting audio context")
                            context = np.array([], dtype=np.float32); prefix = next_prefix
                    except (TypeError, ValueError) as e_ts:
                         logger.warning(f"invalid timestamp for audio context: {e_ts}. resetting audio context.")
                         context = np.array([], dtype=np.float32); prefix = next_prefix
                else:
                    logger.info(f"chunk {chunk_index}: no valid audio context timestamps. resetting audio context.")
                    context = np.array([], dtype=np.float32)
                    prefix = next_prefix # still use text prefix
            else: # chunk_words was empty
                 if decoding_error is None: logger.warning(f"chunk {chunk_index}: no words available, resetting context and prefix")
                 context = np.array([], dtype=np.float32); prefix = delimiter

            # advance to next chunk start position
            offset_samples = end_sample

            # explicit cleanup
            del mel, enc_out, hidden_states, chunk_tokens, decoding_error, output_tokens, text_token_ids, chunk_words, align_trace, new_tokens, chunk_raw_text, initial_past_kv_state
            if 'attentions' in locals(): del attentions
            if 'layer_attentions' in locals(): del layer_attentions
            if 'align_outs' in locals(): del align_outs
            if verbose: logger.info(f"--- end of batch chunk {chunk_index} processing ---")
        # --- end of batch mode while loop ---

        # --- batch mode: construct final result ---
        final_transcript = " ".join(raw_chunk_texts).strip()
        final_result = {
            "text": final_transcript,
            "words": all_words,
            "tokens": all_tokens,
            "language": language,
        }
        logger.info("transcription finished.")
        return final_result # return the dictionary
