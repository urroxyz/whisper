import logging
import numpy as np
# keep torch import if perform_word_alignment might need it internally
# import torch # torch is not directly used here, can be removed if align doesn't need it implicitly
import re
import traceback
import math
import unicodedata

# use consistent imports relative to the package structure
from .audio.load import load_audio_and_resample, calculate_mel_for_segment
from .model.download import get_onnx_model_paths
from .model.load import load_onnx_models, get_tokenizer_for_model
from .align import (
    perform_word_alignment,
    _get_alignment_heads,
    _ALIGNMENT_HEADS,
    AUDIO_TIME_PER_TOKEN,
    SAMPLE_RATE, # use sample rate defined in align.py or here
)

logger = logging.getLogger("urro_whisper") # consistent logger name
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    # basic formatter, can be customized
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.WARNING) # default level

# constants for chunking
CHUNK_LENGTH_SAMPLES = 30 * SAMPLE_RATE  # 30 seconds * 16000 hz
CONTEXT_WORDS = 5  # number of words for textual context
MAX_CONTEXT_SEARCH_WORDS = 15 # how far back to look for valid timestamps for audio context
MIN_CHUNK_SAMPLES = int(0.2 * SAMPLE_RATE) # minimum audio length (e.g., 0.2s) to process a chunk

def whisperer(
    model: str,
    audio: str, # path or numpy array
    language: str = "en",
    speaker_delimiter: str = " -", # initial delimiter only, critical for speaker segmentation
    onnx_encoder: str = None,
    onnx_decoder: str = None,
    verbose: bool = False,
    max_tokens: int = 448, # max total sequence length limit for decoder, including prompt/prefix
    onnx_providers: list = None,
    exclude_providers_on_error: list = ['CoreMLExecutionProvider'] # default excludes coreml
):
    """
    transcribes audio using whisper onnx models with chunking, context handling,
    and forced alignment (if available). outputs raw concatenated transcript.
    """
    # hardcode task to transcribe
    task = "transcribe"

    if verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)
    logger.info(f"starting transcription for: {audio if isinstance(audio, str) else 'numpy array'}".lower())
    # log message updated to remove task
    logger.info(f"using model shorthand: {model}, language: {language}".lower())

    # check if user intends speaker diarization with an incompatible model
    is_speaker_diarization_delimiter = "speaker 1" in speaker_delimiter.lower() or "person 1" in speaker_delimiter.lower()
    is_large_or_medium_model = "medium" in model.lower() or "large" in model.lower()

    if is_speaker_diarization_delimiter and not is_large_or_medium_model:
        logger.warning("speaker diarization with `SPEAKER()` and `PERSON()`is only supported by model sizes medium, large-v1, large-v2, and large-v3. smaller models may produce unpredictable results".lower())

    # load and prepare audio
    # load audio here
    if isinstance(audio, str):
        audio_data, actual_sr = load_audio_and_resample(audio, verbose=verbose)
    elif isinstance(audio, np.ndarray):
        audio_data = audio.astype(np.float32)
        actual_sr = SAMPLE_RATE # assume 16khz
        logger.info("using provided numpy array as audio input".lower())
        if audio.ndim > 1:
             logger.warning("input numpy array has >1 dimension, averaging channels".lower())
             audio_data = audio_data.mean(axis=-1) # ensure it becomes 1d
    else:
        raise TypeError("Input 'audio' must be a file path (str) or a numpy array")

    if actual_sr != SAMPLE_RATE:
        logger.error(f"audio data sample rate ({actual_sr}) must be {SAMPLE_RATE}".lower())
        raise ValueError(f"Incorrect audio sample rate: {actual_sr}")

    total_samples = len(audio_data)
    duration = total_samples / SAMPLE_RATE
    logger.info(f"total audio duration: {duration:.2f}s".lower())

    # load models and tokenizer
    encoder_path, decoder_path = get_onnx_model_paths(
        model, onnx_encoder, onnx_decoder, verbose=verbose
    )
    encoder_sess, decoder_sess = load_onnx_models(
        encoder_path, decoder_path, onnx_providers=onnx_providers, verbose=verbose,
        exclude_providers=exclude_providers_on_error # pass exclusion list
    )
    tokenizer, is_multilingual = get_tokenizer_for_model(model, language, task)
    eos_token = tokenizer.eot # <|endoftext|> token id
    n_mels_required = 128 if "large-v3" in model else 80 # determine n_mels based on model

    # initialize accumulators
    all_words = [] # renamed from 'words' for clarity
    raw_chunk_texts = [] # stores raw decoded text for each chunk
    all_tokens = [] # renamed from 'tokens'
    offset_samples = 0
    chunk_index = 0

    # initial state for chunk 1
    prefix = speaker_delimiter # critical: initialize with the desired speaker delimiter
    context = np.array([], dtype=np.float32) # simplified: context_audio -> context

    # main chunking loop
    while offset_samples < total_samples:
        # --- initialize per-iteration variables to prevent unboundlocalerror ---
        mel = None; enc_out = None; hidden_states = None; chunk_tokens = None
        decoding_error = None; output_tokens = None; text_token_ids = [] # initialize as empty list
        chunk_words = []; align_trace = None; attentions = [] # initialize as empty list
        layer_attentions = []; align_outs = None; new_tokens = []
        chunk_raw_text = "" # initialize raw text for the chunk

        chunk_index += 1
        start_sample = offset_samples
        end_sample = min(offset_samples + CHUNK_LENGTH_SAMPLES, total_samples)
        chunk_duration = (end_sample - start_sample) / SAMPLE_RATE

        if (end_sample - start_sample) < MIN_CHUNK_SAMPLES and chunk_index > 1:
             logger.info(f"skipping very short final segment ({chunk_duration:.2f}s)".lower())
             break

        logger.info(f"processing chunk {chunk_index}: samples {start_sample}-{end_sample} ({chunk_duration:.2f}s)".lower())

        # prepare audio segment: context + current chunk
        chunk_audio = audio_data[start_sample:end_sample]
        segment = np.concatenate([context, chunk_audio])
        segment_dur = len(segment) / SAMPLE_RATE
        context_dur = len(context) / SAMPLE_RATE

        if len(segment) < MIN_CHUNK_SAMPLES: # check combined length
            logger.warning(f"chunk {chunk_index}: segment audio too short ({segment_dur:.2f}s) after adding context, skipping".lower())
            offset_samples = end_sample
            context = np.array([], dtype=np.float32)
            prefix = speaker_delimiter # reset prefix
            # explicitly clean up (optional, python gc handles this)
            del mel, enc_out, hidden_states, chunk_tokens, decoding_error
            del output_tokens, text_token_ids, chunk_words, align_trace
            del attentions, layer_attentions, align_outs, new_tokens, chunk_raw_text
            continue

        # feature extraction
        try:
            mel = calculate_mel_for_segment(segment, model, n_mels_required, verbose=verbose)
        except Exception as e:
            logger.error(f"mel calculation failed for chunk {chunk_index}: {e}\n{traceback.format_exc()}".lower())
            offset_samples = end_sample; context = np.array([], dtype=np.float32); prefix = speaker_delimiter
            del mel # cleanup optional
            continue

        # run encoder
        enc_input_name = encoder_sess.get_inputs()[0].name
        try:
            enc_out = encoder_sess.run(None, {enc_input_name: mel.astype(np.float32)})
            hidden_states = enc_out[0]
            n_frames = hidden_states.shape[1]
        except Exception as e:
             logger.error(f"onnx encoder failed for chunk {chunk_index}: {e}\n{traceback.format_exc()}".lower())
             offset_samples = end_sample; context = np.array([], dtype=np.float32); prefix = speaker_delimiter
             # explicitly clean up
             del mel, enc_out, hidden_states
             continue

        # prepare decoder prompt
        prompt_tokens = [tokenizer.sot]
        if is_multilingual:
            lang_token_str = f"<|{language}|>"
            try: prompt_tokens.append(tokenizer.encode(lang_token_str, allowed_special="all")[0])
            except Exception: pass # ignore if language token isn't found/needed

        # use hardcoded task "transcribe" for token
        task_token_str = "<|transcribe|>"
        try: prompt_tokens.append(tokenizer.encode(task_token_str, allowed_special="all")[0])
        except Exception as e: logger.error(f"failed to encode task token: {e}".lower()); raise
        # timestamps are always enabled, add timestamp_begin token
        prompt_tokens.append(tokenizer.timestamp_begin) # use <|0.00|>

        # encode the current prefix text (critical for diarization forcing)
        try:
            prefix_tokens = tokenizer.encode(prefix, allowed_special="all")
            if not prefix_tokens: prefix_tokens = [] # handle empty prefix gracefully
            elif len(prefix_tokens) > 1 and verbose: logger.info(f"chunk {chunk_index}: multi-token prefix: {prefix_tokens} ('{prefix}')".lower())
        except Exception: prefix_tokens = [] # fallback to empty if encoding fails

        # combine initial prompt and the forced prefix tokens
        chunk_tokens = prompt_tokens[:] + prefix_tokens
        prompt_len = len(chunk_tokens) # length of prompt + forced prefix

        # run decoder generation loop
        dec_input_names = [inp.name for inp in decoder_sess.get_inputs()]
        dec_output_names = [out.name for out in decoder_sess.get_outputs()]
        ids_name = "input_ids"
        states_name = "encoder_hidden_states"
        logits_name = "logits"

        # calculate max new tokens allowed based on current prompt/prefix length
        max_new_tokens = max_tokens - len(chunk_tokens)

        if verbose: logger.info(f"chunk {chunk_index}: starting decoder generation (max new tokens={max_new_tokens}, total len limit={max_tokens})".lower())
        decoding_error = None
        # capture generated tokens (excluding eos) in a separate list for this chunk
        new_tokens = [] # already initialized above

        try:
            if max_new_tokens <= 0:
                 logger.warning(f"chunk {chunk_index}: prompt/prefix length ({len(chunk_tokens)}) meets or exceeds max_tokens ({max_tokens}), no tokens generated".lower())
            else:
                current_gen_tokens = chunk_tokens[:] # copy for generation loop
                for step in range(max_new_tokens):
                    input_ids_np = np.array([current_gen_tokens], dtype=np.int64)
                    decoder_inputs = {
                        ids_name: input_ids_np,
                        states_name: hidden_states.astype(np.float32)
                    }
                    # only need logits for greedy decoding
                    decoder_outputs = decoder_sess.run([logits_name], decoder_inputs)

                    logits = decoder_outputs[0]
                    next_token_logits = logits[0, -1, :]
                    next_token = int(np.argmax(next_token_logits))

                    if next_token == eos_token:
                        if verbose: logger.info(f"chunk {chunk_index}: eos token ({eos_token}) detected at step {step+1}".lower())
                        break # stop decoding for this chunk

                    current_gen_tokens.append(next_token)
                    new_tokens.append(next_token) # store only the newly generated tokens

                # check if loop finished by reaching max steps
                if step == max_new_tokens - 1 and next_token != eos_token:
                     logger.info(f"chunk {chunk_index}: reached max generated token limit ({max_new_tokens})".lower())
                # update chunk_tokens to include the generated ones for alignment context
                chunk_tokens = current_gen_tokens[:]

        except Exception as e:
            decoding_error = e
            logger.error(f"error during decoder generation loop for chunk {chunk_index}: {decoding_error}\n{traceback.format_exc()}".lower())

        # append the valid generated tokens from this chunk to the global list
        all_tokens.extend(new_tokens)

        # post-processing and alignment
        chunk_words = [] # re-initialize for safety (already done above)
        align_trace = None
        text_token_ids = [] # re-initialize for safety (already done above)
        chunk_raw_text = "" # re-initialize for safety (already done above)

        try:
            # get tokens generated after the initial prompt and forced prefix
            output_tokens = new_tokens # use the tokens actually generated in this chunk
            # filter out special/timestamp tokens for text content
            # ensure text_token_ids is defined here before potential use
            text_token_ids = [t for t in output_tokens if t < tokenizer.sot]

            # --- decode raw text for this chunk regardless of alignment ---
            if text_token_ids:
                try:
                    # use tokenizer.decode to get the untokenized string
                    chunk_raw_text = tokenizer.decode(text_token_ids).strip()
                    if verbose: logger.info(f"chunk {chunk_index} raw decoded text: '{chunk_raw_text}'")
                    if chunk_raw_text: # add to list if not empty
                        raw_chunk_texts.append(chunk_raw_text)
                except Exception as e_decode:
                    logger.error(f"failed to decode text tokens for chunk {chunk_index}: {e_decode}".lower())
                    # chunk_raw_text remains ""

            if not text_token_ids and decoding_error is None:
                 logger.warning(f"chunk {chunk_index}: no text tokens generated after prompt/prefix".lower())

            # --- alignment (only if possible and text tokens exist) ---
            if text_token_ids:
                hidden_size = hidden_states.shape[-1]
                size_map = {384: "tiny", 512: "base", 768: "small", 1024: "medium", 1280: "large"}
                model_base = size_map.get(hidden_size, model.split("-")[0].split(".")[0]) # try to infer base model name
                # construct full name for alignment head lookup
                model_name = (f"{model_base}.en" if (language == "en" and f"{model_base}.en" in _ALIGNMENT_HEADS) else model_base)
                if model in _ALIGNMENT_HEADS: model_name = model # use exact model name if available

                # check if decoder outputs cross-attentions
                attn_names = [name for name in dec_output_names if "cross_attentions" in name]
                can_align = bool(attn_names)
                attentions = [] # re-initialize inner scope variable (already done above)

                if can_align:
                    if verbose: logger.info(f"chunk {chunk_index}: cross-attentions found, attempting alignment.".lower())
                    # determine decoder layers and heads for alignment
                    num_layers = 0; max_layer_idx = -1
                    for name in attn_names: match = re.search(r"\.(\d+)", name);
                    if match: max_layer_idx = max(max_layer_idx, int(match.group(1)))
                    num_layers = max_layer_idx + 1
                    if num_layers == 0: # fallback if regex fails
                         layer_map = {"tiny": 4, "base": 6, "small": 12, "medium": 24, "large": 32}; num_layers = layer_map.get(model_base, 6)
                    num_heads = hidden_size // 64 # standard whisper head size
                    align_heads = _get_alignment_heads(model_name, num_layers, num_heads)

                    if align_heads is None: logger.warning(f"could not load alignment heads for {model_name}, alignment may be less accurate".lower())
                    if verbose: logger.info(f"chunk {chunk_index}: extracting cross-attentions...".lower())

                    layer_attentions = [[] for _ in range(num_layers)] # initialize list of lists for attentions
                    align_failed = False
                    # use prompt+prefix as the initial input for attention extraction
                    align_tokens_input = chunk_tokens[:prompt_len]
                    # tokens to extract attention for are the generated text tokens
                    # **use text_token_ids directly here**

                    try: # wrap attention extraction in try/except
                        # iterate over the *generated text tokens* for which we need attention
                        for token_idx, token_id in enumerate(text_token_ids):
                            # input for getting attention for `token_id` should not include `token_id` itself yet.
                            # the input represents the state *before* generating token_id.
                            align_ids_np = np.array([align_tokens_input], dtype=np.int64)
                            align_inputs = {
                                ids_name: align_ids_np,
                                states_name: hidden_states.astype(np.float32)
                            }
                            # request all cross-attention outputs
                            align_req_outputs = attn_names
                            try: # try/except around the run call itself
                                align_outs = decoder_sess.run(align_req_outputs, align_inputs)
                                # collect attentions for the current predicted token (based on align_tokens_input)
                                for layer_idx, att_tensor in enumerate(align_outs):
                                    # attention is for the last token prediction based on input
                                    # shape: [batch(1), heads, seq_len, key_len (n_frames)] -> want [heads, key_len] for the last token
                                    layer_attentions[layer_idx].append(att_tensor[0, :, -1, :])
                            except Exception as e_inner:
                                align_failed = True; logger.error(f"alignment pass run failed for token {token_idx} (id {token_id}): {e_inner}".lower()); break
                            # append the *current* token_id to the input sequence for the *next* iteration
                            align_tokens_input.append(token_id)

                        # if alignment didn't fail during extraction loop
                        if not align_failed:
                            # check if we collected the expected number of attention vectors per layer
                            expected_len = len(text_token_ids)
                            if all(len(lst) == expected_len for lst in layer_attentions):
                                try: # stack attentions for each layer -> [1, heads, seq_len, key_len]
                                    # seq_len here corresponds to the number of generated text tokens
                                    attentions = [np.stack(layer_atts, axis=1)[np.newaxis, :, :, :] for layer_atts in layer_attentions]
                                    if verbose: logger.info(f"successfully extracted attentions, shape example: {attentions[0].shape}".lower())
                                except Exception as e_stack:
                                    logger.error(f"stacking attentions failed: {e_stack}".lower()); attentions = []
                            else:
                                logger.warning(f"chunk {chunk_index}: attention length mismatch (expected {expected_len}, got varying lengths). alignment aborted.".lower())
                                attentions = [] # abort alignment if lengths mismatch
                        else:
                            # align_failed was true from the inner loop
                            attentions = [] # ensure attentions is empty if extraction failed

                    except Exception as e_outer:
                         logger.error(f"error during attention extraction setup: {e_outer}".lower())
                         attentions = [] # ensure it's empty on outer error

                    # --- perform dtw alignment ---
                    if attentions: # only proceed if attentions were successfully extracted and stacked
                        if verbose: logger.info(f"chunk {chunk_index}: performing dtw alignment".lower())
                        try: # wrap alignment call
                            # pass the full sequence including prompt/prefix and generated tokens
                            # but align only the generated text tokens
                            raw_chunk_words = perform_word_alignment(
                                full_token_sequence=chunk_tokens, # the full sequence including prompt+generated
                                generated_text_tokens_ids=text_token_ids, # only the text tokens generated after prompt
                                cross_attentions_list=attentions, # list of numpy arrays [1, heads, seq_len, key_len] per layer
                                tokenizer=tokenizer, alignment_heads=align_heads,
                                model_n_text_layers=num_layers,
                                n_frames_feature=n_frames, language=language,
                                medfilt_width=7, qk_scale=1.0, debug=verbose
                            )
                            if raw_chunk_words:
                                if verbose: logger.info(f"chunk {chunk_index}: dtw successful.".lower())
                                chunk_start_time = start_sample / SAMPLE_RATE
                                temp_chunk_words = [] # use temporary list to avoid modifying chunk_words directly yet
                                for wt in raw_chunk_words:
                                    start_rel, end_rel = wt.get('start'), wt.get('end')
                                    start_glob, end_glob = None, None
                                    if start_rel is not None:
                                        # adjust timestamps relative to the start of the current chunk's audio
                                        if start_rel >= context_dur: # only include words starting after context audio
                                            start_glob = round(max(0, start_rel - context_dur) + chunk_start_time, 3)
                                        else:
                                            if verbose: logger.info(f"skipping word '{wt.get('word')}' starting in context audio ({start_rel:.3f} < {context_dur:.3f})".lower())
                                            continue # skip words starting during context audio
                                    if end_rel is not None:
                                        # adjust end time relative to the start of the current chunk's audio
                                        end_glob = round(max(0, end_rel - context_dur) + chunk_start_time, 3)
                                        # sanity check: ensure end is not before start
                                        if start_glob is not None and end_glob < start_glob:
                                            if verbose: logger.warning(f"correcting end time < start time for word '{wt.get('word')}' ({start_glob=}, {end_glob=})".lower())
                                            end_glob = start_glob
                                    # use 'text' key consistently
                                    if wt.get('word'): # only add if word exists
                                        temp_chunk_words.append({"text": wt['word'], "start": start_glob, "end": end_glob})
                                chunk_words = temp_chunk_words # assign aligned words
                            else: logger.warning(f"chunk {chunk_index}: dtw alignment failed or returned empty list.".lower())
                        except Exception as e_dtw:
                            logger.error(f"error during perform_word_alignment call: {e_dtw}\n{traceback.format_exc()}".lower())
                            chunk_words = [] # ensure empty on alignment error
                    else:
                        logger.warning(f"chunk {chunk_index}: alignment not performed (attentions could not be extracted or stacked).".lower())
                else:
                     # model does not provide cross-attentions
                     logger.info(f"chunk {chunk_index}: alignment not possible (model does not output cross-attentions).".lower())
                     # chunk_words remains empty, fallback will be used below

                # fallback for context prep if alignment failed/skipped but we have text
                # use the raw text decoded earlier
                if not chunk_words and chunk_raw_text:
                     words_only = chunk_raw_text.split()
                     # create dummy timestamp dicts for context prep
                     chunk_words = [{"text": w, "start": None, "end": None} for w in words_only]
                     logger.warning(f"chunk {chunk_index}: using basic text split for context prep due to alignment failure or unavailability.".lower())

        except Exception as e_post:
            logger.error(f"error during post-processing/alignment setup for chunk {chunk_index}: {e_post}\n{traceback.format_exc()}".lower())
            chunk_words = [] # ensure empty on error

        # aggregate results
        # only add words with valid start timestamps to the global list of timed words
        valid_timed_words = [wt for wt in chunk_words if wt.get('start') is not None]
        all_words.extend(valid_timed_words)
        # raw text was already added to raw_chunk_texts earlier

        # prepare context for next chunk (critical for speaker delimiter consistency)
        context = np.array([], dtype=np.float32) # reset context audio
        next_prefix = speaker_delimiter # default next prefix

        if chunk_words: # use chunk_words (timed or untimed fallback) for context
            # 1. prepare text context (last few words)
            context_word_data = [wt for wt in chunk_words if wt.get('text')][-CONTEXT_WORDS:]
            context_text = [wt['text'] for wt in context_word_data]

            # 2. find last speaker delimiter (e.g., "-", ">>") for continuity
            current_speaker_delimiter = speaker_delimiter # default is the initial delimiter
            potential_delimiters = ["-", ">>", "<"] # add other delimiters if needed
            # add speaker/person patterns if they were intended
            if is_speaker_diarization_delimiter:
                # very basic check, might need refinement if delimiters are complex
                potential_delimiters.extend(["speaker 1", "person 1"])

            # search back reasonably far in this chunk's words
            prefix_search_limit = min(len(chunk_words), MAX_CONTEXT_SEARCH_WORDS * 2)
            for wt in reversed(chunk_words[-prefix_search_limit:]):
                 word_text = wt.get('text', '')
                 # need a robust way to check if word_text *is* a delimiter, not just contains part of one
                 # exact match or starts_with might be better depending on expected delimiter format
                 for delim in potential_delimiters:
                     # using strip() and checking for exact match is safer
                     if word_text.strip() == delim.strip():
                         current_speaker_delimiter = word_text + " " # include space after delimiter
                         break # found the most recent delimiter
                 else: # if inner loop didn't break
                     continue # continue outer loop
                 break # if inner loop broke (delimiter found)

            # clean up context text parts (remove punctuation from last word if needed)
            if context_text:
                 last_word = context_text[-1]
                 # remove common punctuation, keep core word
                 clean_last_word = "".join(c for c in last_word if not unicodedata.category(c).startswith('P')).strip()
                 if clean_last_word: context_text[-1] = clean_last_word
                 elif len(context_text) > 1: context_text.pop() # remove if last word was only punctuation

            # combine delimiter and text for next chunk's prompt
            next_prefix = (current_speaker_delimiter + " ".join(context_text)).strip()
            if not next_prefix: next_prefix = speaker_delimiter # ensure it's never empty

            # 3. prepare audio context (based on timestamps of context words - requires alignment)
            first_ctx_word = None; last_ctx_word = None
            # search back for words with valid timestamps
            ts_search_limit = min(len(chunk_words), MAX_CONTEXT_SEARCH_WORDS)
            # only consider words that have 'start' and 'end' times for audio context
            ts_ctx_words = [wt for wt in chunk_words[-ts_search_limit:] if wt.get('start') is not None and wt.get('end') is not None]

            if ts_ctx_words:
                 last_ctx_word = ts_ctx_words[-1]
                 # find the start word corresponding to the textual context (among timed words)
                 start_index_in_filtered = max(0, len(ts_ctx_words) - CONTEXT_WORDS)
                 first_ctx_word = ts_ctx_words[start_index_in_filtered]

            if first_ctx_word and last_ctx_word:
                # get audio slice based on the global timestamps of the context words
                # ensure start/end are valid floats before using them
                try:
                    ctx_start_time = float(first_ctx_word['start'])
                    ctx_end_time = float(last_ctx_word['end'])
                    ctx_start_sample = max(0, math.floor(ctx_start_time * SAMPLE_RATE))
                    ctx_end_sample = min(total_samples, math.ceil(ctx_end_time * SAMPLE_RATE))

                    if ctx_end_sample > ctx_start_sample:
                        context = audio_data[ctx_start_sample:ctx_end_sample]
                        # limit context audio length (e.g., 5 seconds max)
                        max_context_s = 5.0
                        if len(context) > max_context_s * SAMPLE_RATE:
                            logger.warning(f"trimming context audio > {max_context_s}s".lower())
                            context = context[-int(max_context_s * SAMPLE_RATE):]
                        prefix = next_prefix # set prefix for next chunk using text context
                        if verbose: logger.info(f"chunk {chunk_index}: prepared context audio ({len(context)/SAMPLE_RATE:.2f}s) and prefix '{prefix}'".lower())
                    else:
                        logger.warning(f"empty context audio slice calculated (start={ctx_start_time}, end={ctx_end_time}), resetting context".lower())
                        context = np.array([], dtype=np.float32); prefix = speaker_delimiter # reset audio context and use default prefix
                except (TypeError, ValueError) as e_ts:
                     logger.warning(f"invalid timestamp found for audio context calculation: {e_ts}. resetting context.".lower())
                     context = np.array([], dtype=np.float32); prefix = speaker_delimiter
            else:
                logger.warning(f"chunk {chunk_index}: no valid audio context timestamps found (alignment likely failed or unavailable), resetting audio context. using text prefix: '{next_prefix}'".lower())
                context = np.array([], dtype=np.float32) # reset audio context
                prefix = next_prefix # still use the text-based prefix
        else:
             # no words generated or alignment failed completely resulting in empty chunk_words
             if decoding_error is None: logger.warning(f"chunk {chunk_index}: no words available, resetting context and prefix".lower())
             context = np.array([], dtype=np.float32); prefix = speaker_delimiter

        # move to the next chunk start position
        offset_samples = end_sample

        # explicit cleanup at end of loop iteration (optional)
        del mel, enc_out, hidden_states, chunk_tokens, decoding_error
        del output_tokens, text_token_ids, chunk_words, align_trace
        del new_tokens, chunk_raw_text
        if 'attentions' in locals(): del attentions # only delete if it exists
        if 'layer_attentions' in locals(): del layer_attentions
        if 'align_outs' in locals(): del align_outs
        if verbose: logger.info(f"--- end of chunk {chunk_index} processing ---".lower())

    # --- final result combination ---

    # combine raw text chunks into the final transcript string
    # simple join with spaces is usually sufficient and handles punctuation reasonably
    final_transcript = " ".join(raw_chunk_texts).strip()

    final_result = {
        "text": final_transcript,  # the raw, concatenated transcript
        "words": all_words,        # list of word dicts with timestamps (if alignment worked)
        "tokens": all_tokens,      # list of all generated token ids
        "language": language,
    }
    logger.info("transcription finished.")

    return final_result

