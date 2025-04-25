# Whisper with Urro `WHISPER + URRO`

Multilingual automatic speech recognition (ASR) with speaker segmentation (SS) / speaker diarization (SD) and word-level timestamps (WLT)

## Installation

### Latest
```shell
pip install git+https://github.com/urroxyz/whisper@v0.3.0
```

### Development
```shell
pip install git+https://github.com/urroxyz/whisper
```

## Introduction
Yes, Whisper *can* segment speakers and timestamp words! And WHISPER + URRO is here to offer an easy solution therefor.

By modifying the thinking process of the OpenAI model, we can force it to delimit new speakers with symbols like hyphens (`-`) or greater-thans (`>`), or even with complete labels such as `[SPEAKER 1]` and `[SPEAKER 2]` to keep track of who is speaking and when.[^acknowledgement-1] By extracting cross-attentions and processesing them with dynamic-time warping, we can reconstruct timestamps on the word level rather than relying on occasional generated time tokens.[^acknowledgement-2]

[^acknowledgement-1]: Unique to WHISPER + URRO.
[^acknowledgement-2]: As explicitly implemented in `whisper-timestamped`, alongside other libraries, such as `openai-whisper`.

## Supported models
### Official

| Size                                           | Parameters | New-speaker segmentation | Speaker diarization | Word-level timestamps |
|------------------------------------------------|------------|--------------------------|---------------------|-----------------------|
| tiny[^link-1]<br>tiny.en[^link-2]              | 39 M       | ✓                        | x                   | ✓                     |
| base[^link-3]<br>base.en[^link-4]              | 74 M       | ✓                        | x                   | ✓                     |
| small[^link-5]<br>small.en[^link-6]            | 244 M      | ✓                        | x                   | ✓                     |
| medium[^link-7]<br>medium.en[^link-8]          | 769 M      | ✓                        | ✓                   | x                     |
| large-v3[^link-11]                             | 1550 M     | ✓                        | ✓                   | x                     |
| large-v3-turbo[^link-12]                       | 809 M      | ✓                        | x                   | ✓                     |

[^link-1]: https://huggingface.co/onnx-community/whisper-tiny_timestamped
[^link-2]: https://huggingface.co/onnx-community/whisper-tiny.en_timestamped
[^link-3]: https://huggingface.co/onnx-community/whisper-base_timestamped
[^link-4]: https://huggingface.co/onnx-community/whisper-base.en_timestamped
[^link-5]: https://huggingface.co/onnx-community/whisper-small_timestamped
[^link-6]: https://huggingface.co/onnx-community/whisper-small.en_timestamped
[^link-7]: https://huggingface.co/onnx-community/whisper-medium-ONNX
[^link-8]: https://huggingface.co/onnx-community/whisper-medium.en-ONNX  
[^link-11]: https://huggingface.co/onnx-community/whisper-large-v3-ONNX
[^link-12]: https://huggingface.co/onnx-community/whisper-large-v3-turbo_timestamped

### Third-party
| Model                   | Parameters | New-speaker segmentation | Speaker diarization | Word-level timestamps |
|-------------------------|------------|--------------------------|---------------------|-----------------------|
| whisper-d-v1a[^link-13] | 1550 M     | ✓                        | ✓                   | x                     |

[^link-13]: [onnx-community/whisper-d-v1a-ONNX](https://huggingface.co/onnx-community/whisper-d-v1a-ONNX)

## Comparison
<details>
    
https://github.com/user-attachments/assets/6da5199c-0eaa-4e39-b511-61a52b4b48f9
    
</details>

| Source                                                                                                                             | Transcript                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
|------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Ground truth**                                                                                                                 | [SPEAKER 1] Down in front. <br><br> [SPEAKER 2] Hey, sit down, that’s wrong of you. <br><br> [SPEAKER 1] The little lady who is to become Mrs. Harvey Yates over my dead body. <br><br> [SPEAKER 3] I know I have the sincere wishes of all my friends… <br><br> and can only tell you how much I appreciate it. <br><br> I think I can honestly say this is the happiest moment of my life. <br><br> Look what I have here. <br><br> It’s a little engagement present just given me by Mr. Yates. |
| **Pretrained model (`medium`)**<br><br>***No speaker labels***                                                                      | Down in front. <br><br> Hey, sit down, ~~that’s fine~~. <br><br> The little lady who is to become Mrs. Harvey Yates over my dead body. <br><br> [APPLAUSE] <br><br> I know I have the sincere wishes of all my friends, <br><br> and can only tell you how much I appreciate it. <br><br> I think I can honestly say this is the happiest moment of my life. <br><br> Look what I have here… <br><br> It’s a little engagement present just given me by Mr. Yates. |
| **Pretrained model (`medium`)**<br>*with WHISPER + URRO*<br><br>`delimiter=SPEAKER()`<br>`prompt=SPEAKERS(3, "en")`<br><br>***Correct speaker labels*** | [SPEAKER 1] Down in front. <br><br> [SPEAKER 2] Hey, sit down, ~~that’s fine~~. <br><br> [SPEAKER 1] The little lady who is to become Mrs. Harvey Yates over my dead body. <br><br> [APPLAUSE] <br><br> [SPEAKER 3] I know I have the sincere wishes of all my friends, <br><br> and can only tell you how much I appreciate it. <br><br> I think I can honestly say this is the happiest moment of my life. <br><br> Look what I have here… <br><br> It’s a little engagement present just given me by Mr. Yates. |
| **Finetuned model (`d-v1a`)**<br><br>***Incorrect speaker labels***                                                               | [S1] Down in front. <br><br> [S2] Hey, sit down, ~~it’s warm~~. <br><br> [S1] The little lady who is to become Mrs. Harvey Yates, over my dead body. <br><br> ~~[S2]~~ I know I have the sincere wishes of all my friends, <br><br> and can only tell you how much I appreciate it. <br><br> I think I can honestly say this is the happiest moment of my life. <br><br> Look what I have here. |
| **Finetuned model (`d-v1a`)**<br>*with WHISPER + URRO*<br><br>`delimiter=SPEAKER(short=True)`<br>`prompt=SPEAKERS(3, "en", short=True)`<br><br>***Correct speaker labels***   | [S1] Down in front. <br><br> [S2] Hey, sit down, ~~it’s warm~~. <br><br> [S1] The little lady who is to become Mrs. Harvey Yates, over my dead body. <br><br> [S3] I know I have the sincere wishes of all my friends, <br><br> and can only tell you how much I appreciate it. <br><br> I think I can honestly say this is the happiest moment of my life. <br><br> Look what I have here. |

## Quickstart

### 1. Import the library

```py
from urro_whisper import whisperer
from urro_whisper.delimiters import HYPHEN, GREATER_THAN, SPEAKER, PERSON
from urro_whisper.prompts import SPEAKERS, PERSONS
```

### 2. Set variables

<p align="center">to segment speakers:</p>

```py
model = "tiny"
audio = "audio.wav"
language = "en"
delimiter = HYPHEN
```
<p align="center">to label speakers:</p>

```py
model = "medium"
audio = "audio.wav"
language = "en"
prompt = SPEAKERS
delimiter = SPEAKER
speakers = 3
```

### 3. Create the `whisperer`

<p align="center">to segment speakers:</p>

```py
result = whisperer(
    model=model,
    audio=audio,
    language=language,
    delimiter=delimiter(),
    verbose=False,
)
```
<p align="center">to label speakers:</p>

```py
result = whisperer(
    model=model,
    audio=audio,
    language=language,
    prompt=prompt(speakers, language),
    delimiter=delimiter(),
    verbose=False,
)
```

### 3. Print results

```py
import re

print("\n--- Transcript ---")
texts = re.split(delimiter.regex, result["text"])
for _, text in enumerate(texts):
    if len(text) > 0:
      print(text)

def format_timestamp(seconds):
    if seconds is None: return "N/A"
    milliseconds = round(seconds * 1000)
    ss = milliseconds // 1000
    ms = milliseconds % 1000
    mm = ss // 60
    ss %= 60
    hh = mm // 60
    mm %= 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}"

try:
    from IPython.display import display, HTML, Audio
    import soundfile as sf
    import math
    import numpy as np
    import librosa

    audio_original, sr_original = sf.read(audio)
    if audio_original.ndim > 1:
        audio_original = audio_original.mean(axis=1)

    target_sample_rate = 16000

    if sr_original != target_sample_rate:
        audio_playback = librosa.resample(
            y=audio_original.astype(np.float32),
            orig_sr=sr_original,
            target_sr=target_sample_rate
        )
    else:
        audio_playback = audio_original.astype(np.float32)

    html_rows = []
    html_rows.append("<tr><th>Timestamp</th><th>Text</th><th>Audio</th></tr>")

    for idx, word_info in enumerate(result["words"]):
        start_time = word_info['start']
        end_time = word_info['end']
        word_text = word_info['text']
        ts_str = f"[{format_timestamp(start_time)} --> {format_timestamp(end_time)}]"
        audio_player_html = "N/A"
        if (
            start_time is not None
            and end_time is not None
            and end_time > start_time
        ):
            start_sample = max(0, math.floor(start_time * target_sample_rate))
            end_sample = min(len(audio_playback), math.ceil(end_time * target_sample_rate))

            if end_sample > start_sample:
                audio_segment = audio_playback[start_sample:end_sample]

                max_abs = np.max(np.abs(audio_segment))
                if max_abs > 1.0:
                    audio_segment = audio_segment / max_abs
                elif max_abs == 0:
                    
                     pass

                try:
                    audio_obj = Audio(data=audio_segment, rate=target_sample_rate, autoplay=False)
                    audio_player_html = audio_obj._repr_html_()
                except Exception as audio_err:
                    print(f"Warning: Could not create audio player for segment '{word_text}': {audio_err}")
                    audio_player_html = "(Error creating player)"

            else:
                audio_player_html = "(empty segment)"
        html_rows.append(
            f"<tr><td>{ts_str}</td><td>{word_text}</td><td>{audio_player_html}</td></tr>"
        )
    html_table = (
        "<table border='1' style='border-collapse: collapse; width: 100%;'>"
        "<thead></thead><tbody>"
        + "".join(html_rows)
        + "</tbody></table>"
    )
    display(HTML(html_table))

except ImportError as e:
    print(f"\nSkipping HTML table generation due to missing libraries: {e}")
    print("You might need to install: pip install ipython soundfile librosa")
    print("\n--- Word-level Timestamps (Text Fallback) ---")
   
    if "words" in result:
        for word_info in result["words"]:
            start = word_info['start']
            end = word_info['end']
            text_ = word_info['text']
            print(f"[{format_timestamp(start)} --> {format_timestamp(end)}]\t{text_}")
    else:
        print("No word timestamp information available in results.")

except FileNotFoundError:
    print(f"\nError: Audio file not found at '{audio}'. Please provide a valid path.")
except Exception as e:
    print(f"\nAn error occurred during HTML table generation or fallback: {e}")
    import traceback
    traceback.print_exc()
```

## Stream
```py
print("\n--- transcript stream ---")
tokens = whisperer(
    model=model,
    audio=audio,
    language=language,
    delimiter=delimiter(),
    prompt=prompt(speakers, language),
    stream=True,
    verbose=False,
)

i = 0
for token in tokens:
    if i == 0:
        print(delimiter() + token, end="", flush=True)
    else: 
        print(token, end="", flush=True)
    i += 1

print("\n--- end of stream ---")
```

## To-Do

- [ ] Regroup word ouput
- [x] Speaker diarization
- [x] User prompting
- [x] Stream text output
- [ ] Align existing transcript
- [ ] Stream audio input

## Acknowledgements

* [openai-whisper](https://github.com/openai/whisper) by OpenAI
    * mel spectrogram handling
* [whisper-timestamped](https://github.com/linto-ai/whisper-timestamped) by Linto AI
    * word-level timestamp extraction
 
## Notes and links
