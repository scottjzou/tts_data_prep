import errno
import json
import os
import re
import shutil
import subprocess
from datetime import datetime

import librosa
import pandas as pd
import soundfile as sf
from pydub import AudioSegment
from tqdm import tqdm
from vtt_to_srt.vtt_to_srt import ConvertFile
from youtube_transcript_api.formatters import JSONFormatter

from .audio import preprocess_wav
from .text_cleaner import Cleaner


class NoSubtitleWarning(UserWarning):
    pass


class YTLogger(object):
    def debug(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        print(msg)


class YTSpeechDataGenerator(object):
    """
    YTSpeechDataGenerator makes it easier to
    generate data for Text to Speech/Speech to Text.  .

    Parameters:

    dataset_name:           Name of the dataset

    output_type:            Format of the metadata file.
                            Supported formats:(csv/json)

    keep_audio_extension:   Whether to keep the audio file name
                            extensions in the metadata file.

    lang:                   The target language of subtitles.

    Available methods:

    download:               Download wavs from YouTube from a .txt file.

    split_audios:           Split the downloaded single wav files into
                            multiple.

    finalize_dataset:       Generate final dataset from the processes
                            audios.

    get_available_langs:    Get list of available languages in which the
                            the subtitles can be downloaded.

    get_total_audio_length: Get the total length of audios.
    """

    def __init__(
        self,
        dataset_name,
        download_dir,
        output_type="csv",
        keep_audio_extension=False,
        lang="en",
        sr=22050,
    ):
        self.lang_map = {
            "af": "Afrikaans",
            "am": "Amharic",
            "ar": "Arabic",
            "az": "Azerbaijani",
            "be": "Belarusian",
            "bg": "Bulgarian",
            "bn": "Bangla",
            "bs": "Bosnian",
            "ca": "Catalan",
            "ceb": "Cebuano",
            "co": "Corsican",
            "cs": "Czech",
            "cy": "Welsh",
            "da": "Danish",
            "de": "German",
            "el": "Greek",
            "en": "English",
            "eo": "Esperanto",
            "es": "Spanish",
            "et": "Estonian",
            "eu": "Basque",
            "fa": "Persian",
            "fi": "Finnish",
            "fil": "Filipino",
            "fr": "French",
            "fy": "Western Frisian",
            "ga": "Irish",
            "gd": "Scottish Gaelic",
            "gl": "Galician",
            "gu": "Gujarati",
            "ha": "Hausa",
            "haw": "Hawaiian",
            "hi": "Hindi",
            "hmn": "Hmong",
            "hr": "Croatian",
            "ht": "Haitian Creole",
            "hu": "Hungarian",
            "hy": "Armenian",
            "id": "Indonesian",
            "ig": "Igbo",
            "is": "Icelandic",
            "it": "Italian",
            "iw": "Hebrew",
            "ja": "Japanese",
            "jv": "Javanese",
            "ka": "Georgian",
            "kk": "Kazakh",
            "km": "Khmer",
            "kn": "Kannada",
            "ko": "Korean",
            "ku": "Kurdish",
            "ky": "Kyrgyz",
            "la": "Latin",
            "lb": "Luxembourgish",
            "lo": "Lao",
            "lt": "Lithuanian",
            "lv": "Latvian",
            "mg": "Malagasy",
            "mi": "Maori",
            "mk": "Macedonian",
            "ml": "Malayalam",
            "mn": "Mongolian",
            "mr": "Marathi",
            "ms": "Malay",
            "mt": "Maltese",
            "my": "Burmese",
            "ne": "Nepali",
            "nl": "Dutch",
            "no": "Norwegian",
            "ny": "Nyanja",
            "or": "Odia",
            "pa": "Punjabi",
            "pl": "Polish",
            "ps": "Pashto",
            "pt": "Portuguese",
            "ro": "Romanian",
            "ru": "Russian",
            "rw": "Kinyarwanda",
            "sd": "Sindhi",
            "si": "Sinhala",
            "sk": "Slovak",
            "sl": "Slovenian",
            "sm": "Samoan",
            "sn": "Shona",
            "so": "Somali",
            "sq": "Albanian",
            "sr": "Serbian",
            "st": "Southern Sotho",
            "su": "Sundanese",
            "sv": "Swedish",
            "sw": "Swahili",
            "ta": "Tamil",
            "te": "Telugu",
            "tg": "Tajik",
            "th": "Thai",
            "tk": "Turkmen",
            "tr": "Turkish",
            "tt": "Tatar",
            "ug": "Uyghur",
            "uk": "Ukrainian",
            "ur": "Urdu",
            "uz": "Uzbek",
            "vi": "Vietnamese",
            "xh": "Xhosa",
            "yi": "Yiddish",
            "yo": "Yoruba",
            "zh-Hans": "Chinese (Simplified)",
            "zh-Hant": "Chinese (Traditional)",
            "zu": "Zulu",
        }

        self.wav_counter = 0
        self.wav_filenames = []
        self.name = dataset_name
        self.root = os.getcwd()
        self.prep_dir = os.path.join(self.root, self.name + "_prep")
        self.dest_dir = os.path.join(self.root, self.name)
        self.download_dir = download_dir
        self.split_dir = os.path.join(self.prep_dir, "split")
        self.concat_dir = os.path.join(self.prep_dir, "concatenated")
        self.filenames_txt = os.path.join(self.download_dir, "files.txt")
        self.split_audios_csv = os.path.join(self.split_dir, "split.csv")
        self.len_dataset = 0
        self.keep_audio_extension = keep_audio_extension
        self.sr = sr
        if output_type not in ["csv", "json"]:
            raise Exception(
                "Invalid output type. Supported output files are 'csv'/'json'"
            )
        else:
            self.output_type = output_type
        self.cleaner = Cleaner()
        self.transcript_formatter = JSONFormatter()

        if not os.path.exists(self.prep_dir):
            print(f"Creating directory '{self.name}_prep'..")
            print(f"Creating directory '{self.name}_prep/split'")
            print(f"Creating directory '{self.name}_prep/concatenated'")
            os.mkdir(self.prep_dir)
            os.mkdir(self.split_dir)
            os.mkdir(self.concat_dir)

        if not os.path.exists(self.dest_dir):
            print(f"Creating directory '{self.name}'..")
            print(f"Creating directory '{self.name}/wavs'")
            print(f"Creating directory '{self.name}/txts'")
            os.mkdir(self.dest_dir)
            os.mkdir(os.path.join(self.dest_dir, "wavs"))
            os.mkdir(os.path.join(self.dest_dir, "txts"))

        if lang not in self.lang_map:
            raise Exception(f"The language '{lang}' isn't supported at present.")

        self.dataset_lang = lang

        self.ydl_opts = {
            "format": "bestaudio/best",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                }
            ],
            "logger": YTLogger(),
        }

    def get_available_langs(self):
        print("List of supported languages:\n")
        for key, lang in self.lang_map.items():
            print(key, ":", lang)

    def get_video_id(self, url, pattern="(\/|%3D|v=)([0-9A-z-_]{11})([%#?&]|$)"):
        matches = re.findall(pattern, url)
        if matches != []:
            try:
                return matches[0][1]
            except:
                return []
        else:
            return []

    @staticmethod
    def fix_json_trans(trans):
        return [
            {
                "start": trans[ix]["start"],
                "end": trans[ix + 1]["start"],
                "text": trans[ix]["text"],
            }
            if ix != len(trans) - 1
            else {
                "start": trans[ix]["start"],
                "end": trans[ix]["start"] + trans[ix]["duration"],
                "text": trans[ix]["text"],
            }
            for ix in range(len(trans))
            # add more filter based on your subtitle
            if trans[ix]["text"] not in ["[Music]", "[applause]", "(Applause.)"]
        ]

    @staticmethod
    def convert_time(seconds):
        seconds = seconds % (24 * 3600)
        hour = seconds // 3600
        seconds %= 3600
        minutes = seconds // 60
        seconds %= 60

        return "%d:%02d:%02d" % (hour, minutes, seconds)

    @staticmethod
    def parse_time(time_string):
        hours = int(re.findall(r"(\d+):\d+:\d+,\d+", time_string)[0])
        minutes = int(re.findall(r"\d+:(\d+):\d+,\d+", time_string)[0])
        seconds = int(re.findall(r"\d+:\d+:(\d+),\d+", time_string)[0])
        milliseconds = int(re.findall(r"\d+:\d+:\d+,(\d+)", time_string)[0])

        return (hours * 3600 + minutes * 60 + seconds) * 1000 + milliseconds

    @staticmethod
    def parse_srt(srt_string):
        # Original : https://github.com/pgrabovets/srt-to-json
        srt_list = []

        for line in srt_string.split("\n\n"):
            if line:
                try:
                    line.split("\n")
                    _ = int(re.match(r"\d+", line).group())

                    pos = re.search(r"\d+:\d+:\d+,\d+ --> \d+:\d+:\d+,\d+", line).end() + 1
                    content = line[pos:]
                    start_time_string = re.findall(
                        r"(\d+:\d+:\d+,\d+) --> \d+:\d+:\d+,\d+", line
                    )[0]
                    end_time_string = re.findall(
                        r"\d+:\d+:\d+,\d+ --> (\d+:\d+:\d+,\d+)", line
                    )[0]
                    start_time = YTSpeechDataGenerator.parse_time(start_time_string)
                    end_time = YTSpeechDataGenerator.parse_time(end_time_string)

                    srt_list.append(
                        {
                            "text": content.replace("\n", " "),
                            "start": start_time / 1000,
                            "duration": (end_time - start_time) / 1000,
                        }
                    )
                except Exception:
                    print("bad line" + line)
                    continue

        return srt_list

    def split_audios(self):
        """
        Split the downloaded videos into smaller chunks.
        """
        if os.path.exists(self.filenames_txt) and os.path.isfile(self.filenames_txt):
            files_list = open(self.filenames_txt).read().strip().split("\n")
            files_list = files_list[1:]
            check_ffmpeg = subprocess.run(
                ["ffmpeg"], stderr=subprocess.STDOUT, stdout=subprocess.PIPE
            )

            files_pbar = tqdm(files_list)
            for line in files_pbar:
                filename, subtitle, trim_min_begin, trim_min_end = line.split(",")
                caption_json = None
                out_filename = filename.replace(".wav", ".json")
                files_pbar.set_description("Processing %s" % filename)
                if subtitle.lower().endswith(".vtt"):
                    tqdm.write(f"Detected VTT captions. Converting to json..")
                    ConvertFile(os.path.join(self.download_dir, subtitle), "utf-8").convert()
                    file_contents = open(
                        os.path.join(self.download_dir, subtitle[:-3] + 'srt'),
                        mode="r",
                        encoding="utf-8",
                    ).read()
                    caption_json = YTSpeechDataGenerator.parse_srt(file_contents.strip())
                elif subtitle.lower().endswith(".srt"):
                    tqdm.write(f"Detected SRT captions. Converting to json..")
                    file_contents = open(
                        os.path.join(self.download_dir, subtitle),
                        mode="r",
                        encoding="utf-8",
                    ).read()
                    caption_json = YTSpeechDataGenerator.parse_srt(file_contents.strip())
                elif subtitle.lower().endswith(".json"):
                    pass
                else:
                    raise Exception(
                        "Invalid subtitle type. Supported subtitle types are 'vtt'/'srt'"
                    )
                if caption_json:
                    caption_json = self.fix_json_trans(caption_json)
                    open(
                        os.path.join(self.download_dir, out_filename),
                        "w",
                        encoding="utf-8",
                    ).write(json.dumps(caption_json, indent=2, sort_keys=True))
                    tqdm.write(
                        f"Writing json captions for {filename} to '{out_filename}'."
                    )
                trim_min_end = int(trim_min_end)
                trim_min_begin = int(trim_min_begin)
                filename = filename[:-4]
                cnt = 0
                if not caption_json:
                    with open(
                        os.path.join(self.download_dir, subtitle)
                    ) as json_cap:
                        captions = json.loads(json_cap.read())
                else:
                    captions = caption_json
                for ix in range(len(captions)):
                    cap = captions[ix]
                    text = cap["text"]
                    start = cap["start"]
                    end = cap["end"]

                    t = datetime.strptime(
                        self.convert_time(start), "%H:%M:%S"
                    ).time()
                    if trim_min_end > 0:

                        t2 = datetime.strptime(
                            self.convert_time(end), "%H:%M:%S"
                        ).time()

                        if (
                            t.minute >= trim_min_begin
                            and t2.minute <= trim_min_end
                            and text != "\n"
                        ):
                            text = " ".join(text.split('\n"'))
                            text.replace("--", "")
                            text.replace('"', "")

                            new_name = filename + "-" + str(cnt)

                            cmd = [
                                "ffmpeg",
                                "-i",
                                f"{os.path.join(self.download_dir,filename)}.wav",
                                "-ss",
                                str(start),
                                "-to",
                                str(end),
                                "-c",
                                "copy",
                                f"{os.path.join(self.split_dir,new_name)}.wav",
                            ]

                            call = subprocess.run(cmd, stderr=subprocess.STDOUT)

                            with open(
                                os.path.join(self.split_dir, new_name + ".txt"), "w"
                            ) as f:
                                f.write(text)
                    else:
                        if t.minute >= trim_min_begin and text != "\n":
                            text = " ".join(text.split("\n"))
                            new_name = filename + "-" + str(cnt)

                            cmd = [
                                "ffmpeg",
                                "-i",
                                f"{os.path.join(self.download_dir,filename)}.wav",
                                "-ss",
                                str(start),
                                "-to",
                                str(end),
                                "-c",
                                "copy",
                                f"{os.path.join(self.split_dir,new_name)}.wav",
                            ]

                            call = subprocess.run(cmd, stderr=subprocess.STDOUT)

                            with open(
                                os.path.join(self.split_dir, new_name + ".txt"), "w"
                            ) as f:
                                f.write(text)
                    cnt += 1

                tqdm.write(
                    f"Completed splitting audios and texts to '{self.split_dir}'"
                )

                files_pbar = tqdm(files_list)

                tqdm.write(f"Verifying split audios and their transcriptions.")

                df = []
                for name in files_pbar:
                    filename, subtitle, trim_min_begin, trim_min_end = line.split(",")
                    files_pbar.set_description("Processing %s" % filename)
                    fname = filename[:-4]
                    files = os.listdir(self.split_dir)
                    for ix in range(len(files)):
                        current_file = fname + "-" + str(ix) + ".txt"
                        current_wav = current_file.replace(".txt", ".wav")
                        try:
                            current_text = (
                                open(os.path.join(self.split_dir, current_file))
                                .read()
                                .strip('-"')
                            )

                            wav, sr = librosa.load(
                                os.path.join(self.split_dir, current_wav)
                            )
                            length = wav.shape[0] / sr

                            if current_text != "" and length > 1:
                                df.append([current_wav, current_text, round(length, 2)])
                        except:
                            pass

                df = pd.DataFrame(
                    df, columns=["wav_file_name", "transcription", "length"]
                )
                df.to_csv(self.split_audios_csv, sep="|", index=None)

                tqdm.write(
                    f"Completed verifying audios and their transcriptions in '{self.split_dir}'."
                )
                tqdm.write(f"You can find files data in '{self.split_audios_csv}'")
        else:
            print(
                f"ERROR - Couldn't find file 'files.txt'. Make sure it is placed in {self.download_dir}"
            )
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), "files.txt"
            )

    def concat_audios(self, concat_count=5):
        """
        Joins the chunk of audio files into
        audios of recognizable length.
        """
        if os.path.exists(self.split_audios_csv) and os.path.isfile(
            self.split_audios_csv
        ):
            tqdm.write(f"Reading audio data from 'split.csv'.")
            df = pd.read_csv(self.split_audios_csv, sep="|")
            filtered_df = df[df["length"] <= self.min_audio_length]
            long_audios = df[df["length"] > self.min_audio_length]
            print(filtered_df)
            print(long_audios)

            name_ix = 0
            tqdm.write(f"Processing audios shorter than {max_limit} seconds..")
            for ix in tqdm(range(0, filtered_df.shape[0], 2)):
                current_audio = filtered_df.iloc[ix][0]
                print(f"short {current_audio} {ix}")
                text = filtered_df.iloc[ix][1]
                combined_sounds = AudioSegment.from_wav(
                    os.path.join(self.split_dir, current_audio)
                )
                combined_sounds += sound1
                try:
                    for count_ix in range(ix + 1, ix + concat_count):
                        next_audio = filtered_df.iloc[count_ix][0]
                        print(f"short 2 {next_audio} {ix}")
                        sound2 = AudioSegment.from_wav(
                            os.path.join(self.split_dir, next_audio)
                        )
                        text += " " + filtered_df.iloc[count_ix][1]
                        combined_sounds += sound2

                except IndexError:
                    print("IndexError, "+ix)
                finally:
                    text = text.strip()
                    new_name = f"{self.name}-{name_ix}"
                    combined_sounds.set_frame_rate(self.sr)
                    combined_sounds.export(
                        os.path.join(self.concat_dir, new_name + ".wav"), format="wav"
                    )
                    with open(
                        os.path.join(self.concat_dir, new_name + ".txt"), "w"
                    ) as f:
                        f.write(text)
                    name_ix += 1

            tqdm.write(f"Processing audios longer than {max_limit} seconds..")

            for ix in tqdm(range(0, long_audios.shape[0])):
                current_audio = filtered_df.iloc[ix][0]
                print(f"long {current_audio} {ix}")
                text = filtered_df.iloc[ix][1].strip()
                new_name = f"{self.name}-{name_ix}"
                combined_sounds = AudioSegment.from_wav(
                    os.path.join(self.split_dir, current_audio)
                )
                combined_sounds.set_frame_rate(self.sr)
                combined_sounds.export(
                    os.path.join(self.concat_dir, new_name + ".wav"), format="wav"
                )
                with open(os.path.join(self.concat_dir, new_name + ".txt"), "w") as f:
                    f.write(text)
                name_ix += 1

            tqdm.write(
                f"Completed concatenating audios and their transcriptions in '{self.concat_dir}'."
            )
        else:
            tqdm.write(
                f"ERROR - Couldn't find file 'split.csv'. Make sure it is placed in {self.split_dir}"
            )
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), "split.csv"
            )


    def get_total_audio_length(self):
        """
        Returns the total number of preprocessed audio
        in seconds.
        """

        tqdm.write(
            f"Collected {round(self.len_dataset/3600, 2)}hours ({int(self.len_dataset)} seconds) of audio."
        )
        return int(self.len_dataset)

    def finalize_dataset(self):
        """
        Trims silence from audio files
        and creates a medatada file in csv/json format.
        """
        tqdm.write(f"Trimming silence from audios in '{self.split_dir}'.")
        concat_dir = [
            wav for wav in os.listdir(self.concat_dir) if wav.endswith(".wav")
        ]
        _ = [wav.replace(".wav", ".txt") for wav in concat_dir]

        filtered_audios = []
        filtered_txts = []
        audio_lens = []

        for ix in tqdm(range(len(concat_dir))):
            audio = concat_dir[ix]
            wav, sr = librosa.load(os.path.join(self.split_dir, audio))
            silence_removed = preprocess_wav(wav)
            trimmed_length = silence_removed.shape[0] / sr
            audio_lens.append(trimmed_length)


            self.len_dataset += trimmed_length
            sf.write(os.path.join(self.dest_dir, "wavs", audio), silence_removed, sr, 'PCM_24')
            filtered_audios.append(audio)
            filtered_txts.append(audio.replace(".wav", ".txt"))

        for text in filtered_txts:
            shutil.copyfile(
                os.path.join(self.split_dir, text),
                os.path.join(self.dest_dir, "txts", text),
            )

        trimmed = []

        for wav, trans in zip(filtered_audios, filtered_txts):
            with open(os.path.join(self.split_dir, trans)) as f:
                text = f.read().strip()
            trimmed.append([wav, text])

        trimmed = pd.DataFrame(trimmed, columns=["wav_file_name", "transcription"])

        if not self.keep_audio_extension:
            trimmed["wav_file_name"] = trimmed["wav_file_name"].apply(
                lambda x: x.replace(".wav", "")
            )

        if self.output_type == "csv":
            trimmed["transcription_utf"] = trimmed["transcription"].apply(
                lambda x: self.cleaner.clean_english_text(x)
            )
            trimmed.to_csv(
                os.path.join(self.dest_dir, "metadata.csv"),
                sep="|",
                index=None,
                header=None,
            )
            tqdm.write(
                f"Dataset '{self.name}' has been generated. Wav files are placed in '{self.dest_dir}/wavs'. Transcription files are placed in '{self.dest_dir}/txts'."
            )
            tqdm.write(f"Metadata is placed in '{self.dest_dir}' as 'metadata.csv'.")
        elif self.output_type == "json":
            data = {}
            for ix in range(trimmed.shape[0]):
                name = trimmed.iloc[ix][0]
                text = trimmed.iloc[ix][1]
                data[name] = text
            with open(os.path.join(self.dest_dir, "alignment.json"), "w") as f:
                json.dump(data, f)
            tqdm.write(
                f"Dataset '{self.name}' has been generated. Wav files are placed in '{self.dest_dir}/wavs'. Transcription files are placed in '{self.dest_dir}/txts'."
            )
            tqdm.write(f"Metadata is placed in '{self.dest_dir}' as 'alignment.json'.")

        self.get_total_audio_length()

    def prepare_dataset(
        self,
        sr=22050,
        min_audio_length=7,
        max_audio_length=30,
    ):
        """
        A wrapper method for:
          split_audios
          finalize_dataset

        Use downloaded as wav files(optional),
        splits the audios into chunks, joins the
        junks into reasonable audios and trims silence
        from the audios. Creates a metadata file as csv/json
        after the dataset has been generated.

        Parameters:

              min_audio_length: The minimum length of audio files.

              max_audio_length: The maximum length of audio files.
        """
        self.min_audio_length = min_audio_length
        self.max_audio_length = max_audio_length
        self.sr = sr
        self.split_audios()
        self.concat_audios()
        self.finalize_dataset()
