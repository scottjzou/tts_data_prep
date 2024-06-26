# Youtube Speech Data Generator

A python library to generate speech dataset based on downloaded youtube audio and subtitles.
Forked and improved based on https://github.com/hetpandya/youtube_tts_data_generator/ which is outdated

## Installation
Make sure [ffmpeg](https://ffmpeg.org/download.html#get-packages) is installed and is set to the system path.
```bash
$ brew install ffmpeg
$ pip install requirements.txt
```

## Prep
find the video you wanted on youtube, make sure it has a high quality subtitle

find the subtitle via 
```bash
yt-dlp --list-subs    https://www.youtube.com/watch\?v\=Q6YvsrGILrw
# output
# Language       Name                             Formats
# en-dQs7zDoAYDs English - PostTV Closed Captions vtt, ttml, srv3, srv2, srv1, json3

#use the listed language to download both video and subtitles
yt-dlp -x -f bestaudio --write-subs --sub-langs "en-dQs7zDoAYDs" --audio-format wav --sub-format vtt https://www.youtube.com/watch\?v\=Q6YvsrGILrw              
```bash
## Generation

```python
from youtube_tts_data_generator import YTSpeechDataGenerator
# First create a YTSpeechDataGenerator instance with download_dir pointed to downloaded audio and subtitles:
# make sure a files.txt that links audio with associated subtitles is present in the directory
generator = YTSpeechDataGenerator(dataset_name='elon', download_dir='path-to-downloaded-wav-and-vtt')

generator.prepare_dataset()
# The above will take care about creating your dataset, creating a metadata file and trimming silence from the audios.

```

## Usage
<!--ts-->
- Initializing the generator:
  ```generator = YTSpeechDataGenerator(dataset_name='elon', download_dir='path-to-downloaded-wav-and-vtt')```
  - Parameters:
    - *dataset_name*: 
      - The name of the dataset you'd like to give. 
      - A directory structure like this will be created:
        ```
        ├───your_dataset
        │   ├───txts
        │   └───wavs
        └───your_dataset_prep
            ├───concatenated
            └───split
        ```
    - *download_dir*: 
      - The path where your audio and subtitles are stored
         ```
         └───download_dir
            ├───audio.wav
            ├───audio.vtt
            └───files.txt
         ```
      - a 'files.txt' must be present so that the generator can parse files correctly
      - the 'files.txt' should follow the following format:
      
        ```
        filename,subtitle,trim_min_begin,trim_min_end
        audio.wav,subtitle.srt,0,0
        audio2.wav,subtitle.vtt,5,6
        ```
    - *output_type*: 
      - The type of the metadata to be created after the dataset has been generated.
      - Supported types: csv/json
      - Default output type is set to *csv*
      - The csv file follows the format of [LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/)
      - The json file follows this format:
        ```
        {
            "your_dataset1.wav": "This is an example text",
            "your_dataset2.wav": "This is an another example text",
        }
        ```
    - *keep_audio_extension*:
      - Whether to keep the audio file extension in the metadata file
      - Default value is set to *False*
    - *lang*:
      - The key for the target language in which the subtitles have to be downloaded.
      - Default value is set to *en*
      - *Tip* - check list of available languages and their keys using: `generator.get_available_langs()`
    - *sr*:
      - Sample Rate to keep of the audios.
      - Default value is set to *22050*
 
- Methods:
  - split_audios():
    - This method splits all the wav files into smaller chunks according to the duration of the text in the subtitles.
    - Saves transcriptions as '.txt' file for each of the chunks.
    - Example - ```generator.split_audios()```
  - concat_audios():
    - Since the split audios are based on the duration of their subtitles, they might not be so long. This method joins the split files into recognizable ones.
    - Parameters:
      - *max_limit*: 
        - The upper limit of length of the audios that should be concated. The rest will be kept as they are.
        - The default value is set to *7*
      - *concat_count*: 
        - The number of consecutive audios that should be concated together. 
        - The default value is set to *2*
    - Example - ```generator.concat_audios()```
  - finalize_dataset():
    - Trims silence the joined audios since the data has been collected from YouTube and generates the final dataset after finishing all the preprocessing.
    - Parameters:
      - *min_audio_length*:
        - The minumum length of the speech that should be kept. The rest will be ignored.
        - The default value is set set to *5*.
      - *max_audio_length*:
        - The maximum length of the speech that should be kept. The rest will be ignored.
        - The default value is set set to *14*.        
    - Example - ```generator.finalize_dataset(min_audio_length=6)```
  - get_available_langs():
    - Get list of available languages in which the subtitles can be downloaded.
    - Example - ```generator.get_available_langs()```
  - get_total_audio_length():
    - Returns the total amount of preprocessed speech data collected by the generator.
    - Example - ```generator.get_total_audio_length()```
  - prepare_dataset():
    - A wrapper method for *download()*,*split_audios()*,*concat_audios()* and *finalize_dataset()*.
    - If you do not wish to use the above methods, you can directly call *prepare_dataset()*. It will handle all your data generation.
    - Parameters:
      - *sr*:
        - Sample Rate to keep of the audios.
        - Default value is set to *22050*  
      - *max_concat_limit*: 
        - The upper limit of length of the audios that should be concated. The rest will be kept as they are.
        - The default value is set to *7*
      - *concat_count*: 
        - The number of consecutive audios that should be concated together. 
        - The default value is set to *2*
      - *min_audio_length*:
        - The minumum length of the speech that should be kept. The rest will be ignored.
        - The default value is set set to *5*.        
      - *max_audio_length*:
        - The maximum length of the speech that should be kept. The rest will be ignored.
        - The default value is set set to *14*.        
    - Example - ```generator.prepare_dataset(min_audio_length=6)```
<!--te-->

## Final dataset structure
Once the dataset has been created, the structure under 'your_dataset' directory should look like:
```
your_dataset
├───txts
│   ├───your_dataset1.txt
│   └───your_dataset2.txt
├───wavs
│    ├───your_dataset1.wav
│    └───your_dataset2.wav
└───metadata.csv/alignment.json
```
