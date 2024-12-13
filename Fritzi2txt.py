# Workflow
# 1. Download VOD audio file only m4a (aac) > use Twitch Downloader > see also additional helper code below
# 2. DO_CONVERT_WAV: convert audio file from m4a to wav
# 3. DO_SPLIT_AUDIO: cut wav file into smaller files, if hardware has not enough power. 1h files are create for e.g. google colab > use T4 GPU for faster results
# 4. DO_TRANSCRIPT: Check which model size to use: tiny, small, medium, large
# 5. At the same time, an attempt is made to recognize the number of speakers and to assign them to what has been said in the audio track
#
# NOTE: as each audio part is transcribed independently, the labeled speakers differ from file to file!
# The code should automatically recognize whether CUDA is present. If not, everything is calculated on CPU (CPU is very slow ~1h for 5-10 Minutes of audio vs. 30-40 minutes for 1h audio with GPU)

# Additional helper code
# convert from m4a to wav
# ffmpeg.exe -i "NAME_of_FILE.m4a" "NAME_of_FILE.wav"

# Split into x second pieces
# ffmpeg.exe -i "NAME_of_FILE.wav" -f segment -segment_time 3600 -c copy "NAME_of_FILE_part%03d.wav"
     

# Needed packages
# install for speaker recognition
#pip install pyannote.audio

#install openai whisper offline use
#pip install openai-whisper

# used for splitting wav
#pip install pydub
     


################################################################################
################################################################################
################################################################################
### Set here ###
#path to audio file
audio_path = './Fritzi/'
audio_file = '[12-12-24]_Saviarr_-_Sachen machen [SEKTOR]_cameo'
HF_TOKEN = "hf_YOURKEYHERE"

audio_file_split_size = 60  #size in minutes per file

#Switch
MODELLSIZE = 'large'   #small, medium, large
DO_CONVERT_WAV = 0
DO_SPLIT_AUDIO = 0
DO_TRANSCRIPT = 1
################################################################################
################################################################################
################################################################################


#convert m4a to wav/mp3
if DO_CONVERT_WAV == 1:
    from pydub import AudioSegment
    m4a_file = audio_path + audio_file + '.m4a'
    wav_filename = audio_path + audio_file + '.wav'

    sound = AudioSegment.from_file(m4a_file, format='m4a')
    file_handle = sound.export(wav_filename, format='wav')
    print('done')

     
################################################################################


### Splittet die Wav Datei in x Minuten Teile. Dies kann helfen, wenn die Hardware nicht soviel Performance hat
from pydub import AudioSegment
import math
def split_audio(file_path, segment_length=audio_file_split_size*60*1000):  # x minutes in milliseconds
	# Load the audio file
	audio = AudioSegment.from_file(file_path)

	# Get the total length of the audio file
	total_length = len(audio)

	# Calculate the number of segments needed
	num_segments = math.ceil(total_length / segment_length)

	# Loop through and create each segment
	for i in range(num_segments):
		start_time = i * segment_length
		end_time = min((i + 1) * segment_length, total_length)  # Ensure the last segment does not exceed total length
		segment = audio[start_time:end_time]

		# Generate the output file name
		output_file = f"{file_path[:-4]}_part{i+1}.wav"

		# Export the segment as an MP3 file
		segment.export(output_file, format="wav")
		print(f"Exported: {output_file}")


if DO_SPLIT_AUDIO == 1:
	split_audio(audio_path + audio_file + '.wav')

################################################################################


if DO_TRANSCRIPT == 1:
  import os
  import whisper
  import time
  import torch
  import torchaudio
  from pyannote.audio import Pipeline
  from pyannote.audio.pipelines.utils.hook import ProgressHook
  from pyannote.core import Segment
  import functools

  # Funktion zum Formatieren der Zeit im Format hh:mm:ss
  def format_time(seconds):
      """Konvertiere eine Zeit in Sekunden in das Format hh:mm:ss"""
      hours = int(seconds // 3600)
      minutes = int((seconds % 3600) // 60)
      seconds = int(seconds % 60)
      return f"{hours:02}:{minutes:02}:{seconds:02}"

  # Funktion zur Ermittlung der Anzahl der Parts
  def get_num_parts(audio_path, audio_file):
      """Ermittelt die Anzahl der Part-Dateien im Verzeichnis"""
      parts = [f for f in os.listdir(audio_path) if f.startswith(audio_file) and f.endswith(".wav")]
      parts.sort()  # Optional, falls du die Dateien sortieren möchtest
      return parts

  # Funktion zum Prüfen, ob alle Part-Dateien vorhanden sind
  def check_missing_parts(parts, num_parts):
      missing_parts = []
      for i in range(1, num_parts + 1):
          part_name = f"{audio_file}_part{i}.wav"
          if part_name not in parts:
              missing_parts.append(part_name)
      return missing_parts

  # Tik: Startzeitpunkt
  print("Tik: Start der Ausführung")
  start_time = time.time()

  # CUDA-Speicher freigeben
  torch.cuda.empty_cache()

  # Prüfe, ob CUDA verfügbar ist
  torch.cuda.is_available()
  DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
  # Höhere Genauigkeit, benötigt aber die richtige Hardware
  #if torch.cuda.is_available():
  #    torch.backends.cuda.matmul.allow_tf32 = True
  #    torch.backends.cudnn.allow_tf32 = True

  # Umgebungsvariable für CUDA-Speicherverwaltung
  import os
  os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

  # Lade das Whisper-Modell
  model = whisper.load_model(MODELLSIZE, device=DEVICE)  # 'small' Modell ohne fp16

  # Lade die Sprecherdiarisierungs-Pipeline
  pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN)
  pipeline.to(torch.device(DEVICE))

  # Ermittle die vorhandenen Teile
  parts = get_num_parts(audio_path, audio_file)

  # Prüfe auf fehlende Parts
  missing_parts = check_missing_parts(parts, 10)  # Hier 10 anpassen, falls du eine andere maximale Anzahl von Parts erwartest
  if missing_parts:
      print(f"Warnung: Die folgenden Part-Dateien fehlen: {', '.join(missing_parts)}")

  # Durchlaufe alle Parts und führe Transkription und Sprecherdiarisierung durch
  for part in parts:
      part_index = part.split("_")[-1].replace(".wav", "")  # Extrahiere die Part-Nummer aus dem Dateinamen

      print(f"Verarbeite {part}...")

      # Lade die Audiodatei
      waveform, sample_rate = torchaudio.load(os.path.join(audio_path, part))

      # Durchführen der Sprecherdiarisierung
      with ProgressHook() as hook:
          diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate}, hook=hook)

      # Transkription der Audiodatei mit Whisper
      result = model.transcribe(os.path.join(audio_path, part), language="de", task="transcribe", verbose=True, word_timestamps=True)

      # Ausgabe in eine Textdatei exportieren
      output_file = os.path.join(audio_path, f"{audio_file}_part{part_index}_mit_speakern.txt")

      # Öffnen der Datei im Schreibmodus
      with open(output_file, "w", encoding="utf-8") as file:
          file.write(f"Transkription Staffel 2 mit Sprechererkennung von {part}:\n")

          # Durchlaufe die Segmente (jeder Abschnitt der Audiodatei)
          for segment in result["segments"]:
              starting_time = format_time(segment['start'])
              ending_time = format_time(segment['end'])
              text = segment['text']

              # Bestimme den Sprecher
              segment_time = Segment(segment['start'], segment['end'])
              speaker = None
              for turn, _, speaker_name in diarization.itertracks(yield_label=True):
                  if turn.start <= segment_time.end and turn.end >= segment_time.start:
                      speaker = speaker_name
                      break

              # Wenn ein Sprecher erkannt wurde, nutze diesen, ansonsten setze 'Unbekannt'
              speaker_label = f"Sprecher {speaker}" if speaker else "Unbekannt"

              # Schreibe Zeitstempel und Text mit Sprecherkennung
              file.write(f"{starting_time} - {ending_time} - {speaker_label}: {text}\n")

      print(f"Die Transkription mit Sprechererkennung wurde erfolgreich in die Datei '{output_file}' exportiert.")

  # Berechne die Laufzeit
  end_time = time.time()
  elapsed_time = end_time - start_time
  hours = int(elapsed_time // 3600)
  minutes = int((elapsed_time % 3600) // 60)
  seconds = int(elapsed_time % 60)

  # Ausgabe im Format hh:mm:ss
  print(f"Laufzeit: {hours:02}:{minutes:02}:{seconds:02}")
