from flask import Flask, abort, request
import argostranslate.translate
import whisper
from tempfile import NamedTemporaryFile
from translate import Translate
from diarization import metadata_audio_file, embedding_batch, speaker_segmentation, segment_embedding, convert_to_wav
# Load the Whisper model:
model = whisper.load_model('medium')

from_code = "hi"
to_code = "en"

Translate(from_code, to_code)



app = Flask(__name__)

@app.route('/', methods=['POST'])
def handler():
    if not request.files:
        # If the user didn't submit any files, return a 400 (Bad Request) error.
        abort(400)

    # For each file, let's store the results in a list of dictionaries.
    results = []

    # Loop over every file that the user submitted.
    for filename, handle in request.files.items():
        # Create a temporary file.
        # The location of the temporary file is available in `temp.name`.
        temp = NamedTemporaryFile()
        # Write the user's uploaded file to the temporary file.
        # The file will get deleted when it drops out of scope.
        handle.save(temp)
        # Let's get the transcript of the temporary file.
        result = model.transcribe(temp.name)
        
        # Translate
        translatedText = argostranslate.translate.translate(result['text'], from_code, to_code)
        
        # Diarization Steps
        segments = result["segments"]
        path = convert_to_wav(temp.name)
        frames, rate, duration = metadata_audio_file(path)
        embeddings = embedding_batch(segments,duration, path)
        segments , speaker_separation  = speaker_segmentation(embeddings, segments)
        
        # Now we can store the result object for this file.
        results.append({
            'filename': filename,
            'transcript': result['text'],
            'translatedText': translatedText,
            "speaker_1": speaker_separation["speaker_1"], 
            "speaker_2": speaker_separation["speaker_2"]
        })

    # This will be automatically converted to JSON.
    return {'results': results}


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, threaded=True, port=5002)