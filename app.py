from flask import Flask, abort, request
import whisper
from tempfile import NamedTemporaryFile

# Load the Whisper model:
model = whisper.load_model('medium')

import argostranslate.package
import argostranslate.translate

from_code = "hi"
to_code = "en"

# Download and install Argos Translate package
argostranslate.package.update_package_index()
available_packages = argostranslate.package.get_available_packages()
package_to_install = next(
    filter(
        lambda x: x.from_code == from_code and x.to_code == to_code, available_packages
    )
)
argostranslate.package.install_from_path(package_to_install.download())





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
        # Now we can store the result object for this file.
        results.append({
            'filename': filename,
            'transcript': result['text'],
            'translatedText': translatedText
        })

    # This will be automatically converted to JSON.
    return {'results': results}


if __name__ == "__main__":
    app.run(debug=True, threaded=True, port=5001)