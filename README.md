# DocVerify — Vercel-ready document tools + local AI-text detector

DocVerify includes document text extraction, text/file similarity comparison, image-to-PDF conversion, PDF-to-image ZIP conversion, and a browser camera scanner. The AI-generated-text detector is designed to run a **locally trained model**. It does not call Hugging Face inference, OpenAI, or any paid inference API, and needs no API key.

## AI detector

This complete ZIP includes a trained model at `models/ai_detector.joblib`. The model was trained on 24,000 samples from the public English dataset `artem9k/ai-text-detection-pile`. The app uses the bundled model locally and does not require an inference API key. If you retrain the model, keep the saved bundle structure (`pipeline` and `metadata`) and update the model file.

The included training script uses the public English dataset [`artem9k/ai-text-detection-pile`](https://huggingface.co/datasets/artem9k/ai-text-detection-pile), whose dataset card lists an MIT license and human/AI labels. It is about 1.97 GB in full, but the script streams a balanced sample and does not download the entire dataset. Review the dataset card and upstream terms before use. Dataset: https://huggingface.co/datasets/artem9k/ai-text-detection-pile

### Train for free in Google Colab

1. Download and extract this project ZIP locally, or upload the ZIP to Colab.
2. Open https://colab.research.google.com/ and create a new notebook.
3. Upload the project ZIP in the first cell of `colab_train.ipynb`, or follow these cells manually:

```python
from google.colab import files
uploaded = files.upload()  # choose docverify-local-ai-detector.zip
```

```python
import zipfile, os
zip_name = next(name for name in uploaded if name.endswith('.zip'))
os.makedirs('/content/docverify', exist_ok=True)
with zipfile.ZipFile(zip_name) as z:
    z.extractall('/content/docverify')
# If the archive contains a single top-level folder, locate the script:
import glob
script = glob.glob('/content/docverify/**/train_ai_detector.py', recursive=True)[0]
project = os.path.dirname(script)
print(project)
```

```python
%cd $project
!pip -q install -r requirements-train.txt
!python train_ai_detector.py --max-per-class 12000
```

4. Training streams up to 12,000 human-written and 12,000 AI-generated samples, uses a stratified 80/20 train/test split, and reports accuracy, precision, recall, F1, ROC-AUC, and a confusion matrix. This is a starting evaluation, not a proof of real-world performance.
5. Download the model artifact:

```python
from google.colab import files
files.download('models/ai_detector.joblib')
```

6. Put the downloaded file into your local project's `models/` folder, so the path is exactly `models/ai_detector.joblib`.
7. Keep `models/evaluation.json` as a training report if useful. Do not upload the full dataset or raw training examples to GitHub.

The provided `colab_train.ipynb` automates the steps above.

### Training configuration

- Dataset: `artem9k/ai-text-detection-pile` (English; dataset card lists MIT license).
- Features: word-level TF-IDF unigrams and bigrams.
- Classifier: Logistic Regression.
- Labels: `0 = human-written`, `1 = AI-generated`.
- Default sample: 12,000 examples per class (24,000 total before train/test split).
- You can change the sample size, e.g. `--max-per-class 5000` for a quicker first run. Very small training samples are not recommended for evaluation.

## Deploy to Vercel

1. Add `models/ai_detector.joblib` to the project. It is created by the training step; it is not pre-trained in this ZIP.
2. Push the project files to your GitHub repository, including `train_ai_detector.py`, `requirements-train.txt`, `colab_train.ipynb`, and the trained artifact. Do not commit dataset files, environment secrets, or personal documents.
3. Import/update the repository in Vercel and deploy. Runtime dependencies are listed in `requirements.txt`; training-only dependencies are in `requirements-train.txt`.
4. No environment variables or API keys are required for the detector.
5. After deployment, test `/health`, then paste an English text sample (at least 100 characters) in the AI detector UI.
6. If you replace the model file, redeploy so the new artifact is bundled. The model is cached for each warm Python instance.

### Vercel constraints

- Vercel Python Functions have a 500 MB maximum uncompressed bundle size; check the final model and dependency bundle if deployment fails. See https://vercel.com/docs/functions/runtimes/python and https://vercel.com/docs/functions/limitations.
- Vercel Functions have a 4.5 MB request/response payload limit. AI text is sent as JSON and the endpoint limits it to 20,000 characters. Existing document conversion routes may fail for larger uploads; lower `MAX_UPLOAD_MB` does not override Vercel's platform limit.
- The current configuration is intended for Vercel's Python runtime and Flask entrypoint at `api/index.py`.

## Detector limitations

AI-generated-text detection is an uncertain classification task. A TF-IDF model can learn vocabulary, topic, length, or dataset-source artifacts rather than authorship. It can misclassify human writing, edited AI text, short samples, writing from different domains, or outputs from newer models. The displayed number is a classifier score—not a calibrated probability that someone used AI and not proof of misconduct. Do not use it as the sole basis for academic, employment, or other high-stakes decisions.

For responsible evaluation, test on a separate dataset and on human and AI writing from the target domain. Consider reporting false-positive rate and performance by text length/domain, and adjust thresholds only using a separate validation set.

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5000. If you have not trained and placed the model artifact, all other tools work, while the detector explains that model setup is incomplete.

## Tests

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## API

- `GET /health`
- `POST /api/ai-detect` — JSON `{ "text": "..." }`; requires the local model artifact.
- `POST /api/plagiarism/text` — JSON `{ "text1": "...", "text2": "..." }`; compares only the supplied texts, not the internet.
- `POST /api/plagiarism/compare` — multipart fields `file1`, `file2`.
- `POST /api/extract` — multipart field `file`.
- `POST /api/convert/images-to-pdf` — multipart field(s) `files`.
- `POST /api/convert/pdf-to-images` — multipart field `file`.


## Deployment notes

- Runtime uses Python 3.12 and `scikit-learn==1.6.1`, matching the model artifact's training version.
- The model is bundled at `models/ai_detector.joblib` and included in the Vercel Python function via `vercel.json`.
- No Hugging Face inference token or paid inference API is used.
- Vercel's local filesystem is ephemeral. This project does not promise persistent document history across serverless invocations.
- Test locally before pushing: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && python app.py`.
- AI detection is an experimental estimate, not proof of authorship. The reported held-out metrics can overstate real-world performance if test examples resemble training data; do not use the score alone for disciplinary or high-stakes decisions.
