Voice-Based Cognitive Decline Pattern Detection
This project aims to detect cognitive decline patterns, such as those associated with dementia, using audio and natural language processing (NLP) features extracted from speech samples. Leveraging the DementiaBank Pitt Corpus, the pipeline processes audio files, extracts features like pause count, speech rate, and lexical diversity, and applies unsupervised learning (K-means and Isolation Forest) to identify potential patterns.
Features

Preprocesses audio files and generates transcripts using faster-whisper.
Extracts audio features (e.g., pause count, speech rate) and NLP features (e.g., hesitation count, lexical diversity) with librosa and spacy.
Applies K-means clustering and Isolation Forest for anomaly detection.
Generates visualizations (e.g., pause count boxplot, speech rate scatter) using matplotlib and seaborn.
Saves results in CSV format and model pickles for further analysis.

Requirements

Python 3.10
Required packages (install via requirements.txt):
librosa==0.10.1
noisereduce==3.0.2
faster-whisper==1.0.3
nltk==3.8.1
spacy==3.7.2
sentence-transformers==2.7.0
huggingface-hub==0.23.0
transformers==4.41.0
scikit-learn==1.3.2
pandas==2.1.4
numpy==1.26.2
matplotlib==3.8.2
seaborn==0.13.0


Additional setup:python -m spacy download en_core_web_sm
python -m nltk.downloader punkt



Installation

Clone the repository:git clone https://github.com/VarshithGaddam/Voice-Based-Cognitive-Decline-Pattern-Detection.git
cd Voice-Based-Cognitive-Decline-Pattern-Detection


Create a virtual environment:python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install dependencies:pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt



Usage

Place audio files (e.g., .wav from DementiaBank Pitt Corpus) in the data/raw_samples/ directory.
Run the pipeline:python main.py


Check results in the results/ directory:
results.csv: Feature and modeling output.
results/visualizations/: Generated plots (e.g., pause_count_boxplot.png).
results/report.md: Project report.



Project Structure

src/: Source code (e.g., pipeline.py, feature_extraction.py).
data/: Input data (e.g., raw_samples/, processed/).
results/: Output files (e.g., results.csv, visualizations/).
notebooks/: Jupyter notebooks for analysis (e.g., analysis.ipynb).
requirements.txt: Dependency list.
README.md: This file.
venv/: Virtual environment (optional, add to .gitignore).

Visualizations
The pipeline generates four visualizations:

Pause Count by Cluster: Boxplot of pause frequency.
Speech Rate vs. Pause Count: Scatter plot with cluster hue and risk score size.
Hesitation Count by Cluster: Boxplot of speech disfluencies.
Lexical Diversity vs. Incomplete Sentences: Scatter plot of linguistic features.

Contributing
Feel free to fork this repository, submit issues, or create pull requests. Suggestions for improving feature extraction or visualization are welcome!
License
MIT License - See LICENSE for details.
Acknowledgments

DementiaBank Pitt Corpus for audio data.
xAI and various open-source libraries for enabling this project.

Contact
For questions or collaboration, reach out to varshith.gaddam@example.com or open an issue on this repository.
