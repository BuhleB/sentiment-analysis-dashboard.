# Sentiment Analysis Dashboard

This interactive dashboard allows users to analyze the sentiment of text data, providing multi-class sentiment classification (positive, negative, neutral), confidence scoring, and keyword extraction. It supports direct text entry, file uploads (TXT, CSV), and batch processing. The results are visualized through sentiment distribution charts and can be compared side-by-side. Users can export their analysis results in CSV, JSON, or PDF formats.

## Features

*   **Text Input:** Direct entry of text for immediate analysis.
*   **File Upload:** Upload `.txt` files for single text analysis or `.csv` files for batch processing of multiple texts.
*   **Multi-class Sentiment Classification:** Categorizes text into positive, negative, or neutral sentiments.
*   **Confidence Scoring:** Provides a confidence score for each sentiment classification.
*   **Keyword Extraction:** Identifies and highlights key phrases that drive the sentiment of the text.
*   **Batch Processing:** Efficiently analyzes multiple texts at once.
*   **Sentiment Distribution Visualization:** Pie charts and bar charts to show the overall sentiment breakdown and average confidence scores.
*   **Comparative Analysis:** Allows users to select and compare multiple analysis results side-by-side.
*   **Explanation Features:** Provides insights into why a specific sentiment was assigned, based on extracted keywords.
*   **Export Options:** Export results in CSV, JSON, and PDF formats.

## Technical Specifications

*   **NLP API Integration:** Utilizes Hugging Face Transformers for sentiment analysis (using `distilbert-base-uncased-finetuned-sst-2-english`) and keyword extraction (using KeyBERT with a `sentence-transformers` model).
*   **Web Interface:** Built with Streamlit for an interactive and responsive user experience.
*   **Error Handling:** Robust error handling for API failures and invalid inputs.
*   **Efficient Processing:** Appropriate batching for efficient text processing.

## Setup and Installation

1.  **Clone the repository (or create the files manually):**

    ```bash
    git clone <repository_url>
    cd sentiment-analysis-dashboard
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    *Note: If `requirements.txt` is not provided, install the following packages manually:*
    ```bash
    pip install streamlit pandas plotly transformers keybert sentence-transformers fpdf
    ```

## Usage

1.  **Run the Streamlit application:**

    ```bash
    streamlit run dashboard.py
    ```

2.  **Access the dashboard:** Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

3.  **Choose an input method:**
    *   **Direct Text Entry:** Type or paste text directly into the input box and click "Analyze Text."
    *   **File Upload:** Upload a `.txt` file for single analysis or a `.csv` file containing a column of texts for batch analysis. Select the appropriate text column for CSV files.
    *   **Batch Processing:** Enter multiple texts, one per line, into the text area and click "Analyze Batch."

4.  **View Results:** Analysis results will be displayed, including sentiment, confidence scores, and extracted keywords. Summary visualizations will show sentiment distribution.

5.  **Comparative Analysis:** Select multiple results from the table to view a side-by-side comparison.

6.  **Export Results:** Choose your desired format (CSV, JSON, or PDF) and click "Download Results" to save your analysis.

## Model Limitations and Confidence Thresholds

*   **Sentiment Model:** The `distilbert-base-uncased-finetuned-sst-2-english` model is trained on movie reviews and may perform differently on other domains (e.g., highly technical text, sarcasm).
*   **Keyword Extraction:** KeyBERT relies on the quality of embeddings from `sentence-transformers` and may not always capture nuanced or domain-specific keyphrases perfectly.
*   **Confidence Thresholds:** A confidence score below 0.7-0.8 might indicate that the model is less certain about its prediction. Users should interpret results with lower confidence scores cautiously.

## Future Enhancements

*   Support for more NLP models and languages.
*   Integration with external data sources (e.g., social media APIs).
*   Advanced sentiment analysis features (e.g., aspect-based sentiment analysis).
*   User authentication and result persistence.
*   Improved UI/UX for large datasets.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

