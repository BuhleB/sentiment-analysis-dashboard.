import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sentiment_analyzer import SentimentAnalyzer
import json
import csv
import io
from fpdf import FPDF

# Initialize the sentiment analyzer
@st.cache_resource
def load_analyzer():
    return SentimentAnalyzer()

def main():
    st.set_page_config(
        page_title="Sentiment Analysis Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📊 Sentiment Analysis Dashboard")
    st.markdown("Analyze emotional tone in text data with multi-class sentiment classification and keyword extraction")
    
    # Initialize analyzer
    analyzer = load_analyzer()
    
    # Sidebar for input options
    st.sidebar.header("Input Options")
    input_method = st.sidebar.radio(
        "Choose input method:",
        ["Direct Text Entry", "File Upload", "Batch Processing"]
    )
    
    # Initialize session state for storing results
    if 'results' not in st.session_state:
        st.session_state.results = []
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if input_method == "Direct Text Entry":
            st.header("📝 Text Input")
            text_input = st.text_area(
                "Enter text to analyze:",
                height=150,
                placeholder="Type or paste your text here..."
            )
            
            if st.button("Analyze Text", type="primary"):
                if text_input.strip():
                    with st.spinner("Analyzing sentiment and extracting keywords..."):
                        sentiment, scores = analyzer.analyze_sentiment(text_input)
                        keywords = analyzer.extract_keywords(text_input)
                        
                        if sentiment == "ERROR":
                            st.error("Sentiment analysis failed. Please check the `sentiment_analyzer.py` logs for details.")
                        else:
                            result = {
                                'text': text_input,
                                'sentiment': sentiment,
                                'scores': scores,
                                'keywords': keywords,
                                'timestamp': pd.Timestamp.now()
                            }
                            
                            st.session_state.results.append(result)
                            
                            # Display results
                            display_single_result(result)
                else:
                    st.warning("Please enter some text to analyze.")
        
        elif input_method == "File Upload":
            st.header("📁 File Upload")
            uploaded_file = st.file_uploader(
                "Choose a text file",
                type=['txt', 'csv'],
                help="Upload a .txt file or .csv file with text data"
            )
            
            if uploaded_file is not None:
                if uploaded_file.type == "text/plain":
                    text_content = str(uploaded_file.read(), "utf-8")
                    st.text_area("File content:", text_content, height=200, disabled=True)
                    
                    if st.button("Analyze File Content", type="primary"):
                        with st.spinner("Analyzing file content..."):
                            sentiment, scores = analyzer.analyze_sentiment(text_content)
                            keywords = analyzer.extract_keywords(text_content)
                            
                            if sentiment == "ERROR":
                                st.error("Sentiment analysis failed for file content. Please check the `sentiment_analyzer.py` logs for details.")
                            else:
                                result = {
                                    'text': text_content,
                                    'sentiment': sentiment,
                                    'scores': scores,
                                    'keywords': keywords,
                                    'timestamp': pd.Timestamp.now(),
                                    'source': uploaded_file.name
                                }
                                
                                st.session_state.results.append(result)
                                display_single_result(result)
                
                elif uploaded_file.type == "text/csv":
                    df = pd.read_csv(uploaded_file)
                    st.dataframe(df.head())
                    
                    text_column = st.selectbox(
                        "Select the text column:",
                        df.columns.tolist()
                    )
                    
                    if st.button("Analyze CSV Data", type="primary"):
                        process_batch_data(df, text_column, analyzer)
        
        elif input_method == "Batch Processing":
            st.header("🔄 Batch Processing")
            batch_texts = st.text_area(
                "Enter multiple texts (one per line):",
                height=200,
                placeholder="Text 1\nText 2\nText 3..."
            )
            
            if st.button("Analyze Batch", type="primary"):
                if batch_texts.strip():
                    texts = [text.strip() for text in batch_texts.split('\n') if text.strip()]
                    process_batch_texts(texts, analyzer)
                else:
                    st.warning("Please enter some texts to analyze.")
    
    with col2:
        st.header("📈 Analysis Summary")
        
        if st.session_state.results:
            df_results = pd.DataFrame(st.session_state.results)
            
            # Filter out error results for visualizations
            df_results_filtered = df_results[df_results['sentiment'] != 'ERROR']

            if not df_results_filtered.empty:
                # Sentiment distribution Pie Chart
                sentiment_counts = df_results_filtered['sentiment'].value_counts()
                fig_pie = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    title="Sentiment Distribution",
                    color_discrete_map={
                        'POSITIVE': '#2E8B57',
                        'NEGATIVE': '#DC143C',
                        'NEUTRAL': '#4682B4'
                    }
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                
                # Average Confidence Scores Bar Chart
                normalized_scores = []
                for _, row in df_results_filtered.iterrows():
                    scores = row['scores']
                    normalized_scores.append({
                        'POSITIVE': scores.get('POSITIVE', 0.0),
                        'NEGATIVE': scores.get('NEGATIVE', 0.0),
                        'NEUTRAL': scores.get('NEUTRAL', 0.0)
                    })
                
                df_normalized_scores = pd.DataFrame(normalized_scores)
                avg_confidence = df_normalized_scores.mean().reset_index()
                avg_confidence.columns = ['Sentiment Type', 'Average Confidence']
                
                fig_bar = px.bar(
                    avg_confidence,
                    x='Sentiment Type',
                    y='Average Confidence',
                    title='Average Confidence Scores by Sentiment Type',
                    color='Sentiment Type',
                    color_discrete_map={
                        'POSITIVE': '#2E8B57',
                        'NEGATIVE': '#DC143C',
                        'NEUTRAL': '#4682B4'
                    },
                    range_y=[0, 1]
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("No valid analysis results to display summary. All analyses might have resulted in errors.")
            
            if st.button("Clear All Results", type="secondary"):
                st.session_state.results = []
                st.rerun()
        else:
            st.info("No analysis results yet. Start by analyzing some text!")
    
    if st.session_state.results:
        st.header("📋 Analysis Results")
        display_results_table()
        
        st.header("⚖️ Comparative Analysis")
        compare_results()

        st.header("💾 Export Results")
        export_format = st.selectbox(
            "Choose export format:",
            ["CSV", "JSON", "PDF"]
        )
        
        if st.button("Download Results"):
            download_results(export_format)

def display_single_result(result):
    """Display a single analysis result with sentiment, confidence, keywords, and explanation."""
    if result['sentiment'] == "ERROR":
        st.error("Analysis failed for this text.")
        return

    st.success("Analysis Complete!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sentiment Analysis")
        sentiment = result['sentiment']
        confidence = result['scores'].get(sentiment, 0.0)
        
        color = {
            'POSITIVE': 'green',
            'NEGATIVE': 'red',
            'NEUTRAL': 'blue'
        }.get(sentiment, 'gray')
        
        st.markdown(f"**Sentiment:** :{color}[{sentiment}]")
        st.markdown(f"**Confidence:** {confidence:.2%}")
        
        st.subheader("Confidence Scores")
        sorted_scores = sorted(result['scores'].items(), key=lambda item: item[1], reverse=True)
        for sent, score in sorted_scores:
            st.progress(score, text=f"{sent}: {score:.2%}")
    
    with col2:
        st.subheader("Keywords & Explanation")
        if result['keywords']:
            keywords_html = " ".join([
                f'<span style="background-color: #e1f5fe; padding: 2px 8px; margin: 2px; border-radius: 12px; font-size: 12px;">{kw}</span>'
                for kw in result['keywords']
            ])
            st.markdown(keywords_html, unsafe_allow_html=True)
            
            st.markdown("---")
            explanation = ""
            if sentiment == 'POSITIVE':
                explanation = "The sentiment is predominantly positive, likely driven by keywords expressing satisfaction, approval, or good quality."
            elif sentiment == 'NEGATIVE':
                explanation = "The sentiment is predominantly negative, likely influenced by keywords indicating dissatisfaction, issues, or poor quality."
            elif sentiment == 'NEUTRAL':
                explanation = "The sentiment is neutral, suggesting the text contains balanced information or lacks strong emotional indicators."
            st.markdown(f"**Explanation:** {explanation}")
        else:
            st.info("No keywords extracted, making it harder to explain the sentiment drivers.")

def process_batch_texts(texts, analyzer):
    """Process multiple texts in batch and store results."""
    progress_bar = st.progress(0)
    
    for i, text in enumerate(texts):
        with st.spinner(f"Processing text {i+1}/{len(texts)}..."):
            sentiment, scores = analyzer.analyze_sentiment(text)
            keywords = analyzer.extract_keywords(text)
            
            result = {
                'text': text,
                'sentiment': sentiment,
                'scores': scores,
                'keywords': keywords,
                'timestamp': pd.Timestamp.now()
            }
            
            st.session_state.results.append(result)
            progress_bar.progress((i + 1) / len(texts))
    
    st.success(f"Batch processing complete! Analyzed {len(texts)} texts.")

def process_batch_data(df, text_column, analyzer):
    """Process CSV data and store results."""
    progress_bar = st.progress(0)
    
    for i, row in df.iterrows():
        text = str(row[text_column])
        with st.spinner(f"Processing row {i+1}/{len(df)}..."):
            sentiment, scores = analyzer.analyze_sentiment(text)
            keywords = analyzer.extract_keywords(text)
            
            result = {
                'text': text,
                'sentiment': sentiment,
                'scores': scores,
                'keywords': keywords,
                'timestamp': pd.Timestamp.now(),
                'source': f'CSV row {i+1}'
            }
            
            st.session_state.results.append(result)
            progress_bar.progress((i + 1) / len(df))
    
    st.success(f"CSV processing complete! Analyzed {len(df)} rows.")

def display_results_table():
    """Display all analysis results in a table format."""
    df_results = pd.DataFrame(st.session_state.results)
    
    display_df = df_results.copy()
    display_df['text_preview'] = display_df['text'].str[:100] + '...'
    display_df['confidence'] = display_df.apply(lambda row: row['scores'].get(row['sentiment'], 0.0), axis=1)
    display_df['keywords_str'] = display_df['keywords'].apply(lambda x: ', '.join(x) if x else 'None')
    
    display_columns = ['text_preview', 'sentiment', 'confidence', 'keywords_str', 'timestamp']
    if 'source' in display_df.columns:
        display_columns.insert(-1, 'source')
    
    st.dataframe(
        display_df[display_columns],
        column_config={
            'text_preview': 'Text Preview',
            'sentiment': 'Sentiment',
            'confidence': st.column_config.ProgressColumn(
                'Confidence',
                min_value=0,
                max_value=1,
                format="%.2f"
            ),
            'keywords_str': 'Keywords',
            'timestamp': 'Timestamp',
            'source': 'Source'
        },
        use_container_width=True
    )

def compare_results():
    """Allow users to select and compare multiple analysis results side-by-side."""
    if not st.session_state.results:
        st.info("Analyze some texts first to enable comparative analysis.")
        return

    df_results = pd.DataFrame(st.session_state.results)
    df_results['display_name'] = df_results.apply(lambda row: f"{row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} - {row['text'][:50]}...", axis=1)

    selected_indices = st.multiselect(
        "Select results to compare:",
        options=range(len(df_results)),
        format_func=lambda x: df_results.loc[x, 'display_name']
    )

    if len(selected_indices) > 1:
        st.subheader("Side-by-Side Comparison")
        compare_cols = st.columns(len(selected_indices))

        for i, idx in enumerate(selected_indices):
            with compare_cols[i]:
                result = st.session_state.results[idx]
                st.markdown(f"**Text:** {result['text'][:150]}...")
                
                sentiment = result['sentiment']
                confidence = result['scores'].get(sentiment, 0.0)
                color = {
                    'POSITIVE': 'green',
                    'NEGATIVE': 'red',
                    'NEUTRAL': 'blue'
                }.get(sentiment, 'gray')
                st.markdown(f"**Sentiment:** :{color}[{sentiment}]")
                st.markdown(f"**Confidence:** {confidence:.2%}")

                st.markdown("**Keywords:**")
                if result['keywords']:
                    keywords_html = " ".join([
                        f'<span style="background-color: #e1f5fe; padding: 2px 8px; margin: 2px; border-radius: 12px; font-size: 12px;">{kw}</span>'
                        for kw in result['keywords']
                    ])
                    st.markdown(keywords_html, unsafe_allow_html=True)
                else:
                    st.info("No keywords extracted")
    elif len(selected_indices) == 1:
        st.info("Select at least two results for comparative analysis.")

def download_results(format_type):
    """Generate download for results in CSV, JSON, or PDF format."""
    df_results = pd.DataFrame(st.session_state.results)
    
    if format_type == "CSV":
        csv_data = []
        for _, row in df_results.iterrows():
            csv_row = {
                'text': row['text'],
                'sentiment': row['sentiment'],
                'confidence': row['scores'].get(row['sentiment'], 0.0),
                'keywords': ', '.join(row['keywords']) if row['keywords'] else '',
                'timestamp': row['timestamp']
            }
            if 'source' in row:
                csv_row['source'] = row['source']
            csv_data.append(csv_row)
        
        csv_df = pd.DataFrame(csv_data)
        csv_buffer = io.StringIO()
        csv_df.to_csv(csv_buffer, index=False)
        
        st.download_button(
            label="Download CSV",
            data=csv_buffer.getvalue(),
            file_name="sentiment_analysis_results.csv",
            mime="text/csv"
        )
    
    elif format_type == "JSON":
        json_data = []
        for result in st.session_state.results:
            json_result = result.copy()
            json_result['timestamp'] = str(json_result['timestamp'])
            json_data.append(json_result)
        
        json_str = json.dumps(json_data, indent=2)
        
        st.download_button(
            label="Download JSON",
            data=json_str,
            file_name="sentiment_analysis_results.json",
            mime="application/json"
        )
    
    elif format_type == "PDF":
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        pdf.cell(200, 10, txt="Sentiment Analysis Results", ln=True, align="C")
        pdf.ln(10)
        
        for i, result in enumerate(st.session_state.results):
            pdf.set_font("Arial", 'B', size=10)
            pdf.cell(200, 10, txt=f"Analysis Result {i+1}:", ln=True)
            pdf.set_font("Arial", size=10)
            pdf.multi_cell(0, 5, txt=f"Text: {result['text']}")
            pdf.cell(0, 5, txt=f"Sentiment: {result['sentiment']}", ln=True)
            pdf.cell(0, 5, txt=f"Confidence: {result['scores'].get(result['sentiment'], 0.0):.2%}", ln=True)
            pdf.multi_cell(0, 5, txt=f"Keywords: {', '.join(result['keywords']) if result['keywords'] else 'None'}")
            pdf.cell(0, 5, txt=f"Timestamp: {result['timestamp']}", ln=True)
            if 'source' in result:
                pdf.cell(0, 5, txt=f"Source: {result['source']}", ln=True)
            pdf.ln(5)
        
        pdf_output = pdf.output(dest='S').encode('latin-1')
        
        st.download_button(
            label="Download PDF",
            data=pdf_output,
            file_name="sentiment_analysis_results.pdf",
            mime="application/pdf"
        )

if __name__ == "__main__":
    main()

