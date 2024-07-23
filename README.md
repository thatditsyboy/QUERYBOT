# Streamlit Chatbot with File Support(QUERYBOT)

## Description

This project is a Streamlit-based chatbot application that supports interaction with various types of files, including MYSQL, MONGO DB, TXT, PDF, CSV and Excel files. It utilizes Google Generative AI for natural language processing and LangChain for handling and querying the text data.

## Features

- **File Uploads**: Upload and process MYSQL, MONGO DB, TXT, PDF, CSV and Excel  files.
- **Natural Language Querying**: Ask questions based on the content of uploaded files.
- **Data Visualization**: Visualize data from CSV and Excel files.
- **Interactive UI**: Streamlit-based interface for user interaction.

## Requirements

- Python 3.x
- Streamlit
- LangChain
- Google Generative AI SDK
- FAISS
- PyPDF2
- pandas
- openpyxl
- python-docx

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2. **Create a virtual environment and activate it:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Set up environment variables:**

    Create a `.env` file in the root directory with your Google API key:

    ```plaintext
    GOOGLE_API_KEY=your_google_api_key
    ```

## Usage

1. **Run the Streamlit application:**

    ```bash
    streamlit run appui.py
    ```

2. **Upload Files:**

    Use the sidebar to upload TXT, PDF, CSV, Excel, or Word files.

3. **Ask Questions:**

    Type your queries in the chat input field to get answers based on the uploaded files.

## Code Structure

- `appui.py`: Main Streamlit application code.
- `requirements.txt`: List of required Python packages.
- `.env`: Environment file for sensitive information (not included in the repository).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

If you'd like to contribute to this project, please fork the repository and submit a pull request with your changes.

## Contact

For any questions or suggestions, please open an issue on GitHub or contact [bt22cse033@iiitn.ac.in](bt22cse033@iiitn.ac.in).

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [LangChain](https://github.com/langchain/langchain)
- [Google Generative AI](https://developers.google.com/ai)
- [FAISS](https://github.com/facebookresearch/faiss)
- [PyPDF2](https://pythonhosted.org/PyPDF2/)
- [pandas](https://pandas.pydata.org/)
- [openpyxl](https://openpyxl.readthedocs.io/en/stable/)
- [python-docx](https://python-docx.readthedocs.io/en/latest/)
