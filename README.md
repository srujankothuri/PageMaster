# PageMaster

PageMaster is a powerful application designed to enhance your interaction with PDFs. It allows users to upload any PDF, generate a concise summary, and interact with a chatbot to ask questions related to the document. Additionally, the application provides suggested questions tailored to each PDF, helping users gain a deeper understanding of the content.

### Key Features
- **PDF Upload & Summarization**: Automatically generates a summary of the uploaded PDF.
- **Interactive Chatbot**: Engage in a conversation with the chatbot to ask questions related to the document.
- **Suggested Questions**: Pre-generated questions based on the PDF to guide exploration.
- **Summary Length Control**: A summary slider allows users to adjust the summary length in tokens.

### Technologies Used
- **LangChain**: Converts document content into embeddings and stores them in a vector database (FAISS).
- **FAISS (Facebook AI Similarity Search)**: Efficiently stores and retrieves document embeddings.
- **Streamlit**: Provides an intuitive and interactive user interface.

### Installation & Setup
Follow these steps to set up and run PageMaster:

1. Clone the repository:
   ```sh
   git clone https://github.com/srujankothuri/PageMaster
   ```
2. Navigate to the project directory:
   ```sh
   cd PageMaster
   ```
3. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Run the application:
   ```sh
   streamlit run app.py
   ```

### Usage
- Upload a PDF file.
- View the generated summary.
- Use the chatbot to ask questions about the document.
- Explore suggested questions for deeper insights.
- Adjust the summary length using the slider.

Enjoy using **PageMaster** to simplify document exploration and understanding! 🚀
