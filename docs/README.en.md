# FinSynth-Data-Processing

![FinSynth-Data-Processing](FinSynth-Data-Processing.png)

## Project Introduction

Used to create fine-tuning datasets to train our [Fintech-Dreamer/FinSynth: Financial Large Model Interaction Platform Based on Open WebUI Framework](https://github.com/Fintech-Dreamer/FinSynth).

## How to Run

Download the repository

```powershell
git clone https://github.com/Fintech-Dreamer/FinSynth-Data-Processing.git
cd FinSynth-Data-Processing
```

Run the project (Note: Modify the main and params files as per the instructions below)

```powershell
conda create -n FinSynth_data_processing python=3.11
conda activate FinSynth_data_processing
pip install -r requirements.txt -U
python main.py
```

Modify the files

1. Create your own params.py file, refer to params_example.py for details

2. Modify the main program (if \_\_name\_\_ == "\_\_main\_\_")

   - Initial run (Generate chatbot dataset)

       ```python
           file_converter = FileConverter(["demo.pdf"])
           file_converter.file_to_json_lists()
           file_converter.json_lists_to_QA_pairs("chatbot", time_sleep=0, lable=10, embed_model_name="BAAI/bge-m3")
       ```

   - Run multiple files (Generate chatbot dataset)

     ```python
         file_converter = FileConverter(["demo.pdf","demo.csv"])
         file_converter.file_to_json_lists()
         file_converter.json_lists_to_QA_pairs("chatbot", time_sleep=0, lable=10, embed_model_name="BAAI/bge-m3")
     ```

   - Generate fraud detection or contract compliance dataset

     ```python
         file_converter = FileConverter(["demo.pdf"])
         file_converter.file_to_json_lists()
         file_converter.json_lists_to_QA_pairs("fraud", time_sleep=0)
         # file_converter.json_lists_to_QA_pairs("compliance", time_sleep=0)
     ```

   - Save the generated structured json and final question-answer csv

     ```python
         file_converter = FileConverter(["demo.pdf","demo.csv"])
         file_converter.file_to_json_lists()
         file_converter.save_json_lists()
         file_converter.json_lists_to_QA_pairs("chatbot", time_sleep=0, lable=10, embed_model_name="BAAI/bge-m3")
         file_converter.save_QA_pairs()
     ```

   - Read the generated structured json directly to generate question-answer pairs

     ```python
         file_converter = FileConverter([])
         file_converter.read_json_lists("output.json_1.json")
         file_converter.json_lists_to_QA_pairs("chatbot", time_sleep=0, lable=10, embed_model_name="BAAI/bge-m3")
         file_converter.save_QA_pairs()
     ```

## Others

Running the file may require downloading models from huggingface, which might need a VPN.

File types that can be converted:

- Text files (.txt, .md, etc.)
- PDF documents
- Word files (.doc, .docx)
- PowerPoint presentations (.ppt, .pptx)
- Image files (.jpg, .png, etc.)
- HTML webpages
- XML files
- Audio files

### Fine-tuning Models

[Chatbot](https://huggingface.co/Fintech-Dreamer/FinSynth_model_chatbot)

[Fraud Detection](https://huggingface.co/Fintech-Dreamer/FinSynth_model_fraud)

[Compliance Monitoring](https://huggingface.co/Fintech-Dreamer/FinSynth_model_compliance)

### Fine-tuning Datasets

[Dataset](https://huggingface.co/datasets/Fintech-Dreamer/FinSynth_data)

## Technical Details

- Use [Unstructured](https://docs.unstructured.io/welcome) to first chunk unstructured documents and store them in **json lists**
- Use the large model to generate question-answer pairs from each chunk and finally store them in csv files
- **When generating chatbots, use RAG technology to enhance generation, the process is as follows**
  1. **First, read the already structured json lists, which are many chunked sections, and perform basic processing on them.**
  2. **Generate all questions using all chunks first.**
  3. **Iterate over each question, then select the corresponding main text and the previous and next lable (custom parameter) article chunks, use vector retrieval technology (RAG) to find the knowledge background corresponding to the question, and finally generate higher-quality answers.**