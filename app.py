import os
import gradio as gr
import pathlib
import torch
import faiss
from sentence_transformers import SentenceTransformer

from brain import encode_image, analyze_image_with_query
from patientvoice import record_audio, transcribe_with_groq
from doctorvoice import text_to_speech_with_gtts, text_to_speech_with_elevenlabs
from dotenv import load_dotenv
load_dotenv()
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Initialize embeddings model
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str, device: str = None):
        self.model = SentenceTransformer(model_name, device=device)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        embedding = self.model.encode(text, convert_to_tensor=False)
        return embedding.tolist()

embeddings = SentenceTransformerEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device=device
)

# Define vectorstore paths consistently
VECTORSTORE_DIR = "vectorstore/db_faiss"
vectorstore_path = pathlib.Path(VECTORSTORE_DIR)

# Create vectorstore directory if it doesn't exist
vectorstore_path.mkdir(parents=True, exist_ok=True)

if not (vectorstore_path / "index.faiss").exists():
    print("Creating new vectorstore...")
    # Load and split the PDF
    loader = PyPDFLoader("medical.pdf")
    documents = loader.load()
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=100,
        length_function=len,
    )
    texts = text_splitter.split_documents(documents)
    
    # Create and save the vectorstore
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    # If CUDA is available, convert index to GPU
    if device == "cuda":
        res = faiss.StandardGpuResources()  # Initialize GPU resources
        index = vectorstore.index
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)  # Move to GPU
        vectorstore.index = gpu_index
    
    # Save the vectorstore
    vectorstore.save_local(VECTORSTORE_DIR)
    print("Vectorstore created and saved successfully.")
else:
    print("Loading existing vectorstore...")
    # Load existing vectorstore
    vectorstore = FAISS.load_local(
        folder_path=VECTORSTORE_DIR,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    
    # If CUDA is available, convert loaded index to GPU
    if device == "cuda":
        res = faiss.StandardGpuResources()  # Initialize GPU resources
        index = vectorstore.index
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)  # Move to GPU
        vectorstore.index = gpu_index
    print("Vectorstore loaded successfully.")

def get_relevant_context(query):
    try:
        # Search the vector store for relevant documents
        docs = vectorstore.similarity_search(query, k=2)
        
        # Extract and combine the content from retrieved documents
        context = "\n".join([doc.page_content for doc in docs])
        
        return context
    except Exception as e:
        print(f"Error in similarity search: {e}")
        return "Could not retrieve relevant context."

# Update system prompt to include retrieved context
def get_enhanced_prompt(query, context):
    enhanced_prompt = f"""You have to act as a professional doctor, i know you are not but this is for learning purpose.
    Use the following medical context to inform your response: {context}
    What's in this image?. Do you find anything wrong with it medically? 
    If you make a differential, suggest some remedies for them. Donot add any numbers or special characters in 
    your response. Your response should be in one long paragraph. Also always answer as if you are answering to a real person.
    Donot say 'In the image I see' but say 'With what I see, I think you have ....'
    Dont respond as an AI model in markdown, your answer should mimic that of an actual doctor not an AI bot, 
    Keep your answer concise (max 2 sentences). No preamble, start your answer right away please.
    Question from patient: {query}"""
    return enhanced_prompt

def process_inputs(audio_filepath, image_filepath):
    speech_to_text_output = transcribe_with_groq(GROQ_API_KEY=os.environ.get("GROQ_API_KEY"), 
                                                 audio_filepath=audio_filepath,
                                                 stt_model="whisper-large-v3")

    # Get relevant context from the vector store
    context = get_relevant_context(speech_to_text_output)
    
    # Handle the image input
    if image_filepath:
        enhanced_prompt = get_enhanced_prompt(speech_to_text_output, context)
        doctor_response = analyze_image_with_query(query=enhanced_prompt, encoded_image=encode_image(image_filepath), model="llama-3.2-11b-vision-preview")
    else:
        doctor_response = "No image provided for me to analyze"

    # Generate audio response and return the filepath
    output_filepath = "output_audio.mp3"
    voice_of_doctor = text_to_speech_with_elevenlabs(input_text=doctor_response, output_filepath=output_filepath)

    return speech_to_text_output, doctor_response, output_filepath


# Create the interface
iface = gr.Interface(
    fn=process_inputs,
    inputs=[
        gr.Audio(sources=["microphone"], type="filepath"),
        gr.Image(type="filepath")
    ],
    outputs=[
        gr.Textbox(label="Speech to Text"),
        gr.Textbox(label="Doctor's Response"),
        gr.Audio(label="Doctor's Voice")
    ],
    title="AI Doctor with Vision and Voice"
)

iface.launch(debug=True)