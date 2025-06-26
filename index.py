import re
import pandas as pd
from collections import defaultdict

# from sklearn.ensemble import RandomForestClassifier
# from fuzzywuzzy import fuzz
from langchain_cohere import CohereEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

# from crewai import Agent, Task, Crew, LLM
from langchain_groq import ChatGroq
import uuid
import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed


load_dotenv()

config = {
    "mongo_url": os.getenv("MONGO_URL"),
    "pinecone_api_key": os.getenv("PINECONE_API_KEY"),
    "cohere_api_key": os.getenv("COHERE_API_KEY"),
    "groq_api_key": os.getenv("GROQ_API_KEY"),
}

os.environ["PINECONE_API_KEY"] = config["pinecone_api_key"]
os.environ["GROQ_API_KEY"] = config["groq_api_key"]
os.environ["COHERE_API_KEY"] = config["cohere_api_key"]

# llm = ChatGroq(
#     api_key=config["groq_api_key"],
#     model="llama3-8b-8192",
# )

llm = ChatOpenAI(
    base_url="https://api.sambanova.ai/v1/",
    api_key=os.getenv("SAMBANOVA_API_KEY"),
    streaming=True,
    model="Meta-Llama-3.3-70B-Instruct",
)

INDIAN_ADDRESS_TERMS = {
    "gali",
    "marg",
    "road",
    "street",
    "nagar",
    "sector",
    "block",
    "phase",
    "enclave",
    "market",
    "bazaar",
    "puri",
    "pur",
    "ganj",
    "gunj",
    "mandi",
    "mohalla",
    "villa",
    "complex",
    "society",
    "plaza",
}

SPELLING_CORRECTION_THRESHOLD = 90


# class AddressAISystem:
#     def __init__(self, dataset_path):
#         self.dataset = pd.read_csv(dataset_path)
#         self.grouped_data = self.preprocess_dataset(self.dataset)
#         self.model = RandomForestClassifier()

#     def preprocess_dataset(self, dataset):
#         grouped_data = defaultdict(list)
#         for _, row in dataset.iterrows():
#             grouped_data[row["Place"].lower()].append(
#                 {
#                     "place": row["Place"],
#                     "district": row["District"],
#                     "state": row["State"],
#                     "pincode": row["Pincode"],
#                 }
#             )
#         return grouped_data

#     def extract_components(self, address):
#         address = re.sub(r"\b\d{6}\b", "", address)
#         components = re.findall(r"\b[\w\'-]+\b", address.lower())
#         return components

#     def find_best_match(self, components):
#         best_match = None
#         best_score = 0

#         for place, details in self.grouped_data.items():
#             score = max(
#                 fuzz.partial_ratio(place, " ".join(components))
#                 for i in range(len(components))
#             )
#             if score > best_score:
#                 best_score = score
#                 best_match = details[0]

#         return best_match

#     def construct_corrected_address(self, original_address, best_match):
#         match = re.match(
#             r"([\w\s-]+?)(\s*,?\s*\w+\s+\w+)$", original_address, re.IGNORECASE
#         )
#         if match:
#             prefix = match.group(1).strip()
#         else:
#             prefix = original_address.split(",")[0].strip()

#         corrected_address = f"{prefix}, {best_match['place']}, {best_match['district']}, {best_match['pincode']}"
#         return corrected_address

#     def process_address(self, address):
#         components = self.extract_components(address)
#         best_match = self.find_best_match(components)

#         if best_match:
#             corrected_address = self.construct_corrected_address(address, best_match)
#             return {
#                 "original_address": address,
#                 "corrected_address": corrected_address,
#                 "place": best_match["place"],
#                 "district": best_match["district"],
#                 "state": best_match["state"],
#                 "pincode": best_match["pincode"],
#             }
#         else:
#             return {
#                 "original_address": address,
#                 "corrected_address": None,
#                 "error": "No match found",
#             }

#     def process_addresses_bulk(self, addresses):
#       return [self.process_address(address) for address in addresses]


class AddressAISystem:

    def __init__(self):
        self.embeddings = CohereEmbeddings(
            cohere_api_key=config["cohere_api_key"], model="embed-english-v3.0"
        )

        # addresses = self.load_csv_to_pinecone("documents/Book2.csv")

        pc = Pinecone(api_key=config["pinecone_api_key"])

        address_index_name = "address-ai"
        feedback_index_name = "address-feedback"

        existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

        if address_index_name not in existing_indexes:
            pc.create_index(
                name=address_index_name,
                dimension=1024,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            while not pc.describe_index(address_index_name).status["ready"]:
                time.sleep(1)

        if feedback_index_name not in existing_indexes:
            pc.create_index(
                name=feedback_index_name,
                dimension=1024,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            while not pc.describe_index(feedback_index_name).status["ready"]:
                time.sleep(1)

        address_index = pc.Index(address_index_name)
        feedback_index = pc.Index(feedback_index_name)

        self.vectorstore = PineconeVectorStore(
            index=address_index, embedding=self.embeddings
        )

        self.feedback_store = PineconeVectorStore(
            index=feedback_index, embedding=self.embeddings
        )

        # ids = [str(uuid.uuid4()) for _ in range(len(addresses))]

        # self.vectorstore.add_documents(documents=addresses, ids=ids)

        self.extract_area_prompt = ChatPromptTemplate.from_template(
            template="""You are an Indian address correction expert. Your task is to extract the area/colony from the given address.

            Rules:
            1. Extract only the area/colony part of the address
            2. Maintain proper capitalization
            3. Do not include any other parts of the address

            Input Address: {input_address}

            Return only the area/colony in this format:
            [Area/Colony]""",
        )

        self.correct_area_prompt = ChatPromptTemplate.from_template(
            template="""You are an Indian address correction expert. Your task is to correct the area/colony name from the given address.

            Rules:
            1. Correct any typos or variations in the area/colony name
            2. Maintain proper capitalization
            3. Do not include any other parts of the address

            Input Area/Colony: {input_area_colony}

            Return only the corrected area/colony in this format:
            [Corrected Area/Colony]""",
        )

        self.correct_address_prompt = ChatPromptTemplate.from_template(
            template="""You are an Indian address correction expert. Your task is to correct and format the given address.

            Rules:
            1. Keep the response concise and only return the corrected address
            2. Maintain proper capitalization
            3. Use commas to separate address components
            4. Include the pincode at the end
            5. Do not add any explanatory text before or after the address
            6. If you're unsure about any part, maintain the original text
            7. If the pincode doesn't match the area, prefer the area's correct pincode
            8. Ensure that pincode must be correct according to the area/colony. Please check it on your own after correcting the whole address.

            Input Address: {input_address}

            Reference Addresses (use these for context):
            {similar_addresses}

            Previous Corrections (consider this feedback):
            {feedback_history}

            Return only the corrected address in this format:
            [Street Number and Name], [Area/Colony], [City/District], [State] [Pincode]""",
        )

        self.extract_area_chain = self.extract_area_prompt | llm | StrOutputParser()
        self.correct_area_chain = self.correct_area_prompt | llm | StrOutputParser()
        self.correct_address_chain = (
            self.correct_address_prompt | llm | StrOutputParser()
        )

    def load_csv_to_pinecone(self, csv_path):
        """Load CSV data into Pinecone vector store"""
        loader = CSVLoader(csv_path)
        doc = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        addresses = splitter.split_documents(doc)
        return addresses

    def get_similar_addresses(self, query_address, k=5):
        """Get similar addresses from Pinecone"""
        results = self.vectorstore.similarity_search(query_address, k=k)
        similar_addresses = [doc.page_content for doc in results]
        # print("Similar Addresses:", similar_addresses)  # Debugging line
        return similar_addresses

    def get_feedback_history(self, query_address, k=3):
        """Get relevant feedback history"""
        results = self.feedback_store.similarity_search(query_address, k=k)
        return [doc.page_content for doc in results]

    def store_feedback(self, original_address, corrected_address, user_feedback):
        """Store user feedback with better formatting"""
        try:
            feedback_entry = f"""
                Original: {original_address}
                Correction: {corrected_address}
                Feedback: {user_feedback}
                Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """

            self.feedback_store.add_texts([feedback_entry])
            return True
        except Exception as e:
            print(f"Error storing feedback: {str(e)}")
            return False

    def process_address(self, input_address):

        try:
            area_colony = self.extract_area_chain.invoke(
                {"input_address": input_address}
            )
            # print("Extracted Area/Colony:", area_colony)

            corrected_area_colony = self.correct_area_chain.invoke(
                {"input_area_colony": area_colony}
            )
            # print("Corrected Area/Colony:", corrected_area_colony)

            similar_addresses = self.get_similar_addresses(corrected_area_colony)
            formatted_similar = "\n".join([f"- {addr}" for addr in similar_addresses])

            feedback_history = self.get_feedback_history(corrected_area_colony)
            formatted_feedback = "\n".join([f"- {fb}" for fb in feedback_history])

            corrected_address = self.correct_address_chain.invoke(
                {
                    "input_address": input_address,
                    "similar_addresses": formatted_similar,
                    "feedback_history": formatted_feedback,
                }
            )

            if (
                len(corrected_address.strip()) < 10
                or len(corrected_address.strip()) > 200
            ):
                return {
                    "original_address": input_address,
                    "error": "Invalid response length. Please try again.",
                }

            return {
                "original_address": input_address,
                "corrected_address": corrected_address.strip(),
            }

        except Exception as e:
            print(f"Error in address correction: {str(e)}")
            return {"original_address": input_address, "error": str(e)}

    def process_addresses_bulk(self, addresses):
        """Process multiple addresses in bulk with concurrency"""
        results = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_address = {
                executor.submit(self.process_address, address): address
                for address in addresses
            }
            for future in as_completed(future_to_address):
                address = future_to_address[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({"original_address": address, "error": str(e)})
        return results
