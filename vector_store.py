import os
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
from llama_index.legacy import SimpleDirectoryReader, StorageContext, ServiceContext
from llama_index.legacy.indices.vector_store import VectorStoreIndex
from llama_iris import IRISVectorStore
from dotenv import load_dotenv
import json
import logging
import glob

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# IRIS connection parameters
USERNAME = 'demo'
PASSWORD = 'demo'
HOSTNAME = os.getenv('IRIS_HOSTNAME', 'localhost')
PORT = '1972'
NAMESPACE = 'USER'
CONNECTION_STRING = f"iris://{USERNAME}:{PASSWORD}@{HOSTNAME}:{PORT}/{NAMESPACE}"

class VitalSignsStore:
    def __init__(self):
        self.vector_store = IRISVectorStore.from_params(
            connection_string=CONNECTION_STRING,
            table_name="vital_signs_results",
            embed_dim=1536,  # openai embedding dimension
        )
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.index = None
        self._initialize_index()

    def _initialize_index(self):
        """Initialize or load the vector index."""
        try:
            self.index = VectorStoreIndex.from_vector_store(vector_store=self.vector_store)
        except Exception as e:
            logger.info("Creating new vector index...")
            self.index = VectorStoreIndex([], storage_context=self.storage_context)

    def save_result(self, user_id: int, heart_rate: float, spo2: float, metrics: Dict[str, Any]) -> str:
        """
        Save vital signs result to vector store.
        
        Args:
            user_id: Telegram user ID
            heart_rate: Heart rate in BPM
            spo2: Blood oxygen level in percentage
            metrics: Dictionary containing additional metrics
            
        Returns:
            str: Document ID of the saved result
        """
        # Create result document
        timestamp = datetime.now().isoformat()
        result_data = {
            "user_id": user_id,
            "timestamp": timestamp,
            "heart_rate": heart_rate,
            "spo2": spo2,
            "metrics": metrics,
            "type": "measurement",  # Add type to distinguish from analysis
            "analysis": None  # Placeholder for future analysis
        }
        
        # Save to JSON file first
        results_dir = "vital_signs_results"
        os.makedirs(results_dir, exist_ok=True)
        file_path = os.path.join(results_dir, f"result_{user_id}_{timestamp}.json")
        with open(file_path, 'w') as f:
            json.dump(result_data, f)
        
        # Load document into vector store
        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
        doc_id = documents[0].doc_id
        
        # Add to vector index
        self.index.insert(documents[0], storage_context=self.storage_context)
        
        return doc_id, file_path  # Return both ID and path

    def save_analysis(self, user_id: int, result_file_path: str, analysis: str) -> None:
        """
        Save ChatGPT analysis for a result.
        
        Args:
            user_id: Telegram user ID
            result_file_path: Path to the original result file
            analysis: ChatGPT's analysis text
        """
        try:
            # Read the original result
            with open(result_file_path, 'r') as f:
                result_data = json.load(f)
            
            # Add analysis to the result
            result_data['analysis'] = {
                'timestamp': datetime.now().isoformat(),
                'content': analysis,
                'type': 'analysis'
            }
            
            # Save updated result
            with open(result_file_path, 'w') as f:
                json.dump(result_data, f)
            
            # Update vector store
            documents = SimpleDirectoryReader(input_files=[result_file_path]).load_data()
            self.index.insert(documents[0], storage_context=self.storage_context)
            
        except Exception as e:
            logger.error(f"Error saving analysis: {str(e)}")
            raise

    def get_latest_result(self, user_id: int) -> Tuple[Dict[str, Any], str]:
        """
        Retrieve the latest vital signs result for a user.
        
        Args:
            user_id: Telegram user ID
            
        Returns:
            Tuple[Dict[str, Any], str]: (Latest vital signs result, file path)
        """
        try:
            # Get all result files for this user
            results_dir = "vital_signs_results"
            user_files = glob.glob(os.path.join(results_dir, f"result_{user_id}_*.json"))
            
            if not user_files:
                logger.warning(f"No results found for user {user_id}")
                return None, None
            
            # Get the most recent file
            latest_file = max(user_files, key=os.path.getctime)
            
            # Read and parse the JSON file
            with open(latest_file, 'r') as f:
                result_data = json.load(f)
            
            return result_data, latest_file
            
        except Exception as e:
            logger.error(f"Error retrieving latest result: {str(e)}")
            return None, None

    def get_user_history(self, user_id: int, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve the most recent vital signs results for a user.
        
        Args:
            user_id: Telegram user ID
            limit: Maximum number of results to return (default: 5)
            
        Returns:
            List[Dict[str, Any]]: List of vital signs results with their analyses
        """
        try:
            # Get all result files for this user
            results_dir = "vital_signs_results"
            user_files = glob.glob(os.path.join(results_dir, f"result_{user_id}_*.json"))
            
            if not user_files:
                logger.warning(f"No results found for user {user_id}")
                return []
            
            # Sort files by creation time (newest first)
            user_files.sort(key=os.path.getctime, reverse=True)
            
            # Load the most recent results
            results = []
            for file_path in user_files[:limit]:
                try:
                    with open(file_path, 'r') as f:
                        result_data = json.load(f)
                        results.append(result_data)
                except Exception as e:
                    logger.error(f"Error reading file {file_path}: {str(e)}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving user history: {str(e)}")
            return []

    def get_summary_data(self, user_id: int) -> Dict[str, Any]:
        """
        Get summary statistics from user's vital signs history.
        
        Args:
            user_id: Telegram user ID
            
        Returns:
            Dict[str, Any]: Summary statistics
        """
        try:
            # Get recent history
            history = self.get_user_history(user_id)
            if not history:
                return None

            # Calculate summary statistics
            heart_rates = [r['heart_rate'] for r in history]
            spo2_values = [r['spo2'] for r in history]
            quality_scores = [r['metrics']['peak_quality'] for r in history]
            
            summary = {
                'measurements': {
                    'count': len(history),
                    'period': {
                        'start': history[-1]['timestamp'],
                        'end': history[0]['timestamp']
                    },
                    'heart_rate': {
                        'latest': heart_rates[0],
                        'min': min(heart_rates),
                        'max': max(heart_rates),
                        'avg': sum(heart_rates) / len(heart_rates)
                    },
                    'spo2': {
                        'latest': spo2_values[0],
                        'min': min(spo2_values),
                        'max': max(spo2_values),
                        'avg': sum(spo2_values) / len(spo2_values)
                    },
                    'quality': {
                        'latest': quality_scores[0],
                        'avg': sum(quality_scores) / len(quality_scores)
                    }
                },
                'analyses': [
                    {
                        'timestamp': r['timestamp'],
                        'analysis': r['analysis']['content'] if r.get('analysis') else None
                    }
                    for r in history
                    if r.get('analysis')
                ]
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary data: {str(e)}")
            return None