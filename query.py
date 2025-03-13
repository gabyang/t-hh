import os
from openai import OpenAI
from dotenv import load_dotenv
import logging
from vector_store import VitalSignsStore
from typing import Tuple
from datetime import datetime

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize OpenAI client and vector store
client = OpenAI(api_key=OPENAI_API_KEY)
vital_signs_store = VitalSignsStore()

def save_vital_signs(user_id: int, heart_rate: float, spo2: float, metrics: dict) -> Tuple[str, str]:
    """
    Save vital signs result to vector store.
    
    Args:
        user_id: Telegram user ID
        heart_rate: Heart rate in BPM
        spo2: Blood oxygen level in percentage
        metrics: Dictionary containing additional metrics
        
    Returns:
        Tuple[str, str]: (Document ID of the saved result, file path)
    """
    return vital_signs_store.save_result(user_id, heart_rate, spo2, metrics)

def analyze_vital_signs(user_id: int) -> str:
    """
    Analyze vital signs using ChatGPT and provide medical insights.
    
    Args:
        user_id: Telegram user ID
        
    Returns:
        str: Analysis and recommendations from ChatGPT
    """
    try:
        # Get latest result from vector store
        result, result_file_path = vital_signs_store.get_latest_result(user_id)
        if not result:
            return "No previous results found. Please send a video first!"

        # Extract vital signs data
        heart_rate = result['heart_rate']
        spo2 = result['spo2']
        metrics = result['metrics']
        timestamp = result['timestamp']

        # Construct the prompt for ChatGPT
        prompt = f"""As a medical AI assistant, please analyze these vital signs and provide insights for a caregiver:

Heart Rate: {heart_rate:.1f} BPM
Blood Oxygen (SpO2): {spo2:.1f}%
Signal Quality: {metrics['peak_quality']:.2f}/1.00
Pulse Amplitude: {metrics['amplitude']:.3f}
Mean Interval: {metrics['mean_interval']:.1f}ms
Heart Rate Variability: {metrics['variability']*100:.1f}%
Status: {metrics['status']}
Measurement Time: {timestamp}

Please provide:
1. Are these readings normal for an elderly person?
2. Should the caregiver be concerned?
3. What actions should the caregiver take?
4. When should they seek medical attention?

Please keep the response clear and simple for non-medical caregivers."""

        # Get response from ChatGPT
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful medical AI assistant providing guidance to non-medical caregivers about vital signs. Keep responses clear, simple, and focused on practical advice."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )

        analysis = response.choices[0].message.content

        # Save the analysis with the result
        vital_signs_store.save_analysis(user_id, result_file_path, analysis)

        return analysis

    except Exception as e:
        logger.error(f"Error analyzing vital signs with ChatGPT: {str(e)}")
        return "I apologize, but I'm unable to analyze the vital signs at this moment. Please try again later or consult a healthcare professional for immediate concerns."

def summarize_vital_signs(user_id: int) -> str:
    """
    Generate a summary of the user's vital signs history.
    
    Args:
        user_id: Telegram user ID
        
    Returns:
        str: Summary analysis from ChatGPT
    """
    try:
        # Get summary data
        summary = vital_signs_store.get_summary_data(user_id)
        if not summary:
            return "No history found. Please record some measurements first!"

        # Format the data for ChatGPT
        measurements = summary['measurements']
        period_start = datetime.fromisoformat(measurements['period']['start'])
        period_end = datetime.fromisoformat(measurements['period']['end'])
        
        prompt = f"""As a medical AI assistant, please analyze this vital signs history and provide a summary for a caregiver:

Period: From {period_start.strftime('%Y-%m-%d %H:%M')} to {period_end.strftime('%Y-%m-%d %H:%M')}
Number of Measurements: {measurements['count']}

Heart Rate Statistics:
- Latest: {measurements['heart_rate']['latest']:.1f} BPM
- Range: {measurements['heart_rate']['min']:.1f} - {measurements['heart_rate']['max']:.1f} BPM
- Average: {measurements['heart_rate']['avg']:.1f} BPM

Blood Oxygen (SpO2) Statistics:
- Latest: {measurements['spo2']['latest']:.1f}%
- Range: {measurements['spo2']['min']:.1f}% - {measurements['spo2']['max']:.1f}%
- Average: {measurements['spo2']['avg']:.1f}%

Signal Quality:
- Latest: {measurements['quality']['latest']:.2f}/1.00
- Average: {measurements['quality']['avg']:.2f}/1.00

Previous Analyses:
{chr(10).join(f"- {a['timestamp']}: {a['analysis'][:100]}..." for a in summary['analyses'] if a['analysis'])}

Please provide:
1. Overall trend analysis of vital signs
2. Notable changes or patterns
3. Recommendations for the caregiver
4. When to seek medical attention based on these trends

Keep the response clear and simple for non-medical caregivers."""

        # Get response from ChatGPT
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful medical AI assistant providing guidance to non-medical caregivers about vital signs trends. Keep responses clear, simple, and focused on practical advice. Highlight important changes and provide actionable recommendations."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=800
        )

        return response.choices[0].message.content

    except Exception as e:
        logger.error(f"Error generating vital signs summary: {str(e)}")
        return "I apologize, but I'm unable to generate a summary at this moment. Please try again later or consult a healthcare professional for immediate concerns."