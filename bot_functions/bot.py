import os
import logging
import asyncio
import shutil
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from process_video import VideoProcessor
from query import analyze_vital_signs, save_vital_signs, summarize_vital_signs
from typing import Tuple, Dict
import time

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')

# Initialize video processor with default model
video_processor = VideoProcessor(model_type='bp4d_physnet')

# Store last video path for each user
user_last_video: Dict[int, str] = {}

def get_help_message() -> str:
    """Get the help message for the bot."""
    return (
        "📋 How to use this bot:\n\n"
        "1. Send a short video (max 10 seconds) of your face\n"
        "2. Make sure your face is clearly visible and well-lit\n"
        "3. Stay still while recording the video\n"
        "4. Wait for the analysis results\n"
        "5. Use /reanalyze to process the same video again\n"
        "6. Use /askresults to get medical insights about the results\n"
        "7. Use /summarize to get a trend analysis of recent measurements\n\n"
        "Commands:\n"
        "/start - Start the bot\n"
        "/help - Show this help message\n"
        "/reanalyze - Process your last video again\n"
        "/askresults - Get medical insights about your results\n"
        "/summarize - Get a summary of your recent measurements"
    )

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a welcome message when the command /start is issued."""
    welcome_message = (
        "👋 Welcome to the Vital Signs Monitor Bot!\n\n"
        "Send me a short video of your face, and I'll analyze it to estimate your:\n"
        "❤️ Heart Rate\n"
        "🫁 Blood Oxygen Level (SpO2)\n\n"
        "Make sure your face is well-lit and clearly visible in the video.\n"
        "You can also use /reanalyze to process your last video again!\n\n"
        "Type /help for detailed instructions."
    )
    await update.message.reply_text(welcome_message)

async def process_video_file(file_path: str) -> Tuple[float, float, dict]:
    """
    Process a video file to extract heart rate and SpO2.
    
    Args:
        file_path: Path to the video file
        
    Returns:
        Tuple[float, float, dict]: (heart_rate, spo2, metrics)
    """
    try:
        # Create videos directory if it doesn't exist
        videos_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'videos')
        os.makedirs(videos_dir, exist_ok=True)
        
        # Move the video file to videos directory
        video_filename = os.path.basename(file_path)
        new_video_path = os.path.join(videos_dir, video_filename)
        shutil.move(file_path, new_video_path)
        
        # Process video and get results
        heart_rate, bvp_analysis = video_processor.process_video_file(new_video_path, extract_frames=True)
        
        # Calculate SpO2 from BVP analysis if available
        spo2 = 0.0
        if bvp_analysis and bvp_analysis.get('valid', False):
            # Use amplitude as a rough proxy for SpO2
            # This is a simplified estimation - you may want to implement a more sophisticated calculation
            amplitude = bvp_analysis.get('amplitude', 0.0)
            if amplitude > 0:
                spo2 = min(99.0, 90.0 + (amplitude * 5))  # Simple linear mapping
        
        # Extract additional metrics
        metrics = {
            'amplitude': bvp_analysis.get('amplitude', 0.0),
            'peak_quality': bvp_analysis.get('peak_quality', 0.0),
            'mean_interval': bvp_analysis.get('mean_interval', 0.0),
            'variability': bvp_analysis.get('interval_cv', 0.0),
            'status': bvp_analysis.get('message', 'Unknown'),
        }
        
        return heart_rate, spo2, metrics
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise ValueError("Failed to process video")
    finally:
        # Clean up extracted frames from data directory
        if os.path.exists(video_processor.data_dir):
            try:
                shutil.rmtree(video_processor.data_dir)
            except Exception as e:
                logger.error(f"Error cleaning up data directory: {str(e)}")

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming videos."""
    processing_message = None
    file_path = None
    
    try:
        # Send initial processing message
        processing_message = await update.message.reply_text(
            "🔄 Processing your video... Please wait."
        )
        
        # Get the file
        if update.message.video:
            # Check video duration (max 10 seconds)
            if update.message.video.duration > 10:
                await processing_message.edit_text(
                    "❌ Video is too long. Please send a video shorter than 10 seconds.\n\n"
                    "For best results:\n"
                    "- Keep videos between 5-10 seconds\n"
                    "- Ensure good lighting\n"
                    "- Keep your face clearly visible\n\n"
                    "Need help? Type /help for instructions."
                )
                return
            
            file = await update.message.video.get_file()
        else:
            await processing_message.edit_text(
                "❌ Please send a video of your face.\n\n"
                "For best results:\n"
                "- Record 5-10 seconds of video\n"
                "- Ensure good lighting\n"
                "- Keep your face clearly visible\n\n"
                "Need help? Type /help for instructions."
            )
            return

        # Download the file with a unique name
        timestamp = int(time.time())
        file_extension = '.mp4'
        file_path = f"/tmp/video_{timestamp}{file_extension}"
        await file.download_to_drive(custom_path=file_path)
        
        await processing_message.edit_text("🔄 Analyzing video frames... This may take a moment.")
        heart_rate, spo2, metrics = await process_video_file(file_path)

        # Store the video path for this user
        user_id = update.effective_user.id
        if os.path.exists(os.path.join('videos', os.path.basename(file_path))):
            user_last_video[user_id] = os.path.join('videos', os.path.basename(file_path))

        if heart_rate == 0.0 or metrics['peak_quality'] < 0.5:
            await processing_message.edit_text(
                "❌ Could not detect clear vital signs.\n\n"
                "Please try again with:\n"
                "- Better lighting\n"
                "- Clearer face visibility\n"
                "- More stable camera position\n"
                "- Record for at least 5 seconds\n\n"
                "You can also try /reanalyze to process the same video again.\n"
                "Need help? Type /help for instructions."
            )
            return

        # Save results to vector store
        save_vital_signs(user_id, heart_rate, spo2, metrics)

        # Format and send detailed results
        result_message = (
            "✅ Analysis Complete!\n\n"
            f"❤️ Heart Rate: {heart_rate:.1f} BPM\n"
            f"🫁 SpO2: {spo2:.1f}%\n\n"
            "📊 Detailed Metrics:\n"
            f"• Signal Quality: {metrics['peak_quality']:.2f}/1.00\n"
            f"• Pulse Amplitude: {metrics['amplitude']:.3f}\n"
            f"• Mean Interval: {metrics['mean_interval']:.1f}ms\n"
            f"• Heart Rate Variability: {metrics['variability']*100:.1f}%\n"
            f"• Status: {metrics['status']}\n\n"
            "Note: These measurements are estimates and should not be used for medical purposes.\n\n"
            "Options:\n"
            "• Send another video for a new measurement\n"
            "• Use /reanalyze to process this video again\n"
            "• Use /askresults to get medical insights about these results"
        )
        await processing_message.edit_text(result_message)

    except ValueError as e:
        if processing_message:
            await processing_message.edit_text(
                f"❌ {str(e)}\n\n"
                "Please try again with:\n"
                "- Better lighting\n"
                "- Clearer face visibility\n"
                "- More stable camera position\n"
                "- Record for at least 5 seconds\n\n"
                "You can also try /reanalyze to process the same video again.\n"
                "Need help? Type /help for instructions."
            )
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        if processing_message:
            await processing_message.edit_text(
                "❌ An unexpected error occurred while processing your file.\n\n"
                "Please try again with clearer lighting or a more stable position.\n\n"
                "Need help? Type /help for instructions."
            )
    finally:
        # Clean up downloaded file
        if file_path and os.path.exists(file_path):
            os.remove(file_path)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a help message when the command /help is issued."""
    await update.message.reply_text(get_help_message())

async def reanalyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Reanalyze the last video sent by the user."""
    user_id = update.effective_user.id
    
    if user_id not in user_last_video:
        await update.message.reply_text(
            "❌ No previous video found.\n\n"
            "Please send a video first, then use /reanalyze to process it again!"
        )
        return
        
    video_path = user_last_video[user_id]
    if not os.path.exists(video_path):
        await update.message.reply_text(
            "❌ Previous video file not found.\n\n"
            "Please send a new video to analyze!"
        )
        user_last_video.pop(user_id)
        return
        
    processing_message = await update.message.reply_text(
        "🔄 Reanalyzing your previous video... This may take a moment."
    )
    
    try:
        heart_rate, spo2, metrics = await process_video_file(video_path)
        
        if heart_rate == 0.0 or metrics['peak_quality'] < 0.5:
            await processing_message.edit_text(
                "❌ Could not detect clear vital signs in the reanalysis.\n\n"
                "Please try recording a new video with:\n"
                "- Better lighting\n"
                "- Clearer face visibility\n"
                "- More stable camera position\n"
                "- Record for at least 5 seconds\n\n"
                "Need help? Type /help for instructions."
            )
            return
            
        result_message = (
            "✅ Reanalysis Complete!\n\n"
            f"❤️ Heart Rate: {heart_rate:.1f} BPM\n"
            f"🫁 SpO2: {spo2:.1f}%\n\n"
            "📊 Detailed Metrics:\n"
            f"• Signal Quality: {metrics['peak_quality']:.2f}/1.00\n"
            f"• Pulse Amplitude: {metrics['amplitude']:.3f}\n"
            f"• Mean Interval: {metrics['mean_interval']:.1f}ms\n"
            f"• Heart Rate Variability: {metrics['variability']*100:.1f}%\n"
            f"• Status: {metrics['status']}\n\n"
            "Note: These measurements are estimates and should not be used for medical purposes.\n\n"
            "Options:\n"
            "• Send a new video for a new measurement\n"
            "• Use /reanalyze to process this video again"
        )
        await processing_message.edit_text(result_message)
        
    except Exception as e:
        logger.error(f"Error reanalyzing video: {str(e)}")
        await processing_message.edit_text(
            "❌ An error occurred while reanalyzing the video.\n\n"
            "Please try sending a new video instead.\n"
            "Need help? Type /help for instructions."
        )

async def askresults_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Analyze the last results using ChatGPT and provide medical insights."""
    user_id = update.effective_user.id
    
    processing_message = await update.message.reply_text(
        "🤖 Analyzing your results with medical AI... This may take a moment."
    )
    
    try:
        # Get medical insights from ChatGPT using vector store
        analysis = analyze_vital_signs(user_id)
        
        # Send the analysis
        await processing_message.edit_text(
            f"📊 Medical Analysis:\n\n{analysis}\n\n"
            "Note: This analysis is for informational purposes only and should not replace professional medical advice.\n"
            "Always consult healthcare professionals for medical decisions."
        )
        
    except Exception as e:
        logger.error(f"Error analyzing results: {str(e)}")
        await processing_message.edit_text(
            "❌ An error occurred while analyzing the results.\n\n"
            "Please try again later or consult a healthcare professional.\n"
            "Need help? Type /help for instructions."
        )

async def summarize_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generate a summary of recent vital signs measurements."""
    user_id = update.effective_user.id
    
    processing_message = await update.message.reply_text(
        "🔄 Analyzing your measurement history... This may take a moment."
    )
    
    try:
        # Get summary analysis from ChatGPT
        summary = summarize_vital_signs(user_id)
        
        # Send the summary
        await processing_message.edit_text(
            f"📊 Vital Signs History Summary:\n\n{summary}\n\n"
            "Note: This analysis is for informational purposes only and should not replace professional medical advice.\n"
            "Always consult healthcare professionals for medical decisions."
        )
        
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        await processing_message.edit_text(
            "❌ An error occurred while generating your summary.\n\n"
            "Please try again later or consult a healthcare professional.\n"
            "Need help? Type /help for instructions."
        )

async def handle_invalid_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle invalid inputs (text, stickers, etc.)"""
    await update.message.reply_text(
        "❌ I can only process videos of faces.\n\n"
        "To use this bot:\n"
        "1. Send a video (5-10 seconds) of your face\n"
        "2. Wait for the analysis\n"
        "3. Use /reanalyze to process the same video again\n\n"
        "Need help? Type /help for instructions."
    )

def main():
    """Start the bot."""
    if not TOKEN:
        logger.error("No token found! Make sure to set TELEGRAM_BOT_TOKEN in .env file")
        return
    
    # Create application and add handlers
    app = Application.builder().token(TOKEN).build()
    
    # Add command handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("reanalyze", reanalyze_command))
    app.add_handler(CommandHandler("askresults", askresults_command))
    app.add_handler(CommandHandler("summarize", summarize_command))
    
    # Add message handlers
    app.add_handler(MessageHandler(filters.VIDEO, handle_image))
    
    # Add handler for invalid inputs (must be last)
    app.add_handler(MessageHandler(filters.ALL, handle_invalid_input))
    
    # Start the bot
    logger.info("Bot started!")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
