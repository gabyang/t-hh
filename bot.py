import os
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')

def get_help_message() -> str:
    """Get the help message for the bot."""
    return (
        "📋 How to use this bot:\n\n"
        "1. Send a photo or short video of your face\n"
        "2. Make sure your face is clearly visible and well-lit\n"
        "3. Stay still while taking the photo/video\n"
        "4. Wait for the analysis results\n\n"
        "Commands:\n"
        "/start - Start the bot\n"
        "/help - Show this help message"
    )

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a welcome message when the command /start is issued."""
    welcome_message = (
        "👋 Welcome to the Vital Signs Monitor Bot!\n\n"
        "Send me a photo or video of your face, and I'll analyze it to estimate your:\n"
        "❤️ Heart Rate\n"
        "🫁 Blood Oxygen Level (SpO2)\n\n"
        "Make sure your face is well-lit and clearly visible in the image/video.\n\n"
        "Type /help for detailed instructions."
    )
    await update.message.reply_text(welcome_message)

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming photos and videos."""
    try:
        # Send initial processing message
        processing_message = await update.message.reply_text(
            "🔄 Processing your image... Please wait."
        )
        
        # Get the file
        if update.message.photo:
            file = await update.message.photo[-1].get_file()
        elif update.message.video:
            # Check video duration (max 10 seconds)
            if update.message.video.duration > 10:
                await processing_message.edit_text(
                    "❌ Video is too long. Please send a video shorter than 10 seconds.\n\n"
                    "Need help? Type /help for instructions."
                )
                return
            file = await update.message.video.get_file()
        else:
            await processing_message.edit_text(
                "❌ Please send a photo or video of your face.\n\n"
                "Need help? Type /help for instructions."
            )
            return
        
        # Download and process the file
        try:
            file_path = await file.download_to_drive()
        except Exception as e:
            logger.error(f"Error downloading file: {str(e)}")
            await processing_message.edit_text(
                "❌ Failed to download your file. Please try again.\n\n"
                "Need help? Type /help for instructions."
            )
            return
            
        try:
            heart_rate, spo2 = await process_rppg(file_path)
        except ValueError as e:
            if "No face detected" in str(e):
                await processing_message.edit_text(
                    "❌ Could not detect a face in the image.\n\n"
                    "Please ensure:\n"
                    "- Your face is clearly visible\n"
                    "- The lighting is good\n"
                    "- You're looking directly at the camera\n\n"
                    "Need help? Type /help for instructions."
                )
            else:
                await processing_message.edit_text(
                    f"❌ Error processing image: {str(e)}\n\n"
                    "Need help? Type /help for instructions."
                )
            return
        finally:
            # Clean up downloaded file
            if os.path.exists(file_path):
                os.remove(file_path)
        
        if heart_rate == 0.0:
            await processing_message.edit_text(
                "❌ Could not calculate vital signs from the image/video.\n\n"
                "Please ensure:\n"
                "- Your face is clearly visible and well-lit\n"
                "- You're staying still during the video\n"
                "- There's good lighting\n\n"
                "Need help? Type /help for instructions."
            )
            return
        
        # Format and send results
        result_message = (
            "✅ Analysis Complete!\n\n"
            f"❤️ Heart Rate: {heart_rate:.1f} BPM\n"
            f"🫁 SpO2: {spo2:.1f}%\n\n"
            "Note: These measurements are estimates and should not be used for medical purposes.\n\n"
            "Want to measure again? Just send another photo or video!"
        )
        await processing_message.edit_text(result_message)
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        await processing_message.edit_text(
            "❌ An unexpected error occurred while processing your image.\n\n"
            "Please try again with:\n"
            "- Better lighting\n"
            "- Clearer face visibility\n"
            "- More stable camera position\n\n"
            "Need help? Type /help for instructions."
        )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a help message when the command /help is issued."""
    await update.message.reply_text(get_help_message())

async def handle_invalid_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle invalid inputs (text, stickers, etc.)"""
    await update.message.reply_text(
        "❌ I can only process photos and videos of faces.\n\n"
        "Please send me either:\n"
        "- A clear photo of your face\n"
        "- A short video (max 10 seconds) of your face\n\n"
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
    
    # Add message handlers
    app.add_handler(MessageHandler(filters.PHOTO | filters.VIDEO, handle_image))
    
    # Add handler for invalid inputs (must be last)
    app.add_handler(MessageHandler(filters.ALL, handle_invalid_input))
    
    # Start the bot
    logger.info("Bot started!")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
