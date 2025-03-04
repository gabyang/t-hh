import os
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv
from model_processing import process_rppg

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a welcome message when the command /start is issued."""
    welcome_message = (
        "👋 Welcome to the Vital Signs Monitor Bot!\n\n"
        "Send me a photo or video of your face, and I'll analyze it to estimate your:\n"
        "❤️ Heart Rate\n"
        "🫁 Blood Oxygen Level (SpO2)\n\n"
        "Make sure your face is well-lit and clearly visible in the image/video."
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
            file = await update.message.video.get_file()
        else:
            await processing_message.edit_text("❌ Please send a photo or video.")
            return
        
        # Download and process the file
        file_path = await file.download_to_drive()
        heart_rate, spo2 = await process_rppg(file_path)
        
        # Clean up downloaded file
        os.remove(file_path)
        
        if heart_rate == 0.0:
            await processing_message.edit_text(
                "❌ Could not detect a face in the image. Please ensure your face is clearly visible and well-lit."
            )
            return
        
        # Format and send results
        result_message = (
            "✅ Analysis Complete!\n\n"
            f"❤️ Heart Rate: {heart_rate:.1f} BPM\n"
            f"🫁 SpO2: {spo2:.1f}%\n\n"
            "Note: These measurements are estimates and should not be used for medical purposes."
        )
        await processing_message.edit_text(result_message)
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        await processing_message.edit_text(
            "❌ An error occurred while processing your image. Please try again."
        )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a help message when the command /help is issued."""
    help_text = (
        "📋 How to use this bot:\n\n"
        "1. Send a photo or short video of your face\n"
        "2. Make sure your face is clearly visible and well-lit\n"
        "3. Stay still while taking the photo/video\n"
        "4. Wait for the analysis results\n\n"
        "Commands:\n"
        "/start - Start the bot\n"
        "/help - Show this help message"
    )
    await update.message.reply_text(help_text)

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
    
    # Start the bot
    logger.info("Bot started!")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
