#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from datetime import datetime
import pytz
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

# Токен бота
BOT_TOKEN = "8091069297:AAFbR4Jzdu32qatYNOTq9AqqcUm1564OKQ4"

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /start"""
    user = update.effective_user
    await update.message.reply_text(
        f"Привет, {user.first_name}! 👋\n\n"
        "Я бот, который отправляет текущее время и ID пира.\n"
        "Просто напиши мне любое сообщение!"
    )

async def send_time_and_peer_id(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отправляет текущее время и ID пира"""
    
    # Получение текущего времени в московском часовом поясе
    moscow_tz = pytz.timezone('Europe/Moscow')
    current_time = datetime.now(moscow_tz).strftime("%Y-%m-%d %H:%M:%S %Z")
    
    # Получение информации о пире (пользователе)
    user = update.effective_user
    chat = update.effective_chat
    
    # Формирование ответа
    response = (
        f"🕐 Текущее время: {current_time}\n"
        f"👤 ID пира (пользователя): {user.id}\n"
        f"💬 ID чата: {chat.id}\n"
        f"📝 Имя пользователя: {user.first_name}"
    )
    
    if user.last_name:
        response += f" {user.last_name}"
    
    if user.username:
        response += f"\n🔗 Username: @{user.username}"
    
    await update.message.reply_text(response)

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик ошибок"""
    logger.error(msg="Exception while handling an update:", exc_info=context.error)

def main() -> None:
    """Основная функция запуска бота"""
    # Создание приложения
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Регистрация обработчиков
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, send_time_and_peer_id))
    
    # Регистрация обработчика ошибок
    application.add_error_handler(error_handler)
    
    # Запуск бота
    logger.info("Запуск бота...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()