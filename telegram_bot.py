#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
import io
from datetime import datetime
import pytz
import pytesseract
import cv2
import numpy as np
from PIL import Image
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

# Токен бота
BOT_TOKEN = "8091069297:AAFbR4Jzdu32qatYNOTq9AqqcUm1564OKQ4"

# Инициализация детектора лиц OpenCV
face_cascade = None

def get_face_detector():
    """Ленивая инициализация детектора лиц"""
    global face_cascade
    if face_cascade is None:
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except Exception as e:
            logger.error(f"Ошибка инициализации детектора лиц: {e}")
            face_cascade = False
    return face_cascade

# Простой анализ на основе яркости и контраста
def analyze_simple_emotion(face_region):
    """Простой анализ настроения на основе яркости и других параметров"""
    gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
    
    # Анализ яркости
    mean_brightness = np.mean(gray_face)
    
    # Анализ контраста
    contrast = np.std(gray_face)
    
    # Простые эвристики для определения настроения
    if mean_brightness > 140 and contrast > 30:
        return "Счастливое 😊", 0.7
    elif mean_brightness < 100:
        return "Грустное 😢", 0.6
    elif contrast > 50:
        return "Напряженное 😟", 0.5
    else:
        return "Спокойное 😐", 0.6

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /start"""
    user = update.effective_user
    await update.message.reply_text(
        f"Привет, {user.first_name}! 👋\n\n"
        "Я бот, который:\n"
        "• Отправляет текущее время и ID пира на любое текстовое сообщение\n"
        "• Извлекает текст из изображений с помощью OCR 📸➡️📝\n"
        "• Анализирует лица людей на портретах 😊😢😡\n\n"
        "Просто напиши мне сообщение или отправь картинку!"
    )

async def analyze_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Анализирует изображение: извлекает текст и определяет лица"""
    try:
        # Получаем самое большое изображение
        photo = update.message.photo[-1]
        
        # Скачиваем файл
        file = await context.bot.get_file(photo.file_id)
        
        # Получаем байты изображения
        file_bytes = io.BytesIO()
        await file.download_to_memory(file_bytes)
        file_bytes.seek(0)
        
        # Открываем изображение с помощью PIL
        pil_image = Image.open(file_bytes)
        
        # Формируем ответ
        response_parts = ["📸 Изображение обработано!\n"]
        
        # Анализ лиц
        detector = get_face_detector()
        if detector and detector is not False:
            try:
                # Конвертируем PIL изображение в numpy array для OpenCV
                opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
                
                # Детекция лиц
                faces = detector.detectMultiScale(gray, 1.1, 4)
                
                if len(faces) > 0:
                    response_parts.append(f"👥 Обнаружено лиц: {len(faces)}")
                    
                    for i, (x, y, w, h) in enumerate(faces, 1):
                        # Извлекаем область лица
                        face_region = opencv_image[y:y+h, x:x+w]
                        
                        # Простой анализ настроения
                        emotion, confidence = analyze_simple_emotion(face_region)
                        
                        response_parts.append(f"  {i}. Настроение: {emotion} ({confidence:.1%})")
                        
                        # Дополнительная информация о размере лица
                        face_size = w * h
                        if face_size > 10000:
                            response_parts.append(f"     Размер: крупный план")
                        elif face_size > 5000:
                            response_parts.append(f"     Размер: средний план")
                        else:
                            response_parts.append(f"     Размер: мелкий план")
                else:
                    response_parts.append("👤 Лица на изображении не обнаружены")
            except Exception as e:
                logger.error(f"Ошибка анализа лиц: {e}")
                response_parts.append("⚠️ Анализ лиц временно недоступен")
        else:
            response_parts.append("⚠️ Анализ лиц недоступен")
        
        # OCR для извлечения текста
        file_bytes.seek(0)  # Сбрасываем указатель
        pil_image_for_ocr = Image.open(file_bytes)
        extracted_text = pytesseract.image_to_string(pil_image_for_ocr, lang='rus+eng').strip()
        
        # OCR результат
        if extracted_text:
            response_parts.append(f"\n🔍 Извлеченный текст:\n```\n{extracted_text}\n```")
        else:
            response_parts.append("\n📝 Текст на изображении не обнаружен")
        
        response = "\n".join(response_parts)
        await update.message.reply_text(response, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Ошибка при обработке изображения: {e}")
        await update.message.reply_text(
            "❌ Произошла ошибка при обработке изображения. "
            "Попробуйте отправить другое изображение."
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
    
    # Обработчик для изображений (OCR и анализ лиц)
    application.add_handler(MessageHandler(filters.PHOTO, analyze_image))
    
    # Обработчик для текстовых сообщений (исключая команды)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, send_time_and_peer_id))
    
    # Регистрация обработчика ошибок
    application.add_error_handler(error_handler)
    
    # Запуск бота
    logger.info("Запуск бота с поддержкой OCR и анализа лиц...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()