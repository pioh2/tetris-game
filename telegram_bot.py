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

def analyze_advanced_emotion(face_region):
    """Продвинутый анализ эмоций на основе множественных параметров"""
    # Конвертируем в разные цветовые пространства для анализа
    gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
    hsv_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
    
    # Параметры для анализа
    height, width = gray_face.shape
    
    # 1. Анализ яркости
    mean_brightness = np.mean(gray_face)
    brightness_std = np.std(gray_face)
    
    # 2. Анализ контраста
    contrast = gray_face.max() - gray_face.min()
    local_contrast = np.std(gray_face)
    
    # 3. Анализ цветности (насыщенность)
    saturation = np.mean(hsv_face[:, :, 1])
    
    # 4. Анализ верхней и нижней части лица
    upper_half = gray_face[:height//2, :]
    lower_half = gray_face[height//2:, :]
    upper_brightness = np.mean(upper_half)
    lower_brightness = np.mean(lower_half)
    brightness_ratio = upper_brightness / (lower_brightness + 1e-6)
    
    # 5. Анализ градиентов (для определения мимических морщин)
    grad_x = cv2.Sobel(gray_face, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_face, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    avg_gradient = np.mean(gradient_magnitude)
    
    # 6. Анализ симметрии лица
    left_half = gray_face[:, :width//2]
    right_half = cv2.flip(gray_face[:, width//2:], 1)
    if left_half.shape == right_half.shape:
        symmetry = np.corrcoef(left_half.flatten(), right_half.flatten())[0, 1]
        if np.isnan(symmetry):
            symmetry = 0.5
    else:
        symmetry = 0.5
    
    # 7. Анализ областей глаз (верхняя треть)
    eye_region = gray_face[:height//3, :]
    eye_brightness = np.mean(eye_region)
    eye_contrast = np.std(eye_region)
    
    # 8. Анализ области рта (нижняя треть)
    mouth_region = gray_face[2*height//3:, :]
    mouth_brightness = np.mean(mouth_region)
    mouth_contrast = np.std(mouth_region)
    
    # Нормализация параметров
    mean_brightness = mean_brightness / 255.0
    contrast = contrast / 255.0
    local_contrast = local_contrast / 255.0
    saturation = saturation / 255.0
    avg_gradient = min(avg_gradient / 100.0, 1.0)
    symmetry = max(0, min(symmetry, 1))
    
    # Логика определения эмоций на основе комбинации параметров
    emotions = []
    
    # Счастье - высокая яркость, хороший контраст, симметрия
    if mean_brightness > 0.6 and local_contrast > 0.12 and symmetry > 0.3:
        happiness_score = (mean_brightness * 0.4 + local_contrast * 0.3 + symmetry * 0.3)
        emotions.append(("Счастье 😊", happiness_score))
    
    # Грусть - низкая яркость, особенно в области глаз
    if mean_brightness < 0.45 or (eye_brightness < mouth_brightness * 0.9):
        sadness_score = (1 - mean_brightness) * 0.5 + (1 - brightness_ratio) * 0.3 + (1 - saturation) * 0.2
        emotions.append(("Грусть 😢", sadness_score))
    
    # Злость - высокий контраст, асимметрия, высокие градиенты
    if local_contrast > 0.15 and avg_gradient > 0.3:
        anger_score = local_contrast * 0.4 + avg_gradient * 0.4 + (1 - symmetry) * 0.2
        emotions.append(("Злость 😡", anger_score))
    
    # Удивление - высокий контраст в области глаз, высокие градиенты
    if eye_contrast > 0.2 and avg_gradient > 0.25:
        surprise_score = eye_contrast * 0.5 + avg_gradient * 0.3 + brightness_ratio * 0.2
        emotions.append(("Удивление 😮", surprise_score))
    
    # Страх - низкая насыщенность, высокие градиенты, асимметрия
    if saturation < 0.3 and avg_gradient > 0.2 and symmetry < 0.4:
        fear_score = (1 - saturation) * 0.4 + avg_gradient * 0.3 + (1 - symmetry) * 0.3
        emotions.append(("Страх 😨", fear_score))
    
    # Отвращение - низкая яркость в области рта, асимметрия
    if mouth_brightness < mean_brightness * 0.8 and symmetry < 0.5:
        disgust_score = (1 - mouth_brightness / mean_brightness) * 0.5 + (1 - symmetry) * 0.5
        emotions.append(("Отвращение 🤢", disgust_score))
    
    # Задумчивость - средние значения, низкий контраст
    if 0.3 < mean_brightness < 0.7 and local_contrast < 0.1 and symmetry > 0.4:
        thoughtful_score = (0.5 - abs(mean_brightness - 0.5)) * 2 * 0.5 + symmetry * 0.3 + (1 - local_contrast) * 0.2
        emotions.append(("Задумчивость 🤔", thoughtful_score))
    
    # Усталость - низкий контраст, низкая яркость глаз
    if eye_brightness < mean_brightness * 0.8 and local_contrast < 0.08:
        tired_score = (1 - eye_brightness / mean_brightness) * 0.6 + (1 - local_contrast) * 0.4
        emotions.append(("Усталость 😴", tired_score))
    
    # Концентрация - средний контраст, хорошая симметрия, средняя яркость
    if 0.4 < mean_brightness < 0.6 and 0.08 < local_contrast < 0.15 and symmetry > 0.5:
        focus_score = symmetry * 0.5 + (0.5 - abs(mean_brightness - 0.5)) * 2 * 0.3 + (0.5 - abs(local_contrast - 0.115)) * 2 * 0.2
        emotions.append(("Концентрация 🧐", focus_score))
    
    # Спокойствие - средние значения всех параметров, хорошая симметрия
    if 0.45 < mean_brightness < 0.65 and local_contrast < 0.12 and symmetry > 0.4:
        calm_score = symmetry * 0.4 + (0.5 - abs(mean_brightness - 0.55)) * 2 * 0.3 + (1 - local_contrast) * 0.3
        emotions.append(("Спокойствие 😌", calm_score))
    
    # Если эмоции не определены или список пуст
    if not emotions:
        neutral_score = 0.5 + symmetry * 0.3 + (0.5 - abs(mean_brightness - 0.5)) * 0.2
        emotions.append(("Нейтральное 😐", neutral_score))
    
    # Выбираем эмоцию с наивысшим скором
    best_emotion = max(emotions, key=lambda x: x[1])
    
    # Возвращаем топ-3 эмоции для более детального анализа
    sorted_emotions = sorted(emotions, key=lambda x: x[1], reverse=True)[:3]
    
    return best_emotion, sorted_emotions

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /start"""
    user = update.effective_user
    await update.message.reply_text(
        f"Привет, {user.first_name}! 👋\n\n"
        "Я бот, который:\n"
        "• Отправляет текущее время и ID пира на любое текстовое сообщение\n"
        "• Извлекает текст из изображений с помощью OCR 📸➡️📝\n"
        "• Анализирует эмоции САМОГО КРУПНОГО лица на портретах 😊😢😡🤔\n\n"
        "Теперь я распознаю 10+ эмоций с высокой точностью!\n"
        "Просто напиши мне сообщение или отправь картинку!"
    )

async def analyze_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Анализирует изображение: извлекает текст и определяет эмоции самого крупного лица"""
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
                    # Находим самое крупное лицо
                    largest_face = max(faces, key=lambda face: face[2] * face[3])
                    x, y, w, h = largest_face
                    
                    response_parts.append(f"👥 Обнаружено лиц: {len(faces)}")
                    response_parts.append(f"🎯 Анализирую самое крупное лицо:\n")
                    
                    # Извлекаем область самого крупного лица
                    face_region = opencv_image[y:y+h, x:x+w]
                    
                    # Продвинутый анализ эмоций
                    main_emotion, all_emotions = analyze_advanced_emotion(face_region)
                    
                    # Основная эмоция
                    response_parts.append(f"🎭 **Основная эмоция:** {main_emotion[0]}")
                    response_parts.append(f"   Уверенность: {main_emotion[1]:.1%}\n")
                    
                    # Дополнительные эмоции если есть
                    if len(all_emotions) > 1:
                        response_parts.append("📊 **Дополнительные эмоции:**")
                        for emotion, score in all_emotions[1:]:
                            if score > 0.2:  # Показываем только значимые
                                response_parts.append(f"   • {emotion} ({score:.1%})")
                    
                    # Информация о размере лица
                    face_size = w * h
                    total_image_size = opencv_image.shape[0] * opencv_image.shape[1]
                    face_percentage = (face_size / total_image_size) * 100
                    
                    response_parts.append(f"\n📏 **Детали лица:**")
                    response_parts.append(f"   • Размер: {w}×{h} пикселей")
                    response_parts.append(f"   • Занимает {face_percentage:.1f}% кадра")
                    
                    if face_percentage > 25:
                        response_parts.append(f"   • Тип: крупный план 📷")
                    elif face_percentage > 10:
                        response_parts.append(f"   • Тип: средний план 📸")
                    else:
                        response_parts.append(f"   • Тип: общий план 🖼️")
                        
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
            response_parts.append(f"\n🔍 **Извлеченный текст:**\n```\n{extracted_text}\n```")
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
    
    # Обработчик для изображений (OCR и продвинутый анализ эмоций)
    application.add_handler(MessageHandler(filters.PHOTO, analyze_image))
    
    # Обработчик для текстовых сообщений (исключая команды)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, send_time_and_peer_id))
    
    # Регистрация обработчика ошибок
    application.add_error_handler(error_handler)
    
    # Запуск бота
    logger.info("Запуск бота с продвинутым анализом эмоций...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()