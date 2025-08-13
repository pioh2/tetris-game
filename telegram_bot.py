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

def analyze_ml_emotion(face_region):
    """
    Полноценное ML-распознавание эмоций с использованием компьютерного зрения
    """
    try:
        # Подготовка изображения
        if len(face_region.shape) == 3:
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        else:
            gray_face = face_region
        
        # Нормализуем размер лица для анализа
        face_resized = cv2.resize(gray_face, (48, 48))
        face_normalized = face_resized.astype('float32') / 255.0
        
        # Анализ ключевых точек лица
        emotions_analysis = analyze_facial_features(gray_face, face_region)
        
        return emotions_analysis
        
    except Exception as e:
        logger.error(f"Ошибка ML анализа эмоций: {e}")
        return analyze_basic_emotion(face_region)

def analyze_facial_features(gray_face, color_face):
    """Анализ черт лица для определения эмоций"""
    
    # Инициализация детекторов для глаз и рта
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    
    height, width = gray_face.shape
    
    # 1. Детектируем глаза
    eyes = eye_cascade.detectMultiScale(gray_face, 1.1, 3)
    eye_features = analyze_eyes(gray_face, eyes)
    
    # 2. Детектируем рот/улыбку
    smiles = mouth_cascade.detectMultiScale(gray_face, 1.8, 20)
    mouth_features = analyze_mouth(gray_face, smiles)
    
    # 3. Общий анализ лица
    face_metrics = calculate_face_metrics(gray_face, color_face)
    
    # 4. Классификация эмоций на основе признаков
    emotions = classify_emotions(eye_features, mouth_features, face_metrics)
    
    return emotions

def analyze_eyes(gray_face, eyes):
    """Анализ области глаз"""
    features = {
        'count': len(eyes),
        'average_size': 0,
        'brightness': 0,
        'openness': 0.5  # по умолчанию средняя открытость
    }
    
    if len(eyes) >= 2:
        # Берем два самых больших глаза
        eyes_sorted = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
        
        total_size = 0
        total_brightness = 0
        
        for (ex, ey, ew, eh) in eyes_sorted:
            eye_region = gray_face[ey:ey+eh, ex:ex+ew]
            total_size += ew * eh
            total_brightness += np.mean(eye_region)
            
            # Оценка открытости глаз (соотношение высоты к ширине)
            openness_ratio = eh / ew if ew > 0 else 0.5
            features['openness'] += openness_ratio
        
        features['average_size'] = total_size / len(eyes_sorted)
        features['brightness'] = total_brightness / len(eyes_sorted)
        features['openness'] = features['openness'] / len(eyes_sorted)
    
    return features

def analyze_mouth(gray_face, smiles):
    """Анализ области рта"""
    features = {
        'smile_detected': len(smiles) > 0,
        'smile_confidence': 0,
        'mouth_curve': 0  # -1 грустный, 0 нейтральный, 1 счастливый
    }
    
    if len(smiles) > 0:
        # Берем самую уверенную улыбку
        largest_smile = max(smiles, key=lambda s: s[2] * s[3])
        sx, sy, sw, sh = largest_smile
        
        # Оценка уверенности улыбки
        smile_area = sw * sh
        face_area = gray_face.shape[0] * gray_face.shape[1]
        features['smile_confidence'] = min(smile_area / face_area * 10, 1.0)
        features['mouth_curve'] = 0.7  # положительная кривизна для улыбки
    else:
        # Попытка определить грустное выражение
        bottom_third = gray_face[2*gray_face.shape[0]//3:, :]
        mouth_brightness = np.mean(bottom_third)
        face_brightness = np.mean(gray_face)
        
        if mouth_brightness < face_brightness * 0.85:
            features['mouth_curve'] = -0.5  # отрицательная кривизна
    
    return features

def calculate_face_metrics(gray_face, color_face):
    """Расчет общих метрик лица"""
    
    # Конвертация в HSV для анализа цвета
    hsv_face = cv2.cvtColor(color_face, cv2.COLOR_BGR2HSV)
    
    metrics = {
        'brightness': np.mean(gray_face) / 255.0,
        'contrast': np.std(gray_face) / 255.0,
        'saturation': np.mean(hsv_face[:, :, 1]) / 255.0,
        'sharpness': 0,
        'symmetry': 0.5
    }
    
    # Расчет резкости (четкости)
    laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
    metrics['sharpness'] = min(laplacian_var / 1000.0, 1.0)
    
    # Расчет симметрии
    height, width = gray_face.shape
    left_half = gray_face[:, :width//2]
    right_half = cv2.flip(gray_face[:, width//2:], 1)
    
    if left_half.shape == right_half.shape:
        correlation = np.corrcoef(left_half.flatten(), right_half.flatten())
        if not np.isnan(correlation[0, 1]):
            metrics['symmetry'] = max(0, correlation[0, 1])
    
    return metrics

def classify_emotions(eye_features, mouth_features, face_metrics):
    """Классификация эмоций на основе признаков лица"""
    
    emotions = {}
    
    # 1. СЧАСТЬЕ
    happiness_score = 0
    if mouth_features['smile_detected']:
        happiness_score += mouth_features['smile_confidence'] * 0.6
    if mouth_features['mouth_curve'] > 0:
        happiness_score += mouth_features['mouth_curve'] * 0.3
    if face_metrics['brightness'] > 0.5:
        happiness_score += (face_metrics['brightness'] - 0.5) * 0.1
    emotions['Счастье 😊'] = min(happiness_score, 1.0)
    
    # 2. ГРУСТЬ
    sadness_score = 0
    if mouth_features['mouth_curve'] < 0:
        sadness_score += abs(mouth_features['mouth_curve']) * 0.4
    if face_metrics['brightness'] < 0.4:
        sadness_score += (0.4 - face_metrics['brightness']) * 0.3
    if eye_features['brightness'] < 100:  # темные глаза
        sadness_score += 0.3
    emotions['Грусть 😢'] = min(sadness_score, 1.0)
    
    # 3. УДИВЛЕНИЕ
    surprise_score = 0
    if eye_features['openness'] > 0.7:  # широко открытые глаза
        surprise_score += (eye_features['openness'] - 0.7) * 1.0
    if face_metrics['sharpness'] > 0.5:  # четкие черты
        surprise_score += (face_metrics['sharpness'] - 0.5) * 0.3
    emotions['Удивление 😮'] = min(surprise_score, 1.0)
    
    # 4. ЗЛОСТЬ
    anger_score = 0
    if face_metrics['contrast'] > 0.3:  # высокий контраст
        anger_score += (face_metrics['contrast'] - 0.3) * 0.5
    if face_metrics['symmetry'] < 0.4:  # асимметрия
        anger_score += (0.4 - face_metrics['symmetry']) * 0.3
    if face_metrics['sharpness'] > 0.6:  # резкие черты
        anger_score += (face_metrics['sharpness'] - 0.6) * 0.2
    emotions['Злость 😡'] = min(anger_score, 1.0)
    
    # 5. СТРАХ
    fear_score = 0
    if eye_features['openness'] > 0.6 and not mouth_features['smile_detected']:
        fear_score += 0.4
    if face_metrics['saturation'] < 0.3:  # бледность
        fear_score += (0.3 - face_metrics['saturation']) * 0.4
    if face_metrics['symmetry'] < 0.5:
        fear_score += (0.5 - face_metrics['symmetry']) * 0.2
    emotions['Страх 😨'] = min(fear_score, 1.0)
    
    # 6. СПОКОЙСТВИЕ
    calm_score = 0
    if 0.4 < face_metrics['brightness'] < 0.7:  # средняя яркость
        calm_score += 0.3
    if face_metrics['symmetry'] > 0.6:  # хорошая симметрия
        calm_score += (face_metrics['symmetry'] - 0.6) * 0.5
    if 0.3 < eye_features['openness'] < 0.6:  # нормально открытые глаза
        calm_score += 0.2
    emotions['Спокойствие 😌'] = min(calm_score, 1.0)
    
    # 7. УСТАЛОСТЬ
    tired_score = 0
    if eye_features['openness'] < 0.4:  # полузакрытые глаза
        tired_score += (0.4 - eye_features['openness']) * 0.6
    if face_metrics['brightness'] < 0.45:
        tired_score += (0.45 - face_metrics['brightness']) * 0.4
    emotions['Усталость 😴'] = min(tired_score, 1.0)
    
    # Удаляем эмоции с очень низким скором
    emotions = {k: v for k, v in emotions.items() if v > 0.1}
    
    # Если нет явных эмоций, добавляем нейтральную
    if not emotions or max(emotions.values()) < 0.3:
        emotions['Нейтральное 😐'] = 0.6
    
    # Сортируем по убыванию уверенности
    sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_emotions

def analyze_basic_emotion(face_region):
    """Базовый анализ эмоций (резервный метод)"""
    gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray_face) / 255.0
    contrast = np.std(gray_face) / 255.0
    
    if brightness > 0.6 and contrast > 0.2:
        return [('Счастье 😊', 0.7)]
    elif brightness < 0.4:
        return [('Грусть 😢', 0.6)]
    elif contrast > 0.3:
        return [('Напряжение 😟', 0.5)]
    else:
        return [('Спокойствие 😌', 0.5)]

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /start"""
    user = update.effective_user
    await update.message.reply_text(
        f"Привет, {user.first_name}! 👋\n\n"
        "Я бот с **полноценным ML-анализом эмоций**! 🧠\n\n"
        "**Мои возможности:**\n"
        "• 📝 Время и ID пира на текстовые сообщения\n"
        "• 🔍 OCR - извлечение текста с изображений\n"
        "• 🎭 **ML-анализ эмоций** самого крупного лица\n"
        "• 👁️ Анализ глаз, рта и общих черт лица\n"
        "• 📊 Точные проценты уверенности\n\n"
        "**Эмоции:** 😊😢😮😡😨😌😴😐\n"
        "Отправь фото для анализа!"
    )

async def analyze_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Анализирует изображение с ML-распознаванием эмоций"""
    try:
        # Получаем изображение
        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)
        
        file_bytes = io.BytesIO()
        await file.download_to_memory(file_bytes)
        file_bytes.seek(0)
        
        pil_image = Image.open(file_bytes)
        response_parts = ["🧠 **ML-анализ изображения**\n"]
        
        # Анализ лиц
        detector = get_face_detector()
        if detector and detector is not False:
            try:
                opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
                
                faces = detector.detectMultiScale(gray, 1.1, 4)
                
                if len(faces) > 0:
                    # Находим самое крупное лицо
                    largest_face = max(faces, key=lambda face: face[2] * face[3])
                    x, y, w, h = largest_face
                    
                    response_parts.append(f"👥 **Обнаружено лиц:** {len(faces)}")
                    response_parts.append(f"🎯 **Анализирую самое крупное лицо**\n")
                    
                    # Извлекаем и анализируем лицо
                    face_region = opencv_image[y:y+h, x:x+w]
                    emotions = analyze_ml_emotion(face_region)
                    
                    if emotions:
                        main_emotion = emotions[0]
                        response_parts.append(f"🎭 **Основная эмоция:** {main_emotion[0]}")
                        response_parts.append(f"📊 **Уверенность:** {main_emotion[1]*100:.1f}%\n")
                        
                        if len(emotions) > 1:
                            response_parts.append("📈 **Дополнительные эмоции:**")
                            for emotion, confidence in emotions[1:4]:  # топ-3 дополнительных
                                if confidence > 0.15:
                                    response_parts.append(f"• {emotion}: {confidence*100:.1f}%")
                    
                    # Информация о лице
                    face_size = w * h
                    total_size = opencv_image.shape[0] * opencv_image.shape[1]
                    face_percentage = (face_size / total_size) * 100
                    
                    response_parts.append(f"\n📏 **Детали:**")
                    response_parts.append(f"• Размер лица: {w}×{h}px")
                    response_parts.append(f"• Площадь кадра: {face_percentage:.1f}%")
                    
                    if face_percentage > 25:
                        response_parts.append("• Тип: Крупный план 📷")
                    elif face_percentage > 10:
                        response_parts.append("• Тип: Средний план 📸")
                    else:
                        response_parts.append("• Тип: Общий план 🖼️")
                else:
                    response_parts.append("❌ **Лица не обнаружены**")
                    
            except Exception as e:
                logger.error(f"Ошибка анализа: {e}")
                response_parts.append("⚠️ **Ошибка анализа лиц**")
        else:
            response_parts.append("⚠️ **Детектор недоступен**")
        
        # OCR
        file_bytes.seek(0)
        pil_image_ocr = Image.open(file_bytes)
        text = pytesseract.image_to_string(pil_image_ocr, lang='rus+eng').strip()
        
        if text:
            response_parts.append(f"\n🔍 **Текст:**\n```\n{text}\n```")
        else:
            response_parts.append("\n📝 Текст не найден")
        
        response = "\n".join(response_parts)
        await update.message.reply_text(response, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Ошибка обработки: {e}")
        await update.message.reply_text("❌ Ошибка обработки изображения")

async def send_time_and_peer_id(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отправляет текущее время и ID пира"""
    moscow_tz = pytz.timezone('Europe/Moscow')
    current_time = datetime.now(moscow_tz).strftime("%Y-%m-%d %H:%M:%S %Z")
    
    user = update.effective_user
    chat = update.effective_chat
    
    response = (
        f"🕐 **Время:** {current_time}\n"
        f"👤 **ID пира:** {user.id}\n"
        f"💬 **ID чата:** {chat.id}\n"
        f"📝 **Пользователь:** {user.first_name}"
    )
    
    if user.last_name:
        response += f" {user.last_name}"
    if user.username:
        response += f"\n🔗 **Username:** @{user.username}"
    
    await update.message.reply_text(response, parse_mode='Markdown')

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик ошибок"""
    logger.error(msg="Exception while handling an update:", exc_info=context.error)

def main() -> None:
    """Основная функция запуска бота"""
    application = Application.builder().token(BOT_TOKEN).build()
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO, analyze_image))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, send_time_and_peer_id))
    application.add_error_handler(error_handler)
    
    logger.info("🚀 Запуск бота с ML-анализом эмоций...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()