import torch

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image


class PriceRecognizer:
    def __init__(self):
        # Директория, где сохранена предобученная модель
        self.model_dir = "./saved_model"

        # Определение устройства: используем GPU, если доступно, иначе CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Загрузка процессора и модели из указанной директории
        self.processor = TrOCRProcessor.from_pretrained(self.model_dir)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.model_dir)

        # Перенос модели на выбранное устройство
        self.model.to(self.device)

    def predict(self, img_path):
        """
        Метод для предсказания числового значения цены по изображению.
        :param img_path: путь к изображению
        :return: числовое значение цены (int)
        """

        # Открываем изображение и преобразуем его в RGB-формат
        image = Image.open(img_path).convert("RGB")

        # Преобразуем изображение в тензоры с помощью процессора модели
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)

        # Генерация текста из изображения с помощью модели
        generated_ids = self.model.generate(pixel_values)

        # Декодируем выход модели в текстовую строку
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Конвертируем текстовое представление цены в целое число
        return self.convert_to_int(generated_text)

    @staticmethod
    def convert_to_int(price):
        """
        Метод для преобразования текстовой строки в числовой формат.
        :param price: строка, содержащая цену (возможно, с лишними символами)
        :return: целое число (0, если цифры отсутствуют)
        """

        # Извлекаем только цифровые символы из строки
        price = ''.join(char for char in price if char.isdigit())

        # Преобразуем в int, если строка не пустая, иначе возвращаем 0
        return int(price) if price != '' else 0
