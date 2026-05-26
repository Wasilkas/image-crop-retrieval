"""Абстракция SSL-эмбеддинга и PyTorch-реализация.

Два публичных объекта:

* :class:`EmbedderProtocol` — ``@runtime_checkable``-протокол, которому должен
  удовлетворять любой эмбеддер.  Отделяет остальной код от PyTorch.

* :class:`TorchEmbedder` — конкретная реализация, загружающая ``.pt``/``.pth``
  чекпоинт и выполняющая CPU-инференс.

Поддерживаемые форматы чекпоинтов
-----------------------------------
Полная модель (pickle)
    ``torch.save(model, path)``  →  ``torch.load(path)`` возвращает ``nn.Module``.
    Используйте, когда определение класса модели *недоступно* в среде инференса
    (класс зашит в pickle).

State-dict
    ``torch.save(model.state_dict(), path)``  →  ``torch.load(path)`` возвращает
    ``dict``.  Требует передачи ``model_class`` в ``TorchEmbedder`` для создания
    экземпляра архитектуры перед загрузкой весов.

.. warning::
    ``torch.load`` с ``weights_only=False`` выполняет произвольный Python-код
    из pickle.  Загружайте чекпоинты только из **доверенных источников**.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, cast, runtime_checkable

import numpy as np
import torch
import torch.nn as nn
from PIL import Image as PILImage


@runtime_checkable
class EmbedderProtocol(Protocol):
    """Минимальный интерфейс, которому должен удовлетворять каждый эмбеддер.

    Реализации должны возвращать **L2-нормализованные** float32-векторы,
    чтобы поиск по inner product был эквивалентен косинусному сходству.
    """

    def embed(self, crops: list[PILImage.Image]) -> np.ndarray:
        """Вычисляет эмбеддинги для батча PIL-изображений.

        Args:
            crops: Непустой список PIL-изображений любого размера и режима.
                   Реализации сами выполняют изменение размера и конвертацию
                   цветового пространства.

        Returns:
            float32-ndarray формы ``(len(crops), embedding_dim)``.
            Строки являются **L2-нормализованными** (единичные векторы).

        Raises:
            ValueError: Если ``crops`` пуст.
        """
        ...


class TorchEmbedder:
    """Обёртка над кастомным PyTorch-чекпоинтом для CPU-инференса SSL-моделей.

    Args:
        checkpoint_path: Путь к файлу ``.pt`` или ``.pth``.
        model_class: Необязательный *класс* ``nn.Module`` (не экземпляр).
            Если задан — чекпоинт считается state-dict, класс инстанцируется
            перед загрузкой весов.
            Если ``None`` — чекпоинт считается полной моделью (pickle).
        input_size: ``(высота, ширина)`` для изменения размера каждого кропа
            перед подачей в модель.  По умолчанию ``(224, 224)``.
        device: Строка устройства PyTorch.  По умолчанию ``"cpu"``.

    Raises:
        FileNotFoundError: Если ``checkpoint_path`` не существует.
        RuntimeError: Если формат чекпоинта не соответствует ожидаемому режиму
            (напр. state-dict при отсутствии ``model_class`` или наоборот).
    """

    def __init__(
        self,
        checkpoint_path: Path,
        model_class: type[nn.Module] | None = None,
        input_size: tuple[int, int] = (224, 224),
        device: str = "cpu",
    ) -> None:
        self._device = torch.device(device)
        self._model: nn.Module = self._load_model(checkpoint_path, model_class)
        self._model.eval()
        self._transform = _build_transform(input_size)

    def _load_model(
        self,
        path: Path,
        model_class: type[nn.Module] | None,
    ) -> nn.Module:
        """Загружает и возвращает модель из *path*.

        Поддерживает форматы полной модели (pickle) и state-dict; формат
        определяется по наличию *model_class*.

        Raises:
            FileNotFoundError: Файл чекпоинта не найден.
            RuntimeError: Несоответствие формата и ожидаемого режима.
        """
        if not path.exists():
            raise FileNotFoundError(f"Чекпоинт не найден: {path}")

        # weights_only=False необходим для pickle полной модели.
        # Загружайте чекпоинты только из доверенных источников (см. docstring модуля).
        payload = torch.load(path, map_location=self._device, weights_only=False)  # noqa: S614

        if model_class is not None:
            if not isinstance(payload, dict):
                raise RuntimeError(
                    f"Ожидался state-dict (dict) в '{path}', "
                    f"но torch.load() вернул {type(payload).__name__}. "
                    "Уберите model_class для режима полной модели."
                )
            model = model_class()
            model.load_state_dict(payload)
            return model.to(self._device)

        if not isinstance(payload, nn.Module):
            raise RuntimeError(
                f"Ожидался nn.Module в '{path}', "
                f"но torch.load() вернул {type(payload).__name__}. "
                "Передайте model_class= для state-dict-чекпоинтов."
            )
        return payload.to(self._device)

    @torch.no_grad()
    def embed(self, crops: list[PILImage.Image]) -> np.ndarray:
        """Вычисляет эмбеддинги батча кропов; возвращает L2-нормализованный
        массив формы ``(N, D)``.

        Args:
            crops: Непустой список PIL-изображений.

        Returns:
            float32-ndarray формы ``(len(crops), embedding_dim)``,
            каждая строка — единичный L2-вектор.

        Raises:
            ValueError: Если *crops* пуст.
        """
        if not crops:
            raise ValueError("crops должен быть непустым списком")

        # Конвертируем в RGB, применяем трансформации, собираем в батч-тензор
        tensors = torch.stack(
            [self._transform(crop.convert("RGB")) for crop in crops]
        )
        tensors = tensors.to(self._device)

        raw: torch.Tensor = self._model(tensors)

        # Некоторые модели возвращают пространственные карты (N, C, H, W) — выравниваем
        if raw.dim() > 2:
            raw = raw.flatten(start_dim=1)

        # L2-нормализация: IndexFlatIP даёт косинусное сходство для единичных векторов
        normed = torch.nn.functional.normalize(raw, p=2, dim=1)
        return normed.cpu().numpy().astype(np.float32)


def _build_transform(input_size: tuple[int, int]) -> torch.nn.Module:
    """Возвращает пайплайн trochvision-трансформаций.

    Пайплайн изменяет размер до *input_size*, переводит в float-тензор и
    нормализует по ImageNet mean/std — разумное умолчание для большинства
    SSL-моделей, обученных на натуральных изображениях.

    Args:
        input_size: Целевой размер ``(высота, ширина)``.

    Returns:
        Экземпляр ``torchvision.transforms.Compose``.
    """
    from torchvision import transforms  # отложенный импорт: torchvision тяжёлая

    # torchvision не имеет py.typed стабов; cast к ожидаемому типу
    return cast(
        torch.nn.Module,
        transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        ),
    )
