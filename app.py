"""Точка входа Streamlit-приложения с поддержкой нескольких страниц.

Запуск::

    uv run streamlit run app.py

Новые страницы добавляются через ``st.Page()`` в списке ``st.navigation``.
Бизнес-логика и код каждой страницы располагаются в ``src/pages/``.
"""

from __future__ import annotations

import streamlit as st

from pages.image_retrieval import render

st.set_page_config(
    page_title="Поиск похожих кропов",
    page_icon="🔍",
    layout="wide",
)

pg = st.navigation(
    [st.Page(render, title="Поиск похожих кропов", icon="🔍")]
)
pg.run()
