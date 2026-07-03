# app/ui/themes.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Stylesheets for the Light and Dark themes.

# Light theme
LIGHT_THEME_STYLESHEET = """
    /* General Widget Styling */
    QWidget {
        background-color: #F0F0F0;
        color: #000000;
        font-family: "JetBrains Mono";
    }
    QMainWindow {
        background-color: #F0F0F0;
    }

    /* Input Widgets */
    QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {
        background-color: #FFFFFF;
        border: 1px solid #C0C0C0;
        border-radius: 4px;
        padding: 1px;
    }
    QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QLineEdit:focus {
        border: 1px solid #0078D7;
    }
    QSpinBox::up-button, QDoubleSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::down-button {
        width: 20px;
    }

    /* Buttons */
    QPushButton, QToolButton {
        background-color: #F0F0F0;
        border: 1px solid #C0C0C0;
        border-radius: 4px;
        padding: 3px;
    }
    QPushButton:hover, QToolButton:hover {
        background-color: #D0D0D0;
    }
    QPushButton:pressed, QToolButton:pressed {
        background-color: #C8C8C8;
    }

    /* CheckBox */
    QCheckBox {
        spacing: 5px;
    }
    QCheckBox::indicator {
        width: 14px;
        height: 14px;
        border: 1px solid #C0C0C0;
        border-radius: 3px;
        background-color: #FFFFFF;
    }
    QCheckBox::indicator:hover {
        border: 1px solid #0078D7;
    }
    QCheckBox::indicator:checked {
        background-color: #0078D7;
        border: 1px solid #0078D7;
    }

    /* Collapsible Section */
    #collapsibleTitleBar {
        background-color: #E0E0E0;
        border: 1px solid #C0C0C0;
        border-radius: 4px;
    }
    #collapsibleTitleLabel { font-weight: bold; }
    #collapsibleContent {
        border: 1px solid #C0C0C0;
        border-top: none;
        border-radius: 0 0 4px 4px;
        padding: 1px;
    }

    /* Other Widgets */
    QSplitter::handle {
        background-color: #C0C0C0;
    }
    QProgressBar {
        border: 1px solid grey;
        border-radius: 5px;
        text-align: center;
    }
    QProgressBar::chunk {
        background-color: #0078D7;
        width: 10px;
    }
    QFrame#presetFrame {
        border: 1px solid #C0C0C0;
        border-radius: 4px;
    }
"""

# Dark theme
DARK_THEME_STYLESHEET = """
    /* General Widget Styling */
    QWidget {
        background-color: #121212;
        color: #E0E0E0;
        font-family: "JetBrains Mono";
    }
    QMainWindow {
        background-color: #0D0D0D;
    }

    /* Input Widgets */
    QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {
        background-color: #3C3C3C;
        border: 1px solid #5A5A5A;
        border-radius: 4px;
        padding: 1px;
        color: #E0E0E0;
    }
    QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QLineEdit:focus {
        border: 1px solid #0078D7;
    }
    QSpinBox::up-button, QDoubleSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::down-button {
        width: 20px;
    }

    /* Buttons */
    QPushButton, QToolButton {
        background-color: #121212;
        border: 1px solid #5A5A5A;
        border-radius: 4px;
        padding: 3px;
    }
    QPushButton:hover, QToolButton:hover {
        background-color: #3A3A3A;
    }
    QPushButton:pressed, QToolButton:pressed {
        background-color: #4A4A4A;
    }

    /* CheckBox */
    QCheckBox {
        spacing: 5px;
    }
    QCheckBox::indicator {
        width: 14px;
        height: 14px;
        border: 1px solid #5A5A5A;
        border-radius: 3px;
        background-color: #3C3C3C;
    }
    QCheckBox::indicator:hover {
        border: 1px solid #0078D7;
    }
    QCheckBox::indicator:checked {
        background-color: #0078D7;
        border: 1px solid #0078D7;
    }

    /* Collapsible Section */
    #collapsibleTitleBar {
        background-color: #252525;
        border: 1px solid #5A5A5A;
        border-radius: 4px;
    }
    #collapsibleTitleLabel { font-weight: bold; }
    #collapsibleContent {
        border: 1px solid #5A5A5A;
        border-top: none;
        border-radius: 0 0 4px 4px;
        padding: 1px;
        background-color: #333333;
    }

    /* Other Widgets */
    QSplitter::handle {
        background-color: #5A5A5A;
    }
    QProgressBar {
        border: 1px solid #5A5A5A;
        border-radius: 5px;
        text-align: center;
        color: #E0E0E0;
    }
    QProgressBar::chunk {
        background-color: #0078D7;
        width: 10px;
    }
    QFrame#presetFrame {
        border: 1px solid #5A5A5A;
        border-radius: 4px;
    }
"""
