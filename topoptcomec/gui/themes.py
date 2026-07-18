# topoptcomec/ui/themes.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Stylesheets for the Light and Dark themes.
#
# Design: "technical workstation" aesthetic.
# - Sans-serif (Segoe UI / system) for UI text
# - JetBrains Mono for numeric inputs and results
# - Neutral grey surface layers, subtle borders, one accent color

# Light theme
LIGHT_THEME_STYLESHEET = """
    /* General Widget Styling */
    QWidget {
        background-color: #F0F0F0;
        color: #1A1A1A;
        font-family: "Segoe UI", "Inter", "Noto Sans", sans-serif;
    }
    QMainWindow {
        background-color: #F0F0F0;
    }
    QStatusBar {
        background-color: #E8E8E8;
        color: #666666;
        border-top: 1px solid #C8C8C8;
    }
    QToolTip {
        background-color: #FFFFFF;
        color: #1A1A1A;
        border: 1px solid #BFBFBF;
        padding: 2px 5px;
    }

    /* Input Widgets */
    QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {
        background-color: #FFFFFF;
        border: 1px solid #BFBFBF;
        border-radius: 4px;
        padding: 0px 4px;
        min-height: 18px;
        font-family: "JetBrains Mono", "Cascadia Mono", monospace;
        selection-background-color: #2F80ED;
        selection-color: #FFFFFF;
    }
    QComboBox, QLineEdit { min-height: 22px; }
    QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover, QLineEdit:hover {
        border: 1px solid #9A9A9A;
    }
    QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QLineEdit:focus {
        border: 1px solid #2F80ED;
    }
    QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled, QLineEdit:disabled {
        background-color: #E8E8E8;
        color: #9A9A9A;
        border: 1px solid #C8C8C8;
    }
    QSpinBox::up-button, QDoubleSpinBox::up-button,
    QSpinBox::down-button, QDoubleSpinBox::down-button {
        width: 16px;
        height: 6px;
        border: none;
        background: transparent;
    }
    QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
    QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
        background-color: #E0E0E0;
    }
    QComboBox::drop-down {
        border: none;
        width: 18px;
    }
    QComboBox QAbstractItemView {
        background-color: #FFFFFF;
        border: 1px solid #BFBFBF;
        selection-background-color: #DCE9FB;
        selection-color: #1A1A1A;
        outline: none;
    }

    /* Buttons */
    QPushButton, QToolButton {
        background-color: transparent;
        border: 1px solid #BFBFBF;
        border-radius: 4px;
        padding: 0px 8px;
        min-height: 22px;
    }
    QPushButton:hover, QToolButton:hover {
        background-color: #E0E0E0;
        border: 1px solid #9A9A9A;
    }
    QPushButton:pressed, QToolButton:pressed {
        background-color: #D0D0D0;
    }
    QPushButton:disabled, QToolButton:disabled {
        color: #9A9A9A;
        border: 1px solid #C8C8C8;
        background-color: transparent;
    }
    QPushButton:checked, QToolButton:checked {
        background-color: #DDDDDD;
        border: 1px solid #2F80ED;
    }
    QPushButton:flat, QToolButton:flat {
        border: none;
    }

    /* Footer icon buttons (binarize / save): no padding so the icon fills the button */
    QPushButton#footerIconButton, QToolButton#footerIconButton {
        padding: 0px;
    }

    /* Primary action */
    QPushButton#primaryButton {
        background-color: #2F80ED;
        border: 1px solid #2F80ED;
        border-radius: 4px;
        color: #FFFFFF;
        font-weight: bold;
        padding: 4px 12px;
    }
    QPushButton#primaryButton:hover {
        background-color: #4A93F1;
        border: 1px solid #4A93F1;
    }
    QPushButton#primaryButton:pressed {
        background-color: #1F6FD8;
        border: 1px solid #1F6FD8;
    }
    QPushButton#primaryButton:disabled {
        background-color: #B8B8B8;
        border: 1px solid #B8B8B8;
        color: #E8E8E8;
    }

    /* Destructive action */
    QPushButton#dangerButton {
        background-color: #D64545;
        border: 1px solid #D64545;
        border-radius: 4px;
        color: #FFFFFF;
        font-weight: bold;
        padding: 4px 12px;
    }
    QPushButton#dangerButton:hover {
        background-color: #E05A5A;
        border: 1px solid #E05A5A;
    }
    QPushButton#dangerButton:pressed {
        background-color: #B93A3A;
        border: 1px solid #B93A3A;
    }
    QPushButton#dangerButton:disabled {
        background-color: #C8A8A8;
        border: 1px solid #C8A8A8;
        color: #EFE0E0;
    }

    /* CheckBox */
    QCheckBox {
        spacing: 6px;
        background: transparent;
    }
    QCheckBox::indicator {
        width: 13px;
        height: 13px;
        border: 1px solid #9A9A9A;
        border-radius: 3px;
        background-color: #FFFFFF;
    }
    QCheckBox::indicator:hover {
        border: 1px solid #2F80ED;
    }
    QCheckBox::indicator:checked {
        background-color: #2F80ED;
        border: 1px solid #2F80ED;
    }
    QCheckBox::indicator:disabled {
        background-color: #E8E8E8;
        border: 1px solid #C8C8C8;
    }

    /* Collapsible Section */
    #collapsibleTitleBar {
        background-color: #E6E6E6;
        border: none;
        border-radius: 4px;
    }
    #collapsibleTitleBar:hover {
        background-color: #DCDCDC;
    }
    #collapsibleTitleLabel {
        font-weight: bold;
        background: transparent;
    }
    QPushButton#collapsibleToggle, QPushButton#collapsibleEye {
        border: none;
        background: transparent;
        padding: 0px;
    }
    QPushButton#collapsibleToggle:hover, QPushButton#collapsibleEye:hover {
        background-color: #D0D0D0;
        border-radius: 3px;
    }
    #collapsibleContent {
        border: none;
        padding: 1px;
        background-color: #F0F0F0;
    }

    /* Repeated entity cards (regions, forces, supports, materials) */
    #entityCard {
        background-color: #E8E8E8;
        border: 1px solid #D0D0D0;
        border-radius: 4px;
    }
    #entityCard QLabel {
        background: transparent;
    }

    /* Force section labels */
    QLabel#inputForcesLabel {
        color: #C0392B;
        font-weight: bold;
        background: transparent;
    }
    QLabel#outputForcesLabel {
        color: #2F80ED;
        font-weight: bold;
        background: transparent;
    }

    /* Analysis result badges */
    QLabel#resultValue {
        font-family: "JetBrains Mono", "Cascadia Mono", monospace;
        background: transparent;
    }
    QLabel#resultValue[resultState="ok"] { color: #1E8E3E; font-weight: bold; }
    QLabel#resultValue[resultState="bad"] { color: #C0392B; font-weight: bold; }

    /* Menus */
    QMenu {
        background-color: #FFFFFF;
        border: 1px solid #BFBFBF;
        padding: 3px;
    }
    QMenu::item {
        padding: 4px 18px 4px 10px;
        border-radius: 3px;
    }
    QMenu::item:selected {
        background-color: #DCE9FB;
        color: #1A1A1A;
    }
    QMenu::item:disabled {
        color: #9A9A9A;
    }

    /* Scrollbars */
    QScrollBar:vertical {
        background: transparent;
        width: 10px;
        margin: 2px;
    }
    QScrollBar::handle:vertical {
        background-color: #BFBFBF;
        border-radius: 5px;
        min-height: 22px;
    }
    QScrollBar::handle:vertical:hover {
        background-color: #9A9A9A;
    }
    QScrollBar:horizontal {
        background: transparent;
        height: 10px;
        margin: 2px;
    }
    QScrollBar::handle:horizontal {
        background-color: #BFBFBF;
        border-radius: 5px;
        min-width: 24px;
    }
    QScrollBar::handle:horizontal:hover {
        background-color: #9A9A9A;
    }
    QScrollBar::add-line, QScrollBar::sub-line {
        height: 0px;
        width: 0px;
    }
    QScrollBar::add-page, QScrollBar::sub-page {
        background: transparent;
    }

    /* Other Widgets */
    QSplitter::handle {
        background-color: #C8C8C8;
    }
    QSplitter::handle:hover {
        background-color: #9A9A9A;
    }
    QProgressBar {
        background-color: #E8E8E8;
        border: 1px solid #BFBFBF;
        border-radius: 4px;
        text-align: center;
        color: #1A1A1A;
    }
    QProgressBar::chunk {
        background-color: #2F80ED;
        border-radius: 3px;
    }
    QScrollArea {
        border: none;
    }
    QFrame#presetFrame {
        background-color: #E8E8E8;
        border: 1px solid #D0D0D0;
        border-radius: 4px;
    }
    QFrame#presetFrame QLabel {
        background: transparent;
    }
"""

# Dark theme
DARK_THEME_STYLESHEET = """
    /* General Widget Styling */
    QWidget {
        background-color: #050505;
        color: #E8E8E8;
        font-family: "Segoe UI", "Inter", "Noto Sans", sans-serif;
    }
    QMainWindow {
        background-color: #050505;
    }
    QStatusBar {
        background-color: #0A0A0A;
        color: #9A9A9A;
        border-top: 1px solid #2A2A2A;
    }
    QToolTip {
        background-color: #161616;
        color: #E8E8E8;
        border: 1px solid #2A2A2A;
        padding: 2px 5px;
    }

    /* Input Widgets */
    QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {
        background-color: #0A0A0A;
        border: 1px solid #2A2A2A;
        border-radius: 4px;
        padding: 0px 4px;
        min-height: 18px;
        color: #E8E8E8;
        font-family: "JetBrains Mono", "Cascadia Mono", monospace;
        selection-background-color: #4D9FFF;
        selection-color: #050505;
    }
    QComboBox, QLineEdit { min-height: 22px; }
    QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover, QLineEdit:hover {
        border: 1px solid #383838;
    }
    QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QLineEdit:focus {
        border: 1px solid #4D9FFF;
    }
    QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled, QLineEdit:disabled {
        background-color: #0E0E0E;
        color: #6E6E6E;
        border: 1px solid #1A1A1A;
    }
    QSpinBox::up-button, QDoubleSpinBox::up-button,
    QSpinBox::down-button, QDoubleSpinBox::down-button {
        width: 16px;
        height: 6px;
        border: none;
        background: transparent;
    }
    QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
    QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
        background-color: #161616;
    }
    QComboBox::drop-down {
        border: none;
        width: 18px;
    }
    QComboBox QAbstractItemView {
        background-color: #0E0E0E;
        border: 1px solid #2A2A2A;
        selection-background-color: #262626;
        selection-color: #E8E8E8;
        outline: none;
    }

    /* Buttons */
    QPushButton, QToolButton {
        background-color: transparent;
        border: 1px solid #2A2A2A;
        border-radius: 4px;
        padding: 0px 8px;
        min-height: 22px;
    }
    QPushButton:hover, QToolButton:hover {
        background-color: #161616;
        border: 1px solid #383838;
    }
    QPushButton:pressed, QToolButton:pressed {
        background-color: #1A1A1A;
    }
    QPushButton:disabled, QToolButton:disabled {
        color: #6E6E6E;
        border: 1px solid #1A1A1A;
        background-color: transparent;
    }
    QPushButton:checked, QToolButton:checked {
        background-color: #262626;
        border: 1px solid #4D9FFF;
    }
    QPushButton:flat, QToolButton:flat {
        border: none;
    }

    /* Footer icon buttons (binarize / save): no padding so the icon fills the button */
    QPushButton#footerIconButton, QToolButton#footerIconButton {
        padding: 0px;
    }

    /* Primary action */
    QPushButton#primaryButton {
        background-color: #4D9FFF;
        border: 1px solid #4D9FFF;
        border-radius: 4px;
        color: #050505;
        font-weight: bold;
        padding: 4px 12px;
    }
    QPushButton#primaryButton:hover {
        background-color: #6BAFFF;
        border: 1px solid #6BAFFF;
    }
    QPushButton#primaryButton:pressed {
        background-color: #3587EC;
        border: 1px solid #3587EC;
    }
    QPushButton#primaryButton:disabled {
        background-color: #2A2A2A;
        border: 1px solid #2A2A2A;
        color: #6E6E6E;
    }

    /* Destructive action */
    QPushButton#dangerButton {
        background-color: #C0392B;
        border: 1px solid #C0392B;
        border-radius: 4px;
        color: #FFFFFF;
        font-weight: bold;
        padding: 4px 12px;
    }
    QPushButton#dangerButton:hover {
        background-color: #D64545;
        border: 1px solid #D64545;
    }
    QPushButton#dangerButton:pressed {
        background-color: #A93226;
        border: 1px solid #A93226;
    }
    QPushButton#dangerButton:disabled {
        background-color: #4A2A26;
        border: 1px solid #4A2A26;
        color: #9A9A9A;
    }

    /* CheckBox */
    QCheckBox {
        spacing: 6px;
        background: transparent;
    }
    QCheckBox::indicator {
        width: 13px;
        height: 13px;
        border: 1px solid #383838;
        border-radius: 3px;
        background-color: #0A0A0A;
    }
    QCheckBox::indicator:hover {
        border: 1px solid #4D9FFF;
    }
    QCheckBox::indicator:checked {
        background-color: #4D9FFF;
        border: 1px solid #4D9FFF;
    }
    QCheckBox::indicator:disabled {
        background-color: #0E0E0E;
        border: 1px solid #1A1A1A;
    }

    /* Collapsible Section */
    #collapsibleTitleBar {
        background-color: #262626;
        border: none;
        border-radius: 4px;
    }
    #collapsibleTitleBar:hover {
        background-color: #2E2E2E;
    }
    #collapsibleTitleLabel {
        font-weight: bold;
        background: transparent;
    }
    QPushButton#collapsibleToggle, QPushButton#collapsibleEye {
        border: none;
        background: transparent;
        padding: 0px;
    }
    QPushButton#collapsibleToggle:hover, QPushButton#collapsibleEye:hover {
        background-color: #222222;
        border-radius: 3px;
    }
    #collapsibleContent {
        border: none;
        padding: 1px;
        background-color: #050505;
    }

    /* Repeated entity cards (regions, forces, supports, materials) */
    #entityCard {
        background-color: #0E0E0E;
        border: 1px solid #1A1A1A;
        border-radius: 4px;
    }
    #entityCard QLabel {
        background: transparent;
    }

    /* Force section labels */
    QLabel#inputForcesLabel {
        color: #E06C5D;
        font-weight: bold;
        background: transparent;
    }
    QLabel#outputForcesLabel {
        color: #4D9FFF;
        font-weight: bold;
        background: transparent;
    }

    /* Analysis result badges */
    QLabel#resultValue {
        font-family: "JetBrains Mono", "Cascadia Mono", monospace;
        background: transparent;
    }
    QLabel#resultValue[resultState="ok"] { color: #4CAF6E; font-weight: bold; }
    QLabel#resultValue[resultState="bad"] { color: #E06C5D; font-weight: bold; }

    /* Menus */
    QMenu {
        background-color: #0E0E0E;
        border: 1px solid #2A2A2A;
        padding: 3px;
    }
    QMenu::item {
        padding: 4px 18px 4px 10px;
        border-radius: 3px;
    }
    QMenu::item:selected {
        background-color: #262626;
        color: #E8E8E8;
    }
    QMenu::item:disabled {
        color: #6E6E6E;
    }

    /* Scrollbars */
    QScrollBar:vertical {
        background: transparent;
        width: 10px;
        margin: 2px;
    }
    QScrollBar::handle:vertical {
        background-color: #2A2A2A;
        border-radius: 5px;
        min-height: 22px;
    }
    QScrollBar::handle:vertical:hover {
        background-color: #383838;
    }
    QScrollBar:horizontal {
        background: transparent;
        height: 10px;
        margin: 2px;
    }
    QScrollBar::handle:horizontal {
        background-color: #2A2A2A;
        border-radius: 5px;
        min-width: 24px;
    }
    QScrollBar::handle:horizontal:hover {
        background-color: #383838;
    }
    QScrollBar::add-line, QScrollBar::sub-line {
        height: 0px;
        width: 0px;
    }
    QScrollBar::add-page, QScrollBar::sub-page {
        background: transparent;
    }

    /* Other Widgets */
    QSplitter::handle {
        background-color: #1A1A1A;
    }
    QSplitter::handle:hover {
        background-color: #383838;
    }
    QProgressBar {
        background-color: #0E0E0E;
        border: 1px solid #2A2A2A;
        border-radius: 4px;
        text-align: center;
        color: #E8E8E8;
    }
    QProgressBar::chunk {
        background-color: #4D9FFF;
        border-radius: 3px;
    }
    QScrollArea {
        border: none;
    }
    QFrame#presetFrame {
        background-color: #1E1E1E;
        border: 1px solid #2A2A2A;
        border-radius: 4px;
    }
    QFrame#presetFrame QLabel {
        background: transparent;
    }
"""
