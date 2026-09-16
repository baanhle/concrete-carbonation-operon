"""
Carbonation Depth Predictor -- Eq.(1)-only version (Tree-Operon removed).
Six raw mix-design/exposure inputs, one closed-form equation, no routing.
"""
import sys

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFormLayout, QVBoxLayout,
    QDoubleSpinBox, QPushButton, QLabel, QGroupBox,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

import eq1_only_model as model


class PredictorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Carbonation Depth Predictor — Equation (1)')
        self.setMinimumWidth(560)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(12)
        root.setContentsMargins(16, 16, 16, 16)

        title = QLabel('Concrete Carbonation Depth Predictor')
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        root.addWidget(title)

        subtitle = QLabel('Single Operon Global Equation, Eq. (1) — test RMSE 6.884 mm, MAE 4.918 mm')
        subtitle.setStyleSheet('color: #555;')
        root.addWidget(subtitle)

        input_box = QGroupBox('Mix design and exposure inputs')
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        input_box.setLayout(form)

        def make_spin(lo, hi, default, step, decimals=2):
            sb = QDoubleSpinBox()
            sb.setRange(lo, hi)
            sb.setDecimals(decimals)
            sb.setSingleStep(step)
            sb.setValue(default)
            return sb

        self.sb_wc = make_spin(0.0, 3.0, 0.75, 0.01)
        self.sb_wb = make_spin(0.0, 1.5, 0.45, 0.01)
        self.sb_co2 = make_spin(0.0, 100.0, 3.0, 0.1, decimals=2)
        self.sb_rh = make_spin(0.0, 100.0, 70.0, 1.0, decimals=1)
        self.sb_t = make_spin(-10.0, 60.0, 20.0, 0.5, decimals=1)
        self.sb_days = make_spin(1.0, 5000.0, 28.0, 1.0, decimals=1)

        def range_hint(key, fmt='{:g}'):
            lo, hi, _ = model.ENVELOPE[key]
            return f'[{fmt.format(lo)}–{fmt.format(hi)}]'

        form.addRow(f'Water-cement ratio, w/c  (X1, –) {range_hint("w_c", "{:.2f}")}:', self.sb_wc)
        form.addRow(f'Water-binder ratio, w/b  (X2, –) {range_hint("w_b", "{:.2f}")}:', self.sb_wb)
        form.addRow(f'CO₂ concentration  (X4, %) {range_hint("co2", "{:.2f}")}:', self.sb_co2)
        form.addRow(f'Relative humidity, RH  (X5, %) {range_hint("rh", "{:.0f}")}:', self.sb_rh)
        form.addRow(f'Temperature, T  (X6, °C) {range_hint("t", "{:.0f}")}:', self.sb_t)
        form.addRow(f'Exposure duration  (days) {range_hint("exposure_days", "{:.0f}")}:', self.sb_days)

        root.addWidget(input_box)

        self.btn_predict = QPushButton('Predict Carbonation Depth')
        self.btn_predict.setStyleSheet('font-weight: bold; padding: 8px;')
        self.btn_predict.clicked.connect(self.on_predict)
        root.addWidget(self.btn_predict)

        result_box = QGroupBox('Prediction')
        result_layout = QVBoxLayout()
        result_box.setLayout(result_layout)

        self.lbl_depth = QLabel('—')
        depth_font = QFont()
        depth_font.setPointSize(20)
        depth_font.setBold(True)
        self.lbl_depth.setFont(depth_font)

        self.lbl_eq = QLabel('Enter inputs and click Predict.')
        self.lbl_eq.setWordWrap(True)

        self.lbl_warning = QLabel('')
        self.lbl_warning.setWordWrap(True)
        self.lbl_warning.setStyleSheet('color: #8a5a00; background: #fff3cd; padding: 6px; border-radius: 4px;')
        self.lbl_warning.hide()

        result_layout.addWidget(self.lbl_depth)
        result_layout.addWidget(self.lbl_eq)
        result_layout.addWidget(self.lbl_warning)
        root.addWidget(result_box)

        footer = QLabel('Equation (1) is an empirical fit valid within the training envelope; see the '
                         'accompanying manuscript for full derivation and limitations.')
        footer.setWordWrap(True)
        footer.setStyleSheet('color: #777; font-size: 9pt;')
        root.addWidget(footer)

        self.on_predict()

    def on_predict(self):
        result = model.predict(
            w_c=self.sb_wc.value(), w_b=self.sb_wb.value(),
            co2=self.sb_co2.value(), rh=self.sb_rh.value(), t=self.sb_t.value(),
            exposure_days=self.sb_days.value(),
        )
        self.lbl_depth.setText(f"{result['depth_mm']:.2f} mm")
        self.lbl_eq.setText('Predicted via Equation (1), the Single Operon Global Equation.')
        if result['out_of_range']:
            self.lbl_warning.setText('⚠ Extrapolation warning: ' + '; '.join(result['out_of_range']))
            self.lbl_warning.show()
        else:
            self.lbl_warning.hide()


def main():
    app = QApplication(sys.argv)
    win = PredictorWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
