
import warnings
from typing import Sequence
from scipy.stats import f, levene, ncf, ttest_ind
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget, QMessageBox, QComboBox, QTextEdit, QDialog
from PyQt5.QtCore import Qt

def _plot_value(ax, interaction_df, ecological_df, value_fontsize=10):
    length = len(interaction_df.index)
    for i in range(length):
        for j in range(length):
            if not pd.isna(interaction_df.iloc[i, j]):
                num = str(round(interaction_df.iloc[i, j], 2))
                mark = num[-2:] if len(num) == 3 else num[-3:]
                if 'Y' == ecological_df.iloc[i, j]:
                    ax.text(j, i, mark, ha="center", va="center", color="r", fontsize=value_fontsize)
                else:
                    ax.text(j, i, mark, ha="center", va="center", color="k", fontsize=value_fontsize)

class GeoDetector(object):
    def __init__(self, df: pd.DataFrame, y: str, factors: Sequence[str], alpha=0.05):
        self.df = df
        self.y = y
        self.factors = factors
        self.alpha = alpha
        self._check_data(df, y, factors)
        self.factor_df, self.interaction_df, self.ecological_df = None, None, None

    def _check_data(self, df, y, factors):
        for factor in factors:
            if factor not in df.columns:
                raise ValueError(f'Factor [{factor}] is not in data')

        for factor in factors:
            if df[factor].dtype not in ['int64', 'int32', 'int16', 'int8',
                                        'uint64', 'uint32', 'uint16', 'uint8',
                                        'object', 'string']:
                warnings.warn(f"Factor '{factor}' is not of type 'int' or 'str'.")

        if y not in df.columns:
            raise ValueError(f'Factor [{y}] is not in data')

        if any(self.df[y].isnull()):
            raise ValueError("Data has some objects with NULL values.")

    @classmethod
    def _cal_ssw(cls, df: pd.DataFrame, y, factor, extra_factor=None):
        def cal_ssw(df: pd.DataFrame, y):
            length = df.shape[0]
            if length == 1:
                strataVar = 0
                lamda_1st = np.square(df[y].values[0])
                lamda_2nd = df[y].values[0]
            else:
                strataVar = (length - 1) * df[y].var(ddof=1)
                lamda_1st = np.square(df[y].values.mean())
                lamda_2nd = np.sqrt(length) * df[y].values.mean()
            return strataVar, lamda_1st, lamda_2nd

        if extra_factor is None:
            df2 = df[[y, factor]].groupby(factor).apply(cal_ssw, y=y)
        else:
            df2 = df[[y] + list(set([factor, extra_factor]))].groupby([factor, extra_factor]).apply(cal_ssw, y=y)
        df2 = df2.apply(pd.Series)
        df2 = df2.sum()
        strataVarSum, lamda_1st_sum, lamda_2nd_sum = df2.values
        return strataVarSum, lamda_1st_sum, lamda_2nd_sum

    @classmethod
    def _cal_q(cls, df, y, factor, extra_factor=None):
        strataVarSum, lamda_1st_sum, lamda_2nd_sum = cls._cal_ssw(df, y, factor, extra_factor)
        TotalVar = (df.shape[0] - 1) * df[y].var(ddof=1)
        q = 1 - strataVarSum / TotalVar
        return q, lamda_1st_sum, lamda_2nd_sum

    def factor_detector(self):
        self.factor_df = pd.DataFrame(index=["q statistic", "p value"], columns=self.factors, dtype="float32")
        N_var = self.df[self.y].var(ddof=1)
        N_popu = self.df.shape[0]
        for factor in self.factors:
            N_stra = self.df[factor].unique().shape[0]
            q, lamda_1st_sum, lamda_2nd_sum = self._cal_q(self.df, self.y, factor)

            lamda = (lamda_1st_sum - np.square(lamda_2nd_sum) / N_popu) / N_var
            F_value = (N_popu - N_stra) * q / ((N_stra - 1) * (1 - q))
            p_value = ncf.sf(F_value, N_stra - 1, N_popu - N_stra, nc=lamda)

            self.factor_df.loc["q statistic", factor] = q
            self.factor_df.loc["p value", factor] = p_value
        return self.factor_df

    @classmethod
    def _interaction_relationship(cls, df):
        out_df = pd.DataFrame(index=df.index, columns=df.columns)
        length = len(df.index)
        for i in range(length):
            for j in range(i + 1, length):
                factor1, factor2 = df.index[i], df.index[j]
                i_q = df.loc[factor2, factor1]
                q1 = df.loc[factor1, factor1]
                q2 = df.loc[factor2, factor2]

                if (i_q <= q1 and i_q <= q2):
                    outputRls = "Weaken, nonlinear"
                elif (i_q < max(q1, q2) and i_q > min(q1, q2)):
                    outputRls = "Weaken, uni-"
                elif (i_q == (q1 + q2)):
                    outputRls = "Independent"
                elif (i_q > max(q1, q2)):
                    outputRls = "Enhance, bi-"
                elif (i_q > (q1 + q2)):
                    outputRls = "Enhance, nonlinear"

                out_df.loc[factor2, factor1] = outputRls
        return out_df

    def interaction_detector(self, relationship=False):
        self.interaction_df = pd.DataFrame(index=self.factors, columns=self.factors, dtype="float32")
        length = len(self.factors)
        for i in range(length):
            for j in range(i + 1):
                q, _, _ = self._cal_q(self.df, self.y, self.factors[i], self.factors[j])
                self.interaction_df.loc[self.factors[i], self.factors[j]] = q

        if relationship:
            self.interaction_relationship_df = self._interaction_relationship(self.interaction_df)
            return self.interaction_df, self.interaction_relationship_df
        return self.interaction_df

    def ecological_detector(self):
        self.ecological_df = pd.DataFrame(index=self.factors, columns=self.factors, dtype="float32")
        length = len(self.factors)
        for i in range(1, length):
            ssw1, _, _ = self._cal_ssw(self.df, self.y, self.factors[i])
            dfn = self.df[self.factors[i]].notna().sum() - 1
            for j in range(i):
                ssw2, _, _ = self._cal_ssw(self.df, self.y, self.factors[j])
                dfd = self.df[self.factors[j]].notna().sum() - 1
                fval = (dfn * (dfd - 1) * ssw1) / (dfd * (dfn - 1) * ssw2)
                if fval < f.ppf(self.alpha, dfn, dfn):
                    self.ecological_df.loc[self.factors[i], self.factors[j]] = 'Y'
                else:
                    self.ecological_df.loc[self.factors[i], self.factors[j]] = 'N'
        return self.ecological_df

    def risk_detector(self):
        risk_result = dict()
        for factor in self.factors:
            risk_name = self.df.groupby(factor)[self.y].mean()
            strata = np.sort(self.df[factor].unique())
            t_test = np.empty((len(strata), len(strata)))
            t_test.fill(np.nan)
            t_test_strata = pd.DataFrame(t_test, index=strata, columns=strata)
            for i in range(len(strata) - 1):
                for j in range(i + 1, len(strata)):
                    y_i = self.df.loc[self.df[factor] == strata[i], [self.y]]
                    y_j = self.df.loc[self.df[factor] == strata[j], [self.y]]
                    y_i = np.array(y_i).reshape(-1)
                    y_j = np.array(y_j).reshape(-1)
                    # hypothesis testing of variance homogeneity
                    levene_result = levene(y_i, y_j)
                    if levene_result.pvalue < self.alpha:
                        # variance non-homogeneous
                        ttest_result = ttest_ind(y_i, y_j, equal_var=False)
                    else:
                        ttest_result = ttest_ind(y_i, y_j)

                    t_test_strata.iloc[j, i] = ttest_result.pvalue <= self.alpha

            risk_factor = dict(risk=risk_name, ttest_stra=t_test_strata)
            risk_result[factor] = risk_factor
        return risk_result

    def plot(self, tick_fontsize=10, value_fontsize=10, colorbar_fontsize=10, show=True):
        if self.interaction_df is None:
            self.interaction_detector()
        if self.ecological_df is None:
            self.ecological_detector()

        fig, ax = plt.subplots(constrained_layout=True)

        im = ax.imshow(self.interaction_df.values, cmap="YlGnBu", vmin=0, vmax=1)
        _plot_value(ax, self.interaction_df, self.ecological_df, value_fontsize=value_fontsize)

        ax.set_xticks(np.arange(len(self.factors)))
        ax.set_yticks(np.arange(len(self.factors)))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_xticklabels(self.factors, fontsize=tick_fontsize)
        ax.set_yticklabels(self.factors, rotation=45, fontsize=tick_fontsize)
        ax.tick_params(axis='y', pad=0.1)

        colorbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.01, aspect=25, extend="both")
        colorbar.ax.tick_params(labelsize=colorbar_fontsize)

        if show:
            plt.show()
            return ax
        else:
            return ax

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("GeoDetector GUI")
        self.setGeometry(100, 100, 600, 300)

        layout = QVBoxLayout()

        # 用于显示文件路径的标签
        self.label = QLabel("未选择文件", self)
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        # 浏览文件按钮
        self.browse_btn = QPushButton("浏览文件", self)
        self.browse_btn.clicked.connect(self.browse_file)
        layout.addWidget(self.browse_btn)

        # 因子选择下拉菜单
        self.factor_dropdown = QComboBox(self)
        self.factor_dropdown.addItem("选择因子")
        layout.addWidget(self.factor_dropdown)

        # 处理文件按钮
        self.process_btn = QPushButton("处理文件", self)
        self.process_btn.clicked.connect(self.process_file)
        layout.addWidget(self.process_btn)

        # 显示 risk_detector 结果的按钮
        self.risk_btn = QPushButton("显示风险检测结果", self)
        self.risk_btn.clicked.connect(self.show_risk_detector)
        layout.addWidget(self.risk_btn)

        # 用于显示结果的 QTextEdit
        self.result_text = QTextEdit()
        layout.addWidget(self.result_text)

        # 设置中心布局
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "打开文件", "", "CSV 文件 (*.csv);;所有文件 (*)")
        if file_path:
            self.label.setText(file_path)
            self.file_path = file_path
            self.load_data()

    def load_data(self):
        if hasattr(self, 'file_path'):
            try:
                self.df = pd.read_csv(self.file_path)
                self.factor_dropdown.clear()
                self.factor_dropdown.addItem("选择因子")
                self.factor_dropdown.addItems(self.df.columns)
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载数据失败: {str(e)}")
        else:
            QMessageBox.warning(self, "未选择文件", "请先选择一个文件。")

    def process_file(self):
        if hasattr(self, 'file_path'):
            if hasattr(self, 'df'):
                selected_factor = self.factor_dropdown.currentText()
                if selected_factor != "选择因子":
                    y = selected_factor  # 假设目标变量是选中的因子
                    factors = list(self.df.columns)
                    factors.remove(y)

                    try:
                        detector = GeoDetector(self.df, y, factors)
                        factor_df = detector.factor_detector()  # 调用 factor_detector 方法
                        detector.interaction_detector()
                        detector.plot()

                        # 在 GUI 中显示 p 值结果
                        msg = factor_df.to_string()
                        self.result_text.setText(msg)  # 在 QTextEdit 中显示因子检测结果

                    except Exception as e:
                        QMessageBox.critical(self, "错误", f"处理数据失败: {str(e)}")
                else:
                    QMessageBox.warning(self, "未选择因子", "请先选择一个因子。")
            else:
                QMessageBox.warning(self, "未加载数据", "请先加载数据。")
        else:
            QMessageBox.warning(self, "未选择文件", "请先选择一个文件。")

    def show_risk_detector(self):
        if hasattr(self, 'file_path'):
            if hasattr(self, 'df'):
                selected_factor = self.factor_dropdown.currentText()
                if selected_factor != "选择因子":
                    y = selected_factor  # 假设目标变量是选中的因子
                    factors = list(self.df.columns)
                    factors.remove(y)

                    try:
                        detector = GeoDetector(self.df, y, factors)
                        risk_results = detector.risk_detector()

                        # 创建一个新窗口来显示结果
                        dialog = QDialog(self)
                        dialog.setWindowTitle("风险检测结果")
                        dialog.setGeometry(100, 100, 800, 600)

                        layout = QVBoxLayout(dialog)

                        # 使用 QTextEdit 显示结果
                        text_edit = QTextEdit(dialog)
                        text_edit.setReadOnly(True)

                        risk_msg = ""
                        for factor, result in risk_results.items():
                            risk_msg += f"Factor: {factor}\nRisk:\n{result['risk'].to_string()}\n"
                            risk_msg += f"T-Test Results:\n{result['ttest_stra'].to_string()}\n\n"

                        text_edit.setText(risk_msg)
                        layout.addWidget(text_edit)

                        dialog.setLayout(layout)
                        dialog.exec_()

                    except Exception as e:
                        QMessageBox.critical(self, "错误", f"风险检测失败: {str(e)}")
                else:
                    QMessageBox.warning(self, "未选择因子", "请先选择一个因子。")
            else:
                QMessageBox.warning(self, "未加载数据", "请先加载数据。")
        else:
            QMessageBox.warning(self, "未选择文件", "请先选择一个文件。")

# 主函数
def main():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
