from pathlib import Path
from urllib.request import urlretrieve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from threadpoolctl import threadpool_limits
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
SEED = 42
NUM = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
COLS = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
CAT = [c for c in COLS[:-1] if c not in NUM]


def read_data(name):
    path = ROOT / name
    if not path.exists():
        urlretrieve('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/' + name, path)
    df = pd.read_csv(path, names=COLS, skipinitialspace=True, comment='|', na_values='?')
    y = df.pop('income').str.strip().str.rstrip('.').map({'<=50K': 0, '>50K': 1})
    if y.isna().any():
        raise ValueError('Неизвестная метка класса')
    return df, y


def prepare():
    x, y = read_data('adult.data')
    xt, yt = read_data('adult.test')
    xtr, xv, ytr, yv = train_test_split(x, y, test_size=0.2, stratify=y, random_state=SEED)
    prep = ColumnTransformer([
        ('numeric', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), NUM),
        ('category', make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore')), CAT)
    ])
    print('Размеры выборок:', len(ytr), len(yv), len(yt))
    return prep.fit_transform(xtr), prep.transform(xv), prep.transform(xt), ytr, yv, yt


def create_model(name, params):
    if name == 'Perceptron':
        return Perceptron(penalty='l2', max_iter=300, random_state=SEED, **params)
    return MLPClassifier(hidden_layer_sizes=(32,), max_iter=200, random_state=SEED, **params)


def main():
    OUT.mkdir(exist_ok=True)
    xtr, xv, xt, ytr, yv, yt = prepare()
    configs = {
        'Perceptron': ({'eta0': .01, 'alpha': .0001}, {'eta0': [.001, .01, .1], 'alpha': [.0001, .001, .01]}),
        'MLPClassifier': ({'learning_rate_init': .001, 'alpha': .0001, 'solver': 'adam'}, {'learning_rate_init': [.0001, .001, .01], 'alpha': [.0001, .001, .01], 'solver': ['adam', 'sgd', 'lbfgs']})
    }
    rows, summary = [], []
    for name, (base, grids) in configs.items():
        model = create_model(name, base).fit(xtr, ytr)
        summary.append(dict(model=name, stage='baseline', train=accuracy_score(ytr, model.predict(xtr)), validation=accuracy_score(yv, model.predict(xv)), test=accuracy_score(yt, model.predict(xt)), params=str(base)))
        current = base.copy()
        for param, values in grids.items():
            trials = []
            for value in values:
                settings = dict(current, **{param: value})
                candidate = create_model(name, settings).fit(xtr, ytr)
                score = accuracy_score(yv, candidate.predict(xv))
                trials.append((score, value))
                rows.append(dict(model=name, parameter=param, value=str(value), validation=score, settings=str(settings)))
                print(name, param, value, round(score, 4), flush=True)
            current[param] = max(trials, key=lambda t: t[0])[1]
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot([str(v) for _, v in trials], [s for s, _ in trials], marker='o')
            ax.set(xlabel=param, ylabel='Accuracy (validation)', title=name + ': ' + param, ylim=(0, 1))
            ax.grid(alpha=.25)
            fig.tight_layout()
            fig.savefig(OUT / f'{name}_{param}.png', dpi=150)
            plt.close(fig)
        model = create_model(name, current).fit(xtr, ytr)
        pred = model.predict(xt)
        summary.append(dict(model=name, stage='tuned', train=accuracy_score(ytr, model.predict(xtr)), validation=accuracy_score(yv, model.predict(xv)), test=accuracy_score(yt, pred), params=str(current)))
        ConfusionMatrixDisplay.from_predictions(yt, pred, display_labels=['<=50K', '>50K'])
        plt.title(name)
        plt.tight_layout()
        plt.savefig(OUT / f'{name}_confusion.png', dpi=150)
        plt.close()
    pd.DataFrame(rows).to_csv(OUT / 'experiments.csv', index=False)
    table = pd.DataFrame(summary)
    table.to_csv(OUT / 'summary.csv', index=False)
    print(table.to_string(index=False))
    report = ['# Лабораторная работа №4. Вариант 2 — Adult', '', 'Цель: сравнить Perceptron и MLPClassifier при прогнозировании дохода.', '', 'Обработка: исходный тест UCI сохранён; 20% обучающего файла выделено для валидации. Пропуски заполнены, числовые признаки стандартизированы, категориальные закодированы OneHotEncoder. Преобразования обучены только на тренировочных данных.', '', 'Подбор параметров последовательный: каждый следующий параметр исследуется при уже выбранных предыдущих. Критерий — accuracy на валидации. Perceptron использует L2-регуляризацию; solver исследуется только для MLP.', '', '## Результаты', '', '```', table.to_string(index=False), '```', '', '## Эксперименты']
    for (name, param), group in pd.DataFrame(rows).groupby(['model','parameter'], sort=False):
        best = group.loc[group.validation.idxmax()]
        report += ['', f'### {name}: {param}', '', '```', group[['value','validation']].to_string(index=False), '```', '', f'Лучшее значение на данном этапе: {best.value}, accuracy = {best.validation:.4f}.', '', f'![График]({name}_{param}.png)']
    report += ['', '## Выводы']
    for name in configs:
        a, b = table[table.model == name].test.tolist()
        report += [f'- {name}: исходная точность {a:.4f}, после подбора {b:.4f}; изменение {b-a:+.4f}.', f'  ![Матрица ошибок]({name}_confusion.png)']
    winner = table[table.stage == 'tuned'].sort_values('validation', ascending=False).iloc[0]
    report += ['', f'По валидации предпочтительна модель {winner.model}. Её точность на тесте: {winner.test:.4f}. Последовательный подбор не гарантирует глобального оптимума.']
    (OUT / 'report.md').write_text('\n'.join(report), encoding='utf-8')


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
