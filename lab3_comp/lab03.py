import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def split_data(X, y, test_size=0.2, seed=42):
    indices = np.random.default_rng(seed).permutation(len(X))
    n_test = int(len(X) * test_size)
    test, train = indices[:n_test], indices[n_test:]
    return X[train], X[test], y[train], y[test]


def load_data(path):
    df = pd.read_csv(path)
    X = df[[f'WAP{i:03d}' for i in range(1, 521)]].to_numpy(dtype=float, copy=True)
    X[X == 100] = -110
    y = df['LONGITUDE'].to_numpy(dtype=float, copy=True)
    return X, y


def make_model(degree=1, alpha=None):
    return Pipeline([
        ('select', SelectKBest(f_regression, k=8)),
        ('scale_input', StandardScaler()),
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('scale_poly', StandardScaler()),
        ('regression', LinearRegression() if alpha is None else Ridge(alpha=alpha))
    ])


def metrics(model, X, y):
    pred = model.predict(X)
    return {'R2': r2_score(y, pred), 'MAE': mean_absolute_error(y, pred),
            'RMSE': np.sqrt(mean_squared_error(y, pred))}


def plot_scores(values, train, test, validation, xlabel, path, log=False):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(values, train, 'o-', label='Train')
    ax.plot(values, validation, '^-', label='Validation')
    ax.plot(values, test, 's-', label='Test')
    if log:
        ax.set_xscale('log')
    else:
        ax.set_xticks(values)
    ax.set(xlabel=xlabel, ylabel='R²', title=f'R² vs {xlabel}')
    ax.grid(True, alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "csv",
    type=Path,
    nargs="?",
    default=Path(__file__).resolve().parent / "trainingData.csv"
    )
    parser.add_argument('--output', type=Path, default=Path('results'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    X, y = load_data(args.csv)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_fit, X_val, y_fit, y_val = split_data(X_train, y_train, .2, 43)
    print(f'Train: {len(X_train)}, test: {len(X_test)}, validation: {len(X_val)} (inside train)')
    baseline = Pipeline([('scale', StandardScaler()), ('regression', LinearRegression())])
    baseline.fit(X_train, y_train)
    print('Linear regression (520 features):', metrics(baseline, X_test, y_test))
    rows = []
    def experiment(degree, alpha=None):
        model = make_model(degree, alpha)
        model.fit(X_fit, y_fit)
        val = model.score(X_val, y_val)
        model.fit(X_train, y_train)
        row = dict(degree=degree, alpha=alpha, train=model.score(X_train, y_train),
                   validation=val, test=model.score(X_test, y_test))
        rows.append(row)
        print(row)
        return row
    degrees = list(range(1, 5))
    poly = [experiment(d) for d in degrees]
    best_degree = max(poly, key=lambda r: r['validation'])['degree']
    plot_scores(degrees, [r['train'] for r in poly], [r['test'] for r in poly],
                [r['validation'] for r in poly], 'Polynomial degree', args.output / 'degree.png')
    alphas = np.logspace(-3, 5, 9)
    ridge = [experiment(best_degree, float(a)) for a in alphas]
    best_alpha = max(ridge, key=lambda r: r['validation'])['alpha']
    plot_scores(alphas, [r['train'] for r in ridge], [r['test'] for r in ridge],
                [r['validation'] for r in ridge], 'Ridge alpha', args.output / 'alpha.png', True)
    final = make_model(best_degree, best_alpha).fit(X_train, y_train)
    print('Best degree:', best_degree, 'Best alpha:', best_alpha)
    print('Final Ridge:', metrics(final, X_test, y_test))
    selected = final.named_steps['select'].get_support(indices=True)
    print('Selected WAP:', [f'WAP{i+1:03d}' for i in selected])
    pd.DataFrame(rows).to_csv(args.output / 'experiments.csv', index=False)


if __name__ == '__main__':
    main()
