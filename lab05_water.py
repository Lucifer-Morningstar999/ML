from pathlib import Path
import argparse
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

SEED = 42


def load_data(path):
    data = pd.read_csv(path, header=None, na_values=['?', ' ?'], skipinitialspace=True)
    dates = data.iloc[:, 0].astype(str)
    features = data.iloc[:, 1:].apply(pd.to_numeric, errors='raise')
    if features.shape[1] != 38:
        raise ValueError('Ожидается 38 показателей после столбца даты')
    features.columns = [f'P{i+1:02}' for i in range(features.shape[1])]
    missing = int(features.isna().sum().sum())
    filled = SimpleImputer(strategy='median').fit_transform(features)
    return dates, StandardScaler().fit_transform(filled), missing


def assess(X, labels):
    valid = labels != -1
    k = len(np.unique(labels[valid]))
    coverage = float(valid.mean())
    result = dict(clusters=k, coverage=coverage, noise=int((~valid).sum()),
                  silhouette=np.nan, CH=np.nan, DB=np.nan, score=np.nan)
    if 2 <= k < valid.sum():
        a, b = X[valid], labels[valid]
        s = silhouette_score(a, b)
        result.update(silhouette=s, CH=calinski_harabasz_score(a, b),
                      DB=davies_bouldin_score(a, b), score=coverage*(s+1)-1)
    return result


def run(path, out):
    out.mkdir(parents=True, exist_ok=True)
    dates, X, missing = load_data(path)
    print(f'Наблюдений: {len(X)}, признаков: {X.shape[1]}, пропусков: {missing}')
    jobs = []
    for k in range(2, 11):
        jobs.append(('KMeans', f'k={k}', KMeans(n_clusters=k, n_init=20, random_state=SEED)))
    for k, link in itertools.product(range(2, 11), ['ward', 'complete', 'average', 'single']):
        jobs.append(('Agglomerative', f'k={k}, {link}', AgglomerativeClustering(n_clusters=k, linkage=link)))
    # Радиусы DBSCAN выбираются по распределению расстояний до соседей.
    for m in [4, 6, 10]:
        distances = NearestNeighbors(n_neighbors=m).fit(X).kneighbors(X)[0][:, -1]
        for eps in np.unique(np.quantile(distances, [.35, .5, .65, .8, .9, .95])):
            if eps > 0:
                jobs.append(('DBSCAN', f'eps={eps:.4f}, min_samples={m}', DBSCAN(eps=eps, min_samples=m)))
    rows, predictions = [], []
    for name, params, model in jobs:
        labels = model.fit_predict(X)
        predictions.append(labels)
        rows.append(dict(algorithm=name, parameters=params, **assess(X, labels)))
    table = pd.DataFrame(rows)
    table.to_csv(out/'experiments.csv', index=False, encoding='utf-8-sig')
    eligible = table.dropna(subset=['score'])
    if eligible.empty:
        raise RuntimeError('Не получено допустимого разбиения')
    best = eligible.loc[eligible.groupby('algorithm')['score'].idxmax()].sort_values('score', ascending=False)
    best.to_csv(out/'summary.csv', index=False, encoding='utf-8-sig')
    print(best.to_string(index=False))
    winner = best.iloc[0]
    labels = predictions[winner.name]
    pd.DataFrame({'date': dates, 'cluster': labels}).to_csv(out/'clusters.csv', index=False)
    def save(name):
        plt.tight_layout(); plt.savefig(out/name, dpi=150); plt.close()
    km = table[table.algorithm == 'KMeans']
    plt.figure(); plt.plot(range(2,11), km.silhouette, 'o-'); plt.xlabel('Число кластеров'); plt.ylabel('Silhouette'); plt.grid(alpha=.3); save('01_kmeans.png')
    ag = table[table.algorithm == 'Agglomerative']
    plt.figure()
    for link in ['ward','complete','average','single']:
        plt.plot(range(2,11), ag[ag.parameters.str.endswith(', '+link)].silhouette, 'o-', label=link)
    plt.xlabel('Число кластеров'); plt.ylabel('Silhouette'); plt.legend(); save('02_linkage.png')
    db = table[table.algorithm == 'DBSCAN']
    plt.figure()
    for m in [4,6,10]:
        sub = db[db.parameters.str.endswith(f'min_samples={m}')]
        eps = sub.parameters.str.extract(r'eps=([0-9.]+)')[0].astype(float)
        plt.plot(eps, sub.score, 'o-', label=f'min_samples={m}')
    plt.xlabel('eps'); plt.ylabel('Скорректированный Silhouette'); plt.legend(); save('03_dbscan.png')
    plt.figure(); plt.bar(best.algorithm, best.score); plt.ylabel('Итоговая оценка'); save('04_comparison.png')
    coords = PCA(n_components=2).fit_transform(X)
    plt.figure()
    for c in np.unique(labels):
        mask = labels == c
        plt.scatter(*coords[mask].T, s=18, label='Шум' if c == -1 else f'Кластер {c}')
    plt.xlabel('PC1'); plt.ylabel('PC2'); plt.legend(); save('05_pca.png')
    report = ['# Лабораторная работа №5. Вариант 2', '## Цель', 'Выделение групп состояний станции очистки воды методами кластеризации.', '## Подготовка данных', f'Наблюдений: {len(X)}. Признаков: {X.shape[1]}. Пропусков: {missing}. Дата исключена из признаков. Пропуски заполнены медианами, показатели стандартизированы.', '## Эксперименты', 'KMeans: k от 2 до 10; агломеративный метод: те же k и четыре способа объединения; DBSCAN: min_samples 4, 6, 10 и радиусы по квантилям расстояний до соседей.', 'Silhouette и CH максимизируются, DB минимизируется. Для DBSCAN метрики вычисляются без шума. Для общего сравнения используется score = coverage × (silhouette + 1) − 1, чтобы учитывать исключённые точки. Разбиения менее чем на два кластера не оцениваются.', '| Алгоритм | Параметры | Silhouette | CH | DB | Покрытие | Score |', '|---|---|---:|---:|---:|---:|---:|']
    for _, r in best.iterrows():
        report.append(f"| {r.algorithm} | {r.parameters} | {r.silhouette:.4f} | {r.CH:.2f} | {r.DB:.4f} | {r.coverage:.2%} | {r.score:.4f} |")
    for name in ['01_kmeans','02_linkage','03_dbscan','04_comparison','05_pca']:
        report.append(f'\n![{name}]({name}.png)\n')
    report += ['## Вывод', f'По выбранному критерию лучшим оказался {winner.algorithm} ({winner.parameters}), число кластеров: {winner.clusters}, оценка: {winner.score:.4f}.', 'Результат относится к исследованной сетке параметров. Без эталонных меток нельзя подтвердить соответствие кластеров реальным аварийным или штатным режимам. PCA используется только для визуализации.']
    (out/'report.md').write_text('\n\n'.join(report), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path(__file__).with_name('water-treatment.data'))
    parser.add_argument('--out', type=Path, default=Path(__file__).with_name('results_lab05'))
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error('Скачайте water-treatment.data из UCI и положите рядом с программой или укажите --data')
    with threadpool_limits(limits=2):
        run(args.data, args.out)
