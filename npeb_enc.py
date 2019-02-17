from kw_mle import KWMLE

def npeb_enc(cat_features, target):
    train = cat_features.join(target)

    target_grouped = train.groupby(feature)[target]

    counts_grouped = target_grouped.count().values.flatten()
    means_grouped = target_grouped.mean().values.flatten()
    stds_grouped = target_grouped.std().values.flatten() / np.sqrt(counts_grouped)

    kw_mle = KWMLE(means_grouped, stds=stds_grouped)
    kw_mle.fit()

    enc = kw_mle.prediction(means_grouped, stds=stds_grouped)
    enc_dict = dict(zip(target_grouped.mean().index, enc))
    X_test[feature + '_enc'] = X_test[feature].map(enc_dict)