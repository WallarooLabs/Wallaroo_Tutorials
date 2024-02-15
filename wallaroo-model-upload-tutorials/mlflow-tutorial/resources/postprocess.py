def _load_pyfunc(data_path):
    return PostProcessor()


class PostProcessor:
    def predict(self, dataframe):
        from scipy.stats import zscore

        return dataframe.apply(zscore)