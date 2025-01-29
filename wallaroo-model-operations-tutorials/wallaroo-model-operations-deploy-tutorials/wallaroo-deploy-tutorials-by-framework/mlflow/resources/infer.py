def load_model(path):
    import statsmodels.iolib.api as smio

    return smio.load_pickle(path)


def _load_pyfunc(data_path):
    return SarimaxWrapper(load_model(data_path))


class SarimaxWrapper:
    def __init__(self, model) -> "SarimaxWrapper":
        self.model = model

    def predict(self, dataframe):
        return self.model.forecast(steps=len(dataframe), exog=dataframe)