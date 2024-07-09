import joblib

class Model:
    model = None

    def saveModel(self, filePath):
        joblib.dump(self.model, filePath)

    def loadModel(self, filePath):
        self.model = joblib.load(filePath)