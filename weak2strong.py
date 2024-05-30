from load_data import get_data
from load_model import get_model, step_forward
import numpy as np
from w2s_utils import get_layer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

norm_prompt_path = './exp_data/normal_prompt.csv'
jailbreak_prompt_path = './exp_data/malicious_prompt.csv'
malicious_prompt_path = './exp_data/jailbreak_prompt.csv'


def load_exp_data(shuffle_seed=None):
    normal_inputs = get_data(norm_prompt_path, shuffle_seed)
    malicious_inputs = get_data(jailbreak_prompt_path, shuffle_seed)
    jailbreak_inputs = get_data(malicious_prompt_path, shuffle_seed)
    return normal_inputs, malicious_inputs, jailbreak_inputs


class Weak2StrongClassifier:
    def __init__(self, return_dict=False):
        self.return_dict=return_dict

    @staticmethod
    def _process_data(forward_info):
        features = []
        labels = []
        for key, value in forward_info.items():
            for hidden_state in value["hidden_states"]:
                features.append(hidden_state.flatten())
                labels.append(value["label"])

        features = np.array(features)
        labels = np.array(labels)
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
        return X_train, X_test, y_train, y_test

    def svm(self, forward_info):
        X_train, X_test, y_train, y_test = self._process_data(forward_info)
        svm_model = SVC(kernel='linear')
        svm_model.fit(X_train, y_train)
        y_pred = svm_model.predict(X_test)
        report = None
        if not self.return_dict:
            print("SVM Test Classification Report:")
            print(classification_report(y_test, y_pred))
        else:
            report = classification_report(y_test, y_pred)
        return X_test, y_pred, report

    def mlp(self, forward_info):
        X_train, X_test, y_train, y_test = self._process_data(forward_info)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, alpha=0.01,
                            solver='adam', verbose=0, random_state=42,
                            learning_rate_init=.01)

        mlp.fit(X_train_scaled, y_train)
        y_pred = mlp.predict(X_test_scaled)
        report = None
        if not self.return_dict:
            print("SVM Test Classification Report:")
            print(classification_report(y_test, y_pred))
        else:
            report = classification_report(y_test, y_pred)
        return X_test, y_pred, report


class Weak2StrongExplanation:
    def __init__(self, model_name, layer_nums=32):
        self.model, self.tokenizer = get_model(model_name)
        self.layer_sums = layer_nums + 1
        self.forward_info = {}

    def get_forward_info(self, inputs_dataset):
        offset = len(self.forward_info)
        for _, i in enumerate(inputs_dataset):
            list_hs, tl_pair = step_forward(self.model, self.tokenizer, i)
            last_hs = [hs[:, -1, :] for hs in list_hs]
            self.forward_info[_ + offset] = {"hidden_states": last_hs, "top-value_pair": tl_pair, "label": 0}

    def explain(self, dataset_list, classify_list=None):
        if classify_list is None:
            classify_list = ["svm", "mlp"]
        forward_info = {}
        for dataset in dataset_list:
            self.get_forward_info(dataset)

        classifier = Weak2StrongClassifier(return_dict=True)

        rep_dict = {}
        if "svm" in classify_list:
            rep_dict["svm"] = {}
            for _ in range(0, self.layer_sums):
                x, y, rep = classifier.svm(get_layer(self.forward_info, _))
                print(rep)
                rep_dict["svm"][_] = rep

        if "mlp" in classify_list:
            rep_dict["mlp"] = {}
            for _ in range(0, self.layer_sums):
                x, y, rep = classifier.mlp(get_layer(self.forward_info, _))
                print(rep)
                rep_dict["mlp"][_] = rep
