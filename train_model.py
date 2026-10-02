"""Etapes 1 et 2 : entrainer un modele simple sur iris et le sauvegarder avec joblib.

Lancer : python train_model.py
Produit : model.joblib dans le meme dossier que ce script.
"""
from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Le modele est ecrit a cote du script, pas dans le dossier courant :
# comme ca, peu importe d'ou on lance la commande, on sait ou il atterrit.
MODEL_PATH = Path(__file__).with_name("model.joblib")


def main() -> None:
    # iris : 150 fleurs, 4 mesures en cm (longueur/largeur du sepale et du petale),
    # 3 especes a predire (setosa, versicolor, virginica).
    iris = load_iris()
    X, y = iris.data, iris.target

    # On garde 20 % des donnees de cote pour mesurer le modele sur des fleurs
    # qu'il n'a jamais vues. stratify=y garde la meme proportion des 3 especes
    # dans le train et dans le test. random_state fige le tirage : meme resultat
    # a chaque lancement.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Regression logistique : simple, rapide, suffisante sur iris.
    # max_iter=1000 evite l'avertissement "n'a pas converge" du solveur.
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Accuracy sur le jeu de test : {accuracy:.3f}")

    # On sauvegarde le modele ET les noms des classes ensemble : l'API pourra
    # renvoyer "setosa" au lieu de 0 sans recoder la liste a la main.
    joblib.dump({"model": model, "target_names": list(iris.target_names)}, MODEL_PATH)
    print(f"Modele sauvegarde dans {MODEL_PATH}")


if __name__ == "__main__":
    main()
