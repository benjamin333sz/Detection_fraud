import os
from pathlib import Path
import pandas as pd
import numpy as np

import customtkinter
from tkinter import filedialog, ttk

import matplotlib
matplotlib.use("Agg")  # important pour générer des images sans bloquer l'UI
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix


# =========================
# 1) MOTEUR IA (pipeline)
# =========================

def load_csv_semicolon(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    # Normalisation minimale
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" in df.columns:
        # adapte si format différent
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    return df

def build_boxplot_by_year(data: pd.DataFrame, out_path: Path) -> None:
    # suppose colonnes "date" et "clot"
    df = data.copy()
    df = df.dropna(subset=["date", "clot"])
    df["year"] = df["date"].dt.year

    plt.figure(figsize=(14, 6))
    df.boxplot(column="clot", by="year", figsize=(14, 6))
    plt.suptitle("Box plots des prix de clôture par année")
    plt.xlabel("Année")
    plt.ylabel("Prix de clôture")
    plt.tight_layout()
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close()

def compute_features_and_labels(data_true: pd.DataFrame, data_input: pd.DataFrame) -> pd.DataFrame:
    """
    Retourne df prêt pour les modèles:
      - colonne aberrante (label)
      - ret, vol20, clot_mean20, clot_std20
      - z_clot, is_anomaly_rule_z
      - rel_error, is_anomaly_rule
      - residual
    """
    # Harmoniser et garder les colonnes nécessaires
    df = data_input.copy()

    # check colonnes attendues
    for col in ["date", "clot"]:
        if col not in df.columns:
            raise ValueError(f"Colonne manquante dans data_input: {col}")
        if col not in data_true.columns:
            raise ValueError(f"Colonne manquante dans data_true: {col}")

    # Tri
    df = df.sort_values("date").reset_index(drop=True)
    data_true = data_true.sort_values("date").reset_index(drop=True)

    # Label aberrante (diff de clot)
    # (si les dates sont les mêmes et alignées; sinon il faudra merge sur date)
    if len(df) != len(data_true):
        # Alignement plus robuste : merge sur date
        merged = df.merge(data_true[["date", "clot"]].rename(columns={"clot": "clot_true"}),
                          on="date", how="inner")
        merged["aberrante"] = merged["clot_true"] != merged["clot"]
        df = merged.drop(columns=["clot_true"])
    else:
        df["aberrante"] = data_true["clot"].values != df["clot"].values

    # Features
    df["ret"] = df["clot"].pct_change()
    df["vol20"] = df["ret"].rolling(20).std()

    df["clot_mean20"] = df["clot"].rolling(20).mean()
    df["clot_std20"] = df["clot"].rolling(20).std()

    # sous-df sans NA sur features clés
    df2 = df.dropna(subset=["ret", "vol20", "aberrante", "clot_mean20", "clot_std20"]).copy()

    # Z-score
    df2["z_clot"] = np.abs((df2["clot"] - df2["clot_mean20"]) / df2["clot_std20"].replace(0, np.nan))
    df2["z_clot"] = df2["z_clot"].fillna(0)
    df2["is_anomaly_rule_z"] = df2["z_clot"] > 6

    # Erreur relative locale
    df2["rel_error_prev"] = np.abs(df2["clot"] - df2["clot"].shift(1)) / df2["clot"].shift(1)
    df2["rel_error_next"] = np.abs(df2["clot"] - df2["clot"].shift(-1)) / df2["clot"]
    df2["rel_error"] = df2[["rel_error_prev", "rel_error_next"]].min(axis=1)
    df2["is_anomaly_rule"] = df2["rel_error"] > 0.15

    # Résidu naïf
    df2["clot_pred"] = df2["clot"].shift(1)
    df2["residual"] = np.abs(df2["clot"] - df2["clot_pred"])

    return df2

def run_isolation_forest_on_rel_error(df: pd.DataFrame, expected_anomalies: int = 10) -> pd.DataFrame:
    """
    Entraîne Isolation Forest sur rel_error (standardisé) et ajoute:
      - is_anomaly_iforest
      - iforest_score
    """
    dfx = df.dropna(subset=["rel_error"]).copy()
    X = dfx[["rel_error"]].values

    # Standardisation
    mu = X.mean(axis=0)
    std = X.std(axis=0, ddof=0)
    std = np.where(std == 0, 1e-12, std)
    Xn = (X - mu) / std

    contamination = expected_anomalies / max(len(dfx), 1)

    iso = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=42
    )
    iso.fit(Xn)
    pred = iso.predict(Xn)  # -1 anomalie, 1 normal
    dfx["is_anomaly_iforest"] = (pred == -1)
    dfx["iforest_score"] = iso.decision_function(Xn)

    # Réinjecter dans df original (alignement index)
    df_out = df.copy()
    df_out["is_anomaly_iforest"] = False
    df_out["iforest_score"] = np.nan
    df_out.loc[dfx.index, "is_anomaly_iforest"] = dfx["is_anomaly_iforest"]
    df_out.loc[dfx.index, "iforest_score"] = dfx["iforest_score"]

    return df_out

def compute_reports(df: pd.DataFrame) -> dict:
    """
    Retourne un dict avec classification_report + confusion_matrix pour:
      - Z-score
      - erreur relative
      - Isolation Forest
    """
    y_true = df["aberrante"].astype(int)

    y_z = df["is_anomaly_rule_z"].astype(int)
    y_rel = df["is_anomaly_rule"].astype(int)
    y_if = df["is_anomaly_iforest"].astype(int)

    reports = {
        "zscore": {
            "report": classification_report(y_true, y_z, digits=4),
            "cm": confusion_matrix(y_true, y_z)
        },
        "relative": {
            "report": classification_report(y_true, y_rel, digits=4),
            "cm": confusion_matrix(y_true, y_rel)
        },
        "iforest": {
            "report": classification_report(y_true, y_if, digits=4),
            "cm": confusion_matrix(y_true, y_if)
        }
    }
    return reports


# =========================
# 2) UI : Multi-pages
# =========================

class AppState:
    """Petite structure pour partager les résultats entre pages."""
    def __init__(self):
        self.true_path = None
        self.input_path = None

        self.data_true = None
        self.data_input = None
        self.df = None

        self.boxplot_path = Path("img/boxplot_par_annee.png")
        self.reports = None


class ImportPage(customtkinter.CTkFrame):
    def __init__(self, master, state: AppState, go_to_boxplot, go_to_results):
        super().__init__(master)
        self.state = state
        self.go_to_boxplot = go_to_boxplot
        self.go_to_results = go_to_results

        self.grid_columnconfigure(0, weight=1)

        title = customtkinter.CTkLabel(self, text="Import des données", font=customtkinter.CTkFont(size=20, weight="bold"))
        title.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")

        self.info = customtkinter.CTkLabel(
            self,
            text="1) Choisir le CSV réel (True)\n2) Choisir le CSV d'entraînement (Input)\n3) Lancer l'analyse",
            justify="left"
        )
        self.info.grid(row=1, column=0, padx=20, pady=(0, 15), sticky="w")

        self.btn_true = customtkinter.CTkButton(self, text="Choisir CSV réel (True)", command=self.pick_true)
        self.btn_true.grid(row=2, column=0, padx=20, pady=8, sticky="ew")

        self.lbl_true = customtkinter.CTkLabel(self, text="Aucun fichier sélectionné")
        self.lbl_true.grid(row=3, column=0, padx=20, pady=(0, 10), sticky="w")

        self.btn_input = customtkinter.CTkButton(self, text="Choisir CSV entraînement (Input)", command=self.pick_input)
        self.btn_input.grid(row=4, column=0, padx=20, pady=8, sticky="ew")

        self.lbl_input = customtkinter.CTkLabel(self, text="Aucun fichier sélectionné")
        self.lbl_input.grid(row=5, column=0, padx=20, pady=(0, 10), sticky="w")

        self.status = customtkinter.CTkLabel(self, text="", text_color="gray70", justify="left")
        self.status.grid(row=6, column=0, padx=20, pady=(5, 5), sticky="w")

        self.run_btn = customtkinter.CTkButton(self, text="Lancer l'analyse", command=self.run_pipeline)
        self.run_btn.grid(row=7, column=0, padx=20, pady=(10, 10), sticky="ew")

        self.nav_frame = customtkinter.CTkFrame(self, fg_color="transparent")
        self.nav_frame.grid(row=8, column=0, padx=20, pady=(10, 20), sticky="ew")
        self.nav_frame.grid_columnconfigure((0, 1), weight=1)

        self.to_box = customtkinter.CTkButton(self.nav_frame, text="Voir Boxplot", command=self.go_to_boxplot, state="disabled")
        self.to_box.grid(row=0, column=0, padx=(0, 10), sticky="ew")

        self.to_res = customtkinter.CTkButton(self.nav_frame, text="Voir Résultats", command=self.go_to_results, state="disabled")
        self.to_res.grid(row=0, column=1, padx=(10, 0), sticky="ew")

    def pick_true(self):
        p = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("Tous", "*.*")])
        if p:
            self.state.true_path = p
            self.lbl_true.configure(text=os.path.basename(p))

    def pick_input(self):
        p = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("Tous", "*.*")])
        if p:
            self.state.input_path = p
            self.lbl_input.configure(text=os.path.basename(p))

    def run_pipeline(self):
        if not self.state.true_path or not self.state.input_path:
            self.status.configure(text="⚠️ Sélectionne les deux fichiers CSV (True + Input) avant de lancer.")
            return

        try:
            self.status.configure(text="Chargement des données...")
            self.update_idletasks()

            data_true = load_csv_semicolon(self.state.true_path)
            data_input = load_csv_semicolon(self.state.input_path)

            # stockage
            self.state.data_true = data_true
            self.state.data_input = data_input

            self.status.configure(text="Construction des features + labels...")
            self.update_idletasks()

            df = compute_features_and_labels(data_true, data_input)

            self.status.configure(text="Isolation Forest...")
            self.update_idletasks()

            df = run_isolation_forest_on_rel_error(df, expected_anomalies=10)
            self.state.df = df

            # Boxplot
            Path("img").mkdir(exist_ok=True)
            self.status.configure(text="Génération du boxplot...")
            self.update_idletasks()
            build_boxplot_by_year(data_input, self.state.boxplot_path)

            # Reports
            self.status.configure(text="Génération des rapports...")
            self.update_idletasks()
            self.state.reports = compute_reports(df)

            self.status.configure(text="✅ Analyse terminée.")
            self.to_box.configure(state="normal")
            self.to_res.configure(state="normal")

        except Exception as e:
            self.status.configure(text=f"❌ Erreur: {e}")


class BoxplotPage(customtkinter.CTkFrame):
    def __init__(self, master, state: AppState):
        super().__init__(master)
        self.state = state
        self.grid_columnconfigure(0, weight=1)

        title = customtkinter.CTkLabel(self, text="Boxplot (moustaches)", font=customtkinter.CTkFont(size=20, weight="bold"))
        title.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")

        self.desc = customtkinter.CTkLabel(self, text="Affichage du boxplot généré (prix de clôture par année).", justify="left")
        self.desc.grid(row=1, column=0, padx=20, pady=(0, 10), sticky="w")

        # Affichage image (simple)
        self.img_label = customtkinter.CTkLabel(self, text="(Boxplot non généré)")
        self.img_label.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")

        self.refresh_btn = customtkinter.CTkButton(self, text="Rafraîchir l'image", command=self.refresh)
        self.refresh_btn.grid(row=3, column=0, padx=20, pady=(0, 20), sticky="ew")

        self._ctk_img = None

    def refresh(self):
        p = self.state.boxplot_path
        if not p.exists():
            self.img_label.configure(text="Boxplot non trouvé. Lance d'abord l'analyse.")
            return

        # customtkinter sait charger via CTkImage si Pillow est dispo
        try:
            from PIL import Image
            img = Image.open(p)
            self._ctk_img = customtkinter.CTkImage(light_image=img, dark_image=img, size=(980, 420))
            self.img_label.configure(image=self._ctk_img, text="")
        except Exception:
            # fallback texte si Pillow absent
            self.img_label.configure(text=f"Image générée: {p.resolve()}\n(installe Pillow pour l'afficher dans l'UI)")


class ResultsPage(customtkinter.CTkFrame):
    def __init__(self, master, state: AppState):
        super().__init__(master)
        self.state = state
        self.grid_columnconfigure(0, weight=1)

        title = customtkinter.CTkLabel(self, text="Résultats (Z-score / Relative / Isolation Forest)", font=customtkinter.CTkFont(size=20, weight="bold"))
        title.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")

        self.text = customtkinter.CTkTextbox(self, height=520)
        self.text.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="nsew")
        self.text.insert("1.0", "Lance l'analyse depuis la page Import pour afficher les résultats.")
        self.text.configure(state="disabled")

        self.refresh_btn = customtkinter.CTkButton(self, text="Rafraîchir", command=self.refresh)
        self.refresh_btn.grid(row=2, column=0, padx=20, pady=(0, 20), sticky="ew")

    def refresh(self):
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")

        if self.state.df is None or self.state.reports is None:
            self.text.insert("1.0", "Aucun résultat. Lance d'abord l'analyse depuis la page Import.")
            self.text.configure(state="disabled")
            return

        df = self.state.df
        reps = self.state.reports

        # Résumé counts
        n = len(df)
        n_true = int(df["aberrante"].sum())
        n_z = int(df["is_anomaly_rule_z"].sum())
        n_rel = int(df["is_anomaly_rule"].sum())
        n_if = int(df["is_anomaly_iforest"].sum())

        out = []
        out.append("=== Résumé ===")
        out.append(f"Observations analysées : {n}")
        out.append(f"Anomalies réelles (label aberrante) : {n_true}")
        out.append(f"Détectées Z-score : {n_z}")
        out.append(f"Détectées erreur relative : {n_rel}")
        out.append(f"Détectées Isolation Forest : {n_if}")
        out.append("")

        for key, label in [("zscore", "Z-SCORE"), ("relative", "ERREUR RELATIVE"), ("iforest", "ISOLATION FOREST")]:
            out.append(f"=== {label} ===")
            out.append("Confusion matrix [ [TN FP]\n                  [FN TP] ] :")
            out.append(str(reps[key]["cm"]))
            out.append("")
            out.append("Classification report :")
            out.append(reps[key]["report"])
            out.append("")

        # afficher quelques anomalies IF (top score faible)
        an = df[df["is_anomaly_iforest"]].copy()
        if len(an) > 0:
            an = an.sort_values("iforest_score").head(15)
            out.append("=== Top anomalies (Isolation Forest) ===")
            out.append(an[["date", "clot", "rel_error", "iforest_score"]].to_string(index=False))
            out.append("")

        self.text.insert("1.0", "\n".join(out))
        self.text.configure(state="disabled")


class MainApp(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.title("IA Finance - Détection d'anomalies")
        self.geometry("1100x700")

        self.app_state = AppState()

        # Layout principal
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Barre de nav
        self.nav = customtkinter.CTkFrame(self, width=220)
        self.nav.grid(row=0, column=0, sticky="nsw")
        self.nav.grid_rowconfigure(5, weight=1)

        # ✅ IMPORTANT : créer le container AVANT de créer les pages
        self.pages_container = customtkinter.CTkFrame(self)
        self.pages_container.grid(row=0, column=1, sticky="nsew")
        self.pages_container.grid_rowconfigure(0, weight=1)
        self.pages_container.grid_columnconfigure(0, weight=1)

        # Pages (après création du container)
        self.import_page = ImportPage(self.pages_container, self.app_state, self.show_boxplot, self.show_results)
        self.boxplot_page = BoxplotPage(self.pages_container, self.app_state)
        self.results_page = ResultsPage(self.pages_container, self.app_state)

        for p in (self.import_page, self.boxplot_page, self.results_page):
            p.grid(row=0, column=0, sticky="nsew")

        # Boutons nav
        customtkinter.CTkLabel(self.nav, text="Navigation",
                               font=customtkinter.CTkFont(size=18, weight="bold")
                               ).grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")

        customtkinter.CTkButton(self.nav, text="1) Import", command=self.show_import)\
            .grid(row=1, column=0, padx=20, pady=10, sticky="ew")

        customtkinter.CTkButton(self.nav, text="2) Boxplot", command=self.show_boxplot)\
            .grid(row=2, column=0, padx=20, pady=10, sticky="ew")

        customtkinter.CTkButton(self.nav, text="3) Résultats", command=self.show_results)\
            .grid(row=3, column=0, padx=20, pady=10, sticky="ew")

        self.show_import()


    def show_import(self):
        self.import_page.tkraise()

    def show_boxplot(self):
        self.boxplot_page.refresh()
        self.boxplot_page.tkraise()

    def show_results(self):
        self.results_page.refresh()
        self.results_page.tkraise()


if __name__ == "__main__":
    customtkinter.set_appearance_mode("dark")
    customtkinter.set_default_color_theme("blue")
    app = MainApp()
    app.mainloop()
