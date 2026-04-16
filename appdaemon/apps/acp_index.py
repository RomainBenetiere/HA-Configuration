import appdaemon.plugins.hass.hassapi as hass
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import chi2
from datetime import datetime, timezone

class ACPIndex(hass.Hass):

    def initialize(self):
        self.log("Démarrage de l'app Capteur d'Anomalie (Mahalanobis)")
        
        # Récupération des paramètres passés dans apps.yaml
        self.history_days = self.args.get("history_days", 30)
        self.window_minutes = 15  # +/- 15 minutes
        
        # Entités configurées
        self.numeric_entities = [
            "sensor.salon_temperature_corrigee",
            "sensor.jardin_temperature_corrigee",
            "sensor.bur_box_temperature_corrigee",
            "sensor.bur_mezzanine_temperature_corrigee",
            "sensor.champ_parents_temperature_corrigee",
            "sensor.garage_temperature_corrigee",
            "sensor.piscine_temperature_de_l_eau",
            "sensor.salon_humidite_corrigee",
            "sensor.bur_box_humidite_corrigee",
            "sensor.bur_mezzanine_humidite_corrigee",
            "sensor.champ_parents_humidite_corrigee",
            "sensor.garage_humidite_corrigee",
            "sensor.power_network",
            "sensor.q_th",
            "sensor.cop_instantane",
            "sensor.elec3_cm180_88_b2_puissance_instantanee",
            "sensor.duty_cycle_1h",
            "sensor.piscine_ph_lisse",
            "sensor.piscine_orp_redox_lissee",
            "sensor.piscine_fc_estime",
            "sensor.info_nh3_box_env",
            "sensor.info_pm25_box_env",
            "sensor.info_pressure_box_env",
            "sensor.info_temperature_box_env",
            "sensor.info_humidite_box_env",
        ]

        self.binary_entities = [
            "binary_sensor.pac_en_marche",
            "binary_sensor.piscine_relais_pompe_hw",
            "binary_sensor.piscine_relais_robot_hw",
            "binary_sensor.piscine_injection_acide_hw",
            "binary_sensor.piscine_electrolyseur_hw",
            "binary_sensor.info_lum_chb_art",
            "binary_sensor.info_lum_chb_elliot",
            "binary_sensor.info_radiateur_box",
            "binary_sensor.power_excess",
            "binary_sensor.power_shortage",
        ]

        self.categorical_entities = [
            "sensor.piscine_etat_du_cycle_actuel",
        ]

        # Lancer le calcul toutes les 15 minutes, et une fois dans 5 secondes au démarrage
        self.run_every(self.run_anomaly_detection, "now+15", 15 * 60)
        self.run_in(self.run_anomaly_detection, 5)

    def run_anomaly_detection(self, kwargs):
        self.log(f"Début du calcul Mahalanobis (Historique: {self.history_days}j, Fenêtre: +/-{self.window_minutes}min)")
        all_entities = self.numeric_entities + self.binary_entities + self.categorical_entities
        
        history = {}
        for entity_id in all_entities:
            try:
                # AppDaemon get_history retourne format: [[{state1}, {state2}]]
                entity_history = self.get_history(entity_id=entity_id, days=self.history_days)
                if entity_history and len(entity_history) > 0:
                    hist_list = entity_history[0] if isinstance(entity_history[0], list) else entity_history
                    
                    states = []
                    for entry in hist_list:
                        ts = entry.get("last_updated") or entry.get("last_changed")
                        val = entry.get("state")
                        if ts and val not in (None, "unavailable", "unknown"):
                            states.append((ts, val))
                    
                    if states:
                        history[entity_id] = states
            except Exception as e:
                self.log(f"Erreur lors de la récupération de {entity_id}: {e}", level="WARNING")

        if not history:
            self.log("Aucune donnée d'historique récupérée. Abandon.")
            return

        df = self.history_to_dataframe(history)
        if df.empty:
            self.log("DataFrame vide après conversion. Abandon.")
            return

        # Filtrage temporel: garder uniquement les lignes dans +/- 15 min de l'heure actuelle
        now_time = datetime.now(timezone.utc)
        target_hour = now_time.hour
        target_minute = now_time.minute
        
        # On calcule les minutes écoulées depuis minuit pour la comparaison
        target_total_minutes = target_hour * 60 + target_minute
        
        def is_in_window(ts):
            # Formule robuste aux jours qui passent (minuit)
            ts_minutes = ts.hour * 60 + ts.minute
            diff = abs(ts_minutes - target_total_minutes)
            # Gérer le passage par minuit (ex: target=23h50, ts=00h05 -> diff=1435)
            if diff > 720: # plus de 12h d'écart -> on prend l'autre côté de minuit
                diff = 1440 - diff
            return diff <= self.window_minutes

        df_ref = df[df.index.map(is_in_window)]
        
        if df_ref.empty:
            self.log("DataFrame de référence vide après filtrage temporel. Abandon.")
            return
            
        self.log(f"Données de référence historiques (fenêtre temporelle) : {df_ref.shape[0]} lignes × {df_ref.shape[1]} colonnes")

        X, feature_names = self.build_feature_matrix(df_ref)
        if X.size == 0 or X.shape[1] < 2 or X.shape[0] < 10:
            self.log("Matrice de features de référence vide ou insuffisante. Abandon.")
            return

        # Construction du scaler et de la PCA sur df_ref
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        n_components = min(X_scaled.shape[0], X_scaled.shape[1])
        pca = PCA(n_components=min(n_components, max(2, int(X_scaled.shape[1] * 0.9))))
        pca.fit(X_scaled)
        
        # Retenir les composantes pour 95% de variance
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        n_keep = int(np.searchsorted(cumvar, 0.95) + 1)
        n_keep = max(1, min(n_keep, X_scaled.shape[1]))
        
        self.log(f"ACP : {n_keep} composantes retenues ({cumvar[n_keep-1]*100:.1f}% variance expliquée)")

        # Récupérer l'état complet à l'instant présent (la dernière ligne de l'historique brut)
        # On doit utiliser le pre-processing des mêmes colonnes pour obtenir x_now
        X_all, feature_names_all = self.build_feature_matrix(df)
        current_original = X_all[-1, :]
        current_scaled = scaler.transform(current_original.reshape(1, -1))
        
        # Projection PCA
        t_now = pca.transform(current_scaled)[0, :n_keep]
        variances = pca.explained_variance_[:n_keep]
        
        # Calcul Distance de Mahalanobis (T2)
        T2_now = np.sum((t_now ** 2) / variances)
        
        # Seuil de contrôle statistique (99%)
        threshold = chi2.ppf(0.99, df=n_keep)
        
        criticality_ratio = T2_now / threshold
        
        # Diagnostic de Contribution (si ça dépasse le seuil critique)
        top_variable = "unknown"
        if criticality_ratio > 1.0:
            P = pca.components_[:n_keep, :]
            # Contributions estimées par la méthode de reconstruction des scores pondérés.
            weighted_components_sum = np.dot((t_now / variances), P)
            c_j = current_scaled[0] * weighted_components_sum
            contributions = np.abs(c_j)
            
            if len(contributions) > 0 and len(feature_names) == len(contributions):
                top_idx = int(np.argmax(contributions))
                top_variable = feature_names[top_idx]
                
        now_iso = datetime.now(timezone.utc).isoformat()
        
        # --- MISE A JOUR DES CAPTEURS HA ---
        
        # 1. Status Mahalanobis (Distance Absolue)
        self.set_state("sensor.mahalanobis_status", 
            state=str(round(float(T2_now), 2)), 
            attributes={
                "unit_of_measurement": "",
                "friendly_name": "T2 Mahalanobis",
                "icon": "mdi:chart-scatter-plot",
                "state_class": "measurement",
                "derniere_mise_a_jour": now_iso
            })
            
        # 2. Criticality Ratio (Distance / Seuil)
        self.set_state("sensor.anomaly_criticality", 
            state=str(round(float(criticality_ratio), 2)), 
            attributes={
                "unit_of_measurement": "x seuil",
                "friendly_name": "Criticité de l'Anomalie",
                "icon": "mdi:alert-decagram",
                "state_class": "measurement",
                "seuil_99": round(float(threshold), 2),
                "composantes_retenues": int(n_keep),
                "derniere_mise_a_jour": now_iso
            })
            
        # 3. Source de l'anomalie
        self.set_state("sensor.anomaly_source", 
            state=str(top_variable), 
            attributes={
                "friendly_name": "Source de l'Anomalie",
                "icon": "mdi:feature-search",
                "derniere_mise_a_jour": now_iso
            })

        self.log(f"Résultat Mahalanobis publié : T2={T2_now:.2f} (Seuil={threshold:.2f}, Criticité={criticality_ratio:.2f}x). Source: {top_variable}")

    # --- Pipeline de transformation des données ---

    def history_to_dataframe(self, history, resample_interval="15min"):
        series_dict = {}
        for entity_id, states in history.items():
            timestamps = []
            values = []
            for ts_str, val in states:
                try:
                    ts = pd.to_datetime(ts_str, utc=True)
                    timestamps.append(ts)
                    values.append(val)
                except (ValueError, TypeError):
                    continue

            if timestamps:
                s = pd.Series(values, index=pd.DatetimeIndex(timestamps), name=entity_id)
                s = s[~s.index.duplicated(keep="last")]
                s = s.sort_index()
                series_dict[entity_id] = s

        if not series_dict:
            return pd.DataFrame()

        df = pd.DataFrame(series_dict)
        # Using forward fill and interpolation for resampling
        df = df.resample(resample_interval).last()
        
        return df

    def encode_numeric(self, df_col):
        values = pd.to_numeric(df_col, errors="coerce")
        values = values.interpolate(method="linear", limit_direction="both")
        mean_val = values.mean()
        if pd.isna(mean_val):
            mean_val = 0.0
        values = values.fillna(mean_val)

        if values.std() == 0 or values.isna().all():
            return None, []

        return values.values.reshape(-1, 1), [df_col.name]

    def encode_binary(self, df_col):
        mapping = {"on": 1.0, "off": 0.0, "true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0}
        values = df_col.map(lambda x: mapping.get(str(x).lower(), np.nan))
        values = values.ffill().fillna(0.0)

        if values.std() == 0:
            return None, []

        return values.values.reshape(-1, 1), [df_col.name]

    def encode_categorical(self, df_col):
        df_col = df_col.ffill().fillna("unknown")
        dummies = pd.get_dummies(df_col, prefix=df_col.name)
        
        non_const_cols = [c for c in dummies.columns if dummies[c].std() > 0]
        if not non_const_cols:
            return None, []

        dummies = dummies[non_const_cols]
        return dummies.values, list(dummies.columns)

    def build_feature_matrix(self, df):
        blocks = []
        feature_names = []

        for eid in self.numeric_entities:
            if eid in df.columns:
                result, names = self.encode_numeric(df[eid])
                if result is not None:
                    blocks.append(result)
                    feature_names.extend(names)

        for eid in self.binary_entities:
            if eid in df.columns:
                result, names = self.encode_binary(df[eid])
                if result is not None:
                    blocks.append(result)
                    feature_names.extend(names)

        for eid in self.categorical_entities:
            if eid in df.columns:
                result, names = self.encode_categorical(df[eid])
                if result is not None:
                    blocks.append(result)
                    feature_names.extend(names)

        if not blocks:
            return np.array([]), []

        n_rows = blocks[0].shape[0]
        valid_blocks = [b for b in blocks if b.shape[0] == n_rows]
        X = np.hstack(valid_blocks)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X, feature_names
