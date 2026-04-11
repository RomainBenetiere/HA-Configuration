import appdaemon.plugins.hass.hassapi as hass
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from datetime import datetime, timezone

class ACPIndex(hass.Hass):

    def initialize(self):
        self.log("Démarrage de l'app Indice de Déviance Domotique (ACP)")
        
        # Récupération des paramètres passés dans apps.yaml
        self.history_days = self.args.get("history_days", 30)
        self.variance_threshold = self.args.get("variance_threshold", 0.80)
        
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
        self.run_every(self.run_acp, "now+15", 15 * 60)
        self.run_in(self.run_acp, 5)

    def run_acp(self, kwargs):
        self.log(f"Début du calcul de l'ACP (Historique: {self.history_days}j)")
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
            
        self.log(f"DataFrame : {df.shape[0]} lignes × {df.shape[1]} colonnes")

        X, feature_names = self.build_feature_matrix(df)
        if X.size == 0 or X.shape[1] < 2:
            self.log("Matrice de features vide ou insuffisante. Abandon.")
            return

        result = self.compute_deviance_score(X, feature_names)
        if result:
            now_iso = datetime.now(timezone.utc).isoformat()
            attributes = {
                "unit_of_measurement": "%",
                "friendly_name": "Indice Global ACP",
                "icon": "mdi:chart-bell-curve-cumulative",
                "state_class": "measurement",
                "principale_variable": result["principale_variable"],
                "variance_expliquee": result["variance_expliquee"],
                "entites_analysees": result["entites_analysees"],
                "composantes_retenues": result["composantes_retenues"],
                "derniere_mise_a_jour": now_iso,
                "entites_disponibles": len(history),
            }
            # Push the computed sensor to HA
            self.set_state("sensor.indice_global_acp", state=str(result["score"]), attributes=attributes)
            self.log(f"Score ACP publié : {result['score']}% (Variable dominante: {result['principale_variable']})")


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

    def encode_numeric(self, series):
        values = pd.to_numeric(series, errors="coerce")
        values = values.interpolate(method="linear", limit_direction="both")
        values = values.fillna(values.mean())

        if values.std() == 0 or values.isna().all():
            return None, []

        scaler = StandardScaler()
        scaled = scaler.fit_transform(values.values.reshape(-1, 1))
        return scaled, [series.name]

    def encode_binary(self, series):
        mapping = {"on": 1.0, "off": 0.0, "true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0}
        values = series.map(lambda x: mapping.get(str(x).lower(), np.nan))
        values = values.fillna(method="ffill").fillna(0.0)

        if values.std() == 0:
            return None, []

        return values.values.reshape(-1, 1), [series.name]

    def encode_categorical(self, series):
        series = series.fillna(method="ffill").fillna("unknown")
        dummies = pd.get_dummies(series, prefix=series.name)
        
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
        self.log(f"Matrice finale : {X.shape[0]} observations × {X.shape[1]} features")
        return X, feature_names

    def compute_deviance_score(self, X, feature_names):
        if X.shape[0] < 10 or X.shape[1] < 2:
            self.log(f"Données insuffisantes : {X.shape}")
            return None

        n_components = min(X.shape[0], X.shape[1])
        pca = PCA(n_components=min(n_components, max(2, int(X.shape[1] * 0.9))))
        X_pca = pca.fit_transform(X)

        cumvar = np.cumsum(pca.explained_variance_ratio_)
        n_keep = int(np.searchsorted(cumvar, self.variance_threshold) + 1)
        n_keep = max(1, min(n_keep, X_pca.shape[1]))

        X_reduced = X_pca[:, :n_keep]
        variance_explained = cumvar[n_keep - 1] * 100

        self.log(f"ACP : {n_keep} composantes retenues ({variance_explained:.1f}% variance)")

        distances = np.linalg.norm(X_reduced, axis=1)
        current_dist = distances[-1]
        max_dist = np.max(distances)

        final_score = 0.0 if max_dist == 0 else (current_dist / max_dist) * 100

        loadings = pca.components_[:n_keep, :]
        current_original = X[-1, :]
        mean_abs_loadings = np.mean(np.abs(loadings), axis=0)
        contributions = np.abs(current_original) * mean_abs_loadings

        top_variable = "inconnu"
        if len(contributions) > 0 and len(feature_names) == len(contributions):
            top_variable = feature_names[int(np.argmax(contributions))]

        return {
            "score": round(float(final_score), 1),
            "principale_variable": top_variable,
            "variance_expliquee": f"{variance_explained:.1f}%",
            "composantes_retenues": int(n_keep),
            "entites_analysees": int(X.shape[1]),
        }
