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
        
        # Entités configurées (selon la nouvelle liste)
        self.numeric_entities = [
            "sensor.envoy_122301074544_production_d_electricite_actuelle",
            "sensor.salon_temperature_corrigee",
            "sensor.chambre_temperature_corrigee",
            "sensor.carillon_force_du_signal",
            "sensor.jardin_temperature_corrigee",
            "sensor.envoy_122301074544_consommation_d_energie_totale",
            "sensor.envoy_122301074544_lifetime_net_energy_consumption",
            "sensor.pac_energy",
            "sensor.suivipac_pac_cabinet_temp",
            "sensor.envoy_122301074544_production_d_energie_totale",
            "sensor.envoy_122301074544_lifetime_net_energy_production",
            "sensor.q_th",
            "sensor.delta_eau_modele",
            "sensor.chambre_humidite_corrigee",
            "sensor.openweathermap_feels_like_temperature",
            "sensor.openweathermap_temperature",
            "sensor.energy_current_hour",
            "sensor.power_production_now",
            "sensor.q_th_energy",
            "sensor.openweathermap_uv_index",
            "sensor.bur_buan_point_de_rosee",
            "sensor.garage_temperature_corrigee",
            "sensor.bur_buan_temperature_corrigee",
            "sensor.energy_next_hour",
            "sensor.envoy_122301074544_lifetime_balanced_net_energy_consumption",
            "sensor.salon_point_de_rosee",
            "sensor.chambre_point_de_rosee",
            "sensor.garage_humidite_absolue",
            "sensor.salon_humidite_absolue",
            "sensor.delta_eau_ecart",
            "sensor.chambre_humidite_absolue",
            "sensor.garage_point_de_rosee",
            "sensor.garage_humidite_corrigee",
            "sensor.bur_box_temperature_corrigee",
            "sensor.cycles_24h",
            "sensor.bur_buan_humidite_absolue",
            "sensor.energy_production_today",
            "sensor.thgr810_thgn800_72_02_temperature",
            "sensor.thermostat_box_bureau_ema_temperature",
            "sensor.portail_wifi_signal_sensor",
            "sensor.energy_production_today_remaining",
            "sensor.openweathermap_humidity",
            "sensor.energy_production_tomorrow",
            "sensor.ecart_depart_vs_consigne",
            "sensor.jardin_humidite_corrigee",
            "sensor.q_th_daily",
            "sensor.q_th_daily_2",
            "sensor.suivipac_temppac_out",
            "sensor.thgr810_thgn800_72_02_humidite",
            "sensor.bur_box_humidite_corrigee",
            "sensor.envoy_122301074544_production_d_energie_du_jour",
            "sensor.sun_solar_elevation",
            "sensor.cop_journalier",
            "sensor.bur_box_point_de_rosee",
            "sensor.bur_box_humidite_absolue",
            "sensor.suivipac_temppac_in",
            "sensor.jardin_point_de_rosee"
        ]

        self.binary_entities = [
            "binary_sensor.workday_sensor",
            "switch.portail_close_portal",
            "switch.portail_open_portal",
            "binary_sensor.pac_en_marche",
            "switch.chauffe_eau_michelou",
            "automation.ouvrir_volet_chambre_matin"
        ]

        self.categorical_entities = [
            "cover.volet_roulant_cuisine",
            "cover.volets_roulants_buanderie",
            "cover.volet_roulant_cham_parents",
            "cover.volet_roulant_cham_elliot",
            "cover.volet_roulant_cham_arthur",
            "cover.volet_roulant_sam_2",
            "cover.volet_roulant_sam_1"
        ]

        # Lancer le calcul toutes les 15 minutes, et une fois dans 5 secondes au démarrage
        self.run_every(self.run_anomaly_detection, "now+15", 15 * 60)
        self.run_in(self.run_anomaly_detection, 5)

    def run_anomaly_detection(self, kwargs):
        self.log(f"Début du calcul Mahalanobis (Global ACP - Historique: {self.history_days}j)")
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

        # Construction de la matrice de features sur TOUT le dataframe (Modèle Global)
        X_all, feature_names = self.build_feature_matrix(df)
        if X_all.size == 0 or X_all.shape[1] < 2 or X_all.shape[0] < 10:
            self.log("Matrice de base vide ou insuffisante. Abandon.")
            return

        self.log(f"Données de référence (Modèle Global) : {X_all.shape[0]} observations × {X_all.shape[1]} colonnes")

        # Construction du scaler et de la PCA sur X_all directement (ACP Globale)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_all)

        n_components = min(X_scaled.shape[0], X_scaled.shape[1])
        pca = PCA(n_components=min(n_components, max(2, int(X_scaled.shape[1] * 0.9))))
        pca.fit(X_scaled)
        
        # Retenir les composantes pour 95% de variance
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        n_keep = int(np.searchsorted(cumvar, 0.95) + 1)
        n_keep = max(1, min(n_keep, X_scaled.shape[1]))
        
        self.log(f"ACP : {n_keep} composantes retenues ({cumvar[n_keep-1]*100:.1f}% variance expliquée)")

        # Récupérer l'état complet à l'instant présent (la dernière ligne de l'historique brut)
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
        top_3_contributors_str = "unknown"
        
        if criticality_ratio > 1.0:
            P = pca.components_[:n_keep, :]
            # Contributions estimées par la méthode de reconstruction des scores pondérés.
            weighted_components_sum = np.dot((t_now / variances), P)
            c_j = current_scaled[0] * weighted_components_sum
            contributions = np.abs(c_j)
            
            if len(contributions) > 0 and len(feature_names) == len(contributions):
                forbidden_features = ["hour_sin", "hour_cos", "day_sin", "day_cos"]
                valid_indices = [i for i, name in enumerate(feature_names) if name not in forbidden_features]
                
                if valid_indices:
                    valid_contributions = contributions[valid_indices]
                    valid_names = [feature_names[i] for i in valid_indices]
                    
                    # Tris des variables explicatives physiques
                    sorted_rel_indices = np.argsort(valid_contributions)[::-1]
                    top_variable = valid_names[sorted_rel_indices[0]]
                    
                    top_3_list = [valid_names[i] for i in sorted_rel_indices[:min(3, len(valid_names))]]
                    top_3_contributors_str = ", ".join(top_3_list)
                
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
                "top_3_contributors": top_3_contributors_str,
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
        
        # Ingénierie des variables temporelles (Cyclic Features)
        try:
            local_dt = df.index.tz_convert('Europe/Paris')
        except Exception:
            local_dt = df.index
            
        minutes_of_day = local_dt.hour * 60 + local_dt.minute
        df['hour_sin'] = np.sin(2 * np.pi * minutes_of_day / 1440.0)
        df['hour_cos'] = np.cos(2 * np.pi * minutes_of_day / 1440.0)
        
        day_of_week = local_dt.dayofweek
        df['day_sin'] = np.sin(2 * np.pi * day_of_week / 7.0)
        df['day_cos'] = np.cos(2 * np.pi * day_of_week / 7.0)
        
        return df

    def encode_numeric(self, df_col):
        values = pd.to_numeric(df_col, errors="coerce")
        values = values.interpolate(method="linear", limit_direction="both")
        mean_val = values.mean()
        if pd.isna(mean_val):
            mean_val = 0.0
        values = values.fillna(mean_val)

        return values.values.reshape(-1, 1), [df_col.name]

    def encode_binary(self, df_col):
        mapping = {"on": 1.0, "off": 0.0, "true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0}
        values = df_col.map(lambda x: mapping.get(str(x).lower(), np.nan))
        values = values.ffill().fillna(0.0)

        return values.values.reshape(-1, 1), [df_col.name]

    def encode_categorical(self, df_col):
        df_col = df_col.ffill().fillna("unknown")
        dummies = pd.get_dummies(df_col, prefix=df_col.name)
        
        return dummies.values, list(dummies.columns)

    def build_feature_matrix(self, df):
        blocks = []
        feature_names = []

        cyclic_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
        for eid in self.numeric_entities + cyclic_features:
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
