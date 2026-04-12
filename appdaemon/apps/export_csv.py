import appdaemon.plugins.hass.hassapi as hass
import csv
import datetime

class ExportCSV(hass.Hass):
    def initialize(self):
        self.log("Initialisation du module d'exportation CSV...")
        
        # Ce module peut mettre du temps à cracher la BDD, on le lance en asynchrone 10 sec après le démarrage
        self.run_in(self.start_export, 10)

    def start_export(self, kwargs):
        days_to_export = self.args.get("history_days", 30)
        output_file = self.args.get("output_file", "/homeassistant/export_historique.csv")
        
        # Récupère absolument tous les entity_id existants
        self.log("Récupération de la liste complète des entités...")
        all_states = self.get_state()
        if not all_states:
            self.log("Erreur : impossible de récupérer l'état des entités.", level="ERROR")
            return
            
        entity_ids = list(all_states.keys())
        self.log(f"{len(entity_ids)} entités trouvées. Démarrage de la requête de base de données...")

        total_lines = 0
        total_entities_processed = 0

        try:
            with open(output_file, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # Header
                writer.writerow(["Date", "Entity", "Valeur"])
                
                # On traite chaque capteur un par un pour ne pas faire crasher la RAM avec un appel géant à la DB
                for entity in entity_ids:
                    try:
                        history = self.get_history(entity_id=entity, days=days_to_export)
                        if history and len(history) > 0:
                            hist_list = history[0] if isinstance(history[0], list) else history
                            for entry in hist_list:
                                timestamp = entry.get("last_updated") or entry.get("last_changed")
                                state = entry.get("state")
                                
                                # On nettoie un peu si les stats sont vides
                                if state not in (None, "unavailable", "unknown"):
                                    writer.writerow([timestamp, entity, state])
                                    total_lines += 1
                                    
                            total_entities_processed += 1
                            
                    except Exception as e:
                        self.log(f"Erreur d'extraction d'historique pour {entity} : {e}", level="WARNING")
                        
                    # Un petit log tous les 50 capteurs pour montrer la progression
                    if total_entities_processed > 0 and total_entities_processed % 50 == 0:
                        self.log(f"Progression : {total_entities_processed}/{len(entity_ids)} entités traitées ({total_lines} lignes écrites)...")

            self.log(f"✅ EXPORT RÉUSSI ! {total_lines} lignes écrites au total depuis {total_entities_processed} capteurs.")
            self.log(f"📁 Fichier d'export disponible ici : {output_file}")
            
        except Exception as file_error:
            self.log(f"❌ Erreur critique lors de l'écriture du fichier : {file_error}", level="ERROR")
