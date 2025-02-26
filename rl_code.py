import ollama
import subprocess
import os
import time
import re
import logging
import json
from typing import Tuple, Optional, Dict, List, Any
from pathlib import Path

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("reinforcement_learning.log"),
        logging.StreamHandler()
    ]
)

# Konstanten
DOCKER_IMAGE = "python:3.11-slim"  # Slim-Variante für schnellere Ausführung
CODE_START = "### CODE START ###"
CODE_END = "### CODE END ###"
MAX_ITERATIONS = 10  # Maximale Anzahl von Iterationen
TIMEOUT_SECONDS = 30  # Zeitlimit für Code-Ausführung
MODELS = {
    "primary": "deepseek-r1:32b",
    "fallback": "llama3:latest"  # Fallback-Modell, falls das primäre Modell Probleme hat
}

class RLCodeOptimizer:
    """Klasse für die Optimierung von Code durch Reinforcement Learning."""
    
    def __init__(self, task_description: str, work_dir: str = None):
        """
        Initialisiert den Code-Optimierer.
        
        Args:
            task_description: Beschreibung der zu lösenden Aufgabe
            work_dir: Arbeitsverzeichnis für temporäre Dateien (Standard: aktuelles Verzeichnis)
        """
        self.task_description = task_description
        self.work_dir = Path(work_dir or os.getcwd())
        self.script_path = self.work_dir / "script.py"
        self.history = []  # Verlauf aller Iterationen
        self.best_code = None  # Bester Code bisher
        self.best_output = None  # Bester Output bisher
        
        # Stelle sicher, dass Docker verfügbar ist
        self._check_docker_availability()
        
        # Stelle sicher, dass das Ollama-Modell lokal verfügbar ist
        self._check_model_availability()

    def _check_docker_availability(self):
        """Überprüft, ob Docker verfügbar ist."""
        try:
            subprocess.run(["docker", "--version"], check=True, capture_output=True)
            logging.info("Docker ist verfügbar.")
        except (subprocess.SubprocessError, FileNotFoundError):
            logging.error("Docker ist nicht verfügbar. Bitte installieren Sie Docker.")
            raise RuntimeError("Docker ist erforderlich für die Ausführung dieses Skripts.")
    
    def _check_model_availability(self):
        """Überprüft, ob das Ollama-Modell lokal verfügbar ist."""
        try:
            models = ollama.list()
            available_models = [model['name'] for model in models.get('models', [])]
            
            for role, model_name in MODELS.items():
                model_base = model_name.split(':')[0]
                if not any(model.startswith(model_base) for model in available_models):
                    logging.warning(f"{role.capitalize()}-Modell '{model_name}' ist möglicherweise nicht verfügbar.")
                    
            logging.info(f"Verfügbare Modelle überprüft.")
        except Exception as e:
            logging.warning(f"Konnte Modellverfügbarkeit nicht überprüfen: {e}")
    
    def generate_prompt(self, feedback: str = None) -> str:
        """
        Generiert den Prompt für die KI.
        
        Args:
            feedback: Feedback zur vorherigen Iteration (optional)
            
        Returns:
            Formatierter Prompt für die KI
        """
        base_prompt = f"""
Du bist eine KI, die Python-Code generiert und optimiert.  
Du hast KEINEN menschlichen Zugriff auf Feedback.  
Du musst selbst entscheiden, ob dein Code gut ist oder ob er verbessert werden muss.

**Wichtiges Kriterium:**
- Dein Code sollte robust sein und die Aufgabe effektiv lösen
- Vermeide unnötige Komplexität, aber implementiere wichtige Fehlerbehandlung
- Achte auf Performance und Speicherverbrauch
- Verwende bewährte Algorithmen und Datenstrukturen

**WICHTIG:** Du darfst KEINE Benutzereingaben (`input()`) verwenden.  
Falls `input()` genutzt wird, muss es durch eine selbstgewählte feste Variable ersetzt werden.  
Du bist in einer geschlossenen Umgebung und kannst keine Hilfe erwarten. Du bist allein für die Optimierung verantwortlich.  

**Formatierung:**  
- Teile deine Gedanken zuerst als normale Ausgabe.  
- Danach schreibe den Code zwischen {CODE_START} und {CODE_END}.  
- Schreibe NUR reinen ausführbaren Code ohne Markdown-Blöcke (```python ... ```).  

Die Aufgabe lautet: {self.task_description}  
"""

        if feedback:
            base_prompt += f"\n\nHier ist das Feedback zur vorherigen Iteration:\n{feedback}"
            
            if self.best_code:
                base_prompt += f"\n\nDer bisher beste Code war:\n{CODE_START}\n{self.best_code}\n{CODE_END}"

        return base_prompt

    def generate_code(self, feedback: str = None) -> Tuple[str, Optional[str]]:
        """
        Generiert Code mit Ollama.
        
        Args:
            feedback: Feedback zur vorherigen Iteration (optional)
            
        Returns:
            Tuple mit Thinking-Prozess und generiertem Code
        """
        prompt = self.generate_prompt(feedback)
        
        try:
            response = ollama.chat(model=MODELS["primary"], messages=[{"role": "user", "content": prompt}])
            content = response["message"]["content"]
        except Exception as e:
            logging.warning(f"Fehler bei Verwendung des primären Modells: {e}")
            logging.info(f"Versuche Fallback-Modell...")
            try:
                response = ollama.chat(model=MODELS["fallback"], messages=[{"role": "user", "content": prompt}])
                content = response["message"]["content"]
            except Exception as e2:
                logging.error(f"Auch Fallback-Modell fehlgeschlagen: {e2}")
                return "Modell-Fehler aufgetreten", None

        # Thinking-Teil extrahieren
        thinking = content.split(CODE_START)[0].strip() if CODE_START in content else content

        # Code extrahieren & Markdown-Reste entfernen
        code_match = re.search(rf"{CODE_START}(.*?){CODE_END}", content, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
            code = re.sub(r"```python|```", "", code).strip()  # Entfernt Markdown-Formatierung
            
            # Überprüfe, ob `input()` vorkommt
            if "input(" in code:
                logging.warning("Der generierte Code enthält `input()`, was nicht erlaubt ist.")
                return thinking, None
        else:
            logging.warning("Kein Code zwischen den Markierungen gefunden.")
            code = None  # Kein Code erhalten

        return thinking, code

    def run_code_in_docker(self, code: str) -> Tuple[str, str]:
        """
        Führt den Code im Docker-Container aus.
        
        Args:
            code: Auszuführender Python-Code
            
        Returns:
            Tuple mit stdout und stderr der Ausführung
        """
        # Speichere den Code in einer Datei
        self.script_path.write_text(code)
        
        # Bereite Docker-Befehl vor
        cmd = [
            "docker", "run", "--rm",
            "--network=none",  # Keine Netzwerkkonnektivität
            "--memory=512m",  # Speicherlimit
            "--cpus=1",       # CPU-Limit
            "-v", f"{self.work_dir.absolute()}:/app",
            DOCKER_IMAGE,
            "timeout", str(TIMEOUT_SECONDS),  # Zeitlimit für die Ausführung
            "python", "/app/script.py"
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS + 5  # Etwas zusätzliche Zeit für Docker-Overhead
            )
            return result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return "", "Timeout: Die Codeausführung wurde nach {} Sekunden abgebrochen.".format(TIMEOUT_SECONDS)
        except Exception as e:
            return "", f"Fehler bei der Docker-Ausführung: {str(e)}"

    def evaluate_result(self, output: str, error: str) -> Tuple[str, float]:
        """
        Evaluiert das Ergebnis der Code-Ausführung.
        
        Args:
            output: Ausgabe des Codes
            error: Fehlermeldungen (falls vorhanden)
            
        Returns:
            Tuple mit Bewertungsergebnis und Score (0-1)
        """
        # Einfache heuristische Bewertung
        score = 0.0
        
        if error:
            if "Timeout" in error:
                return "Der Code hat das Zeitlimit überschritten.", 0.1
            return f"Der Code hat Fehler erzeugt: {error}", 0.2
        
        # Grundlegende Punktzahl für fehlerfreie Ausführung
        score = 0.7
        
        # Spezifische Aufgabenevaluierung könnte hier ergänzt werden
        # z.B. für Pi-Approximation: überprüfe Genauigkeit
        
        # Überprüfe Output auf Nützlichkeit
        if output and len(output) > 0:
            score += 0.2
            
            # Für Pi-Approximation: versuche, den Wert zu extrahieren und zu überprüfen
            if "pi" in self.task_description.lower():
                pi_pattern = r"(\d+\.\d+)"
                pi_matches = re.findall(pi_pattern, output)
                if pi_matches:
                    try:
                        pi_value = float(pi_matches[0])
                        pi_accuracy = abs(pi_value - 3.14159265358979323846)
                        
                        if pi_accuracy < 0.00001:
                            score += 0.1  # Sehr genau
                        elif pi_accuracy < 0.0001:
                            score += 0.05  # Genau
                    except:
                        pass
        
        return "Erfolgreiche Ausführung" if score > 0.7 else "Ausführung ohne aussagekräftigen Output", score

    def ask_if_satisfied(self, output: str, error: str, evaluation: str, score: float) -> str:
        """
        Fragt die KI, ob das Skript bereits optimal ist.
        
        Args:
            output: Ausgabe des Codes
            error: Fehlermeldungen (falls vorhanden)
            evaluation: Bewertung des Ergebnisses
            score: Bewertungsscore (0-1)
            
        Returns:
            Antwort der KI
        """
        prompt = f"""
Das Skript wurde getestet.

📜 **Terminal-Output:**
{output}

⚠️ **Fehler:**
{error if error else 'Keine Fehler'}

🔍 **Bewertung:** {evaluation} (Score: {score:.2f}/1.0)

Du hast KEINEN menschlichen Zugriff auf Feedback.
Analysiere den Output und die Fehler sorgfältig.

Falls dein Code perfekt ist, schreibe: "Ja, das Skript ist optimal."  
Falls du Verbesserungen siehst, beschreibe genau:
1. Was sind die Probleme?
2. Wie können sie behoben werden?
3. Welche spezifischen Änderungen schlägst du vor?

Sei präzise in deiner Analyse.
"""
        try:
            response = ollama.chat(model=MODELS["primary"], messages=[{"role": "user", "content": prompt}])
            return response["message"]["content"].strip()
        except Exception as e:
            logging.warning(f"Fehler bei der Zufriedenheitsabfrage: {e}")
            return "Aufgrund eines Fehlers konnte die Zufriedenheit nicht abgefragt werden. Optimierung fortsetzen."

    def is_satisfied(self, response: str) -> bool:
        """
        Prüft, ob die KI mit dem Ergebnis zufrieden ist.
        
        Args:
            response: Antwort der KI
            
        Returns:
            True, wenn die KI zufrieden ist, sonst False
        """
        return "Ja, das Skript ist optimal" in response

    def save_iteration(self, iteration: int, thinking: str, code: str, 
                       output: str, error: str, satisfaction: str,
                       evaluation: str, score: float) -> None:
        """
        Speichert die aktuelle Iteration im Verlauf.
        
        Args:
            iteration: Nummer der Iteration
            thinking: Thinking-Prozess der KI
            code: Generierter Code
            output: Ausgabe des Codes
            error: Fehlermeldungen
            satisfaction: Zufriedenheitsantwort der KI
            evaluation: Bewertung des Ergebnisses
            score: Bewertungsscore
        """
        iteration_data = {
            "iteration": iteration,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "thinking": thinking,
            "code": code,
            "output": output,
            "error": error,
            "satisfaction": satisfaction,
            "evaluation": evaluation,
            "score": score
        }
        
        self.history.append(iteration_data)
        
        # Update bester Code, wenn der aktuelle besser ist
        if self.best_code is None or score > self.history[-2]["score"] if len(self.history) > 1 else 0:
            self.best_code = code
            self.best_output = output
            logging.info(f"Neuer bester Code in Iteration {iteration} mit Score {score:.2f}")

    def save_results(self) -> None:
        """Speichert alle Ergebnisse in einer JSON-Datei."""
        output_path = self.work_dir / "rl_results.json"
        with open(output_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        best_code_path = self.work_dir / "best_code.py"
        if self.best_code:
            with open(best_code_path, 'w') as f:
                f.write(self.best_code)
                
        logging.info(f"Ergebnisse gespeichert in {output_path} und {best_code_path}")

    def optimize_code(self) -> str:
        """
        Führt den Reinforcement-Learning-Prozess durch.
        
        Returns:
            Der optimierte Code
        """
        feedback = None
        
        for iteration in range(1, MAX_ITERATIONS + 1):
            logging.info(f"Starte Iteration {iteration}/{MAX_ITERATIONS}")
            
            # Code generieren
            thinking, code = self.generate_code(feedback)
            logging.info(f"Thinking:\n{thinking[:200]}...")
            
            if code is None:
                logging.warning("Kein gültiger Code erhalten.")
                feedback = "Der generierte Code war ungültig. Bitte stelle sicher, dass du gültigen Python-Code zwischen den Markierungen generierst."
                continue
            
            # Code ausführen
            output, error = self.run_code_in_docker(code)
            
            # Ergebnis evaluieren
            evaluation, score = self.evaluate_result(output, error)
            logging.info(f"Bewertung: {evaluation} (Score: {score:.2f})")
            
            # KI um Feedback bitten
            satisfaction_response = self.ask_if_satisfied(output, error, evaluation, score)
            satisfied = self.is_satisfied(satisfaction_response)
            
            # Iteration speichern
            self.save_iteration(iteration, thinking, code, output, error, 
                              satisfaction_response, evaluation, score)
            
            # Abbruchbedingung: KI ist zufrieden oder maximale Iterationen erreicht
            if satisfied:
                logging.info("Die KI ist mit dem Ergebnis zufrieden. Beende Optimierung.")
                break
            
            # Feedback für nächste Iteration
            feedback = f"""
Bewertung: {evaluation} (Score: {score:.2f})

KI-Feedback:
{satisfaction_response}
"""
            
            logging.info(f"Iteration {iteration} abgeschlossen. Fortfahren mit nächster Iteration.")
            time.sleep(1)  # Kleine Pause
        
        # Ergebnisse speichern
        self.save_results()
        
        return self.best_code

# Hauptfunktion
def main():
    """Hauptfunktion zum Starten des Optimierungsprozesses."""
    # Aufgabe definieren
    task_description = """
Schreibe ein Python-Skript, das die Zahl Pi approximiert. Stelle sicher, dass keine Benutzereingabe (`input()`) erforderlich ist. Die Werte müssen automatisch gewählt werden.

Das Skript sollte mehrere verschiedene Approximationsalgorithmen implementieren und deren Ergebnisse vergleichen:
1. Monte-Carlo-Methode
2. Leibniz-Reihe
3. Nilakantha-Reihe
4. Eine weitere Methode deiner Wahl

Zeige für jede Methode die Approximation und die Rechenzeit an. Achte auf Effizienz und numerische Stabilität.
"""

    # Optimierer initialisieren und starten
    optimizer = RLCodeOptimizer(task_description)
    optimized_code = optimizer.optimize_code()
    
    print("\n\n=== OPTIMIERTER CODE ===\n")
    print(optimized_code)
    print("\n=== ENDE DES OPTIMIERTEN CODES ===")

if __name__ == "__main__":
    main()
