import ollama
import subprocess
import os
import time
import re

# Docker-Image für Python
DOCKER_IMAGE = "python:3.11"

# Kennwort für Code-Markierung
CODE_START = "### CODE START ###"
CODE_END = "### CODE END ###"

# Prompt-Vorlage für die KI
PROMPT_TEMPLATE = f"""
Du bist eine KI, die Python-Code generiert und optimiert.  
Du hast KEINEN menschlichen Zugriff auf Feedback.  
Du musst selbst entscheiden, ob dein Code gut ist oder ob er verbessert werden muss.  
Falls dein Code nicht perfekt ist, versuche ihn zu verbessern.  
Falls du zufrieden bist, schreibe: "Ja, das Skript ist optimal."  
Falls du Verbesserungen siehst, analysiere das Problem und überarbeite den Code.  

**WICHTIG:** Du darfst KEINE Benutzereingaben (`input()`) verwenden.  
Falls `input()` genutzt wird, muss es durch eine selbstgewählte feste Variable ersetzt werden.  
Du bist in einer geschlossenen Umgebung und kannst keine Hilfe erwarten. Du bist allein für die Optimierung verantwortlich.  

**Formatierung:**  
- Teile deine Gedanken zuerst als normale Ausgabe.  
- Danach schreibe den Code zwischen {CODE_START} und {CODE_END}.  
- Schreibe NUR reinen ausführbaren Code ohne Markdown-Blöcke (```python ... ```).  

Die Aufgabe lautet: {{}}  
"""

# Funktion: Generiert Code mit Ollama
def generate_code(task_description):
    prompt = PROMPT_TEMPLATE.format(task_description)
    response = ollama.chat(model="deepseek-r1:32b", messages=[{"role": "user", "content": prompt}])
    
    content = response["message"]["content"]

    # Thinking-Teil extrahieren
    thinking = content.split(CODE_START)[0].strip()

    # Code extrahieren & Markdown-Reste entfernen
    code_match = re.search(rf"{CODE_START}(.*?){CODE_END}", content, re.DOTALL)
    if code_match:
        code = code_match.group(1).strip()
        code = re.sub(r"```python|```", "", code).strip()  # Entfernt Markdown-Formatierung
        
        # Überprüfe, ob `input()` vorkommt
        if "input(" in code:
            print("\n⚠️ Der generierte Code enthält `input()`, was nicht erlaubt ist. Die KI muss ihn ersetzen.")
            return thinking, None
    else:
        code = None  # Kein Code erhalten

    return thinking, code

# Funktion: Führt den Code im Docker-Container aus
def run_code_in_docker(code):
    script_path = os.path.join(os.getcwd(), "script.py")

    # Speichere den Code in einer Datei
    with open(script_path, "w") as f:
        f.write(code)

    # Starte den Docker-Container und führe das Skript aus
    result = subprocess.run(
        ["docker", "run", "--rm", "-v", f"{os.getcwd()}:/app", DOCKER_IMAGE, "python", "/app/script.py"],
        capture_output=True, text=True
    )

    return result.stdout, result.stderr

# Funktion: Fragt die KI, ob das Skript bereits optimal ist
def ask_if_satisfied(task_description, output, error):
    prompt = f"""
Das Skript wurde getestet.

📜 **Terminal-Output:**
{output}

⚠️ **Fehler:**
{error if error else 'Keine Fehler'}

Du hast KEINEN menschlichen Zugriff auf Feedback.  
Falls dein Code nicht perfekt ist, musst du ihn verbessern.  
Falls du zufrieden bist, schreibe: "Ja, das Skript ist optimal."  
Falls du Verbesserungen siehst, analysiere das Problem und überarbeite den Code.  
"""
    response = ollama.chat(model="deepseek-r1:32b", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"].strip()

# Hauptschleife für das Reinforcement Learning der KI
def reinforcement_loop():
    task_description = "Schreibe ein Python-Skript, das die Zahl Pi approximiert. Stelle sicher, dass keine Benutzereingabe (`input()`) erforderlich ist. Die Werte müssen automatisch gewählt werden."

    iteration = 0
    while True:
        iteration += 1
        print(f"\n🔄 Iteration {iteration}...")

        thinking, code = generate_code(task_description)

        print("\n🧠 Thinking:\n", thinking)

        if code is None:
            print("\n⚠️ Kein gültiger Code erhalten. Die KI muss weiter optimieren.")
            continue  # Keine Beendigung, sondern eine weitere Iteration

        print("\n📝 Generierter Code:\n", code)

        output, error = run_code_in_docker(code)
        print("\n📜 Terminal Output:\n", output)

        # KI entscheidet, ob sie fertig ist
        satisfaction_response = ask_if_satisfied(task_description, output, error)
        print("\n🤖 KI-Antwort:", satisfaction_response)

        if "Ja, das Skript ist optimal" in satisfaction_response:
            print("\n✅ Die KI ist zufrieden. Beende den Prozess.")
            break  # Beende die Schleife

        # Falls die KI nicht zufrieden ist, überarbeitet sie den Code
        print("\n🔄 Die KI optimiert weiter, da sie nicht zufrieden ist.")
        task_description = f"{task_description} \nVerbessere den Code: {satisfaction_response}"
        time.sleep(2)  # Verhindert Spam

# Starte den Reinforcement-Learning-Prozess
reinforcement_loop()
