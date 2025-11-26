---
marp: true
theme: agentic-ai
paginate: true
backgroundColor: #fff
---

<!-- _class: lead -->

# Lezione 1
## Introduzione ai Large Language Models e ai Transformers

Corso: Sviluppo di Sistemi Agentici AI per l'Analisi di Campagne Pubblicitarie mediante Framework Langchain

---

## Agenda della Lezione

**Prima Parte: Fondamenti Teorici**
- Definizione e caratteristiche dei Large Language Models
- Architettura Transformer e componenti fondamentali

**Seconda Parte: Panoramica dei Modelli**
- Analisi comparativa dei principali modelli disponibili
- Criteri di selezione per diversi scenari applicativi

**Terza Parte: Applicazioni Pratiche**
- Casi d'uso in ambito aziendale
- Esercitazioni guidate

Durata: 2 ore

---

## Obiettivi di Apprendimento

Al termine di questa lezione gli studenti saranno in grado di:

- Comprendere i principi di funzionamento dei Large Language Models e il loro ruolo nell'elaborazione del linguaggio naturale
- Spiegare i concetti fondamentali dell'architettura Transformer: tokenizzazione, meccanismo di self-attention e context window
- Confrontare le caratteristiche distintive dei principali modelli disponibili sul mercato
- Identificare casi d'uso appropriati per l'applicazione degli LLM in contesti aziendali
- Valutare quale modello risulta più adatto per specifici scenari operativi

---

<!-- _class: lead -->

# Parte 1
## Definizione e Caratteristiche dei Large Language Models

---

## Large Language Models: Definizione

I **Large Language Models** sono modelli di intelligenza artificiale addestrati su vasti corpus testuali con le seguenti capacità:

**Comprensione del linguaggio naturale**
- Analisi semantica e contestuale del testo
- Interpretazione di query complesse

**Generazione di contenuto**
- Produzione di testo coerente e grammaticalmente corretto
- Mantenimento della coerenza contestuale

**Versatilità operativa**
- Esecuzione di task diversificati senza necessità di riprogrammazione
- Adattamento a nuovi compiti mediante few-shot learning

Caratteristica distintiva: apprendimento di pattern linguistici direttamente dai dati, senza dipendenza da regole esplicite predefinite.

---

## Percorso Evolutivo: Dal Machine Learning agli LLM

**Evoluzione Tecnologica**

Machine Learning Tradizionale → Deep Learning → Transformers → Large Language Models

**Timeline**
- 2012-2016: Diffusione del Deep Learning per task specifici
- 2017: Introduzione dell'architettura Transformer
- 2018-2020: Primi modelli pre-addestrati su larga scala
- 2020-oggi: Era dei Large Language Models general-purpose

**Paradigm shift**: transizione da modelli specializzati per singoli task a modelli versatili capaci di adattarsi a molteplici applicazioni senza riaddestramentо.

---

## Capacità Fondamentali degli LLM

<div class="columns">
<div>

### Comprensione del Linguaggio
- Analisi contestuale e semantica
- Disambiguazione di significati multipli
- Riconoscimento e classificazione di entità
- Analisi del sentiment e delle intenzioni

</div>
<div>

### Generazione di Contenuto
- Produzione di testo grammaticalmente corretto
- Traduzioni multilingue
- Sintesi e riassunti di documenti
- Risposte articolate a domande complesse

</div>
</div>

**Capacità emergente**: Few-shot learning - capacità di apprendere nuovi task con un numero limitato di esempi dimostrativi, senza necessità di riaddestramentо del modello.

---

<!-- _class: lead -->

# Parte 2
## L'Architettura Transformer

---

## Transformer: Innovazione Fondamentale

**Paper di riferimento**: "Attention is All You Need" (Vaswani et al., 2017)

**Innovazione principale**
L'introduzione del meccanismo di **self-attention** che consente l'elaborazione parallela di tutti i token di una sequenza, superando i limiti sequenziali delle architetture precedenti.

**Impatto**
Questa architettura ha permesso lo sviluppo di modelli significativamente più efficienti e capaci di catturare relazioni semantiche anche tra elementi distanti nel testo.

**Rilevanza attuale**
Tutti i moderni Large Language Models (GPT, Claude, Llama, Mistral) implementano varianti di questa architettura fondamentale.

---

## Token: L'Unità Base di Elaborazione

**Definizione**
Il token rappresenta l'unità minima di elaborazione testuale nei Large Language Models.

**Tipologie di token**
Un token può corrispondere a:
- Una parola completa (esempio: "pubblicitaria")
- Una porzione di parola (esempio: "pubbli" + "citaria")
- Un singolo carattere o simbolo (esempio: "@", "€")
- Elementi di punteggiatura (esempio: ",", ".")

**Esempio pratico di tokenizzazione**
Testo originale: "Analisi campagna pubblicitaria"
Output tokenizzato: ["Anal", "isi", " campagna", " pubbl", "icitaria"]

---

## Processo di Tokenizzazione: Esempio Dettagliato

**Input testuale**
"La reach della campagna è del 45% sul target A25-54."

**Output dopo tokenizzazione** (circa 15 token)
["La", " reach", " della", " campagna", " è", " del", " ", "45", "%", " sul", " target", " A", "25", "-", "54", "."]

**Considerazione importante per la lingua italiana**
Il rapporto medio è di circa 1 token per 0.75 parole. Questo valore può variare significativamente in base alla morfologia della lingua e alla complessità del vocabolario tecnico utilizzato.

---

## Self-Attention: Meccanismo Fondamentale del Transformer

Il meccanismo di **self-attention** costituisce il componente chiave dell'architettura Transformer e permette al modello di:

**Elaborazione parallela**
Analizzare simultaneamente tutti i token di una sequenza, superando i vincoli di elaborazione sequenziale.

**Pesatura contestuale**
Assegnare coefficienti di attenzione variabili a ciascun token in base alla rilevanza contestuale rispetto agli altri elementi della sequenza.

**Cattura di dipendenze a lungo raggio**
Identificare e modellare relazioni semantiche anche tra token distanti nella sequenza testuale.

**Esempio illustrativo**
Nella frase "Il target della campagna ha superato gli obiettivi", durante l'elaborazione del termine "obiettivi", il modello assegna pesi di attenzione maggiori ai token "campagna" e "target".

---

## Visualizzazione del Meccanismo di Self-Attention

![width:800px](https://jalammar.github.io/images/t/transformer_self-attention_visualization.png)

Fonte: The Illustrated Transformer - Jay Alammar

Le connessioni visualizzate rappresentano i pesi di attenzione che ogni token assegna agli altri elementi della sequenza. L'intensità delle connessioni indica la rilevanza contestuale tra i token.

---

## Context Window: Definizione e Implicazioni

**Definizione**
Il context window rappresenta la quantità massima di testo che un modello può elaborare in una singola inferenza. Questa capacità è misurata in numero di token, non in parole o caratteri.

| Modello        | Context Window | Equivalente Approssimativo |
|----------------|---------------|----------------------------|
| GPT-3.5        | 4,096 token   | circa 3,000 parole         |
| GPT-4          | 128,000 token | circa 96,000 parole        |
| Claude 3       | 200,000 token | circa 150,000 parole       |
| Gemini 2.5 Pro | 1,048,576 token | circa 900,000 parole       |

**Implicazione operativa**
La dimensione del context window determina la lunghezza dei documenti che possono essere analizzati in una singola elaborazione, influenzando significativamente le applicazioni possibili.

---

## Context Window: Applicazioni Pratiche

**Scenario applicativo: Analisi di report mensili**

Report breve (5 pagine, circa 2,000 token)
- Compatibile con tutti i modelli disponibili

Report medio (20 pagine, circa 8,000 token)
- Richiede GPT-4, Claude 3 o modelli equivalenti

Report esteso (100 pagine, circa 40,000 token)
- Richiede modelli con context window esteso come Claude 3

**Regola pratica per la stima**
Una pagina standard di testo corrisponde approssimativamente a 400-500 token, considerando formattazione e spaziatura tipiche di documenti professionali.

---

<!-- _class: lead -->

# Parte 3
## Panoramica dei Principali Large Language Models

---

## GPT (Generative Pre-trained Transformer)

**Sviluppatore**: OpenAI

**Caratteristiche tecniche distintive**
- Famiglia di modelli proprietari accessibili tramite API
- GPT-4 rappresenta attualmente uno degli LLM più avanzati disponibili
- Capacità superiori nel ragionamento complesso e nei task creativi
- Context window fino a 128,000 token

**Scenari applicativi ottimali**
- Generazione di contenuti per campagne marketing
- Analisi complesse che richiedono ragionamento multi-step
- Task creativi e supporto alle attività di brainstorming
- Applicazioni che richiedono comprensione contestuale profonda

---

## Claude

**Sviluppatore**: Anthropic

**Caratteristiche tecniche distintive**
- Architettura progettata con enfasi su sicurezza e affidabilità delle risposte
- Context window particolarmente esteso (200,000 token)
- Performance eccellenti nell'analisi di documenti lunghi e complessi
- Disponibile tramite API con diverse versioni (Haiku, Sonnet, Opus)

**Scenari applicativi ottimali**
- Analisi dettagliata di report estesi e documentazione tecnica
- Summarization accurata di documenti voluminosi
- Task che richiedono particolare precisione e affidabilità
- Applicazioni dove la tracciabilità del ragionamento è critica

---

## Llama

**Sviluppatore**: Meta AI

**Caratteristiche tecniche distintive**
- Famiglia di modelli open-source con licenza permissiva
- Disponibile in diverse dimensioni (7B, 13B, 70B parametri)
- Possibilità di esecuzione in ambiente locale (on-premise)
- Supporto per fine-tuning su domini specifici

**Scenari applicativi ottimali**
- Deployment on-premise per applicazioni con requisiti stringenti di privacy
- Progetti che richiedono personalizzazione estensiva del modello
- Ambienti dove è necessario controllo completo sui dati e sull'infrastruttura
- Applicazioni che beneficiano da ottimizzazioni specifiche per il dominio

---

## Mistral

**Sviluppatore**: Mistral AI (Francia)

**Caratteristiche tecniche distintive**
- Famiglia di modelli europei, sia open-source che proprietari
- Efficienza computazionale superiore rispetto alle dimensioni
- Capacità multilingue particolarmente sviluppate
- Diverse varianti disponibili (Mistral 7B, Mixtral 8x7B, Mistral Large)

**Scenari applicativi ottimali**
- Applicazioni multilingue con focus sul mercato europeo
- Deployment con vincoli di risorse computazionali
- Scenari che richiedono latenza ridotta
- Applicazioni che necessitano di bilanciamento tra performance e costi

---

## Confronto Modelli: Tabella Sinottica

| Caratteristica | GPT-4 | Claude 3 | Llama 3 70B | Mistral Large |
|---------------|-------|----------|-------------|---------------|
| **Tipo** | Proprietario | Proprietario | Open-source | Ibrido |
| **Context** | 128K | 200K | 8K | 32K |
| **Parametri** | ~1.7T | ~200B | 70B | ~100B |
| **Costo** | $$$ | $$$ | Gratuito* | $$ |
| **Deploy** | API | API | Locale/Cloud | API/Locale |

*Self-hosting richiede infrastruttura

---

## Criteri di Selezione del Modello

**Fattori da considerare nella scelta del modello appropriato**

**Dimensione del context window**
Valutare la lunghezza tipica dei documenti da elaborare e scegliere un modello con context window adeguato.

**Budget e sostenibilità economica**
Considerare il costo per token e il volume previsto di elaborazioni per stimare l'impatto economico.

**Requisiti di privacy e sicurezza**
Per dati sensibili, valutare soluzioni open-source con deployment on-premise.

**Capacità multilingue**
Verificare il supporto per le lingue necessarie e la qualità delle performance in ciascuna lingua.

**Requisiti di latenza**
Considerare i tempi di risposta richiesti dall'applicazione e le caratteristiche di throughput necessarie.

Non esiste un modello universalmente superiore, ma è necessario selezionare la soluzione più adatta al caso d'uso specifico.

---

<!-- _class: lead -->

# Parte 4
## Applicazioni Pratiche in Ambito Business

---

## Applicazione 1: Assistenti Conversazionali

**Caso d'uso**: Chatbot per analisi campagne

**Funzionalità**:
```
Utente: "Qual è stata la reach della campagna di marzo?"

Assistente: "La campagna di marzo ha raggiunto una reach 
del 48% sul target Adults 25-54, con 3.2 milioni 
di contatti unici e una frequency media di 3.8."
```

**Vantaggi**:
- Interfaccia naturale per utenti non tecnici
- Disponibilità 24/7
- Riduzione carico su analisti

---

## Applicazione 2: Analisi di Documenti

**Caso d'uso**: Estrazione KPI da report testuali

**Input**: Report PDF/Word non strutturato
**Output**: Dati strutturati (JSON, Excel)

**Esempio**:
```
Report PDF → LLM → {
  "reach": "45%",
  "frequency": 3.8,
  "impressions": 5200000,
  "target": "Adults 25-54"
}
```

**Risparmio**: Da 1 ora di lavoro manuale a 30 secondi automatizzati

---

## Applicazione 3: Generazione di Report

**Caso d'uso**: Trasformazione dati → narrativa

**Input**: Tabelle, metriche, numeri
**Output**: Report narrativo professionale

**Valore**:
- Democratizzazione dell'accesso ai dati
- Risparmio di tempo per analisti
- Standardizzazione della comunicazione
- Personalizzazione automatica per diversi stakeholder

---

## Casi d'Uso nel Settore Advertising

<div class="columns">
<div>

### Analisi delle Performance
- Interpretazione di indicatori chiave di performance
- Identificazione di trend e pattern
- Rilevamento di anomalie
- Benchmarking competitivo

### Analisi del Target
- Profilazione dettagliata dell'audience
- Segmentazione avanzata del mercato
- Estrazione di insights comportamentali

</div>
<div>

### Reporting e Comunicazione
- Sintesi esecutive automatizzate
- Generazione di report dettagliati
- Preparazione di presentazioni
- Sistema di alert automatici

### Ottimizzazione delle Campagne
- Generazione di raccomandazioni strategiche
- Analisi di test A/B
- Ottimizzazione dell'allocazione del budget

</div>
</div>

---

## Il Progetto del Corso: Sistema TTVAM

**Obiettivo**: Creare un sistema agentico che permetta a utenti non tecnici di:

1. Fare domande in linguaggio naturale su campagne pubblicitarie
2. Ottenere automaticamente dati dalle API TTVAM
3. Ricevere analisi e report comprensibili

**Esempio**:
```
Utente: "Mostrami la performance della campagna autunnale 
        su MTV per il target femminile 25-44"

Sistema: [Traduce in chiamata API] → [Analizza dati] 
         → [Genera risposta strutturata]
```

---

## Limitazioni degli LLM: Considerazioni Preliminari

**Aspetti critici da considerare nell'utilizzo degli LLM**

**Allucinazioni**
I modelli possono generare informazioni che appaiono plausibili ma risultano fatalmente inesatte, particolarmente in assenza di verifiche incrociate.

**Knowledge cutoff**
La conoscenza del modello è limitata alla data di conclusione del training, con conseguente impossibilità di accedere a informazioni successive.

**Bias algoritmici**
I modelli possono riflettere e amplificare bias presenti nei dati di addestramento, richiedendo particolare attenzione in contesti sensibili.

**Struttura dei costi**
Le API commerciali applicano tariffe basate sul volume di token elaborati, rendendo necessaria un'attenta pianificazione economica.

**Considerazioni sulla privacy**
L'invio di dati sensibili a API esterne richiede valutazione attenta dei rischi e conformità normativa.

Questi aspetti verranno approfonditi nella prossima lezione.

---

## Sintesi della Lezione

**Contenuti affrontati nella lezione odierna**

**Definizione dei Large Language Models**
Comprensione delle caratteristiche fondamentali e delle capacità dei modelli pre-addestrati su larga scala per l'elaborazione del linguaggio naturale.

**Architettura Transformer**
Analisi dei componenti chiave: tokenizzazione, meccanismo di self-attention e gestione del context window.

**Panoramica dei modelli principali**
Confronto dettagliato tra GPT, Claude, Llama, Mistral e identificazione delle loro caratteristiche distintive.

**Applicazioni in ambito business**
Esplorazione di casi d'uso concreti: assistenti conversazionali, analisi di documenti e generazione automatizzata di report.

**Progetto del corso**
Introduzione al sistema agentico per l'analisi delle campagne pubblicitarie TTVAM.

---

## Anteprima della Prossima Lezione

**Lezione 2: Funzionamento Operativo e Limitazioni degli LLM**

**Argomenti che verranno affrontati**

Tecniche di prompt engineering per ottimizzare l'interazione con i modelli e migliorare la qualità delle risposte ottenute.

Strategie per la gestione delle allucinazioni e la mitigazione dei bias algoritmici presenti nei modelli.

Best practices per l'utilizzo degli LLM in contesti produttivi, con particolare attenzione agli aspetti di affidabilità e monitoraggio.

Identificazione dei limiti intrinseci degli LLM e analisi dei contesti in cui il loro utilizzo risulta inappropriato o subottimale.

---

## Risorse per l'Approfondimento

**Letteratura scientifica di riferimento**
- Vaswani et al. (2017): "Attention is All You Need" - Paper fondamentale sull'architettura Transformer
- Brown et al. (2020): "Language Models are Few-Shot Learners" - Introduzione di GPT-3 e del paradigma few-shot

**Documentazione tecnica ufficiale**
- OpenAI Platform Documentation: https://platform.openai.com/docs/
- Anthropic Claude Documentation: https://docs.anthropic.com/
- Hugging Face NLP Course: https://huggingface.co/learn/nlp-course/

**Materiali didattici supplementari**
- "The Illustrated Transformer" di Jay Alammar - Visualizzazione intuitiva dell'architettura
- Tutorial e guide di Sebastian Raschka sugli LLM - Approfondimenti tecnici avanzati

---
