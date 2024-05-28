# Thesis
I will keep here all important advancements of the thesis work.\
\
[Notion page](https://www.notion.so/Lawyer-LLM-c764972bb5964a0b88e711029cc1ca6e?pvs=4)\
\
Link possibili fonti di dati:\
[Normattiva](https://www.normattiva.it/staticPage/codici): Official webpage of italian governament, this contains all of the italian laws categorized, that can be downloaded as XML, PDF, EPUB and RTF, there is also an "Akaoma ntoso" format, which is apparently adopted also in other countries even if there isn't a real converter online.\
note: For the section "diritto penale", the download of the 4 file types didn't work so i sent an e-mail, hope i won't have to convert the akaoma ntoso file.\
[Scuola Zincani](https://www.formazionegiuridica.org/quiz-autovalutazione-esame-avvocatura): Dataset non disponibile (privato), ma ci sono quiz con alcune domande con riferimenti agli articoli, sarebbe possbile comperarli?\
[API for italian developers](https://developers.italia.it/it/api.html): there might be something\
[Quiz-Concorsi-Online](https://www.quiz-concorsi-online.com/item.php?pgCode=G28I220R466&js_status=js_is_on): There is one quiz of the type we need, but there is no download for that, i wrote an email to them, waiting for an answer.\
[Mininterno](https://www.mininterno.net/begint.asp?idc=527#google_vignette): Another private dataset that seem to be particularly good, but they dont offer any download, i sent an email, but they will probably ask for money.\
[Gazzetta Ufficiale HuggingFace](https://huggingface.co/datasets/mii-llm/gazzetta-ufficiale): Dataset quite full with legislative texts, public and private acts.\
[Agenzia per l'Italia Digitalia](https://dati.gov.it/): Banca dati d'Italia che però sembra non avere dati utili 💀💀💀 ([Data sciencing in Italy](https://forum.italia.it/t/normattiva-open-data/536))\
Quiz sul diritto penale: dataset girato da Cristiano Casadei via Email\
Costituzione: dataset girato da Lorenzo Molfetta via Teams\

Results su i vari set:\
- ECLI: Ricerca meh.\
 POST request molto complessa.
- Eur-Lex: Ha qualcosa.\
 Le richieste avvengono tramite GET ma bloccano le richieste per ragioni di sicurezza.
- Normattiva: Le ricerche con "antinomia" o "antinomie" risultano solamente in leggi che dicono di voler stare attenti alle antinomie.\
 La GET request restituisce un internal server error, utilizzando i permalink offerti sul sito invece, si può ottenere solo un link al sito dell'atto (ma non dei singoli articoli).
- Suprema Corte di Cassazione: disponibili gratuitamente solo gli ultimi 5anni, altrimenti abbonamento -> https://www.italgiure.giustizia.it/sncass/, c'è della roba però è da cpire se è utile\
 pdf ottenibili tramite loro motore di ricerca.
- Consulta Online: anche qui sono presenti diverse decisioni relative alle antinomie\
 anche qui solo file pdf ottenibili tramite loro motore di ricerca.
- HUDOC: simile agli altri
 API non reperibili e analisi delle richieste dalla rete non analizzabili
- Juris: dataset privato
 
\
Download di ogni codice:
[X] Costituzione
[ ] Codice civile
[ ] Codice penale
[X] Codice di procedura civile
[ ] Codice di procedura penale
[ ] Codice del processo amministrativo
[ ] Codice della strada
[ ] Statuto dei lavoratori
[ ] Codice tributario
[ ] Codice di commercio
\
Ideas:
- si potrebbe chiedere ad un LLM di generare dei quiz basandosi sulle leggi


Notes:
Sentence: L'Italia è una Repubblica democratica, fondata sul lavoro. La sovranità appartiene al popolo, che la esercita nelle forme e nei limiti della Costituzione.

LegalBert: 10s
Multilingual Bert: 4.89s
Roberta: 8.35s

Saul: 152s + 47.32s
ChatLaw: 5.04s
Meta-Llama3: 85s + 114.33s
MPT-7B: 74s + 100.37s
Falcon-7B: 74s + 103.44s

NOTA PER LA TESI: NON ABBIAMO UN DATASET, PROBLEMA GENERALE NON SOLO PER NOI (NEL NOSTRO CASO IN PARTICOLARE NON SAPPIAMO PROPRIO SE DUE LEGGI SONO ANTINOMIE)
Per testare il modello migliore fagli rispondere ai quiz
1. Prova a dividere i quiz e mandali a Llama3 8B/70B, Mistral 45B, Minerva, Openllama, Gemma quantizzato chiedendogli "se rilevi una legge cercala" creando una tabella con "chunk | quiz_ref | legge | numero | link"
2. Generare il testo della legge utilizzando quattro modelli differenti e valutali
3. Generare un unica tabella dalle quattro, magari ad alcui è sfuggito
4. Provare su tipi di file diversi xml, rdf... per provare ad estrarre le varie leggi
5. Prova a far generare ai modelli una nuova tabella contenenti una legge simile
6. Fai un controllo a campione anche usando un modello per capire la qualità 
7. Sostituzione del riferimento di una legge con il testo di una legge simile a quella referenziata, se la risposta alla stessa domanda è sbagliata (da parte di più modelli) vuol dire che il significato è sbagliato (o in contrasto)
8. Brute Force (ultimo modo): utilizzo di modelli non open source, 

Prova a passargli sia a mo di domanda+risposta corretta, che chunk di domande
Prova a definirgli come fare la risposta

Una volta finito, chiedi ai tre modelli se esiste una legge contraria o simile emessa da un'altro organo di stato, data il quiz


Pipeline: File con leggi + prompt -> Modello(Llama...) -> Script per cliccare automaticamente sul sito (soluzione al blocco con parametri)