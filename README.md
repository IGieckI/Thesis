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

Results su i vari set:\
- ECLI: Ricerca meh
- Eur-Lex: Ha qualcosa
- Normattiva: Le ricerche con "antinomia" o "antinomie" risultano solamente in leggi che dicono di voler stare attenti alle antinomie
- Suprema Corte di Cassazione: disponibili gratuitamente solo gli ultimi 5anni, altrimenti abbonamento -> https://www.italgiure.giustizia.it/sncass/, c'è della roba però è da cpire se è utile
- Consulta Online: anche qui sono presenti diverse decisioni relative alle antinomie
- HUDOC: Solo 3 documenti disponibili relativi alle antinomie :,(

Ideas:
- si potrebbe chiedere ad un LLM di generare dei quiz basandosi sulle leggi