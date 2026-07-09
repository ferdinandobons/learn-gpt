# Guidance del tutorial LearnGPT

Questo file contiene le regole da seguire durante tutto il percorso didattico
per costruire, passo dopo passo, un piccolo progetto GPT basato su FineWeb-Edu.

## Obiettivo

Costruire gradualmente buona parte del progetto `learn-gpt`, ma in una versione
didattica dentro `LearnGPT`. La primissima parte del corso usa un campione
FineWeb-Edu leggibile e piccolo:

```text
LearnGPT/data/raw/fineweb_edu_sample.txt
```

La parte finale del progetto usa invece FineWeb-Edu processato con tokenizer
BPE in:

```text
LearnGPT/data/processed/fineweb_edu/
```

Lo scopo principale non è solo arrivare al codice finale, ma capire cosa fa
ogni pezzo e perché serve.

## Regole del percorso

1. Procedere piano, uno step alla volta.
2. Scrivere codice in piccoli blocchi copiabili e incollabili.
3. Spiegare ogni nuovo pezzo di codice prima di passare al successivo.
4. Non introdurre concetti avanzati prima che il pezzo precedente sia chiaro.
5. Dopo ogni file o modifica importante, eseguire un test manuale di avvio.
6. Se un comando produce errore, fermarsi e correggere quello prima di andare
   avanti.
7. Usare nomi semplici e leggibili, anche se meno compatti.
8. Preferire codice esplicito a codice troppo furbo.
9. Collegare ogni passaggio al progetto GPT completo: testo, token, batch,
   modello, loss, training, generazione.
10. Rispondere ai dubbi prima di proseguire con nuovi step.
11. All'inizio di ogni file, anche quelli già creati, indicare sempre:
   cosa cambia rispetto ai file precedenti e qual è lo scopo del file.
12. Quando si scrive testo italiano in docstring, commenti didattici o
   indicazioni a inizio file, usare lettere accentate corrette: `è`, `perché`,
   `può`, `più`, `già`, ecc.
13. La root `LearnGPT` deve restare ordinata: contiene `guidance.md`,
   `corso.md`, la cartella `data` per dataset condivisi e processati, la
   cartella `studio/lezioni` per i file numerati, `studio/snapshot` per gli
   snapshot didattici e `progetto_finale` per il codice finale vivo.
14. Ogni lezione deve avere uno snapshot completo nella cartella
   `studio/snapshot` con numero a due cifre, per esempio
   `studio/snapshot/lezione_28`. Gli script dentro `studio/lezioni` devono
   importare dal proprio snapshot, non dal progetto finale. Esempio:
   `from studio.snapshot.lezione_28.tokenizer import create_vocabulary`.
15. Le dipendenze Python del progetto finale vanno dichiarate dentro
   `progetto_finale/requirements.txt`, così la root `LearnGPT` resta pulita.
16. Per eseguire gli script usare il comando `python`, che in questo progetto
   punta al Python gestito da pyenv. Evitare `python3` se richiama il Python di
   sistema senza PyTorch.
17. Mantenere `corso.md` come documento vivo del percorso. Alla fine del corso
   dovrà contenere tutte le lezioni, i pezzi di codice aggiunti passo passo,
   le spiegazioni, i chiarimenti extra e il codice completo della parte
   interessata. Alla fine finale verrà usato come base per generare un PDF.
18. Da ora in poi, ogni nuova lezione e ogni spiegazione extra richiesta
   dall'utente devono essere aggiunte anche a `corso.md`, nella stessa sezione
   della lezione a cui si riferiscono.
19. In `corso.md`, quando una spiegazione descrive un flusso, una pipeline o
   una relazione tra componenti, preferire diagrammi Mermaid rispetto a grafici
   puramente testuali. Usare testo semplice solo quando serve mostrare forme di
   tensori, liste o esempi di output.
20. All'inizio di `corso.md` deve rimanere una mappa generale delle
   trasformazioni del progetto in due versioni:
   - una versione concisa, utile per orientarsi velocemente;
   - una versione estesa, utile per seguire tutte le trasformazioni importanti.
   Quando una nuova lezione aggiunge un passaggio rilevante al flusso dei dati
   o del modello, aggiornare entrambe se necessario. La versione estesa deve
   includere almeno: preparazione dati, tokenizer BPE, `train.bin`/`val.bin`,
   memmap, batch input/target, token embeddings, position embeddings,
   TransformerBlock, attention, MLP, final LayerNorm, `output_head`, logits,
   loss, backward, optimizer, checkpoint e generazione.
21. Quando si crea o modifica una lezione, fare sempre questa checklist:
   aggiornare lo snapshot corrispondente, per esempio
   `studio/snapshot/lezione_28`, aggiornare `progetto_finale` se il codice
   finale evolve, controllare gli import dello script numerato in
   `studio/lezioni`, per esempio `studio/lezioni/28_training_transformer.py`,
   controllare i path del dataset, aggiornare `corso.md`, verificare
   Pylance/import e lanciare almeno lo script della lezione.
22. Quando una lezione prende ispirazione da codice compatto di nanoGPT, mostrare
    sempre il collegamento tra la versione compatta e la versione didattica
    espansa. Lo scopo è permettere a chi studia di riconoscere lo stesso
    meccanismo nei due stili di codice. Per esempio, se nanoGPT scrive:

    ```python
    x = x + self.attn(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x))
    ```

    nel corso bisogna mostrare anche la versione esplicita:

    ```python
    attention_input = self.attention_layer_norm(embeddings)
    attention_output, _ = self.multi_head_attention(attention_input)
    residual_after_attention = embeddings + attention_output

    feed_forward_input = self.feed_forward_layer_norm(residual_after_attention)
    feed_forward_output = self.feed_forward(feed_forward_input)
    residual_after_feed_forward = residual_after_attention + feed_forward_output
    ```

    La spiegazione deve chiarire che la differenza principale è il livello di
    compattezza:

    | Versione | Stile |
    | --- | --- |
    | nanoGPT | codice compatto, adatto a un progetto già maturo |
    | `LearnGPT` | codice più verboso, adatto a seguire ogni passaggio |

    Questa regola vale per tutti i punti in cui il progetto didattico segue la
    direzione di nanoGPT: prima mostrare il codice compatto di riferimento, poi
    scomporlo in passaggi con nomi intermedi chiari.
23. Dentro `studio/snapshot`, ogni snapshot deve contenere solo il codice
    realmente necessario alla lezione corrente. Non trascinare classi storiche
    non usate, per esempio un vecchio modello bigram dentro una lezione sui
    TransformerBlock.
24. Negli script dentro `studio/lezioni`, il modello principale della lezione deve
    essere importato con il nome stabile `LanguageModel` dal proprio snapshot:
    `from studio.snapshot.lezione_28.model import LanguageModel`. Il path
    della lezione indica la versione del modello; il nome della classe resta
    stabile per ridurre rumore mentale durante lo studio.
25. Tutto il codice Python deve usare identificatori in inglese: nomi di
    variabili, funzioni, classi, argomenti e attributi. Le spiegazioni del
    corso, le docstring iniziali, i commenti didattici e i testi stampati
    restano in italiano corretto. Esempio: `text`, `model`, `create_batch`,
    `char_to_id`; non `testo`, `modello`, `crea_batch`, `carattere_a_id`.
26. I moduli Python operativi del progetto devono usare nomi file inglesi,
    semplici e coerenti. Per addestramento e generazione usare `training.py` e
    `generate.py`; non usare `generation.py` o nomi file italiani per questi
    moduli.
27. Dopo modifiche a lezioni, snapshot, `corso.md` o `guidance.md`, eseguire il
    validatore locale:

    ```bash
    python -B LearnGPT/strumenti/validate_learngpt.py
    ```

    Questo validatore è specifico per la struttura scelta in `LearnGPT`, dove
    il corso vive in un unico `corso.md` invece che in file Markdown numerati.
28. I dataset grandi non vanno duplicati dentro ogni snapshot di lezione. Per
    dataset come FineWeb-Edu usare una fonte comune sotto `LearnGPT/data/`,
    preferendo file processati come `train.bin`, `val.bin` e `meta.json`.
    Gli snapshot possono contenere piccoli dati storici solo quando servono a
    mantenere rieseguibili le vecchie lezioni; per dati grandi o definitivi
    bisogna referenziare il dataset condiviso.
29. La progressione didattica sul tokenizer deve essere questa: prima un
    approccio molto grezzo a caratteri, solo per capire che il testo diventa
    numeri; subito dopo un tokenizer BPE più realistico con `tiktoken` e
    FineWeb-Edu; poi batching, modello e Transformer devono iniziare a
    collegarsi alla rappresentazione realistica del progetto finale. FineWeb,
    BPE, `train.bin`, `val.bin` e memmap non devono restare argomenti solo
    finali.
30. L'ultimo script numerato del percorso deve chiamarsi `Final Project`, per
    esempio `42_final_project.py`. Le lezioni precedenti possono introdurre,
    correggere o ottimizzare pezzi del codice, ma il file finale deve essere una
    versione pulita end-to-end e deve rispecchiare il contenuto di
    `progetto_finale/`.
31. Lo snapshot dell'ultima lezione, per esempio
    `studio/snapshot/lezione_42/`, deve essere identico ai moduli finali in
    `progetto_finale/` per tutti i file Python operativi e per
    `requirements.txt`. Se si modifica `progetto_finale/`, bisogna sincronizzare
    anche lo snapshot finale e rilanciare il validatore. Questa regola evita che
    il `Final Project` didattico e il progetto finale reale divergano.
32. `progetto_finale/prepare_data.py` fa parte del progetto finale pulito, non è
    un file accessorio da dimenticare. La pipeline finale comprende quindi anche
    preparazione dati FineWeb-Edu, tokenizer BPE, file binari processati,
    configurazione, device, modello, training, checkpoint e generazione.
33. Dopo una code review o una richiesta di fix generale, seguire il ciclo:
    applicare i fix necessari, fare una review critica dei file toccati e dei
    file collegati, correggere eventuali nuovi finding, poi verificare. Se i fix
    cambiano concetti didattici o comportamento del progetto finale, aggiornare
    anche `corso.md`.
34. Per le verifiche usare preferibilmente `python -B`, così non vengono create
    cartelle `__pycache__` nel progetto didattico. Se un comando come
    `compileall` crea `__pycache__`, rimuovere solo quei file generati prima di
    considerare chiusa la modifica. La cartella `LearnGPT` deve restare pulita.

## Metodo per ogni step

Ogni step deve seguire questa struttura:

1. Spiegazione breve dell'obiettivo.
2. Differenza rispetto ai file precedenti.
3. File da creare o modificare.
4. Codice da copiare e incollare.
5. Spiegazione delle righe importanti.
6. Comando da eseguire.
7. Output atteso o controllo da fare.
8. Collegamento concettuale con il passo successivo.
9. Aggiornamento di `corso.md` con codice, spiegazione e chiarimenti extra.
10. Aggiornamento della mappa generale in `corso.md` se la lezione cambia il
    flusso dei dati, le forme dei tensori o i passaggi del modello.
11. Quando si usa nanoGPT come riferimento, aggiungere un confronto esplicito
    tra il codice compatto originale e la versione didattica di `LearnGPT`,
    usando nomi intermedi leggibili per ogni passaggio.
12. Quando una lezione modifica il modello, mantenere `LanguageModel` come nome
    pubblico importato dagli script di studio e pulire lo snapshot della lezione
    da classi o funzioni non usate dalla lezione stessa.
13. Verificare che il codice Python usi identificatori in inglese, mentre le
    spiegazioni, le docstring iniziali e i commenti didattici restano in
    italiano.
14. Eseguire `python -B LearnGPT/strumenti/validate_learngpt.py` per
    controllare struttura, riferimenti, snapshot e blocchi di codice completi
    nel corso.
15. Se la lezione riguarda dati grandi, verificare che il dataset sia in
    `LearnGPT/data/` e non duplicato nelle cartelle `lezione_NN`.
16. Se la modifica tocca il progetto finale o lo snapshot finale, controllare
    che `progetto_finale/` e l'ultimo snapshot restino allineati. Il validatore
    deve fallire se divergono.
17. Se la modifica è ampia o riguarda regole generali, eseguire anche uno smoke
    test dell'ultima lezione, per esempio:

    ```bash
    python -B LearnGPT/studio/lezioni/42_final_project.py
    ```

    Quando il rischio è più alto, eseguire tutte le lezioni numerate.

## Fonti e riferimenti

Queste regole sono basate su:

- le decisioni operative prese durante il percorso `LearnGPT`;
- `nanoGPT/model.py`, usato come direzione architetturale generale;
- la documentazione locale e le API PyTorch usate dal codice didattico;
- il file `corso.md`, che contiene le spiegazioni e i riferimenti tecnici
  specifici delle lezioni.

## Verifica rapida delle regole

Prima di considerare chiusa una nuova lezione, controllare queste domande:

1. Lo script in `studio/lezioni` importa dallo snapshot della stessa lezione?
2. Lo snapshot contiene solo il codice necessario alla lezione corrente?
3. Il modello principale si chiama `LanguageModel`?
4. Gli identificatori del codice Python sono in inglese?
5. `corso.md` contiene il codice nuovo, la spiegazione e il codice completo
   della parte interessata?
6. La mappa generale del corso va aggiornata per questa modifica?
7. Se cambia il flusso complessivo, sono aggiornati sia il diagramma conciso
   sia quello esteso in `corso.md`?
8. Se cambia il progetto finale, lo snapshot finale è ancora identico ai file
   operativi in `progetto_finale/`?
9. Il validatore locale passa senza errori?
10. La cartella `LearnGPT` è pulita da `__pycache__`, `.pyc` e `.DS_Store`?

## Stato iniziale

La struttura iniziale del progetto è:

```text
LearnGPT/
  guidance.md
  corso.md
  data/
    raw/
      fineweb_edu_sample.txt
    processed/
      fineweb_edu/
        train.bin
        val.bin
        meta.json
  studio/
    lezioni/
      01_leggi_testo.py
      02_tokenizer_caratteri.py
      ...
    snapshot/
      __init__.py
      lezione_01/
        __init__.py
      lezione_02/
        ...
      lezione_28/
        tokenizer.py
        batching.py
        model.py
  progetto_finale/
    __init__.py
    tokenizer.py
    batching.py
    device.py
    model.py
    prepare_data.py
    training.py
    checkpoint.py
    generate.py
    requirements.txt
  strumenti/
    validate_learngpt.py
```

La distinzione tra le cartelle è:

```text
data/             # dataset condivisi, raw opzionale e file processati
studio/lezioni/   # file numerati, didattici, da leggere in ordine
studio/snapshot/  # snapshot completi per ogni lezione
progetto_finale/  # ultima versione pulita del progetto finale
```

Il primo step legge un piccolo campione FineWeb-Edu in formato testo. Gli step
immediatamente successivi mostrano prima la tokenizzazione grezza a caratteri e
poi la tokenizzazione BPE più realistica, prima di passare al modello.
