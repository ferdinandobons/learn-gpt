# LearnGPT

LearnGPT è un progetto didattico dedicato alla costruzione di un modello GPT,
ispirato a [nanoGPT](https://github.com/karpathy/nanoGPT) di Andrej Karpathy.
L'obiettivo è spiegare in modo progressivo le idee principali dei Transformer
decoder-only in stile GPT e applicarle, una alla volta, fino a ottenere un
progetto PyTorch completo e funzionante.

Il repository privilegia la chiarezza didattica. Ogni concetto viene introdotto
in un passaggio specifico: tokenization, batch, embedding, causal self-attention,
multi-head attention, blocchi Transformer, ottimizzazione, checkpoint e
generazione.

Il corso interattivo gratuito è disponibile su
[learngpt.ferdinandobonsegna.com](https://learngpt.ferdinandobonsegna.com/).
Il sito contiene le spiegazioni, la matematica, le rappresentazioni grafiche e
il codice associato a ogni lezione. Questo repository è la fonte pubblica
ufficiale degli script, degli snapshot progressivi e del progetto PyTorch finale
utilizzati dal corso.

## LearnGPT: sito, corso e repository

LearnGPT è composto da due parti coordinate:

- il **sito web**, che presenta il percorso gratuito in italiano e in inglese,
  con 42 lezioni tecniche più una lezione iniziale di orientamento;
- il **repository GitHub**, che contiene il codice Python eseguibile di ogni
  passaggio, lo stato completo del progetto dopo ciascuna lezione e
  l'implementazione finale.

È possibile seguire tutte le spiegazioni direttamente dal sito senza installare
nulla. Il repository serve per eseguire il codice sul proprio computer,
controllare nel dettaglio le modifiche introdotte da ogni lezione e riprodurre
l'intero percorso fino al training e alla generazione.

Il corso viene mantenuto attraverso una guida collegata al codice e due edizioni
grafiche allineate:

- [course_en.md](course_en.md): percorso sintetico in inglese collegato al
  codice delle lezioni;
- [course_en_graphic.md](course_en_graphic.md): edizione inglese completa di
  spiegazioni, matematica, esempi svolti e diagrammi;
- [course_it_graphic.md](course_it_graphic.md): edizione italiana scritta in
  modo naturale, mantenendo in inglese i termini tecnici internazionali;
- [Contratto di scrittura del corso](docs/COURSE_AUTHORING_CONTRACT.md): regole
  per la struttura delle lezioni, la qualità dell'italiano, la responsabilità
  dei pannelli, i link al codice e i controlli di pubblicazione;
- [Procedura completa di training](docs/FINAL_TRAINING_RUNBOOK.md): flusso di
  riferimento per macOS/MPS e Windows/CUDA;
- [Ottimizzazioni del training CUDA](CUDA_TRAINING_OPTIMIZATIONS.md): misurazioni
  su fused attention, batching, logging e utilizzo della VRAM su hardware
  NVIDIA;
- [Guida alla serie video](docs/VIDEO_SERIES_GUIDE.md): struttura didattica dei
  42 passaggi e dell'esperimento finale;
- [Memoria del modello e limiti del training](docs/MODEL_MEMORY_AND_TRAINING_LIMITS.md):
  effetto di parametri, attivazioni, contesto e stato dell'optimizer su un
  training locale con 8 GB di memoria.

Tutte le 42 lezioni rispettano lo stesso criterio di leggibilità nelle due
lingue: una spiegazione continua adatta anche a chi affronta questi concetti per
la prima volta, una sequenza da tre a sette passaggi definita in base alla
complessità reale della trasformazione e un riepilogo esplicito di stato
iniziale, obiettivo, stato finale e vincolo. Le versioni inglese e italiana
devono essere aggiornate insieme. I termini tecnici mantengono la forma inglese
comunemente usata. Sul sito l'italiano è la lingua predefinita e il testo deve
risultare naturale, non una traduzione letterale.

## Contenuto del progetto

- Un percorso organizzato per lezioni per costruire un language model simile a
  GPT.
- Snapshot riproducibili delle lezioni in `study/snapshots/`.
- Un progetto finale completo in `final_project/`.
- Tokenization BPE di GPT-2 con `tiktoken`.
- Preparazione di FineWeb-Edu per il training locale.
- Sottoinsiemi sperimentali casuali e riproducibili derivati dai dati
  processati.
- Caricamento tramite memory mapping di `train.bin` e `val.bin` per dataset
  locali di grandi dimensioni.
- Selezione del dispositivo tra CPU, CUDA e MPS su Apple Silicon.
- Gruppi dell'optimizer AdamW, gradient accumulation, pianificazione del
  learning rate, gradient clipping, inizializzazione in stile GPT, checkpoint
  atomici `best` e `latest`, ripresa del training, proiezione del vocabolario a
  blocchi, controlli di integrità dei gradienti su MPS, diagnostica del contesto
  basata sui target, mixed precision opzionale, fused multi-head QKV attention,
  logging essenziale dell'avanzamento e supporto opzionale per `torch.compile`.

## Struttura del progetto

```text
LearnGPT/
  README.md
  course_en.md
  course_en_graphic.md
  course_it_graphic.md
  CUDA_TRAINING_OPTIMIZATIONS.md

  docs/
    COURSE_AUTHORING_CONTRACT.md
    FINAL_TRAINING_RUNBOOK.md
    MODEL_MEMORY_AND_TRAINING_LIMITS.md
    VIDEO_SERIES_GUIDE.md
    training_workflow.json
    verified_runs/

  data/
    README.md
    study_sample.txt     # versionato, ridotto e usato dalle lezioni
    raw/                 # ignorato da Git
    processed/           # ignorato da Git

  study/
    lessons/             # script numerati delle lezioni
    snapshots/           # snapshot del progetto per ogni lezione

  final_project/
    config.py
    tokenizer.py
    prepare_data.py
    prepare_subset.py
    batching.py
    device.py
    model.py
    training.py
    checkpoint.py
    generate.py
    quality.py
    requirements.txt
    requirements-common.txt

  tools/
    run_all_lessons.py
    validate_learngpt.py

  tests/
    test_final_project.py
```

`study/` contiene il percorso didattico. `final_project/` contiene la versione
completa e aggiornata del progetto. Dataset, checkpoint e file generati dal
modello non vengono versionati in Git.

## Avvio rapido: seguire il corso

Usa Python 3.12 o una versione successiva. Python 3.13 è la versione consigliata
e verificata dalla CI. Clona il repository, quindi crea un ambiente virtuale
locale su macOS o Linux:

```bash
git clone https://github.com/ferdinandobons/learn-gpt.git
cd learn-gpt
```

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Installa le dipendenze:

```bash
python -m pip install -r final_project/requirements.txt
```

Su Windows PowerShell usa direttamente l'interprete dell'ambiente virtuale. Per
eseguire le lezioni su CPU, installa le dipendenze comuni e specifica
esplicitamente il wheel per CPU:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r final_project\requirements-common.txt
.\.venv\Scripts\python.exe -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
```

Se il computer verrà usato per il training con NVIDIA CUDA, installa il wheel
CUDA indicato nella sezione dedicata ai backend invece di quello per CPU.

Verifica la struttura del repository:

```bash
python -B tools/validate_learngpt.py
```

Esegui i test di regressione del progetto finale:

```bash
python -B -m unittest discover -s tests -v
```

Esegui una lezione specifica:

```bash
python -B study/lessons/01_read_text.py
```

Esegui lo smoke test dell'ultima lezione:

```bash
python -B study/lessons/42_final_project.py
```

Esegui il controllo completo del percorso didattico partendo da un clone pulito:

```bash
python -B tools/run_all_lessons.py
```

Consulta il percorso del corso mentre esegui gli script numerati. Le edizioni
grafiche adottano una separazione precisa: la spiegazione centrale descrive il
processo, il pannello Matematica contiene formule e calcoli completi, mentre il
pannello Programmazione contiene sintassi e codice. `study/snapshots/` conserva
lo stato completo del codice per ogni lezione. Le lezioni usano il file
versionato `data/study_sample.txt`; il corpus da 10 GiB serve soltanto per il
training reale. GitHub Actions esegue il validatore, i test di regressione e
tutte le 42 lezioni sia su Linux sia su Windows.

## Installazione di PyTorch in base al backend

Per la maggior parte delle esecuzioni locali è sufficiente
`python -m pip install -r final_project/requirements.txt`. Per un training reale,
installa prima la build di PyTorch adatta all'hardware, quindi le altre
dipendenze del progetto. Il file dei requirements specifica le versioni esatte
usate nell'esecuzione verificata.

Usa il [selettore di installazione ufficiale di PyTorch](https://pytorch.org/get-started/locally/)
per ottenere il comando più recente relativo al backend utilizzato.

<details>
<summary>Installazione di PyTorch per Apple Silicon MPS</summary>

Su macOS con Apple Silicon, installa il wheel standard per macOS già verificato
e le dipendenze rimanenti nelle versioni specificate:

```bash
python -m pip install -r final_project/requirements.txt
```

Verifica MPS da una normale sessione del Terminale:

```bash
python -c "import torch; print(torch.backends.mps.is_built(), torch.backends.mps.is_available()); print(torch.ones(1, device='mps'))"
```

Risultato previsto:

```text
True True
tensor(..., device='mps:0')
```

Se il comando restituisce `True False` in una shell gestita o isolata, ripeti lo
stesso controllo in una normale sessione del Terminale. I processi isolati
possono non avere il permesso di creare un dispositivo Metal anche quando il Mac
supporta MPS.

</details>

<details>
<summary>Installazione di PyTorch per NVIDIA CUDA</summary>

Scegli dal selettore di installazione di PyTorch il wheel CUDA adatto al
computer. Il profilo Windows controllato usa PyTorch 2.12.1 e il wheel per
CUDA 12.6:

```powershell
.\.venv\Scripts\python.exe -m pip uninstall -y torch
.\.venv\Scripts\python.exe -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r final_project\requirements-common.txt
```

La disinstallazione rende esplicito il passaggio da CPU a CUDA. In caso
contrario, `pip` considera già soddisfatta una build CPU installata con lo stesso
numero di versione.

Verifica CUDA:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda device')"
```

Consulta la [procedura completa di training](docs/FINAL_TRAINING_RUNBOOK.md) se
il selettore ufficiale richiede un indice dei wheel diverso per il driver
installato.

</details>

<details>
<summary>Installazione di PyTorch solo per CPU</summary>

Usa questa configurazione quando non è disponibile un backend GPU:

```bash
python -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r final_project/requirements-common.txt
```

Verifica la CPU:

```bash
python -c "import torch; print(torch.__version__); print(torch.ones(1).device)"
```

</details>

## Come eseguire il training: procedura rapida

Il progetto finale esegue il training su
[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) usando
il tokenizer BPE di GPT-2. Il dataset non è incluso nel repository.

La procedura completa di riferimento è documentata nella
[guida al training](docs/FINAL_TRAINING_RUNBOOK.md). I comandi seguenti
costituiscono una sintesi operativa. Consulta la guida per configurazione,
monitoraggio, ripresa del training, uso di Windows PowerShell e risoluzione dei
problemi.

Prepara circa 10 GB di dati tokenizzati:

```bash
python -B final_project/prepare_data.py \
  --target-gb 10 \
  --output-dir data/processed/fineweb_edu
```

Il comando crea:

```text
data/processed/fineweb_edu/
  train.bin
  val.bin
  meta.json
```

Verifica i dati locali di riferimento:

```bash
python -B tools/validate_learngpt.py \
  --training-data-dir data/processed/fineweb_edu
```

Scegli un backend per il training e apri soltanto la sezione corrispondente al
computer utilizzato.

<details>
<summary>Training con Apple Silicon MPS</summary>

Verifica che PyTorch rilevi MPS:

```bash
python -c "import torch; print(torch.backends.mps.is_built(), torch.backends.mps.is_available())"
```

Esegui il controllo da una normale sessione del Terminale. Alcune shell gestite
o isolate possono non avere il permesso di creare un dispositivo Metal e
restituire `True False`, anche quando MPS funziona correttamente al di fuori
dell'ambiente isolato.

Esegui uno smoke test di MPS con un solo passaggio di training ridotto:

```bash
python -m final_project.training \
  --device mps \
  --data-dir data/processed/fineweb_edu \
  --checkpoint-path /tmp/learngpt-mps-smoke.pt \
  --overwrite-checkpoints \
  --context-size 8 \
  --embedding-size 16 \
  --num-heads 4 \
  --num-transformer-blocks 1 \
  --batch-size 1 \
  --gradient-accumulation-steps 1 \
  --training-steps 1 \
  --eval-interval 1 \
  --eval-batches 1 \
  --base-learning-rate 1e-4 \
  --min-learning-rate 1e-5 \
  --warmup-steps 0 \
  --decay-steps 1
```

Genera testo a partire dal checkpoint dello smoke test:

```bash
python -m final_project.generate \
  --device mps \
  --checkpoint-path /tmp/learngpt-mps-smoke.pt \
  --prompt "Once upon" \
  --max-new-tokens 8 \
  --temperature 1.0 \
  --top-k 20
```

### Training reale controllato su un Mac Apple Silicon con 8 GB

Mantieni il corpus processato completo come fonte di riferimento e crea un
esperimento separato e riproducibile da 1 GiB per questa esecuzione con risorse
di calcolo limitate:

```bash
.venv/bin/python -B -m final_project.prepare_subset \
  --source-data-dir data/processed/fineweb_edu \
  --output-dir data/processed/fineweb_edu_experiment_1g \
  --target-gb 1 \
  --validation-ratio 0.01 \
  --seed 1337 \
  --chunk-tokens 65536
```

Il comando legge soltanto `fineweb_edu` e scrive un nuovo dataset locale da
1 GiB. Seleziona blocchi di token non sovrapposti in ordine deterministico:
usando lo stesso dataset sorgente e lo stesso seed si ottiene lo stesso
esperimento.

#### Perché la precedente esecuzione su MPS non ha funzionato

Il problema si trovava nel backward pass, non in FineWeb-Edu o nella
tokenization. Hanno contribuito due comportamenti di MPS. La proiezione unica del
vocabolario `256 -> 50257` restituiva una direzione errata per il gradiente
dell'hidden state, anche quando la loss del forward pass era corretta. Inoltre,
`optimizer.zero_grad(set_to_none=True)` induceva MPS ad allocare nuovi buffer per
i gradienti dei tensori foglia e produceva occasionalmente gradienti molto
elevati. Il gradient clipping a `1.0` ne limitava la dimensione, ma non poteva
correggerne la direzione. Di conseguenza, il modello tendeva a produrre quasi la
stessa distribuzione di token ad alta frequenza per ogni prompt.

La procedura corretta:

- divide la proiezione del vocabolario da 50.257 token in blocchi da non più di
  32.768 elementi;
- alloca buffer persistenti per i gradienti MPS e li azzera senza ricrearli;
- esegue prima del training un backward pass di warm-up il cui risultato viene
  scartato;
- prima del primo aggiornamento dell'optimizer richiede che due backward pass
  MPS identici concordino tra loro e con un riferimento calcolato su CPU;
- rifiuta una norma grezza del gradiente superiore a `100`, ripete gli stessi
  batch fino a tre volte e si arresta senza applicare aggiornamenti se tutti i
  tentativi falliscono.

Non riprendere il training da checkpoint prodotti dalla procedura interessata
dal problema. Il gradient clipping ha nascosto l'errore nei pesi appresi e non
esiste un metodo affidabile per correggerli. Parti da un'inizializzazione casuale
e usa un nuovo percorso di checkpoint mai utilizzato in precedenza.

#### Esecuzione completa da 45.000 passaggi

Questo è il comando controllato per il modello da 17,7 milioni di parametri. Con
batch size 4, contesto 256 e otto micro-batch accumulati, elabora 8.192 token per
aggiornamento dell'optimizer e circa 368,6 milioni di posizioni di token in
45.000 passaggi:

```bash
caffeinate -i .venv/bin/python -B -m final_project.training \
  --device mps \
  --data-dir data/processed/fineweb_edu_experiment_1g \
  --checkpoint-path checkpoints/learngpt-mps-18m-stable-1g-v2.pt \
  --encoding-name gpt2 \
  --seed 1337 \
  --context-size 256 \
  --embedding-size 256 \
  --num-heads 4 \
  --num-transformer-blocks 6 \
  --dropout 0.0 \
  --use-scaled-dot-product-attention \
  --output-chunk-size 32768 \
  --batch-size 4 \
  --gradient-accumulation-steps 8 \
  --training-steps 45000 \
  --eval-interval 250 \
  --eval-batches 20 \
  --base-learning-rate 3e-4 \
  --min-learning-rate 3e-5 \
  --warmup-steps 1000 \
  --decay-steps 45000 \
  --weight-decay 0.05 \
  --gradient-clip 1.0 \
  --max-grad-norm-before-clip 100 \
  --gradient-retry-attempts 3 \
  --context-sensitivity-contexts 32
```

Non aggiungere `--mixed-precision` o `--compile-model` a questa configurazione
MPS. Sono funzionalità opzionali per altri backend e non fanno parte della
procedura verificata.

All'avvio, il training esegue il controllo automatico di ripetibilità su MPS e
di corrispondenza con la CPU. Se i gradienti non concordano, l'esecuzione si
arresta prima del passaggio 1. Durante il training, `grad_norm` indica la norma
grezza misurata prima del clipping; `grad_retries=0` è il risultato normale. Un
errore di integrità persistente interrompe l'esecuzione prima che l'optimizer
possa usare il gradiente errato.

Un'esecuzione MPS completa da 45.000 passaggi ha superato il controllo iniziale
di corrispondenza e non ha richiesto tentativi aggiuntivi nelle valutazioni
salvate. Il checkpoint migliore è stato ottenuto al passaggio 42.750 con
validation loss `4.2894`. Il checkpoint più recente ha raggiunto il passaggio
45.000 con validation loss `4.4524`, norma grezza del gradiente `2.3872` e
`context_loss_gain=+6.1914`. Il risultato in formato strutturato e un esempio
generato con seed definito sono registrati in
`docs/verified_runs/mps-18m-1g-45000.json`.

`context_js` rimane una misura osservativa della variazione della distribuzione
di output tra contesti diversi. Un valore vicino a zero nelle prime fasi del
training non indica necessariamente un errore: un modello appena inizializzato
apprende normalmente le frequenze generali dei token prima di usare in modo
efficace il contesto. `context_loss_gain` tiene conto dei target ed equivale alla
loss con contesti rimescolati meno la loss con i contesti corretti. Un andamento
positivo indica che i contesti reali aiutano a prevedere i token successivi; un
valore vicino a zero nella fase iniziale è previsto. Nessuna delle due metriche
viene usata come criterio rigido di arresto nelle fasi iniziali.

Con l'inizializzazione in stile GPT, la prima loss dovrebbe essere vicina a
`ln(50257)`, quindi circa `10.82`, non nell'ordine delle decine o delle
centinaia. In seguito dovrebbe diminuire su più valutazioni, anche se i singoli
valori di validation possono oscillare.

Non servono pause artificiali tra i passaggi. `caffeinate` impedisce la
sospensione e macOS gestisce già il thermal throttling. Mantieni il Mac su una
superficie rigida che non ostacoli la ventilazione e interrompi l'esecuzione
soltanto se macOS segnala una pressione termica o di memoria persistente.

Il training scrive due checkpoint atomici:

```text
checkpoints/learngpt-mps-18m-stable-1g-v2.pt         # validation loss migliore
checkpoints/learngpt-mps-18m-stable-1g-v2-latest.pt  # ultimo passaggio valutato
```

Se il Terminale o il Mac si interrompono, riprendi soltanto questa nuova
esecuzione verificata. Il numero di passaggi rimane l'obiettivo totale, non
indica altri 45.000 passaggi:

```bash
caffeinate -i .venv/bin/python -B -m final_project.training \
  --device mps \
  --data-dir data/processed/fineweb_edu_experiment_1g \
  --checkpoint-path checkpoints/learngpt-mps-18m-stable-1g-v2.pt \
  --resume-checkpoint-path checkpoints/learngpt-mps-18m-stable-1g-v2-latest.pt \
  --training-steps 45000
```

Esegui una generazione riproducibile:

```bash
.venv/bin/python -B -m final_project.generate \
  --device mps \
  --checkpoint-path checkpoints/learngpt-mps-18m-stable-1g-v2.pt \
  --prompt "Once upon a time" \
  --max-new-tokens 120 \
  --temperature 0.9 \
  --top-k 50 \
  --seed 1337
```

Questa esecuzione addestra un piccolo base language model. Il modello completa
prompt in inglese in modo plausibile, ma non è un assistente addestrato a seguire
istruzioni e non risponde in modo affidabile alle domande. Un comportamento da
assistente richiede una fase successiva di instruction tuning su esempi composti
da prompt e risposta.

Se MPS non è disponibile nel runtime PyTorch corrente, correggi la
configurazione di PyTorch o macOS prima di avviare il training con
`--device mps`.

</details>

<details>
<summary>Training con NVIDIA CUDA</summary>

La procedura Windows/CUDA addestra lo stesso modello controllato usato su MPS:
contesto 256, dimensione dell'embedding 256, 4 head, 6 blocchi, sottoinsieme
riproducibile da 1 GiB, 8.192 token effettivi per aggiornamento dell'optimizer e
un obiettivo totale di 45.000 passaggi. CUDA usa l'autocast FP16 con un
GradScaler salvato nel checkpoint e non applica i tentativi aggiuntivi previsti
per i gradienti MPS. Un overflow FP16 temporaneo riduce la scala e ripete
esattamente lo stesso batch e lo stesso passaggio fino a otto volte, quindi
interrompe l'esecuzione se il problema persiste.

Il comando CUDA di riferimento abilita anche `--fused-attention`, che proietta
insieme Q, K e V per tutte le head ed esegue una singola chiamata SDPA in batch
per ogni blocco. `--log-interval` stampa informazioni essenziali
sull'avanzamento tra gli eventi meno frequenti di validazione e salvataggio dei
checkpoint, controllati da `--eval-interval`.

Segui la procedura PowerShell in due fasi descritta in
[Guida al training: Windows NVIDIA CUDA](docs/FINAL_TRAINING_RUNBOOK.md#6-windows-nvidia-cuda-smoke-gate-and-complete-run).
La procedura esegue un controllo da 20 passaggi con l'architettura reale, quindi
riprende lo stesso checkpoint fino al passaggio 45.000. La sezione comprende
anche i profili per 4, 6 e 8 GiB di VRAM e il relativo comando di generazione.

Per il profilo NVIDIA più grande sottoposto a misurazione e per i dati relativi
a throughput e VRAM, consulta le
[ottimizzazioni del training CUDA](CUDA_TRAINING_OPTIMIZATIONS.md).

Questa procedura per CUDA è stata sottoposta a code review ed è coperta da test
dei checkpoint eseguiti su CPU, compresi lo stato del GradScaler e la gestione
di riduzione e ripetizione dopo un overflow. Prima di considerare verificato un
training prolungato, è comunque necessario completare il controllo finale
sull'hardware NVIDIA di destinazione.

</details>

<details>
<summary>Training di ripiego su CPU</summary>

Quando PyTorch è installato, la CPU è sempre disponibile, ma risulta molto più
lenta per un training reale. Usa questa modalità soprattutto per smoke test o
esecuzioni molto ridotte.

Avvia il training:

```bash
python -m final_project.training \
  --device cpu \
  --data-dir data/processed/fineweb_edu \
  --checkpoint-path checkpoints/learngpt-cpu.pt \
  --context-size 128 \
  --embedding-size 256 \
  --num-heads 4 \
  --num-transformer-blocks 4 \
  --batch-size 2 \
  --gradient-accumulation-steps 1 \
  --training-steps 100 \
  --eval-interval 20 \
  --eval-batches 5
```

Riprendi il training:

```bash
python -m final_project.training \
  --device cpu \
  --data-dir data/processed/fineweb_edu \
  --checkpoint-path checkpoints/learngpt-cpu.pt \
  --resume-checkpoint-path checkpoints/learngpt-cpu-latest.pt \
  --training-steps 200
```

Genera:

```bash
python -m final_project.generate \
  --device cpu \
  --checkpoint-path checkpoints/learngpt-cpu.pt \
  --prompt "Once upon a time" \
  --max-new-tokens 120 \
  --temperature 0.9 \
  --top-k 50 \
  --seed 1337
```

</details>

La CLI del training mostra i runtime Python e PyTorch, il dispositivo
selezionato, la dimensione del dataset, la configurazione del modello e del
training, la validation loss, il learning rate, la norma grezza del gradiente
prima del clipping, il numero di tentativi, la diagnostica del contesto basata
sui target, i conteggi di tentativi e overflow di CUDA AMP, i token elaborati al
secondo e il tempo rimanente stimato. `--log-interval N` stampa metriche
essenziali ogni `N` aggiornamenti senza avviare la validazione o scrivere un
checkpoint.

## Pubblicazione dei checkpoint

I checkpoint possono raggiungere dimensioni elevate e sono quindi ignorati da
Git. Per condividere pubblicamente un modello addestrato, usa gli asset di una
GitHub Release o una piattaforma esterna per modelli invece di aggiungere al
repository file `.pt`, `.pth` o `.ckpt`.

## Relazione con nanoGPT

LearnGPT usa l'implementazione locale `../AndrejKarpathy/nanoGPT` come
riferimento per architettura e training:

- architettura Transformer decoder-only;
- embedding appresi per token e posizione;
- causal self-attention;
- multi-head attention;
- blocchi Transformer pre-LayerNorm;
- residual connection;
- blocchi MLP/feed-forward con GELU;
- pesi condivisi tra token embedding e output, con inizializzazione in stile
  GPT;
- training con AdamW;
- gradient accumulation, clipping, warm-up e decadimento coseno del learning
  rate;
- valutazione su training e validation set, checkpoint e generazione
  autoregressiva.

La differenza principale riguarda la struttura didattica. nanoGPT raggruppa
Q/K/V in un'unica proiezione e include funzioni orientate alla produzione, come
DDP reale, calcolo dell'MFU e importazione dei pesi preaddestrati di GPT-2.
LearnGPT usa proiezioni Q/K/V separate e nomi espliciti per rendere controllabile
la forma di ogni tensore. Mantiene opzionali mixed precision e `torch.compile`,
spiega DDP senza avviarlo e aggiunge sottoinsiemi riproducibili di FineWeb-Edu,
controlli di integrità dei gradienti su MPS e diagnostica del contesto basata sui
target per esperimenti locali con risorse di calcolo limitate.

## Licenza

LearnGPT è distribuito con [licenza MIT](LICENSE).
