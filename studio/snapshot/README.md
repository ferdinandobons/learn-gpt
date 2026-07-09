# Snapshot delle lezioni

Questa cartella contiene uno snapshot del codice di progetto per ogni lezione.

Regola:

```text
studio/lezioni/NN_nome_lezione.py -> importa da studio/snapshot/lezione_NN/
```

Motivo:

- le lezioni vecchie devono restare rieseguibili;
- il codice finale può continuare a evolvere in `progetto_finale/`;
- una lezione non deve dipendere implicitamente da una versione futura del
  progetto.

Quando viene creata una nuova lezione:

1. creare la cartella snapshot con numero a due cifre, per esempio
   `studio/snapshot/lezione_31/`;
2. copiarci o scriverci solo i file di progetto necessari per quella lezione;
3. aggiornare gli import dello script numerato in `studio/lezioni`, per esempio
   `studio/lezioni/31_generate.py`;
4. aggiornare `corso.md`;
5. aggiornare `progetto_finale/` se il progetto finale cambia.

Regole aggiuntive:

- `model.py` dello snapshot non deve contenere classi storiche non usate dalla
  lezione corrente.
- Il modello principale della lezione deve chiamarsi `LanguageModel`.
- Se servono componenti di supporto, per esempio `SelfAttentionHead`,
  `MultiHeadAttention`, `FeedForward` o `TransformerBlock`, mantenerli nello
  snapshot solo quando sono necessari al `LanguageModel` della lezione.
