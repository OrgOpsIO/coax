# coax — Anforderungen aus dem Praxiseinsatz

Gesammelt beim Betrieb einer BFF-Recherche-App gegen ein eigenes Gateway
(vLLM, OpenAI-kompatibel, `max_model_len: 65536`, Qwen3-VL-32B-Instruct-AWQ).
Jede Anforderung nennt den Befund im Code und die gemessene Wirkung.
Stand: coax v0.5.0, Quellstand `7a901bb`.

Reihenfolge = Dringlichkeit, nicht Aufwand.

---

## Blockiert uns jetzt

### 1. `reasoningEffort` durchreichen — pro Aufruf, als Default, pro Alias

**Befund.** Kein Provider schickt ein Feld dafür. Der OpenAI-Body kennt nur
`model`, `max_tokens`, `messages`, `tools`, `tool_choice`
(`src/providers/openai.ts:160-199`), der Anthropic-Body entsprechend
(`src/providers/anthropic.ts:128-180`). Es gibt keinen Weg, das Denken
abzuschalten, ohne coax zu umgehen.

**Messung.** Derselbe Aufruf gegen dasselbe Gateway:

| | Ausgabe-Tokens | Dauer |
|---|---|---|
| ohne Steuerung (Default) | 1243 | 30 s |
| `reasoning_effort: "none"` | 100 | 2,8 s |

**Wirkung bei uns.** 8 von 14 Aufrufen pro Recherche brauchen kein Denken
(Klassifikation, Umformulierung, Formatierung). Gesamtlauf ~19,5 Min → ~3–5 Min.

**Der Alias-Weg ist der wichtige.** Rollen sind in unserer `models.ts` ohnehin
schon getrennt. Hängt die Einstellung am Alias, ist es eine Konfigzeile statt
15 geänderter Aufrufstellen:

```ts
models: {
  "klassifikation": { use: "orgops:qwen3-vl", reasoningEffort: "none" },
  "synthese":       { use: "orgops:qwen3-vl", reasoningEffort: "high" },
}
```

Dazu `defaults.reasoningEffort` (`src/config.ts:39-48`) und ein Feld auf
`ObjectCall` / `TextCall` / `RunCall` (`src/ai.ts:10-102`), das den Alias
überschreibt. Auflösung an einer Stelle: `registry.resolve()`
(`src/registry.ts:74-86`) kennt den Alias schon, gibt aber heute nur den
Provider zurück — die Einstellung müsste mit heraus und in den Request.

**Entschieden.**

- Die Einstellung hängt am Alias, an `defaults` und pro Aufruf (Aufruf >
  Alias > Default). Sie wandert **nicht** in die Provider-Optionen: der
  Provider-Cache-Schlüssel ist `${providerName}:${model}`
  (`src/registry.ts:50`) — zwei Aliase auf dasselbe Modell mit verschiedenem
  Effort teilen sich dieselbe Instanz. Stattdessen gibt `resolve()` neben dem
  Provider ein `callSettings`-Objekt zurück, und `ai.ts` mischt es in den
  Request. Der Request (`BaseRequest`, `src/types.ts:50`) bekommt
  `reasoningEffort?: "none" | "low" | "medium" | "high"`.
- Wire OpenAI: `reasoning_effort` mit genau diesen vier Werten, nur gesendet
  wenn gesetzt — ein Endpoint ohne das Feld sieht es nie.
- Wire Anthropic: `none` → kein `thinking`-Feld; `low`/`medium`/`high` →
  `thinking: { type: "enabled", budget_tokens: 2048 / 8192 / 16384 }`,
  gedeckelt auf `max_tokens − 1024` (Anthropic verlangt Budget < max_tokens).
- **Bewusste Lücke v1:** `reasoningEffort` auf Anthropic-`tools()` wird nicht
  unterstützt (klarer Fehler, kein stilles Ignorieren). Grund: Anthropic
  erwartet Thinking-Blöcke über Tool-Runden hinweg zurück, und
  `Message.content` ist ein blanker String (`src/types.ts:31-37`) — das
  Round-Tripping ist ein eigener Umbau und blockiert unseren Anwendungsfall
  (OpenAI-Wire) nicht.

Wo ein Endpoint das Feld anders buchstabiert (z. B. Qwen via
`chat_template_kwargs`), greift Punkt 2.

---

### 2. `extraBody` — generisch, pro Provider und pro Aufruf

**Befund.** Der Request-Body wird in den Providern literal aufgebaut; alles
außerhalb dieser Literale ist unerreichbar. `temperature` und `top_p` schickt
coax nirgends — Qwen hat für beide ausdrückliche Empfehlungen, wir können sie
nicht setzen.

**Was wir brauchen.**

```ts
providers: {
  orgops: { apiKey, baseURL, api: "openai", extraBody: { temperature: 0.6, top_p: 0.95 } },
},
// und pro Aufruf:
ai.text({ model: "...", prompt, extraBody: { chat_template_kwargs: { enable_thinking: false } } })
```

**Entschieden.**

- Flacher Merge, wie bei `headers` (`src/types.ts:62-68`,
  `src/providers/openai.ts:144-152`): `{ ...coaxBody, ...endpoint.extraBody,
  ...call.extraBody }`. Aufruf schlägt Endpoint schlägt coax.
- `extraBody` **darf** coax' eigene Felder überschreiben (`max_tokens`,
  `tools`, …). Das ist Absicht: eine Notluke, die nicht alles öffnen kann,
  ist keine. Wer `tools` überschreibt, zerschießt sein Tool-Calling — das
  ist sein Recht und steht so in der Doku.
- Kein Whitelisting — der Sinn ist gerade, dass die nächste Gateway-Eigenheit
  keine coax-Version braucht. Mit `extraBody` wäre Punkt 1 notfalls von Hand
  überbrückbar; als Erstklass-Feld gehört `reasoningEffort` trotzdem an den
  Alias.

---

### 3. `toolChoice: 'auto' | 'required' | 'none'` in `RunOptions`

**Befund.** Fest verdrahtet auf `"auto"` — `src/providers/openai.ts:196`
(`dist/index.js:381`). Anthropics `tools()` schickt gar kein `tool_choice`
(`src/providers/anthropic.ts:163-175`), also ebenfalls implizit auto.
`RunOptions` (`src/tools.ts:47-59`) hat kein Feld dafür.

**Wirkung bei uns.** Die erste Rechercherunde antwortet regelmäßig aus dem
Gedächtnis statt nachzuschlagen. Unser Notbehelf ist ein zusätzlicher
Durchgang mit „du hast nichts nachgeschlagen, mach das nochmal" — ein
kompletter Extra-Lauf. Mit `toolChoice: 'required'` für Schritt 0 fällt der
ersatzlos weg.

**Entschieden.**

```ts
type ToolChoice = "auto" | "required" | "none";
// RunOptions / RunCall:
toolChoice?: ToolChoice | ((step: number) => ToolChoice);
```

- Default `"auto"` (heutiges Verhalten). Die Funktionsform ist die empfohlene
  für `"required"`: konstant `"required"` heißt, das Modell darf **nie** mit
  Text antworten — die Abbruchbedingung der Schleife (`!res.calls.length`,
  `src/tools.ts:140`) greift dann nie, und der Lauf endet zwangsläufig im
  `maxSteps`-Fehler. Das wird dokumentiert, nicht verboten: `maxSteps`/Budget
  sind die Bremse, kein Sonderfall im Code.
- Wire OpenAI: Werte literal (`"required"`, `"none"`, `"auto"`).
- Wire Anthropic: `required` → `{ type: "any" }`, `auto` → `{ type: "auto" }`,
  `none` → `{ type: "none" }` — der `tools()`-Pfad
  (`src/providers/anthropic.ts:163-175`) schickt das Feld dann erstmals.
- `ToolsRequest` (`src/types.ts:92-96`) bekommt `toolChoice?: ToolChoice`
  (aufgelöst pro Schritt — die Funktionsform wertet `runTools` aus, der
  Provider sieht nur noch den Wert).

---

## Reißt uns irgendwann den Lauf ab

### 4. `max_tokens` aus dem Restkontext ableiten

**Befund.** coax schickt fix `req.maxTokens ?? opts.maxTokens ?? 8192` — an
sechs Stellen (`src/providers/openai.ts:163,179,193`,
`src/providers/anthropic.ts:134,153,169`). Der Wert ist unabhängig davon, wie
groß der Prompt inzwischen ist.

**Was passiert.** vLLM antwortet hart mit 400, sobald Prompt + `max_tokens`
über `max_model_len` liegen. Wörtlich nachgemessen:

> you requested 60000 output tokens and your prompt contains at least 5537
> input tokens, for a total of at least 65537

Ein Tool-Lauf wächst genau dort hinein: bei 11 gelesenen Knoten trägt der
Transkript-Verlauf (`src/tools.ts:142,180`) den Prompt über die Grenze, und
der Lauf stirbt mitten drin — mit allem, was Punkt 5 beschreibt.

**Entschieden: beide Wege.**

- **Proaktiv:** `contextWindow?: number` auf `ProviderEndpoint`
  (`src/config.ts:8-19`). Wenn gesetzt, deckelt der Provider `max_tokens` auf
  `contextWindow − geschätzteInputTokens − 512`. Schätzung ohne Tokenizer:
  `Zeichen / 3` (bewusst konservativ, Deutsch und Code liegen näher an 3 als
  an 4) plus pauschal 1500 pro Bild. Fällt der Deckel unter 1024, fliegt ein
  klarer Fehler („Prompt füllt das Kontextfenster") statt eines
  Mini-`max_tokens`, das nur Müll produzieren kann.
- **Reaktiv:** den 400 erkennen und **einmal** mit ausgerechnetem
  `max_tokens` nachfassen. Lebt im OpenAI-Provider (der Fehlertext ist
  vLLM-Wortlaut, also gehört der Regex dorthin, nicht in die generische
  Schicht), Erkennung über die beiden Zahlen im Text
  (`requested (\d+) … at least (\d+) input`). Genau ein Nachfassen — der
  zweite 400 ist ein echter Fehler. Ausdrücklich **nicht** in `withRetry`
  (`src/retry.ts`): das behandelt transiente Fehler mit unverändertem
  Request, hier wird der Request korrigiert.
- Beide zusammen: der proaktive Deckel macht den reaktiven Pfad zum
  Sicherheitsnetz für schlechte Schätzungen, nicht zum Normalfall.

---

### 5. Fehler sollen den Teilstand mittragen

**Befund.** In `runTools` wird nur der Abbruch angereichert; jeder andere
Fehler fliegt roh weiter (`src/tools.ts:129-134`):

```ts
} catch (err) {
  throw err instanceof CoaxAbortError ? new CoaxAbortError(addUsage(usage, err.usage), err) : err;
}
```

Damit sind `messages`, `calls`, `steps` und `usage` weg, sobald der Provider
wirft. `CoaxToolError` kann all das schon tragen (`src/tools.ts:78-95`) — es
wird nur bei `maxSteps` und Budget benutzt.

**Was es gekostet hat.** Ein 502 beim vorletzten Aufruf hat 476 Sekunden
Recherche vernichtet: `ai.run` warf, der Verlauf war weg, elf gelesene Knoten
mit ihm. Dasselbe an der `maxSteps`-Grenze — dort *ist* der Zustand am Fehler,
aber unsere App wiederholt trotzdem den ganzen Lauf, weil sie ihn nicht
fortsetzen kann.

**Entschieden.**

- Jeder Provider-Fehler in `runTools` wird in `CoaxToolError` eingepackt,
  Original als `cause`, mit `messages` / `calls` / `steps` / `usage`.
  `CoaxAbortError` bleibt die Ausnahme (eigener Typ, eigene Semantik) —
  bekommt aber zusätzlich `messages`/`calls`, damit auch ein Abbruch
  fortsetzbar ist.
- `CoaxSchemaError` (`src/client.ts:18-29`) bekommt `messages` dazu (trägt
  heute nur `usage` und `lastError`).
- **Das ist ein Verhaltens-Breaking-Change:** wer heute
  `err instanceof APIError` fängt, sieht künftig `CoaxToolError` und muss auf
  `err.cause` schauen. Akzeptiert — 0.x, Changelog-Eintrag, und der Gewinn
  ist genau der Punkt der Übung: `ai.run({ messages: err.messages, ... })`
  setzt dort an, wo es abgerissen ist. Weiterarbeiten statt Wiederholen.

---

### 6. `maxSteps: null` = unbegrenzt

**Befund.** `opts.maxSteps ?? 8` (`src/tools.ts:112`), Schleife bis `maxSteps`
(`:124`), danach `CoaxToolError` (`:183`). Der Typ ist `number | undefined`
(`src/tools.ts:52`, `src/config.ts:47`) — „kein Limit" lässt sich nur als
große Zahl schreiben, also genau das Pseudo-Limit, das die Bibliothek
vermeiden will.

**Entschieden.** `maxSteps?: number | null`, `null` = unbegrenzt — aber nur
in Kombination mit einer echten Bremse: `null` ohne `budget` **und** ohne
`signal` wirft sofort einen klaren Fehler („unbegrenzt braucht ein Budget
oder ein Signal"). Das passt zur Linie der Bibliothek: keine Pseudo-Limits,
aber auch kein stilles „bis das Gateway aufgibt". Budget (`src/budget.ts`)
und Signal sind ehrlichere Abbruchgründe als eine geratene Schrittzahl.

---

## Komfort

### 7. Modell-Autoauflösung

Der ursprüngliche Auslöser: Am Gateway wird das Modell getauscht, die App
steht. `splitRef` (`src/registry.ts:65-71`) verlangt ein literales
`provider:model`.

**Entschieden.** `orgops:auto` → einmalig `GET /v1/models`, erstes Modell
nehmen, im Registry-Cache (`src/registry.ts:31`) merken. Konsequenz:
`resolve()` (`src/registry.ts:74`) wird async, `withFallback`
(`src/ai.ts:167-185`) awaitet — kleiner, aber echter Eingriff in die
Registry-Signatur. Dazu ein Invalidierungspfad: antwortet der Endpoint auf
einen Aufruf mit „model not found", wird der Cache-Eintrag einmal verworfen
und neu aufgelöst — damit überlebt die App auch den Modellwechsel *während*
sie läuft, nicht nur beim Start.

Zweitrangig gegenüber 1–6, aber billig.

### 8. Streaming

**Befund.** `stream: true` kommt im Build kein einziges Mal vor. Der
Anthropic-Provider nutzt `messages.stream(...).finalMessage()`
(`src/providers/anthropic.ts:32-40`), aber nur, um die 10-Minuten-Weigerung
des SDK zu umgehen — nach außen gibt es keine Token.

Nötig für ein ehrliches „denkt nach…" in der Oberfläche. Aber: wenn nach
Punkt 1 acht von vierzehn Aufrufen nicht mehr denken, ist die Totenstille
größtenteils weg. Deshalb zuletzt.

**Entschieden: Callback, kein Iterator — und bewusst schmal.**

```ts
// TextCall und RunCall:
onText?: (delta: string) => void;
```

- **Form:** ein Delta-Callback, kein AsyncIterator und kein zweiter
  Rückgabetyp. Rückgabewerte, `usage`, `onUsage`, Fallback und Retry bleiben
  exakt wie heute — `onText` ist reine Beobachtung, das Endergebnis bleibt
  die Wahrheit. Ein Iterator-API (`for await`) würde den Rückgabetyp jeder
  Methode spalten und Fallback/Retry-Semantik neu definieren; das ist den
  Preis für einen Ladeindikator nicht wert. Für ein SSE-Relay im BFF reicht
  der Callback vollständig (`onText: (d) => res.write(...)`).
- **Geltungsbereich v1:** `ai.text()` streamt den Text; `ai.run()` streamt
  die Textanteile jedes Modell-Zugs (auch die Zwischenzüge — genau da lebt
  die Totenstille). `ai.object()` streamt **nicht**: die Repair-Schleife und
  der aggressive Parser (`src/parse.ts`) brauchen das vollständige Dokument,
  partielles JSON zu streamen wäre gelogen.
- **Wire:** OpenAI `stream: true` + `stream_options: { include_usage: true }`,
  Deltas aus `choices[0].delta`, Tool-Call-Fragmente per `index`
  zusammensetzen. Anthropic: der schon vorhandene `messages.stream()`-Pfad
  bekommt einen Text-Delta-Hook — die Infrastruktur liegt da
  (`src/providers/anthropic.ts:32-40`), sie ist nur nicht angeschlossen.
- **Provider-Vertrag:** `onText?` wandert als optionales Feld auf
  `TextRequest`/`ToolsRequest`. Ein Provider, der es ignoriert, ist korrekt
  (er liefert eben erst am Ende) — Streaming ist eine Capability, keine
  Pflicht. Kein neues Interface-Mitglied.
- **Abbruch:** mitten im Strom → `CoaxAbortError` wie heute; was der
  Callback schon gesehen hat, hat er gesehen — das Endergebnis des
  abgebrochenen Aufrufs gibt es nicht.
- **Grenze:** `onText` feuert auch während eines Fallback-Zweitversuchs
  erneut von vorn. Wer an die UI relayt, muss beim Retry/Fallback den Puffer
  verwerfen — dafür bekommt der Callback später ggf. ein Meta-Argument;
  v1 dokumentiert es nur.

---

## Umsetzungsplan

- **v0.6.0:** Punkte 1, 2, 3, 5, 6 — zusammen bringen sie den Lauf von
  ~19,5 auf ~3–5 Minuten und killen den Extra-Durchgang; 5 ist der
  Verhaltens-Breaking-Change, der den Minor-Sprung rechtfertigt.
- **v0.6.x:** Punkte 4 und 7 — additiv, kein Bruch.
- **v0.7.0:** Punkt 8 (Streaming) — additiv, aber eigener Testaufwand
  (Delta-Zusammensetzung, Abbruch mitten im Strom), deshalb separat.

---

## Nicht auf dieser Liste

Unsere Baustelle, nicht die von coax:

- Änderungsabgleich als deterministischer Einzelaufruf statt Agentenlauf.
- Bündelung mehrerer Werkzeuge pro Runde (umgesetzt, noch nicht committet).
- Durchsatz des Gateways selbst.
