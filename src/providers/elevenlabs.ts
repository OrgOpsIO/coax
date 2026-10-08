import {
  emptyUsage,
  withBilledUsage,
  type AudioFormat,
  type Provider,
  type ProviderResponse,
  type SpeakRequest,
  type SpeakResponse,
  type SpeakStreamResponse,
  type TranscribeRequest,
  type TranscribeResponse,
  type TranscribeTokenRequest,
  type TranscribeTokenResponse,
  type TranscriptWord,
  type Usage,
} from "../types";
import { CoaxAbortError, CoaxUnsupportedError } from "../client";
import { EXTENSIONS } from "./audio";

export interface ElevenLabsOptions {
  /** The model of the reference: a TTS id (`eleven_*`) for speak, a Scribe id (`scribe_*`) for transcribe. */
  model: string;
  /** Required unless `client` is given. coax never falls back to the SDK's `ELEVENLABS_API_KEY` env lookup. */
  apiKey?: string;
  /** Inject an existing SDK client (an `ElevenLabsClient`). Otherwise coax lazily constructs one from `apiKey`
   *  (the SDK ships inside coax and is imported only then). */
  client?: ElevenLabsClientLike;
  /** Host root, without `/v1` — e.g. a data-residency host. Default: the SDK's `https://api.elevenlabs.io`. */
  baseURL?: string;
  /** Headers sent with every request (per-call `headers` are merged over these). */
  headers?: Record<string, string>;
  /** Default voice id for `speak`; a per-call `voice` wins. ElevenLabs has no default voice of its own. */
  voice?: string;
}

/**
 * The part of the SDK's `ElevenLabsClient` coax calls, typed structurally: the SDK's own declarations do not
 * compile with `skipLibCheck: false` (2.71.0), and naming them here would break every consumer of coax that
 * type-checks its libraries — ElevenLabs user or not. A real `ElevenLabsClient` fits as it is.
 */
interface ElevenLabsClientLike {
  // `stream` and `tokens` are optional so a hand-made client that compiled before they existed still compiles; a
  // call that needs one it lacks fails with a plain Error naming it, not a TypeError from deep inside.
  textToSpeech: { convert(voiceId: string, request: unknown, requestOptions?: unknown): unknown; stream?(voiceId: string, request: unknown, requestOptions?: unknown): unknown };
  speechToText: { convert(request: unknown, requestOptions?: unknown): unknown };
  tokens?: { singleUse: { create(tokenType: string, requestOptions?: unknown): unknown } };
}

type RequestOptions ={ headers?: Record<string, string>; abortSignal?: AbortSignal };
type RawWord = { text: string; start?: number | null; end?: number | null; type: string; speakerId?: string | null };
type SpeechCall = (
  voiceId: string,
  body: Record<string, unknown>,
  options?: RequestOptions,
) => { withRawResponse(): Promise<{ data: ReadableStream<Uint8Array>; rawResponse: { headers: Headers } }> };
type AnyClient = {
  textToSpeech: { convert: SpeechCall; stream?: SpeechCall };
  speechToText: {
    convert(body: Record<string, unknown>, options?: RequestOptions): Promise<{ text?: unknown; words?: RawWord[]; audioDurationSecs?: number }>;
  };
  tokens?: { singleUse: { create(tokenType: "realtime_scribe", options?: RequestOptions): Promise<{ token?: unknown }> } };
};

/**
 * coax's codec → the ElevenLabs `output_format` variant. One fixed variant per codec: the API's own
 * default for mp3; Opus has one rate, at the mp3 default's bitrate; wav/pcm at 24 kHz, the highest
 * rate every tier is cleared for (44.1 kHz needs Pro). ElevenLabs offers no aac or flac.
 */
const FORMATS: Partial<Record<AudioFormat, { outputFormat: string; mediaType: string }>> = {
  mp3: { outputFormat: "mp3_44100_128", mediaType: "audio/mpeg" },
  opus: { outputFormat: "opus_48000_128", mediaType: "audio/opus" },
  wav: { outputFormat: "wav_24000", mediaType: "audio/wav" },
  pcm: { outputFormat: "pcm_24000", mediaType: "audio/pcm" },
};

/** The `character-cost` response header is what the call was billed; absent or unparsable → not reported. */
function characters(headers: Headers): number | undefined {
  const raw = headers.get("character-cost")?.trim();
  if (!raw) return undefined;
  const n = Number(raw);
  return Number.isFinite(n) && n >= 0 ? n : undefined;
}

/**
 * The SDK wraps a failed fetch (undici: TypeError "fetch failed", the socket error as its cause) into an
 * ElevenLabsError with no status and no code, so coax's retry cannot tell a dropped connection from a bad
 * request. Lift the network code from the cause chain onto the error; `isTransient` decides as for any vendor
 * (and `withRetry` never retries an aborted call, whatever its code).
 */
function liftNetworkCode(err: unknown): unknown {
  const e = err as (Error & { code?: unknown }) | undefined;
  if (!(e instanceof Error) || e.code != null) return err;
  let cause: unknown = e.cause;
  for (let depth = 0; depth < 4 && cause && typeof cause === "object"; depth++) {
    const code = (cause as { code?: unknown }).code;
    if (typeof code === "string") {
      e.code = code;
      break;
    }
    cause = (cause as { cause?: unknown }).cause;
  }
  return err;
}

/**
 * A speech body that failed after the header: coax books what `character-cost` said, as for a completed speech
 * (whether ElevenLabs bills a cut-off speech is not documented). An abort becomes the provider's own
 * CoaxAbortError, so the reported characters survive `abortedBy()`; anything else (undici's
 * "terminated" on a dropped connection) goes through as it is — not lifted, so never retried into a second bill.
 */
function afterHeader(err: unknown, signal: AbortSignal | undefined, usage: Usage, billed: boolean): unknown {
  const failure = signal?.aborted && !(err instanceof CoaxAbortError) ? new CoaxAbortError(usage, err) : err;
  if (billed && typeof failure === "object" && failure !== null) withBilledUsage(failure, usage);
  return failure;
}

// The SDK's own derivation of the realtime endpoint from the REST host (2.71.0 wrapper/realtime/scribe.js l.78–86).
const realtimeUrl = (baseURL?: string): string =>
  (baseURL ?? "https://api.elevenlabs.io").replace(/\/+$/, "").replace(/^https?:\/\//i, (m) => (m.toLowerCase() === "https://" ? "wss://" : "ws://")) +
  "/v1/speech-to-text/realtime";

/** Spoken words only (no `spacing` / `audio_event` entries), with timing and the speaker label if any. */
function toWords(raw: RawWord[]): TranscriptWord[] {
  const out: TranscriptWord[] = [];
  for (const w of raw) {
    // A word without timing cannot be a TranscriptWord; it is still in `text`.
    if (w.type !== "word" || typeof w.start !== "number" || typeof w.end !== "number") continue;
    out.push({ text: w.text, start: w.start, end: w.end, ...(w.speakerId ? { speaker: w.speakerId } : {}) });
  }
  return out;
}

/**
 * ElevenLabs — voice only: `speak` and `speakStream` (text-to-speech), `transcribe` (Scribe, batch) and
 * `transcribeToken` (Scribe realtime, for the browser). It serves no text or structured output, so those reject
 * with CoaxUnsupportedError, as a missing optional capability does.
 */
export function elevenlabs(opts: ElevenLabsOptions): Provider {
  // The SDK reads ELEVENLABS_API_KEY when no key is passed — a key coax was not configured with must never be used.
  if (!opts.client && !opts.apiKey) throw new Error(`coax: provider "elevenlabs" needs an apiKey`);
  let client: AnyClient | undefined = opts.client as unknown as AnyClient | undefined;

  async function getClient(): Promise<AnyClient> {
    if (client) return client;
    const { ElevenLabsClient } = await import("@elevenlabs/elevenlabs-js");
    // maxRetries 0: coax's retrying() already retries transient errors; both would multiply the attempts.
    client = new ElevenLabsClient({ apiKey: opts.apiKey, maxRetries: 0, ...(opts.baseURL ? { baseUrl: opts.baseURL } : {}) }) as unknown as AnyClient;
    return client;
  }

  const requestOptions = (headers: Record<string, string> | undefined, signal: AbortSignal | undefined): RequestOptions | undefined => {
    const merged = { ...opts.headers, ...headers };
    const out: RequestOptions = {
      ...(Object.keys(merged).length ? { headers: merged } : {}),
      ...(signal ? { abortSignal: signal } : {}),
    };
    return Object.keys(out).length ? out : undefined;
  };

  // Every check runs before the SDK is loaded: the vendor is never billed for a call coax won't honour.
  function speech(req: SpeakRequest): { voice: string; variant: { outputFormat: string; mediaType: string }; body: Record<string, unknown> } {
    const voice = req.voice || opts.voice;
    if (!voice) throw new Error("coax: elevenlabs needs a voice id — pass `voice` to ai.speak() or set `voice` on the endpoint");
    if (req.instructions) throw new CoaxUnsupportedError("delivery instructions (`instructions`)", "elevenlabs");
    const format = req.format ?? "mp3";
    const variant = FORMATS[format];
    if (!variant) throw new CoaxUnsupportedError(`${format} output`, "elevenlabs");
    // The documented range; what the API does outside it (error or clamp) is not — a clamp would change the call.
    if (req.speed != null && !(req.speed >= 0.7 && req.speed <= 1.2)) {
      throw new Error(`coax: elevenlabs speed must be between 0.7 and 1.2 (got ${req.speed})`);
    }
    const body = {
      text: req.input,
      modelId: opts.model,
      outputFormat: variant.outputFormat,
      ...(req.language ? { languageCode: req.language } : {}),
      ...(req.speed != null ? { voiceSettings: { speed: req.speed } } : {}),
    };
    return { voice, variant, body };
  }

  return {
    name: "elevenlabs",
    model: opts.model,

    async structured(): Promise<ProviderResponse> {
      throw new CoaxUnsupportedError("structured output", "elevenlabs");
    },

    async text(): Promise<ProviderResponse> {
      throw new CoaxUnsupportedError("text generation", "elevenlabs");
    },

    async speak(req: SpeakRequest): Promise<SpeakResponse> {
      const { voice, variant, body } = speech(req);
      const c = await getClient();
      const { data, rawResponse } = await c.textToSpeech
        .convert(voice, body, requestOptions(req.headers, req.signal))
        .withRawResponse()
        .catch((err: unknown) => {
          throw liftNetworkCode(err);
        });
      // The SDK resolves at the header (measured), so the bill is known before the body is read — and survives a
      // body that breaks or is aborted after it.
      const billed = characters(rawResponse.headers);
      const usage: Usage = { ...emptyUsage(), ...(billed != null ? { characters: billed } : {}) };
      let audio: Uint8Array;
      try {
        audio = new Uint8Array(await new Response(data).arrayBuffer());
      } catch (err) {
        throw afterHeader(err, req.signal, usage, billed != null);
      }
      // Never an empty success: no audio is a failure, whatever the status said — and it still cost what the header says.
      if (audio.byteLength === 0) {
        const err = new Error("coax: elevenlabs returned no audio");
        throw billed != null ? withBilledUsage(err, usage) : err;
      }
      return { audio, mediaType: variant.mediaType, usage, model: opts.model };
    },

    async speakStream(req: SpeakRequest): Promise<SpeakStreamResponse> {
      const { voice, variant, body } = speech(req);
      const c = await getClient();
      if (!c.textToSpeech.stream) throw new Error("coax: the ElevenLabs client passed as `client` has no textToSpeech.stream");
      const { data, rawResponse } = await c.textToSpeech
        .stream(voice, body, requestOptions(req.headers, req.signal))
        .withRawResponse()
        .catch((err: unknown) => {
          throw liftNetworkCode(err);
        });
      // Whether /stream sends character-cost is not documented: when it does, it is the bill — known before the audio.
      const billed = characters(rawResponse.headers);
      const usage: Usage = { ...emptyUsage(), ...(billed != null ? { characters: billed } : {}) };
      async function* audio(): AsyncGenerator<Uint8Array, Usage, void> {
        let bytes = 0;
        try {
          // A return() while suspended at `yield` leaves this loop, which cancels the body and closes the connection.
          for await (const chunk of data) {
            if (!chunk.byteLength) continue;
            bytes += chunk.byteLength;
            yield chunk;
          }
        } catch (err) {
          throw afterHeader(err, req.signal, usage, billed != null);
        }
        if (bytes === 0) {
          const err = new Error("coax: elevenlabs returned no audio");
          throw billed != null ? withBilledUsage(err, usage) : err;
        }
        return usage;
      }
      return { mediaType: variant.mediaType, model: opts.model, audio: audio(), ...(billed != null ? { billed: usage } : {}) };
    },

    async transcribeToken(req: TranscribeTokenRequest): Promise<TranscribeTokenResponse> {
      const c = await getClient();
      if (!c.tokens?.singleUse?.create) throw new Error("coax: the ElevenLabs client passed as `client` has no tokens.singleUse.create");
      // The one token type the row asks for; batch_scribe / tts_websocket tokens are not coax's to hand out.
      const res = await c.tokens.singleUse.create("realtime_scribe", requestOptions(req.headers, req.signal)).catch((err: unknown) => {
        throw liftNetworkCode(err);
      });
      if (typeof res?.token !== "string" || !res.token) throw new Error("coax: elevenlabs returned no token");
      // Issuing has no documented cost and coax never sees the session it opens: zero units, reported once, so an
      // app can still count the tokens it handed out.
      return { token: res.token, url: realtimeUrl(opts.baseURL), usage: emptyUsage(), model: opts.model };
    },

    async transcribe(req: TranscribeRequest): Promise<TranscribeResponse> {
      // ElevenLabs has no free-text context; keyterms/transcript_edit mean (and cost) something else.
      if (req.prompt) throw new CoaxUnsupportedError("a transcription prompt (`prompt`)", "elevenlabs");
      const { mediaType } = req.audio;
      const filename = req.audio.filename ?? (mediaType ? `audio.${EXTENSIONS[mediaType] ?? "wav"}` : "audio");
      const c = await getClient();
      const res = await c.speechToText
        .convert(
          {
            modelId: opts.model,
            file: { data: req.audio.data, filename, ...(mediaType ? { contentType: mediaType } : {}) },
            ...(req.language ? { languageCode: req.language } : {}),
            ...(req.speakers ? { diarize: true } : {}),
            // Keep "(laughter)"-style tags out of `text`, so it holds what was said — as on every other vendor.
            tagAudioEvents: false,
          },
          requestOptions(req.headers, req.signal),
        )
        .catch((err: unknown) => {
          throw liftNetworkCode(err);
        });
      const seconds = res.audioDurationSecs;
      const billed = typeof seconds === "number" && Number.isFinite(seconds);
      const usage: Usage = { ...emptyUsage(), ...(billed ? { audioSeconds: seconds } : {}) };
      // The multichannel and webhook shapes (never requested by coax) carry no top-level text; a duration on them was billed.
      if (typeof res.text !== "string") {
        const err = new Error("coax: elevenlabs returned no transcript text");
        throw billed ? withBilledUsage(err, usage) : err;
      }
      const words = Array.isArray(res.words) ? toWords(res.words) : undefined;
      return { text: res.text, ...(words ? { words } : {}), usage, model: opts.model };
    },
  };
}
